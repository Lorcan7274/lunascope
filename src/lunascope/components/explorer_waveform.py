
#  --------------------------------------------------------------------
#
#  This file is part of Luna.
#
#  LUNA is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Luna is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Luna. If not, see <http:#www.gnu.org/licenses/>.
#
#  Please see LICENSE.txt for more details.
#
#  --------------------------------------------------------------------

#  --------------------------------------------------------------------
#  Luna / Lunascope  —  Explorer: Waveform (peri-event traces) tab
#  --------------------------------------------------------------------

"""Time-locked peri-event waveform viewer for the currently attached record.

For each event of a chosen annotation class, slices a window of one or
more EDF channels around the event onset / midpoint / offset.  Draws
individual thin traces plus mean ± 95 % CI on top.
"""

import traceback

import numpy as np
import pandas as pd

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from .explorer_base import BG, FG, GRID, _ExplorerTab

TRACE_SUMMARY_THRESHOLD = 2000
QUANTILE_SUBSAMPLE_LIMIT = 4000


def _apply_transform(mat, transform_mode):
    if transform_mode == "rectified":
        return np.abs(mat)
    return mat


# ---------------------------------------------------------------------------
# Pure computation (background thread)
# ---------------------------------------------------------------------------

def _extract_traces(p, ns, annot_class, channels, pre_secs, post_secs, align_to, baseline,
                    transform_mode, summary_mode):
    """
    Extract peri-event signal windows.

    Returns dict:
        traces    – {ch: list of (t_rel np.ndarray, values np.ndarray)}
        t_grid    – common relative-time grid
        mean      – {ch: np.ndarray}
        ci_lo/hi  – {ch: np.ndarray}   (95% CI via ±1.96 SE)
        n_events  – int
        sr        – {ch: float}
    """
    try:
        hdr = p.headers()
    except Exception:
        hdr = None

    # ---- events --------------------------------------------------------
    try:
        ev = p.fetch_annots([annot_class])
    except Exception as e:
        raise RuntimeError(f"Could not fetch annotations for  '{annot_class}': {e}")

    if ev is None or ev.empty:
        raise RuntimeError(f"No events found for annotation  '{annot_class}'.")

    # normalise column names
    col_map = {}
    for col in ev.columns:
        lc = col.lower()
        if lc in ("class", "annotation"): col_map[col] = "Class"
        elif lc == "start":               col_map[col] = "Start"
        elif lc in ("stop", "end"):       col_map[col] = "Stop"
    if col_map:
        ev = ev.rename(columns=col_map)

    ev["Start"] = pd.to_numeric(ev.get("Start", pd.Series()), errors="coerce")
    ev["Stop"]  = pd.to_numeric(ev.get("Stop",  pd.Series()), errors="coerce")
    ev = ev.dropna(subset=["Start", "Stop"])
    if ev.empty:
        raise RuntimeError("No valid event times found.")

    # Alignment point
    if align_to == "start":
        t_aligns = ev["Start"].values.astype(float)
    elif align_to == "stop":
        t_aligns = ev["Stop"].values.astype(float)
    else:  # midpoint
        t_aligns = ((ev["Start"].values + ev["Stop"].values) / 2.0).astype(float)

    n_events = len(t_aligns)

    traces_out: dict[str, list] = {ch: [] for ch in channels}
    sr_out: dict[str, float]    = {}
    t_grid_out: dict[str, np.ndarray] = {}

    for ch in channels:
        # Fetch full-recording signal for this channel once
        try:
            idx = p.s2i([(0.0, float(ns))])
            raw = p.slice(idx, chs=ch, time=True)
            if raw is None or raw[1] is None or len(raw[1]) == 0:
                continue
            arr  = raw[1]
            t_all = arr[:, 0].astype(float)
            v_all = arr[:, 1].astype(float)
        except Exception:
            continue

        if len(t_all) < 2:
            continue
        header_sr = np.nan
        if hdr is not None and not hdr.empty and "CH" in hdr.columns and "SR" in hdr.columns:
            row = hdr.loc[hdr["CH"] == ch]
            if not row.empty:
                try:
                    header_sr = float(row["SR"].iloc[0])
                except Exception:
                    header_sr = np.nan
        inferred_sr = (len(t_all) / float(ns)) if ns else np.nan
        diffs = np.diff(t_all)
        positive_diffs = diffs[diffs > 0]
        sample_dt = float(np.median(positive_diffs)) if positive_diffs.size else np.nan
        effective_sr = (1.0 / sample_dt) if np.isfinite(sample_dt) and sample_dt > 0 else np.nan
        sr = effective_sr if np.isfinite(effective_sr) else (header_sr if np.isfinite(header_sr) else inferred_sr)
        sr_out[ch] = sr

        if not np.isfinite(sr) or sr <= 0:
            continue

        n_pre = int(round(float(pre_secs) * sr))
        n_post = int(round(float(post_secs) * sr))
        offsets = np.arange(-n_pre, n_post + 1, dtype=np.int64)
        expected_rel = offsets.astype(float) / float(sr)
        t_grid_out[ch] = expected_rel

        insert_idx = np.searchsorted(t_all, t_aligns, side="left")
        center_idx = np.clip(insert_idx, 0, len(t_all) - 1)
        between = (insert_idx > 0) & (insert_idx < len(t_all))
        if np.any(between):
            left_idx = insert_idx[between] - 1
            right_idx = insert_idx[between]
            choose_left = (
                np.abs(t_aligns[between] - t_all[left_idx]) <=
                np.abs(t_all[right_idx] - t_aligns[between])
            )
            center_idx[between] = np.where(choose_left, left_idx, right_idx)

        in_bounds = (
            (center_idx + offsets[0] >= 0) &
            (center_idx + offsets[-1] < len(t_all))
        )
        valid_centers = center_idx[in_bounds]

        kept_chunks = []
        dropped_gap = 0
        gap_tol = max(1e-6, abs(sample_dt) * 0.25) if np.isfinite(sample_dt) else 1e-6
        chunk_size = 2048
        for start in range(0, len(valid_centers), chunk_size):
            centers_chunk = valid_centers[start:start + chunk_size]
            idx_mat = centers_chunk[:, None] + offsets[None, :]
            segs = np.asarray(v_all[idx_mat], dtype=float)
            rel = np.asarray(t_all[idx_mat] - t_all[centers_chunk][:, None], dtype=float)
            uniform = np.all(np.abs(rel - expected_rel[None, :]) <= gap_tol, axis=1)
            if not np.any(uniform):
                dropped_gap += int(len(centers_chunk))
                continue
            segs = segs[uniform]
            dropped_gap += int(np.count_nonzero(~uniform))
            if baseline and n_pre > 0 and segs.size:
                segs = segs - segs[:, :n_pre].mean(axis=1, keepdims=True)
            segs = _apply_transform(segs, transform_mode)
            kept_chunks.append(segs)

        if kept_chunks:
            mat = np.vstack(kept_chunks)
            traces_out[ch] = [row for row in mat]

    # ---- summary stats -------------------------------------------------
    mean_out  = {}
    ci_lo_out = {}
    ci_hi_out = {}
    std_out = {}
    var_out = {}
    median_out = {}
    q05_out = {}
    q25_out = {}
    q75_out = {}
    q95_out = {}
    quantile_n_out = {}

    for ch in channels:
        segs = traces_out[ch]
        if not segs:
            continue
        mat = np.vstack(segs)         # shape (n_events, n_grid)
        valid = ~np.isnan(mat)
        counts = np.sum(valid, axis=0)

        sums = np.nansum(mat, axis=0)
        m = np.full(mat.shape[1], np.nan, dtype=float)
        valid_cols = counts > 0
        m[valid_cols] = sums[valid_cols] / counts[valid_cols]

        se = np.full(mat.shape[1], np.nan, dtype=float)
        se[counts == 1] = 0.0
        multi_cols = counts > 1
        if np.any(multi_cols):
            centered = np.where(valid, mat - m, 0.0)
            ss = np.sum(centered * centered, axis=0)
            var = np.full(mat.shape[1], np.nan, dtype=float)
            var[multi_cols] = ss[multi_cols] / (counts[multi_cols] - 1)
            se[multi_cols] = np.sqrt(var[multi_cols] / counts[multi_cols])
        else:
            var = np.full(mat.shape[1], np.nan, dtype=float)

        mean_out[ch]  = m
        std = np.sqrt(var)
        std_out[ch] = std
        var_out[ch] = var
        ci_lo_out[ch] = m - 1.96 * se
        ci_hi_out[ch] = m + 1.96 * se
        if summary_mode == "mean_quantiles":
            if mat.shape[0] > QUANTILE_SUBSAMPLE_LIMIT:
                idx = np.linspace(0, mat.shape[0] - 1, QUANTILE_SUBSAMPLE_LIMIT, dtype=int)
                qmat = mat[idx]
            else:
                qmat = mat
            quantile_n_out[ch] = int(qmat.shape[0])
            median_out[ch] = np.nanmedian(qmat, axis=0)
            q05_out[ch] = np.nanpercentile(qmat, 5, axis=0)
            q25_out[ch] = np.nanpercentile(qmat, 25, axis=0)
            q75_out[ch] = np.nanpercentile(qmat, 75, axis=0)
            q95_out[ch] = np.nanpercentile(qmat, 95, axis=0)
        else:
            quantile_n_out[ch] = int(mat.shape[0])

    return {
        "traces":   traces_out,
        "t_grid":   t_grid_out,
        "mean":     mean_out,
        "ci_lo":    ci_lo_out,
        "ci_hi":    ci_hi_out,
        "std":      std_out,
        "var":      var_out,
        "median":   median_out,
        "q05":      q05_out,
        "q25":      q25_out,
        "q75":      q75_out,
        "q95":      q95_out,
        "quantile_n": quantile_n_out,
        "transform_mode": transform_mode,
        "summary_mode": summary_mode,
        "n_events": n_events,
        "sr":       sr_out,
        "annot_class": annot_class,
        "channels": channels,
        }


# ---------------------------------------------------------------------------
# Tab widget
# ---------------------------------------------------------------------------

class WaveformTab(_ExplorerTab):
    """Peri-event waveform tab (single attached record)."""

    _sig_ok  = QtCore.Signal(object)
    _sig_err = QtCore.Signal(str)

    def __init__(self, ctrl, parent=None):
        super().__init__(ctrl, parent)
        self._last_result = None
        self._pending_units = {}
        self._sig_ok.connect(self._on_ok,  Qt.QueuedConnection)
        self._sig_err.connect(self._on_err, Qt.QueuedConnection)
        self._build_widget()

    # ------------------------------------------------------------------
    # Widget
    # ------------------------------------------------------------------

    def _build_widget(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 4, 6, 4); outer.setSpacing(4)

        # row 1: annotation + channels
        row1 = QWidget(); rl1 = QHBoxLayout(row1)
        rl1.setContentsMargins(0,0,0,0); rl1.setSpacing(6)

        btn_refresh = QPushButton("↻"); btn_refresh.setFixedWidth(30)
        btn_refresh.setToolTip("Reload channels/annotations from current record")

        combo_ann = QComboBox(); combo_ann.setMinimumWidth(120)
        combo_ann.setToolTip("Annotation class to use as events")

        # Multi-select channel combo (reuse soappops widget)
        from .soappops import MultiSelectComboBox
        combo_ch = MultiSelectComboBox()
        combo_ch.setMinimumWidth(140)
        combo_ch.setMaximumWidth(260)
        combo_ch.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo_ch.setToolTip("EDF channels to extract (select multiple)")

        lab_ann = QLabel("Annotation:")
        lab_ch = QLabel("Channels:")
        rl1.addWidget(lab_ann)
        rl1.addWidget(combo_ann)
        rl1.addSpacing(10)
        rl1.addWidget(lab_ch)
        rl1.addWidget(combo_ch)
        rl1.addStretch(1)
        rl1.addWidget(btn_refresh)

        # row 2: window / alignment / baseline / render
        row2 = QWidget(); rl2 = QHBoxLayout(row2)
        rl2.setContentsMargins(0,0,0,0); rl2.setSpacing(6)

        spin_pre = QDoubleSpinBox(); spin_pre.setRange(0, 300); spin_pre.setValue(2)
        spin_pre.setSuffix(" s"); spin_pre.setDecimals(1); spin_pre.setFixedWidth(72)
        spin_pre.setToolTip("Pre-event window (seconds)")

        spin_post = QDoubleSpinBox(); spin_post.setRange(0, 300); spin_post.setValue(5)
        spin_post.setSuffix(" s"); spin_post.setDecimals(1); spin_post.setFixedWidth(72)
        spin_post.setToolTip("Post-event window (seconds)")

        combo_align = QComboBox(); combo_align.setFixedWidth(90)
        for key, lbl in [("start","Start"), ("mid","Midpoint"), ("stop","Stop")]:
            combo_align.addItem(lbl, key)
        combo_align.setToolTip("Align traces to event start / midpoint / stop")

        chk_baseline = QCheckBox("Baseline subtract")
        chk_baseline.setToolTip("Subtract mean of pre-event window from each trace")
        chk_baseline.setChecked(True)

        combo_transform = QComboBox(); combo_transform.setFixedWidth(105)
        combo_transform.addItem("Raw", "raw")
        combo_transform.addItem("Rectified", "rectified")
        combo_transform.setToolTip("Apply a simple per-sample transform before summarizing")

        combo_summary = QComboBox(); combo_summary.setFixedWidth(125)
        combo_summary.addItem("Mean ± CI", "mean_ci")
        combo_summary.addItem("Mean ± SD", "mean_sd")
        combo_summary.addItem("Quantiles", "mean_quantiles")
        combo_summary.addItem("Variance", "variance")
        combo_summary.setToolTip("Summary statistic to plot across event-locked traces")

        btn_render = QPushButton("Render"); btn_render.setFixedWidth(80)
        btn_render.setToolTip("Extract signal windows and draw traces")

        rl2.addWidget(QLabel("Pre:")); rl2.addWidget(spin_pre)
        rl2.addWidget(QLabel("Post:")); rl2.addWidget(spin_post)
        rl2.addWidget(QLabel("Align:")); rl2.addWidget(combo_align)
        rl2.addWidget(QLabel("Value:")); rl2.addWidget(combo_transform)
        rl2.addWidget(QLabel("Show:")); rl2.addWidget(combo_summary)
        rl2.addWidget(chk_baseline)
        rl2.addStretch(1); rl2.addWidget(btn_render)

        # row 3: y-axis controls
        row3 = QWidget(); rl3 = QHBoxLayout(row3)
        rl3.setContentsMargins(0,0,0,0); rl3.setSpacing(6)

        chk_auto_ymin = QCheckBox("Auto min")
        chk_auto_ymin.setChecked(True)
        spin_ymin = QDoubleSpinBox()
        spin_ymin.setRange(-1_000_000_000, 1_000_000_000)
        spin_ymin.setDecimals(2)
        spin_ymin.setSingleStep(5.0)
        spin_ymin.setValue(-100.0)
        spin_ymin.setFixedWidth(92)
        spin_ymin.setEnabled(False)
        spin_ymin.setToolTip("Manual lower y-axis limit")

        chk_auto_ymax = QCheckBox("Auto max")
        chk_auto_ymax.setChecked(True)
        spin_ymax = QDoubleSpinBox()
        spin_ymax.setRange(-1_000_000_000, 1_000_000_000)
        spin_ymax.setDecimals(2)
        spin_ymax.setSingleStep(5.0)
        spin_ymax.setValue(100.0)
        spin_ymax.setFixedWidth(92)
        spin_ymax.setEnabled(False)
        spin_ymax.setToolTip("Manual upper y-axis limit")

        rl3.addWidget(QLabel("Y min:")); rl3.addWidget(spin_ymin)
        rl3.addWidget(chk_auto_ymin)
        rl3.addSpacing(12)
        rl3.addWidget(QLabel("Y max:")); rl3.addWidget(spin_ymax)
        rl3.addWidget(chk_auto_ymax)
        rl3.addStretch(1)

        # canvas host
        canvas_host = QFrame()
        canvas_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        canvas_host.setFrameShape(QFrame.NoFrame)
        canvas_host.setLayout(QVBoxLayout())
        canvas_host.layout().setContentsMargins(0,0,0,0)
        canvas_host.layout().setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self._canvas_host = canvas_host

        canvas_scroll = QScrollArea()
        canvas_scroll.setFrameShape(QFrame.NoFrame)
        canvas_scroll.setWidgetResizable(False)
        canvas_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        canvas_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        canvas_scroll.setAlignment(Qt.AlignTop)
        canvas_scroll.setStyleSheet(
            "QScrollBar:vertical { background:#0d1117; width:12px; margin:0; }"
            "QScrollBar::handle:vertical { background:#4b5563; min-height:28px; border-radius:6px; }"
            "QScrollBar::handle:vertical:hover { background:#6b7280; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0px; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:#111827; }"
        )
        canvas_scroll.setWidget(canvas_host)
        self._canvas_scroll = canvas_scroll
        canvas_scroll.destroyed.connect(self._on_canvas_scroll_destroyed)
        canvas_scroll.viewport().installEventFilter(self)

        outer.addWidget(row1); outer.addWidget(row2); outer.addWidget(row3); outer.addWidget(canvas_scroll, 1)

        # store
        self._root        = root
        self._combo_ann   = combo_ann
        self._combo_ch    = combo_ch
        self._spin_pre    = spin_pre
        self._spin_post   = spin_post
        self._combo_align = combo_align
        self._combo_transform = combo_transform
        self._combo_summary = combo_summary
        self._chk_base    = chk_baseline
        self._chk_auto_ymin = chk_auto_ymin
        self._chk_auto_ymax = chk_auto_ymax
        self._spin_ymin   = spin_ymin
        self._spin_ymax   = spin_ymax

        # wire
        btn_refresh.clicked.connect(self.refresh_controls)
        btn_render.clicked.connect(self._render_trigger)
        chk_auto_ymin.toggled.connect(self._on_y_limit_toggle)
        chk_auto_ymax.toggled.connect(self._on_y_limit_toggle)
        spin_ymin.valueChanged.connect(self._redraw_cached)
        spin_ymax.valueChanged.connect(self._redraw_cached)
        self._save_btn = QPushButton("Export…"); self._save_btn.setFixedWidth(80)
        rl1.addWidget(self._save_btn)
        self._save_btn.clicked.connect(self._save_figure)

    def _set_canvas_height(self, nrows: int | None = None):
        """Let stacked waveform plots grow vertically and scroll instead of clipping."""
        canvas = self._ensure_canvas()
        if canvas is None:
            return
        nrows = max(1, int(nrows or 1))
        # Give every row ~260 px plus a fixed header budget; for a single row
        # this provides a usable minimum so the canvas never collapses to zero.
        # Multi-row canvases are fixed-height (scroll); single-row stretches to
        # fill available space so it uses whatever the dock gives it.
        min_height = 120 + (nrows * 260) + ((nrows - 1) * 24)
        canvas.setMinimumHeight(min_height)
        canvas.setMaximumHeight(min_height if nrows > 1 else 16777215)
        if self._canvas_host is not None:
            self._canvas_host.setMinimumHeight(min_height)
            self._canvas_host.setMaximumHeight(min_height if nrows > 1 else 16777215)
        self._sync_canvas_width()

    # ------------------------------------------------------------------
    # Control refresh (call when switching to this tab)
    # ------------------------------------------------------------------

    def refresh_controls(self):
        """Repopulate annotation and channel combos from ctrl.p."""
        p = getattr(self.ctrl, "p", None)
        if p is None:
            return
        try:
            all_annots = [c for c in (p.edf.annots() or [])
                          if c not in {"N1","N2","N3","R","W","L","?"}]
        except Exception:
            all_annots = []
        cur_ann = self._combo_ann.currentText()
        self._combo_ann.blockSignals(True)
        self._combo_ann.clear()
        self._combo_ann.addItems(all_annots)
        idx = self._combo_ann.findText(cur_ann)
        if idx >= 0:
            self._combo_ann.setCurrentIndex(idx)
        self._combo_ann.blockSignals(False)

        try:
            df_h = p.headers()
            channels = df_h["CH"].tolist() if (df_h is not None and "CH" in df_h.columns) else []
        except Exception:
            channels = []
        self._combo_ch.set_items(channels)

    def _get_channel_units(self, channels):
        """Map channel name to physical unit from EDF headers when available."""
        p = getattr(self.ctrl, "p", None)
        if p is None:
            return {}
        try:
            df_h = p.headers()
        except Exception:
            return {}
        if df_h is None or "CH" not in df_h.columns:
            return {}

        unit_col = next((c for c in ("PDIM", "UNIT", "UNITS") if c in df_h.columns), None)
        if unit_col is None:
            return {}

        units = {}
        for _, row in df_h.iterrows():
            ch = str(row.get("CH", "")).strip()
            if not ch or ch not in channels:
                continue
            raw_unit = row.get(unit_col, "")
            unit = "" if pd.isna(raw_unit) else str(raw_unit).strip()
            units[ch] = unit
        return units

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render_trigger(self):
        p = getattr(self.ctrl, "p", None)
        if p is None:
            QtWidgets.QMessageBox.warning(self._root, "Waveform",
                                          "No record attached.")
            return
        ann = self._combo_ann.currentText()
        chs = self._combo_ch.checked_items()
        if not ann:
            QtWidgets.QMessageBox.warning(self._root, "Waveform",
                                          "Select an annotation class.")
            return
        if not chs:
            QtWidgets.QMessageBox.warning(self._root, "Waveform",
                                          "Select at least one channel.")
            return
        _, _, y_limits_valid = self._get_y_limits()
        if not y_limits_valid:
            QtWidgets.QMessageBox.warning(
                self._root, "Waveform",
                "Manual Y-axis minimum must be smaller than maximum."
            )
            return
        if not self._start_work("Extracting waveforms…"):
            return

        pre      = float(self._spin_pre.value())
        post     = float(self._spin_post.value())
        align_to = self._combo_align.currentData()
        baseline = self._chk_base.isChecked()
        transform_mode = self._combo_transform.currentData()
        summary_mode = self._combo_summary.currentData()
        ns       = float(getattr(self.ctrl, "ns", 0.0))
        self._pending_units = self._get_channel_units(chs)

        fut = self.ctrl._exec.submit(
            _extract_traces, p, ns, ann, chs, pre, post, align_to, baseline,
            transform_mode, summary_mode)
        def _done(_f=fut):
            try:
                self._sig_ok.emit(_f.result())
            except Exception:
                self._sig_err.emit(traceback.format_exc())
        fut.add_done_callback(_done)

    def _on_ok(self, result):
        try:
            result["units"] = dict(self._pending_units)
            self._last_result = result
            self._draw(result)
        finally:
            self._pending_units = {}
            self._end_work()

    def _on_err(self, tb_str):
        try:
            self._pending_units = {}
            QtWidgets.QMessageBox.critical(
                self._root, "Waveform error", tb_str[:800])
        finally:
            self._end_work()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _on_y_limit_toggle(self):
        self._spin_ymin.setEnabled(not self._chk_auto_ymin.isChecked())
        self._spin_ymax.setEnabled(not self._chk_auto_ymax.isChecked())
        self._redraw_cached()

    def _get_y_limits(self):
        y_min = None if self._chk_auto_ymin.isChecked() else float(self._spin_ymin.value())
        y_max = None if self._chk_auto_ymax.isChecked() else float(self._spin_ymax.value())
        if y_min is not None and y_max is not None and y_min >= y_max:
            return None, None, False
        return y_min, y_max, True

    def _redraw_cached(self, *_):
        if self._last_result is not None:
            _, _, y_limits_valid = self._get_y_limits()
            if not y_limits_valid:
                return
            self._draw(self._last_result)

    def _draw(self, result):
        channels  = result["channels"]
        t_grid    = result["t_grid"]
        traces    = result["traces"]
        mean_d    = result["mean"]
        ci_lo     = result["ci_lo"]
        ci_hi     = result["ci_hi"]
        std_d     = result.get("std", {})
        var_d     = result.get("var", {})
        median_d  = result.get("median", {})
        q05_d     = result.get("q05", {})
        q25_d     = result.get("q25", {})
        q75_d     = result.get("q75", {})
        q95_d     = result.get("q95", {})
        transform_mode = result.get("transform_mode", "raw")
        summary_mode = result.get("summary_mode", "mean_ci")
        n_ev      = result["n_events"]
        ann_cls   = result["annot_class"]
        units     = result.get("units", {})

        chs_with_data = [ch for ch in channels if ch in mean_d]
        if not chs_with_data:
            self._set_canvas_height()
            self._render_empty("No signal data extracted.\n"
                               "Check that channels are loaded and event times are valid.")
            return

        n = len(chs_with_data)
        canvas = self._ensure_canvas()
        self._set_canvas_height(n)
        fig = canvas.figure; fig.clear(); fig.patch.set_facecolor(BG)

        axes = fig.subplots(n, 1, squeeze=False)
        fig.subplots_adjust(hspace=0.4, left=0.10, right=0.97,
                            top=0.90, bottom=0.10)
        title_bits = [f"'{ann_cls}'", f"({n_ev} events)"]
        if transform_mode == "rectified":
            title_bits.append("rectified")
        if summary_mode == "variance":
            title_bits.append("variance")
        elif summary_mode == "mean_sd":
            title_bits.append("mean ± SD")
        elif summary_mode == "mean_quantiles":
            title_bits.append("quantiles")
        fig.suptitle(f"Peri-event waveform  |  {'  '.join(title_bits)}",
                     color=FG, fontsize=10, y=0.97)
        y_min, y_max, _ = self._get_y_limits()

        colors = ["#4cc9f0", "#f9844a", "#06d6a0", "#a78bfa",
                  "#ffd166", "#f72585", "#90be6d", "#ff6b6b"]

        for ch_idx, ch in enumerate(chs_with_data):
            ch_t_grid = t_grid.get(ch) if isinstance(t_grid, dict) else t_grid
            if ch_t_grid is None or len(ch_t_grid) == 0:
                continue
            ax  = axes[ch_idx][0]
            col = colors[ch_idx % len(colors)]
            ax.set_facecolor(BG)
            n_traces = len(traces.get(ch, []))
            summary_only = n_traces > TRACE_SUMMARY_THRESHOLD or summary_mode == "variance"

            if not summary_only:
                # Individual traces (very transparent)
                for seg in traces.get(ch, []):
                    ax.plot(ch_t_grid, seg, color=col, linewidth=0.4, alpha=0.15)

            if summary_mode == "variance":
                v = var_d.get(ch)
                if v is not None:
                    ax.plot(ch_t_grid, v, color=col, linewidth=2.0, alpha=0.95)
            elif summary_mode == "mean_quantiles":
                q05 = q05_d.get(ch)
                q25 = q25_d.get(ch)
                q75 = q75_d.get(ch)
                q95 = q95_d.get(ch)
                med = median_d.get(ch)
                if q05 is not None and q95 is not None:
                    ax.fill_between(ch_t_grid, q05, q95, color=col, alpha=0.12, linewidth=0)
                if q25 is not None and q75 is not None:
                    ax.fill_between(ch_t_grid, q25, q75, color=col, alpha=0.24, linewidth=0)
                if med is not None:
                    ax.plot(ch_t_grid, med, color=col, linewidth=2.0, alpha=0.95)
                if summary_only and ch in mean_d:
                    ax.plot(ch_t_grid, mean_d[ch], color="#ffffff", linewidth=0.8, alpha=0.65)
            elif summary_mode == "mean_sd":
                m = mean_d[ch]
                sd = std_d.get(ch)
                if sd is not None:
                    ax.fill_between(ch_t_grid, m - sd, m + sd, color=col, alpha=0.22)
                ax.plot(ch_t_grid, m, color=col, linewidth=1.8)
            else:
                m = mean_d[ch]
                lo = ci_lo[ch]
                hi = ci_hi[ch]
                ax.fill_between(ch_t_grid, lo, hi, color=col, alpha=0.25)
                ax.plot(ch_t_grid, m, color=col, linewidth=1.8)

            # Event-onset line
            ax.axvline(0, color="#ffffff", lw=0.7, ls="--", alpha=0.55)
            ax.axhline(0, color=GRID, lw=0.4, alpha=0.7)
            if len(ch_t_grid) >= 2:
                ax.set_xlim(float(ch_t_grid[0]), float(ch_t_grid[-1]))

            ylabel = units.get(ch, "")
            if summary_mode == "variance" and ylabel:
                ylabel = f"{ylabel}^2"
            self._style_ax(ax, title=ch, ylabel=ylabel)
            if y_min is not None or y_max is not None:
                ax.set_ylim(bottom=y_min, top=y_max)
            if ch_idx < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time relative to event (s)", color=FG, fontsize=8)

        canvas.draw()
