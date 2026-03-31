
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
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from .explorer_base import BG, FG, GRID, _ExplorerTab


# ---------------------------------------------------------------------------
# Pure computation (background thread)
# ---------------------------------------------------------------------------

def _extract_traces(p, ns, annot_class, channels, pre_secs, post_secs, align_to, baseline):
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
    STAGE_CLASSES_SKIP = {"N1", "N2", "N3", "R", "W", "L", "?"}

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

    # ---- common grid ---------------------------------------------------
    # 200 points over the window (sample-rate independent for the grid)
    t_grid = np.linspace(-pre_secs, post_secs, 400)

    traces_out: dict[str, list] = {ch: [] for ch in channels}
    sr_out: dict[str, float]    = {}

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
        sr = len(t_all) / float(ns)
        sr_out[ch] = sr

        for t_ref in t_aligns:
            t0, t1 = t_ref - pre_secs, t_ref + post_secs
            mask = (t_all >= t0) & (t_all <= t1)
            if np.sum(mask) < 5:
                continue
            t_seg = t_all[mask] - t_ref
            v_seg = v_all[mask]

            # baseline subtract (mean of pre-event window)
            if baseline and pre_secs > 0:
                pre_mask = t_seg < 0
                if np.any(pre_mask):
                    v_seg = v_seg - float(np.mean(v_seg[pre_mask]))

            # interpolate onto common grid
            v_interp = np.interp(t_grid, t_seg, v_seg,
                                 left=np.nan, right=np.nan)
            traces_out[ch].append(v_interp)

    # ---- summary stats -------------------------------------------------
    mean_out  = {}
    ci_lo_out = {}
    ci_hi_out = {}

    for ch in channels:
        segs = traces_out[ch]
        if not segs:
            continue
        mat = np.vstack(segs)         # shape (n_events, n_grid)
        m   = np.nanmean(mat, axis=0)
        se  = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(mat), axis=0))
        mean_out[ch]  = m
        ci_lo_out[ch] = m - 1.96 * se
        ci_hi_out[ch] = m + 1.96 * se

    return {
        "traces":   traces_out,
        "t_grid":   t_grid,
        "mean":     mean_out,
        "ci_lo":    ci_lo_out,
        "ci_hi":    ci_hi_out,
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
        combo_ch.setMinimumWidth(220)
        combo_ch.setToolTip("EDF channels to extract (select multiple)")

        rl1.addWidget(QLabel("Annotation:")); rl1.addWidget(combo_ann)
        rl1.addWidget(QLabel("Channels:")); rl1.addWidget(combo_ch, 1)
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

        btn_render = QPushButton("Render"); btn_render.setFixedWidth(80)
        btn_render.setToolTip("Extract signal windows and draw traces")

        rl2.addWidget(QLabel("Pre:")); rl2.addWidget(spin_pre)
        rl2.addWidget(QLabel("Post:")); rl2.addWidget(spin_post)
        rl2.addWidget(QLabel("Align:")); rl2.addWidget(combo_align)
        rl2.addWidget(chk_baseline)
        rl2.addStretch(1); rl2.addWidget(btn_render)

        # canvas host
        canvas_host = QFrame()
        canvas_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas_host.setFrameShape(QFrame.NoFrame)
        canvas_host.setLayout(QVBoxLayout())
        canvas_host.layout().setContentsMargins(0,0,0,0)
        self._canvas_host = canvas_host

        outer.addWidget(row1); outer.addWidget(row2); outer.addWidget(canvas_host, 1)

        # store
        self._root        = root
        self._combo_ann   = combo_ann
        self._combo_ch    = combo_ch
        self._spin_pre    = spin_pre
        self._spin_post   = spin_post
        self._combo_align = combo_align
        self._chk_base    = chk_baseline

        # wire
        btn_refresh.clicked.connect(self.refresh_controls)
        btn_render.clicked.connect(self._render_trigger)
        self._save_btn = QPushButton("Export…"); self._save_btn.setFixedWidth(80)
        rl1.addWidget(self._save_btn)
        self._save_btn.clicked.connect(self._save_figure)

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
        if not self._start_work("Extracting waveforms…"):
            return

        pre      = float(self._spin_pre.value())
        post     = float(self._spin_post.value())
        align_to = self._combo_align.currentData()
        baseline = self._chk_base.isChecked()
        ns       = float(getattr(self.ctrl, "ns", 0.0))

        fut = self.ctrl._exec.submit(
            _extract_traces, p, ns, ann, chs, pre, post, align_to, baseline)
        def _done(_f=fut):
            try:
                self._sig_ok.emit(_f.result())
            except Exception:
                self._sig_err.emit(traceback.format_exc())
        fut.add_done_callback(_done)

    def _on_ok(self, result):
        try:
            self._draw(result)
        finally:
            self._end_work()

    def _on_err(self, tb_str):
        try:
            QtWidgets.QMessageBox.critical(
                self._root, "Waveform error", tb_str[:800])
        finally:
            self._end_work()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self, result):
        channels  = result["channels"]
        t_grid    = result["t_grid"]
        traces    = result["traces"]
        mean_d    = result["mean"]
        ci_lo     = result["ci_lo"]
        ci_hi     = result["ci_hi"]
        n_ev      = result["n_events"]
        ann_cls   = result["annot_class"]

        chs_with_data = [ch for ch in channels if ch in mean_d]
        if not chs_with_data:
            self._render_empty("No signal data extracted.\n"
                               "Check that channels are loaded and event times are valid.")
            return

        n = len(chs_with_data)
        canvas = self._ensure_canvas()
        fig = canvas.figure; fig.clear(); fig.patch.set_facecolor(BG)

        axes = fig.subplots(n, 1, squeeze=False)
        fig.subplots_adjust(hspace=0.4, left=0.10, right=0.97,
                            top=0.90, bottom=0.10)
        fig.suptitle(f"Peri-event waveform  |  '{ann_cls}'  ({n_ev} events)",
                     color=FG, fontsize=10, y=0.97)

        colors = ["#4cc9f0", "#f9844a", "#06d6a0", "#a78bfa",
                  "#ffd166", "#f72585", "#90be6d", "#ff6b6b"]

        for ch_idx, ch in enumerate(chs_with_data):
            ax  = axes[ch_idx][0]
            col = colors[ch_idx % len(colors)]
            ax.set_facecolor(BG)

            # Individual traces (very transparent)
            for seg in traces.get(ch, []):
                ax.plot(t_grid, seg, color=col, linewidth=0.4, alpha=0.15)

            # Mean ± CI
            m   = mean_d[ch]
            lo  = ci_lo[ch]
            hi  = ci_hi[ch]
            ax.fill_between(t_grid, lo, hi, color=col, alpha=0.25)
            ax.plot(t_grid, m, color=col, linewidth=1.8)

            # Event-onset line
            ax.axvline(0, color="#ffffff", lw=0.7, ls="--", alpha=0.55)
            ax.axhline(0, color=GRID, lw=0.4, alpha=0.7)

            self._style_ax(ax, title=ch, ylabel="µV" if ch_idx == 0 else "")
            if ch_idx < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time relative to event (s)", color=FG, fontsize=8)

        canvas.draw()
