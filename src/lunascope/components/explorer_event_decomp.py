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
#  Luna / Lunascope  —  Explorer: Event waveform decomposition tab
#  --------------------------------------------------------------------

"""Waveform QC viewer for collections of annotated events.

The tab treats each event as a waveform vector, computes a basic SVD/PCA,
and keeps all visualization inside the Explorer dock:

* mean + component interpretation
* scatter view of event scores
* residual histogram with threshold filtering
* dimension-sorted event browser

The implementation is intentionally split into:

* event representation and extraction
* decomposition result contract
* dock-local viewer interactions

That keeps the viewer reasonably agnostic to the exact decomposition
backend and makes future multi-channel support less invasive.
"""

from __future__ import annotations

import math
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.widgets import RectangleSelector, SpanSelector

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .explorer_base import BG, FG, GRID, SEP, _ExplorerTab
from ..file_dialogs import save_file_name

DEFAULT_MAX_EVENTS = 1500
DEFAULT_NORM_POINTS = 100
DEFAULT_BROWSER_PCS = 2
MIN_EVENTS_FOR_PCA = 3
MAX_COMPONENTS = 3


def _normalize_events_df(events: pd.DataFrame | None) -> pd.DataFrame:
    if events is None or len(events) == 0:
        return pd.DataFrame()
    df = events.copy()
    col_map = {}
    for col in df.columns:
        lc = str(col).lower()
        if lc in ("class", "annotation"):
            col_map[col] = "Class"
        elif lc == "start":
            col_map[col] = "Start"
        elif lc in ("stop", "end"):
            col_map[col] = "Stop"
    if col_map:
        df = df.rename(columns=col_map)
    if "Start" not in df.columns or "Stop" not in df.columns:
        return pd.DataFrame()
    df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
    df["Stop"] = pd.to_numeric(df["Stop"], errors="coerce")
    df = df.dropna(subset=["Start", "Stop"]).copy()
    if "Class" not in df.columns:
        df["Class"] = ""
    return df.reset_index(drop=True)


def _standard_metadata_columns() -> set[str]:
    return {"Class", "Start", "Stop"}


def _format_meta_value(val) -> str:
    if pd.isna(val):
        return "NA"
    if isinstance(val, (float, np.floating)):
        return f"{float(val):.3f}"
    return str(val)


def _nearest_sample_index(times: np.ndarray, target: float) -> int:
    idx = int(np.searchsorted(times, target, side="left"))
    if idx <= 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    left = idx - 1
    if abs(times[left] - target) <= abs(times[idx] - target):
        return left
    return idx


def _infer_sr(times: np.ndarray, ns: float, hdr: pd.DataFrame | None, channel: str) -> float:
    dt = np.diff(times)
    dt = dt[dt > 0]
    if dt.size:
        sr = 1.0 / float(np.median(dt))
        if np.isfinite(sr) and sr > 0:
            return sr
    if hdr is not None and not hdr.empty and "CH" in hdr.columns and "SR" in hdr.columns:
        row = hdr.loc[hdr["CH"] == channel]
        if not row.empty:
            try:
                sr = float(row["SR"].iloc[0])
                if np.isfinite(sr) and sr > 0:
                    return sr
            except Exception:
                pass
    if ns and len(times) > 1:
        sr = len(times) / float(ns)
        if np.isfinite(sr) and sr > 0:
            return sr
    return float("nan")


def _unit_for_channel(hdr: pd.DataFrame | None, channel: str) -> str:
    if hdr is None or hdr.empty or "CH" not in hdr.columns:
        return ""
    unit_col = next((c for c in ("PDIM", "UNIT", "UNITS") if c in hdr.columns), None)
    if unit_col is None:
        return ""
    row = hdr.loc[hdr["CH"] == channel]
    if row.empty:
        return ""
    raw = row[unit_col].iloc[0]
    return "" if pd.isna(raw) else str(raw).strip()


def _evenly_spaced_indices(total: int, limit: int) -> np.ndarray:
    if total <= 0:
        return np.zeros(0, dtype=int)
    if limit <= 0 or total <= limit:
        return np.arange(total, dtype=int)
    idx = np.linspace(0, total - 1, limit)
    idx = np.unique(np.round(idx).astype(int))
    if idx.size < limit:
        missing = []
        used = set(idx.tolist())
        probe = 0
        while len(missing) + idx.size < limit and probe < total:
            if probe not in used:
                missing.append(probe)
            probe += 1
        if missing:
            idx = np.sort(np.concatenate([idx, np.asarray(missing, dtype=int)]))
    return idx[:limit]


def _resample_segment(times: np.ndarray, values: np.ndarray, start: float, stop: float,
                      n_points: int) -> np.ndarray | None:
    if n_points < 2 or stop <= start:
        return None
    if len(times) < 2 or len(values) < 2:
        return None
    uniq_t, uniq_idx = np.unique(times, return_index=True)
    if uniq_t.size < 2:
        return None
    uniq_v = values[uniq_idx]
    target = np.linspace(start, stop, n_points)
    return np.interp(target, uniq_t, uniq_v).astype(float)


def _extract_representation(times: np.ndarray, values: np.ndarray, start_idx: int, stop_idx: int,
                            anchor_idx: int, representation: str, pre_secs: float, post_secs: float,
                            sr: float, norm_points: int) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if representation == "fixed":
        n_pre = int(round(pre_secs * sr))
        n_post = int(round(post_secs * sr))
        lo = int(anchor_idx - n_pre)
        hi = int(anchor_idx + n_post)
        if lo < 0 or hi >= len(values):
            return None, None, "edge"
        waveform = values[lo:hi + 1].astype(float)
        t_axis = np.arange(-n_pre, n_post + 1, dtype=float) / float(sr)
        return waveform, t_axis, "ok"

    seg_t = times[start_idx:stop_idx + 1]
    seg_v = values[start_idx:stop_idx + 1]
    waveform = _resample_segment(seg_t, seg_v, float(times[start_idx]), float(times[stop_idx]), norm_points)
    if waveform is None:
        return None, None, "short"
    t_axis = np.linspace(0.0, 1.0, norm_points, dtype=float)
    return waveform, t_axis, "ok"


def _compute_event_decomposition(p, ns: float, annot_class: str, channel: str, representation: str,
                                 anchor_mode: str, pre_secs: float, post_secs: float,
                                 norm_points: int, max_events: int, recon_pcs: int,
                                 per_event_mean_center: bool):
    try:
        hdr = p.headers()
    except Exception:
        hdr = None

    events = _normalize_events_df(p.fetch_annots([annot_class]))
    if events.empty:
        raise RuntimeError(f"No events found for annotation '{annot_class}'.")

    idx = p.s2i([(0.0, float(ns))])
    raw = p.slice(idx, chs=channel, time=True)
    arr = None if raw is None else raw[1]
    if arr is None or len(arr) < 3:
        raise RuntimeError(f"No usable signal data returned for channel '{channel}'.")

    times = np.asarray(arr[:, 0], dtype=float)
    values = np.asarray(arr[:, 1], dtype=float)
    if len(times) != len(values) or len(times) < 3:
        raise RuntimeError(f"Signal data for channel '{channel}' is malformed.")

    sr = _infer_sr(times, float(ns), hdr, channel)
    if not np.isfinite(sr) or sr <= 0:
        raise RuntimeError(f"Could not determine sample rate for channel '{channel}'.")

    extra_cols = [c for c in events.columns if c not in _standard_metadata_columns()]
    candidates = []
    skip_counts = {"invalid_times": 0, "outside": 0, "empty": 0, "edge": 0, "short": 0}

    for event_idx, row in events.iterrows():
        start = float(row["Start"])
        stop = float(row["Stop"])
        if not np.isfinite(start) or not np.isfinite(stop):
            skip_counts["invalid_times"] += 1
            continue
        if stop < start:
            if representation == "fixed":
                stop = start
            else:
                skip_counts["invalid_times"] += 1
                continue
        if stop < times[0] or start > times[-1]:
            skip_counts["outside"] += 1
            continue

        start_idx = int(np.searchsorted(times, start, side="left"))
        stop_idx = int(np.searchsorted(times, stop, side="right")) - 1
        start_idx = max(0, min(start_idx, len(times) - 1))
        stop_idx = max(start_idx, min(stop_idx, len(times) - 1))
        if representation != "fixed" and stop_idx <= start_idx:
            skip_counts["empty"] += 1
            continue

        seg_values = values[start_idx:stop_idx + 1]
        if representation != "fixed" and seg_values.size < 2:
            skip_counts["short"] += 1
            continue

        if anchor_mode == "onset":
            anchor_idx = _nearest_sample_index(times, start)
        elif anchor_mode == "midpoint":
            anchor_idx = _nearest_sample_index(times, (start + stop) * 0.5)
        elif anchor_mode == "peak":
            anchor_idx = int(start_idx + np.nanargmax(seg_values))
        else:
            anchor_idx = int(start_idx + np.nanargmin(seg_values))

        if representation == "fixed":
            n_pre = int(round(pre_secs * sr))
            n_post = int(round(post_secs * sr))
            if (anchor_idx - n_pre) < 0 or (anchor_idx + n_post) >= len(values):
                skip_counts["edge"] += 1
                continue
        else:
            seg_t = times[start_idx:stop_idx + 1]
            if np.unique(seg_t).size < 2:
                skip_counts["short"] += 1
                continue

        amp = float(np.nanmax(seg_values) - np.nanmin(seg_values))
        meta = {str(col): row[col] for col in extra_cols}
        candidates.append({
            "orig_index": int(event_idx),
            "start": start,
            "stop": stop,
            "duration": float(stop - start),
            "amplitude": amp,
            "anchor_time": float(times[anchor_idx]),
            "start_idx": int(start_idx),
            "stop_idx": int(stop_idx),
            "anchor_idx": int(anchor_idx),
            "meta": meta,
        })

    if len(candidates) < MIN_EVENTS_FOR_PCA:
        skip_summary = ", ".join(f"{k}={v}" for k, v in skip_counts.items() if v) or "none"
        if representation == "fixed":
            mode_hint = (
                f"Current fixed window needs at least {pre_secs:g}s before and {post_secs:g}s after the "
                f"{anchor_mode} anchor. Point events such as peaks are supported in this mode."
            )
            action_hint = "Try a smaller window, a different anchor, or duration-normalized mode."
        else:
            mode_hint = (
                f"Current duration-normalized mode resamples each event to {int(norm_points)} points "
                "and requires non-zero event duration."
            )
            action_hint = "Try a different channel or annotation class."
        raise RuntimeError(
            f"Too few usable events after extraction for '{annot_class}' on '{channel}'. "
            f"Found {len(events)} matching annotations, but only {len(candidates)} usable events "
            f"(need at least {MIN_EVENTS_FOR_PCA}). "
            f"Rejected counts: {skip_summary}. {mode_hint} {action_hint}"
        )

    chosen = _evenly_spaced_indices(len(candidates), int(max_events))
    active = list(candidates)
    waveforms = []
    active_kept = []
    time_axis = None
    for ev in active:
        waveform, ev_t_axis, status = _extract_representation(
            times, values,
            int(ev["start_idx"]),
            int(ev["stop_idx"]),
            int(ev["anchor_idx"]),
            representation,
            pre_secs,
            post_secs,
            sr,
            norm_points,
        )
        if waveform is None:
            skip_counts[status] = skip_counts.get(status, 0) + 1
            continue
        if time_axis is None:
            time_axis = np.asarray(ev_t_axis, dtype=float)
        waveforms.append(waveform)
        active_kept.append(ev)

    if len(waveforms) < MIN_EVENTS_FOR_PCA:
        raise RuntimeError(
            f"Too few usable events remained after extraction "
            f"({len(waveforms)} kept, need at least {MIN_EVENTS_FOR_PCA}). "
            "Try reducing the fixed window size or changing representation."
        )

    active = active_kept

    matrix = np.vstack(waveforms).astype(float)
    if matrix.shape[0] < MIN_EVENTS_FOR_PCA:
        raise RuntimeError("Too few events available for decomposition.")

    event_offsets = matrix.mean(axis=1, keepdims=True) if per_event_mean_center else np.zeros((matrix.shape[0], 1), dtype=float)
    matrix_for_pca = matrix - event_offsets
    mean_waveform = matrix_for_pca.mean(axis=0)
    centered = matrix_for_pca - mean_waveform
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    rank = min(MAX_COMPONENTS, vt.shape[0], matrix.shape[0])
    components = vt[:rank].copy()
    scores = centered @ components.T

    denom = max(matrix.shape[0] - 1, 1)
    eigenvalues = (s ** 2) / denom if s.size else np.zeros(0, dtype=float)
    total_var = float(eigenvalues.sum()) if eigenvalues.size else 0.0
    explained = np.zeros(rank, dtype=float)
    if total_var > 0 and rank > 0:
        explained = eigenvalues[:rank] / total_var

    recon_rank = max(1, min(int(recon_pcs), rank))
    recon_centered = mean_waveform + (scores[:, :recon_rank] @ components[:recon_rank])
    recon = recon_centered + event_offsets
    residual = np.sqrt(np.mean((matrix - recon) ** 2, axis=1))

    score_stds = np.std(scores[:, :rank], axis=0, ddof=1) if matrix.shape[0] > 1 else np.ones(rank)
    score_stds = np.where(np.isfinite(score_stds) & (score_stds > 0), score_stds, 1.0)

    durations = np.asarray([ev["duration"] for ev in active], dtype=float)
    amplitudes = np.asarray([ev["amplitude"] for ev in active], dtype=float)
    starts = np.asarray([ev["start"] for ev in active], dtype=float)
    stops = np.asarray([ev["stop"] for ev in active], dtype=float)
    anchors = np.asarray([ev["anchor_time"] for ev in active], dtype=float)

    meta_columns = sorted({k for ev in active for k in ev["meta"].keys()})
    metadata = {}
    for col in meta_columns:
        metadata[col] = np.asarray([ev["meta"].get(col) for ev in active], dtype=object)

    numeric_metadata = {}
    categorical_metadata = {}
    for col, arr_vals in metadata.items():
        series = pd.to_numeric(pd.Series(arr_vals), errors="coerce")
        if series.notna().sum() >= max(3, int(math.ceil(0.5 * len(arr_vals)))):
            numeric_metadata[col] = series.to_numpy(dtype=float)
        else:
            categorical_metadata[col] = np.asarray([_format_meta_value(v) for v in arr_vals], dtype=object)

    browser_metrics = {
        "pc1": scores[:, 0] if rank >= 1 else np.zeros(matrix.shape[0], dtype=float),
        "pc2": scores[:, 1] if rank >= 2 else np.zeros(matrix.shape[0], dtype=float),
        "pc3": scores[:, 2] if rank >= 3 else np.zeros(matrix.shape[0], dtype=float),
        "residual": residual,
        "duration": durations,
        "amplitude": amplitudes,
        "start": starts,
    }

    if time_axis is None:
        raise RuntimeError("Could not establish a common waveform axis for the selected representation.")
    value_label = _unit_for_channel(hdr, channel) or "Signal"
    x_label = "Time relative to anchor (s)" if representation == "fixed" else "Normalized event progress"
    anchor_label = anchor_mode if representation == "fixed" else "duration-normalized"

    return {
        "annot_class": annot_class,
        "channel": channel,
        "representation": representation,
        "anchor_mode": anchor_mode,
        "anchor_label": anchor_label,
        "pre_secs": float(pre_secs),
        "post_secs": float(post_secs),
        "norm_points": int(norm_points),
        "per_event_mean_center": bool(per_event_mean_center),
        "sample_rate": float(sr),
        "time_axis": np.asarray(time_axis, dtype=float),
        "x_label": x_label,
        "value_label": value_label,
        "total_matching_events": int(len(events)),
        "usable_events": int(len(candidates)),
        "active_events": int(matrix.shape[0]),
        "display_event_cap": int(max_events),
        "sampled_event_positions": chosen,
        "skip_counts": skip_counts,
        "waveforms": matrix,
        "event_offsets": event_offsets.reshape(-1),
        "pca_input": matrix_for_pca,
        "reconstruction_centered": recon_centered,
        "reconstruction": recon,
        "mean_waveform": mean_waveform,
        "components": components,
        "scores": scores[:, :rank],
        "explained": explained,
        "score_stds": score_stds[:rank],
        "residual": residual,
        "durations": durations,
        "amplitudes": amplitudes,
        "starts": starts,
        "stops": stops,
        "anchors": anchors,
        "metadata": metadata,
        "numeric_metadata": numeric_metadata,
        "categorical_metadata": categorical_metadata,
        "browser_metrics": browser_metrics,
        "recon_rank": recon_rank,
    }


class EventDecompTab(_ExplorerTab):
    """Explorer tab for waveform decomposition and QC."""

    _sig_ok = QtCore.Signal(object)
    _sig_err = QtCore.Signal(str)

    def __init__(self, ctrl, parent=None):
        super().__init__(ctrl, parent)
        self._cache = {}
        self._last_result = None
        self._active_result_key = None
        self._scatter_artist = None
        self._scatter_pick_map = {}
        self._scatter_canvas_ids = []
        self._scatter_selector = None
        self._residual_selector = None
        self._scatter_filter = None
        self._residual_filter = None
        self._filtered_indices = np.zeros(0, dtype=int)
        self._current_event_idx = 0
        self._heatmap_hover_idx = None
        self._axes = {}
        self._plot_hosts = {}
        self._plot_canvases = {}
        self._heatmap_canvas_ids = []
        self._heatmap_preview_canvas = None
        self._build_widget()
        self._sig_ok.connect(self._on_ok, Qt.QueuedConnection)
        self._sig_err.connect(self._on_err, Qt.QueuedConnection)

    def _build_widget(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(4)

        row1 = QWidget()
        rl1 = QHBoxLayout(row1)
        rl1.setContentsMargins(0, 0, 0, 0)
        rl1.setSpacing(6)

        btn_refresh = QPushButton("↻")
        btn_refresh.setFixedWidth(30)
        btn_refresh.setToolTip("Reload annotation classes and channels from current record")

        combo_ann = QComboBox()
        combo_ann.setMinimumWidth(140)
        combo_ann.setToolTip("Annotation class to treat as events")

        from .soappops import MultiSelectComboBox
        combo_ch = MultiSelectComboBox()
        combo_ch.setMinimumWidth(140)
        combo_ch.setMaximumWidth(260)
        combo_ch.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo_ch.setToolTip("Select exactly one signal channel for event extraction")

        combo_mode = QComboBox()
        combo_mode.addItem("Fixed window", "fixed")
        combo_mode.addItem("Duration-normalized", "normalized")
        combo_mode.setFixedWidth(150)

        combo_anchor = QComboBox()
        combo_anchor.addItem("Onset", "onset")
        combo_anchor.addItem("Peak", "peak")
        combo_anchor.addItem("Trough", "trough")
        combo_anchor.addItem("Midpoint", "midpoint")
        combo_anchor.setFixedWidth(110)

        btn_run = QPushButton("Compute")
        btn_run.setFixedWidth(88)
        btn_run.setToolTip("Extract event waveforms and recompute decomposition")

        btn_export = QPushButton("Export…")
        btn_export.setFixedWidth(80)

        rl1.addWidget(QLabel("Annotation:"))
        rl1.addWidget(combo_ann)
        rl1.addSpacing(8)
        rl1.addWidget(QLabel("Channel:"))
        rl1.addWidget(combo_ch)
        rl1.addSpacing(8)
        rl1.addWidget(QLabel("Mode:"))
        rl1.addWidget(combo_mode)
        rl1.addWidget(QLabel("Anchor:"))
        rl1.addWidget(combo_anchor)
        rl1.addStretch(1)
        rl1.addWidget(btn_refresh)
        rl1.addWidget(btn_export)
        rl1.addWidget(btn_run)

        row2 = QWidget()
        rl2 = QHBoxLayout(row2)
        rl2.setContentsMargins(0, 0, 0, 0)
        rl2.setSpacing(6)

        spin_pre = QDoubleSpinBox()
        spin_pre.setRange(0.0, 60.0)
        spin_pre.setValue(1.0)
        spin_pre.setDecimals(2)
        spin_pre.setSuffix(" s")
        spin_pre.setFixedWidth(80)

        spin_post = QDoubleSpinBox()
        spin_post.setRange(0.0, 60.0)
        spin_post.setValue(1.0)
        spin_post.setDecimals(2)
        spin_post.setSuffix(" s")
        spin_post.setFixedWidth(80)

        spin_norm = QSpinBox()
        spin_norm.setRange(16, 2048)
        spin_norm.setValue(DEFAULT_NORM_POINTS)
        spin_norm.setFixedWidth(88)

        spin_limit = QSpinBox()
        spin_limit.setRange(100, 20000)
        spin_limit.setSingleStep(100)
        spin_limit.setValue(DEFAULT_MAX_EVENTS)
        spin_limit.setFixedWidth(92)
        spin_limit.setToolTip("Display cap hint only; PCA/SVD currently uses all usable events")

        spin_recon = QSpinBox()
        spin_recon.setRange(1, MAX_COMPONENTS)
        spin_recon.setValue(DEFAULT_BROWSER_PCS)
        spin_recon.setFixedWidth(60)

        chk_wave_center = QCheckBox("Mean-center waves")
        chk_wave_center.setToolTip("Subtract each event's own mean before PCA/SVD; browser reconstructions are offset back to raw space")

        combo_scatter_x = QComboBox()
        combo_scatter_x.setFixedWidth(92)
        combo_scatter_y = QComboBox()
        combo_scatter_y.setFixedWidth(92)

        combo_color = QComboBox()
        combo_color.setMinimumWidth(150)

        rl2.addWidget(QLabel("Pre:"))
        rl2.addWidget(spin_pre)
        rl2.addWidget(QLabel("Post:"))
        rl2.addWidget(spin_post)
        rl2.addWidget(QLabel("Norm pts:"))
        rl2.addWidget(spin_norm)
        rl2.addWidget(QLabel("Max shown:"))
        rl2.addWidget(spin_limit)
        rl2.addWidget(QLabel("Recon PCs:"))
        rl2.addWidget(spin_recon)
        rl2.addWidget(chk_wave_center)
        rl2.addSpacing(8)
        rl2.addWidget(QLabel("Scatter X:"))
        rl2.addWidget(combo_scatter_x)
        rl2.addWidget(QLabel("Y:"))
        rl2.addWidget(combo_scatter_y)
        rl2.addWidget(QLabel("Color:"))
        rl2.addWidget(combo_color, 1)

        row3 = QWidget()
        rl3 = QHBoxLayout(row3)
        rl3.setContentsMargins(0, 0, 0, 0)
        rl3.setSpacing(6)

        combo_sort = QComboBox()
        combo_sort.setMinimumWidth(130)
        combo_sort.addItem("PC1", "pc1")
        combo_sort.addItem("PC2", "pc2")
        combo_sort.addItem("PC3", "pc3")
        combo_sort.addItem("Residual", "residual")
        combo_sort.addItem("Duration", "duration")
        combo_sort.addItem("Amplitude", "amplitude")

        chk_desc = QCheckBox("Descending")
        chk_desc.setChecked(False)

        btn_prev = QPushButton("Previous")
        btn_prev.setFixedWidth(84)
        btn_next = QPushButton("Next")
        btn_next.setFixedWidth(72)

        spin_pct = QSpinBox()
        spin_pct.setRange(0, 100)
        spin_pct.setValue(50)
        spin_pct.setSuffix(" %")
        spin_pct.setFixedWidth(76)
        btn_jump = QPushButton("Jump")
        btn_jump.setFixedWidth(64)

        chk_raw = QCheckBox("Raw")
        chk_raw.setChecked(True)
        chk_recon = QCheckBox("Recon")
        chk_recon.setChecked(True)
        chk_diff = QCheckBox("Diff")
        chk_diff.setChecked(False)

        btn_clear_scatter = QPushButton("Clear scatter filter")
        btn_clear_scatter.setFixedWidth(128)
        btn_clear_resid = QPushButton("Clear residual filter")
        btn_clear_resid.setFixedWidth(132)

        rl3.addWidget(QLabel("Browser sort:"))
        rl3.addWidget(combo_sort)
        rl3.addWidget(chk_desc)
        rl3.addSpacing(8)
        rl3.addWidget(btn_prev)
        rl3.addWidget(btn_next)
        rl3.addWidget(QLabel("Percentile:"))
        rl3.addWidget(spin_pct)
        rl3.addWidget(btn_jump)
        rl3.addSpacing(8)
        rl3.addWidget(chk_raw)
        rl3.addWidget(chk_recon)
        rl3.addWidget(chk_diff)
        rl3.addStretch(1)
        rl3.addWidget(btn_clear_scatter)
        rl3.addWidget(btn_clear_resid)

        row4 = QWidget()
        rl4 = QHBoxLayout(row4)
        rl4.setContentsMargins(0, 0, 0, 0)
        rl4.setSpacing(10)

        lbl_summary = QLabel("No decomposition loaded.")
        lbl_summary.setStyleSheet("color:#9ca3af;")
        lbl_browser = QLabel("")
        lbl_browser.setStyleSheet("color:#9ca3af;")

        rl4.addWidget(lbl_summary, 1)
        rl4.addWidget(lbl_browser)

        view_tabs = QTabWidget()
        view_tabs.setDocumentMode(True)
        view_tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #30363d; background:#0d1117; }"
            "QTabBar::tab { background:#111827; color:#9ca3af; padding:6px 10px; border:1px solid #30363d; }"
            "QTabBar::tab:selected { color:#e5e7eb; background:#0f172a; }"
        )

        for key, title in (("browser", "Browser"), ("sorted", "Sorted")):
            host = QFrame()
            host.setFrameShape(QFrame.NoFrame)
            host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            host_layout = QVBoxLayout(host)
            host_layout.setContentsMargins(0, 0, 0, 0)
            self._plot_hosts[key] = host
            self._plot_canvases[key] = None
            view_tabs.addTab(host, title)

        outer.addWidget(row1)
        outer.addWidget(row2)
        outer.addWidget(row3)
        outer.addWidget(row4)
        outer.addWidget(view_tabs, 1)

        self._root = root
        self._combo_ann = combo_ann
        self._combo_ch = combo_ch
        self._combo_mode = combo_mode
        self._combo_anchor = combo_anchor
        self._spin_pre = spin_pre
        self._spin_post = spin_post
        self._spin_norm = spin_norm
        self._spin_limit = spin_limit
        self._spin_recon = spin_recon
        self._chk_wave_center = chk_wave_center
        self._combo_scatter_x = combo_scatter_x
        self._combo_scatter_y = combo_scatter_y
        self._combo_color = combo_color
        self._combo_sort = combo_sort
        self._chk_desc = chk_desc
        self._chk_raw = chk_raw
        self._chk_recon = chk_recon
        self._chk_diff = chk_diff
        self._spin_pct = spin_pct
        self._lbl_summary = lbl_summary
        self._lbl_browser = lbl_browser
        self._view_tabs = view_tabs
        self._btn_export = btn_export

        btn_refresh.clicked.connect(self.refresh_controls)
        btn_export.clicked.connect(self._on_export_clicked)
        btn_run.clicked.connect(self._compute_trigger)
        combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        combo_scatter_x.currentIndexChanged.connect(self._redraw_cached)
        combo_scatter_y.currentIndexChanged.connect(self._redraw_cached)
        combo_color.currentIndexChanged.connect(self._redraw_cached)
        combo_sort.currentIndexChanged.connect(self._on_sort_changed)
        chk_desc.stateChanged.connect(self._on_sort_changed)
        chk_raw.stateChanged.connect(self._redraw_cached)
        chk_recon.stateChanged.connect(self._redraw_cached)
        chk_diff.stateChanged.connect(self._redraw_cached)
        chk_wave_center.stateChanged.connect(self._on_decomp_option_changed)
        spin_recon.valueChanged.connect(self._on_recon_changed)
        btn_prev.clicked.connect(lambda: self._step_browser(-1))
        btn_next.clicked.connect(lambda: self._step_browser(1))
        btn_jump.clicked.connect(self._jump_percentile)
        btn_clear_scatter.clicked.connect(self._clear_scatter_filter)
        btn_clear_resid.clicked.connect(self._clear_residual_filter)
        view_tabs.currentChanged.connect(self._redraw_cached)

        self._populate_plot_controls(None)
        self._on_mode_changed()

    def _ensure_plot_canvas(self, key: str):
        canvas = self._plot_canvases.get(key)
        if canvas is not None:
            return canvas
        host = self._plot_hosts.get(key)
        if host is None:
            return None
        from .mplcanvas import MplCanvas
        canvas = MplCanvas(host)
        canvas.installEventFilter(self)
        canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        canvas.customContextMenuRequested.connect(lambda pos, c=canvas: self._context_menu_for(c, pos))
        host.layout().addWidget(canvas)
        self._plot_canvases[key] = canvas
        return canvas

    def _current_plot_key(self) -> str:
        idx = self._view_tabs.currentIndex()
        return ("browser", "sorted")[max(0, min(idx, 1))]

    def _current_canvas(self):
        return self._plot_canvases.get(self._current_plot_key())

    def _set_canvas_height(self, key: str, min_height: int):
        canvas = self._ensure_plot_canvas(key)
        if canvas is None:
            return
        canvas.setMinimumHeight(min_height)
        host = self._plot_hosts.get(key)
        if host is not None:
            host.setMinimumHeight(min_height)
        if key == "browser" and not self._scatter_canvas_ids:
            self._scatter_canvas_ids = [
                canvas.mpl_connect("pick_event", self._on_pick_event),
            ]
        if key == "sorted" and not self._heatmap_canvas_ids:
            self._heatmap_canvas_ids = [
                canvas.mpl_connect("motion_notify_event", self._on_heatmap_hover),
                canvas.mpl_connect("axes_leave_event", self._on_heatmap_leave),
            ]
        return

    def refresh_controls(self):
        p = getattr(self.ctrl, "p", None)
        if p is None:
            return
        try:
            annots = [c for c in (p.edf.annots() or []) if c not in {"N1", "N2", "N3", "R", "W", "L", "?"}]
        except Exception:
            annots = []

        cur_ann = self._combo_ann.currentText()
        self._combo_ann.blockSignals(True)
        self._combo_ann.clear()
        self._combo_ann.addItems(annots)
        idx = self._combo_ann.findText(cur_ann)
        if idx >= 0:
            self._combo_ann.setCurrentIndex(idx)
        self._combo_ann.blockSignals(False)

        try:
            df_h = p.headers()
            channels = df_h["CH"].tolist() if (df_h is not None and "CH" in df_h.columns) else []
        except Exception:
            channels = []

        checked = self._combo_ch.checked_items()
        self._combo_ch.set_items(channels, checked)

    def _on_mode_changed(self, *_):
        fixed = self._combo_mode.currentData() == "fixed"
        self._combo_anchor.setEnabled(fixed)
        self._spin_pre.setEnabled(fixed)
        self._spin_post.setEnabled(fixed)
        self._spin_norm.setEnabled(not fixed)

    def _cache_key(self):
        p = getattr(self.ctrl, "p", None)
        if p is None:
            return None
        checked = tuple(self._combo_ch.checked_items())
        return (
            id(p),
            float(getattr(self.ctrl, "ns", 0.0)),
            self._combo_ann.currentText().strip(),
            checked[0] if checked else "",
            self._combo_mode.currentData(),
            self._combo_anchor.currentData(),
            round(float(self._spin_pre.value()), 4),
            round(float(self._spin_post.value()), 4),
            int(self._spin_norm.value()),
            int(self._spin_limit.value()),
            int(self._spin_recon.value()),
            bool(self._chk_wave_center.isChecked()),
        )

    def _compute_trigger(self):
        p = getattr(self.ctrl, "p", None)
        if p is None:
            QtWidgets.QMessageBox.warning(self._root, "Event Decomposition", "No record attached.")
            return

        ann = self._combo_ann.currentText().strip()
        chs = self._combo_ch.checked_items()
        if not ann:
            QtWidgets.QMessageBox.warning(self._root, "Event Decomposition", "Select an annotation class.")
            return
        if len(chs) != 1:
            QtWidgets.QMessageBox.warning(
                self._root, "Event Decomposition",
                "Select exactly one channel for this first version of the decomposition viewer."
            )
            return

        key = self._cache_key()
        if key in self._cache:
            self._active_result_key = key
            self._scatter_filter = None
            self._residual_filter = None
            self._on_ok(self._cache[key], from_cache=True)
            return

        if not self._start_work("Computing event decomposition…"):
            return

        fut = self.ctrl._exec.submit(
            _compute_event_decomposition,
            p,
            float(getattr(self.ctrl, "ns", 0.0)),
            ann,
            chs[0],
            self._combo_mode.currentData(),
            self._combo_anchor.currentData(),
            float(self._spin_pre.value()),
            float(self._spin_post.value()),
            int(self._spin_norm.value()),
            int(self._spin_limit.value()),
            int(self._spin_recon.value()),
            bool(self._chk_wave_center.isChecked()),
        )

        def _done(_f=fut):
            try:
                self._sig_ok.emit(_f.result())
            except Exception:
                self._sig_err.emit(traceback.format_exc())

        fut.add_done_callback(_done)

    def _on_ok(self, result, from_cache=False):
        try:
            if not from_cache:
                key = self._cache_key()
                if key is not None:
                    self._cache[key] = result
                    self._active_result_key = key
                self._scatter_filter = None
                self._residual_filter = None
                self._heatmap_hover_idx = None
            self._last_result = result
            self._populate_plot_controls(result)
            self._apply_filters(reset_browser=True)
            self._draw()
        finally:
            if not from_cache:
                self._end_work()

    def _on_err(self, tb_str):
        try:
            QtWidgets.QMessageBox.critical(self._root, "Event decomposition error", tb_str[:800])
        finally:
            self._end_work()

    def _populate_plot_controls(self, result):
        scatter_dims = [("PC1", "pc1"), ("PC2", "pc2"), ("PC3", "pc3"), ("Residual", "residual"), ("Duration", "duration")]
        prev_x = self._combo_scatter_x.currentData()
        prev_y = self._combo_scatter_y.currentData()
        self._combo_scatter_x.blockSignals(True)
        self._combo_scatter_y.blockSignals(True)
        self._combo_scatter_x.clear()
        self._combo_scatter_y.clear()
        for label, key in scatter_dims:
            self._combo_scatter_x.addItem(label, key)
            self._combo_scatter_y.addItem(label, key)
        idx_x = self._combo_scatter_x.findData(prev_x)
        idx_y = self._combo_scatter_y.findData(prev_y)
        self._combo_scatter_x.setCurrentIndex(max(0, idx_x if idx_x >= 0 else self._combo_scatter_x.findData("pc1")))
        self._combo_scatter_y.setCurrentIndex(max(0, idx_y if idx_y >= 0 else self._combo_scatter_y.findData("pc2")))
        self._combo_scatter_x.blockSignals(False)
        self._combo_scatter_y.blockSignals(False)

        prev_color = self._combo_color.currentData()
        self._combo_color.blockSignals(True)
        self._combo_color.clear()
        base_items = [
            ("Residual", ("metric", "residual")),
            ("Duration", ("metric", "duration")),
            ("Amplitude", ("metric", "amplitude")),
            ("Start time", ("metric", "start")),
        ]
        for label, data in base_items:
            self._combo_color.addItem(label, data)
        if result is not None:
            for col in sorted(result.get("numeric_metadata", {}).keys()):
                self._combo_color.addItem(f"Metadata: {col}", ("numeric_meta", col))
            for col in sorted(result.get("categorical_metadata", {}).keys()):
                self._combo_color.addItem(f"Metadata: {col}", ("categorical_meta", col))
        idx = self._combo_color.findData(prev_color)
        if idx < 0:
            idx = 0
        self._combo_color.setCurrentIndex(idx)
        self._combo_color.blockSignals(False)

    def _invalidate_cache_and_redraw(self, *_):
        self._active_result_key = None
        if self._last_result is not None:
            self._draw()

    def _on_decomp_option_changed(self, *_):
        self._active_result_key = None
        if self._last_result is not None and not getattr(self.ctrl, "_busy", False):
            self._compute_trigger()

    def _context_menu_for(self, canvas, pos):
        self._canvas = canvas
        self._context_menu(pos)

    def _save_figure(self):
        canvas = self._current_canvas()
        if canvas is None:
            return
        self._canvas = canvas
        super()._save_figure()

    def _on_export_clicked(self):
        menu = QtWidgets.QMenu(self._btn_export)
        act_fig = menu.addAction("Save Figure…")
        act_waves = menu.addAction("Save Waveforms (.npz)…")
        if self._last_result is None:
            act_waves.setEnabled(False)
        picked = menu.exec(self._btn_export.mapToGlobal(self._btn_export.rect().bottomLeft()))
        if picked == act_fig:
            self._save_figure()
        elif picked == act_waves:
            self._save_waveforms_array()

    def _save_waveforms_array(self):
        result = self._last_result
        if result is None:
            QtWidgets.QMessageBox.warning(self._root, "Event Decomposition", "No decomposition loaded.")
            return

        waveforms = np.asarray(result.get("waveforms"))
        if waveforms.ndim not in (2, 3):
            QtWidgets.QMessageBox.critical(
                self._root,
                "Event Decomposition",
                f"Unsupported waveform array shape {tuple(waveforms.shape)}. Expected 2D or 3D.",
            )
            return

        fn, _ = save_file_name(
            self._root,
            "Save Waveforms (NumPy)",
            "event_decomp_waveforms.npz",
            "NumPy archive (*.npz)",
        )
        if not fn:
            return
        if not fn.lower().endswith(".npz"):
            fn = f"{fn}.npz"

        time_axis = np.asarray(result.get("time_axis", np.zeros(0, dtype=float)), dtype=float)
        channel_names = result.get("channels")
        if channel_names is None:
            one_channel = str(result.get("channel", "")).strip()
            channel_names = [one_channel] if one_channel else []
        elif isinstance(channel_names, str):
            channel_names = [channel_names]
        else:
            channel_names = [str(ch) for ch in channel_names]

        labels = {
            "layout": "waveform x sample(time) [ x channel ]",
            "axes": ["waveform", "sample(time)"] + (["channel"] if waveforms.ndim == 3 else []),
            "shape": tuple(int(v) for v in waveforms.shape),
            "channel_names": channel_names,
            "x_label": str(result.get("x_label", "")),
            "value_label": str(result.get("value_label", "")),
            "time_axis_seconds": time_axis,
        }
        try:
            np.savez_compressed(
                fn,
                waveforms=waveforms,
                labels=np.asarray(labels, dtype=object),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._root,
                "Event Decomposition",
                f"Failed to save waveform array:\n{exc}",
            )
            return
        QtWidgets.QMessageBox.information(
            self._root,
            "Event Decomposition",
            f"Saved waveform array to:\n{fn}",
        )

    def _metric_values(self, result, key: str) -> np.ndarray:
        browser_metrics = result.get("browser_metrics", {})
        vals = browser_metrics.get(key)
        if vals is None:
            return np.zeros(result["active_events"], dtype=float)
        return np.asarray(vals, dtype=float)

    def _rebuild_reconstruction(self, cached, rank: int):
        comps = cached.get("components", np.zeros((0, 0), dtype=float))
        scores = cached.get("scores", np.zeros((0, 0), dtype=float))
        mean_waveform = np.asarray(cached.get("mean_waveform", np.zeros(0, dtype=float)), dtype=float)
        offsets = np.asarray(cached.get("event_offsets", np.zeros(scores.shape[0], dtype=float)), dtype=float).reshape(-1, 1)
        rank = max(1, min(int(rank), comps.shape[0], scores.shape[1]))
        recon_centered = mean_waveform + (scores[:, :rank] @ comps[:rank])
        cached["recon_rank"] = rank
        cached["reconstruction_centered"] = recon_centered
        cached["reconstruction"] = recon_centered + offsets
        cached["residual"] = np.sqrt(np.mean((cached["waveforms"] - cached["reconstruction"]) ** 2, axis=1))
        cached["browser_metrics"]["residual"] = cached["residual"]
        return cached

    def _apply_filters(self, reset_browser=False):
        result = self._last_result
        if result is None:
            self._filtered_indices = np.zeros(0, dtype=int)
            self._current_event_idx = 0
            self._update_browser_label()
            return

        n = int(result["active_events"])
        keep = np.ones(n, dtype=bool)

        if self._scatter_filter is not None:
            scatter_keep = np.zeros(n, dtype=bool)
            scatter_keep[np.asarray(self._scatter_filter, dtype=int)] = True
            keep &= scatter_keep

        if self._residual_filter is not None:
            lo, hi = self._residual_filter
            residual = np.asarray(result["residual"], dtype=float)
            keep &= (residual >= lo) & (residual <= hi)

        order_key = self._combo_sort.currentData()
        values = self._metric_values(result, order_key)
        indices = np.where(keep)[0]
        if indices.size:
            order = np.argsort(values[indices], kind="mergesort")
            if self._chk_desc.isChecked():
                order = order[::-1]
            indices = indices[order]
        self._filtered_indices = indices

        if indices.size == 0:
            self._current_event_idx = 0
        elif reset_browser:
            self._current_event_idx = 0
        else:
            self._current_event_idx = max(0, min(self._current_event_idx, indices.size - 1))
        self._update_browser_label()

    def _update_browser_label(self):
        result = self._last_result
        if result is None:
            self._lbl_summary.setText("No decomposition loaded.")
            self._lbl_browser.setText("")
            return

        explained = result.get("explained", np.zeros(0, dtype=float))
        bits = [
            f"{result['annot_class']} / {result['channel']}",
            f"{result['representation']}",
            f"usable {result['usable_events']} of {result['total_matching_events']}",
            f"decomp {result['active_events']}",
        ]
        if result.get("per_event_mean_center"):
            bits.append("wave-centered")
        if explained.size:
            pct = ", ".join(f"PC{i + 1} {100.0 * v:.1f}%" for i, v in enumerate(explained[:3]))
            bits.append(pct)
        self._lbl_summary.setText("  |  ".join(bits))

        if self._filtered_indices.size == 0:
            self._lbl_browser.setText("Filtered browser: 0 events")
        else:
            self._lbl_browser.setText(
                f"Browser: {self._current_event_idx + 1}/{self._filtered_indices.size} "
                f"(of {result['active_events']} decomposed events)"
            )

    def _on_sort_changed(self, *_):
        self._apply_filters(reset_browser=False)
        self._draw()

    def _on_recon_changed(self, *_):
        result = self._last_result
        if result is None:
            return
        key = self._active_result_key
        if key is None:
            return
        cached = self._cache.get(key)
        if cached is None:
            return
        cached = self._rebuild_reconstruction(cached, int(self._spin_recon.value()))
        self._last_result = cached
        self._populate_plot_controls(cached)
        self._apply_filters(reset_browser=False)
        self._draw()

    def _step_browser(self, delta: int):
        if self._filtered_indices.size == 0:
            return
        self._current_event_idx = max(0, min(self._current_event_idx + int(delta), self._filtered_indices.size - 1))
        self._update_browser_label()
        self._draw()

    def _jump_percentile(self):
        if self._filtered_indices.size == 0:
            return
        pct = float(self._spin_pct.value()) / 100.0
        pos = int(round((self._filtered_indices.size - 1) * pct))
        self._current_event_idx = max(0, min(pos, self._filtered_indices.size - 1))
        self._update_browser_label()
        self._draw()

    def _clear_scatter_filter(self):
        self._scatter_filter = None
        self._heatmap_hover_idx = None
        self._apply_filters(reset_browser=True)
        self._draw()

    def _clear_residual_filter(self):
        self._residual_filter = None
        self._heatmap_hover_idx = None
        self._apply_filters(reset_browser=True)
        self._draw()

    def _selected_event_index(self) -> int | None:
        if self._filtered_indices.size == 0:
            return None
        if not (0 <= self._current_event_idx < self._filtered_indices.size):
            return None
        return int(self._filtered_indices[self._current_event_idx])

    def _redraw_cached(self, *_):
        if self._last_result is not None:
            self._draw()

    def _on_pick_event(self, event):
        if self._last_result is None:
            return
        artist = getattr(event, "artist", None)
        mapping = self._scatter_pick_map.get(artist)
        if mapping is None:
            return
        ind = getattr(event, "ind", None)
        if ind is None or len(ind) == 0:
            return
        ev_idx = int(mapping[int(ind[0])])
        matches = np.where(self._filtered_indices == ev_idx)[0]
        if matches.size:
            self._current_event_idx = int(matches[0])
        else:
            self._current_event_idx = 0
        self._update_browser_label()
        self._draw()

    def _on_scatter_select(self, eclick, erelease):
        result = self._last_result
        if result is None:
            return
        if eclick.xdata is None or erelease.xdata is None or eclick.ydata is None or erelease.ydata is None:
            return
        x0, x1 = sorted([float(eclick.xdata), float(erelease.xdata)])
        y0, y1 = sorted([float(eclick.ydata), float(erelease.ydata)])
        if abs(x1 - x0) < 1e-12 or abs(y1 - y0) < 1e-12:
            return
        xvals = self._scatter_axis_values(result, self._combo_scatter_x.currentData())
        yvals = self._scatter_axis_values(result, self._combo_scatter_y.currentData())
        keep = np.where((xvals >= x0) & (xvals <= x1) & (yvals >= y0) & (yvals <= y1))[0]
        self._scatter_filter = keep
        self._apply_filters(reset_browser=True)
        self._draw()

    def _on_residual_select(self, lo, hi):
        result = self._last_result
        if result is None:
            return
        if lo is None or hi is None:
            return
        lo_f, hi_f = sorted([float(lo), float(hi)])
        if not np.isfinite(lo_f) or not np.isfinite(hi_f):
            return
        self._residual_filter = (lo_f, hi_f)
        self._apply_filters(reset_browser=True)
        self._draw()

    def _scatter_axis_values(self, result, key: str) -> np.ndarray:
        if key in ("pc1", "pc2", "pc3"):
            scores = result.get("scores", np.zeros((result["active_events"], 0), dtype=float))
            idx = {"pc1": 0, "pc2": 1, "pc3": 2}[key]
            if scores.shape[1] > idx:
                return np.asarray(scores[:, idx], dtype=float)
            return np.zeros(result["active_events"], dtype=float)
        return self._metric_values(result, key)

    def _browser_title(self, result, ev_idx: int) -> str:
        score_bits = []
        scores = result.get("scores", np.zeros((result["active_events"], 0), dtype=float))
        for i in range(min(scores.shape[1], 3)):
            score_bits.append(f"PC{i + 1}={scores[ev_idx, i]:.3f}")

        meta_bits = [
            f"dur={result['durations'][ev_idx]:.3f}s",
            f"amp={result['amplitudes'][ev_idx]:.3f}",
            f"resid={result['residual'][ev_idx]:.4f}",
        ]
        for col, arr in result.get("categorical_metadata", {}).items():
            val = arr[ev_idx]
            if val not in ("", "NA"):
                meta_bits.append(f"{col}={val}")
                if len(meta_bits) >= 6:
                    break
        return "  |  ".join(score_bits + meta_bits)

    def _clear_plot(self, key: str, msg: str):
        canvas = self._ensure_plot_canvas(key)
        if canvas is None:
            return
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)
        ax.text(0.5, 0.5, msg, color=FG, ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.set_axis_off()
        canvas.draw()

    def _draw(self):
        result = self._last_result
        if result is None:
            for key in ("browser", "sorted"):
                self._clear_plot(key, "No decomposition loaded.")
            return

        self._update_browser_label()
        self._draw_browser_tab(result)
        self._draw_sorted_tab(result)

    def _draw_browser_tab(self, result):
        canvas = self._ensure_plot_canvas("browser")
        if canvas is None:
            return
        self._set_canvas_height("browser", 640)
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(BG)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.75, 1.0], hspace=0.26)
        ax_browser = fig.add_subplot(gs[0, 0])
        detail_gs = gs[1, 0].subgridspec(1, 3, wspace=0.22)
        ax_mean = fig.add_subplot(detail_gs[0, 0])
        ax_scatter = fig.add_subplot(detail_gs[0, 1])
        ax_hist = fig.add_subplot(detail_gs[0, 2])

        self._axes = {
            "mean": ax_mean,
            "scatter": ax_scatter,
            "hist": ax_hist,
        }

        self._draw_browser(ax_browser, result)
        self._draw_mean_components(ax_mean, result)
        self._draw_scatter(ax_scatter, result)
        self._draw_residual_hist(ax_hist, result)
        fig.suptitle(
            f"Browser  |  {result['annot_class']} / {result['channel']}  |  "
            f"{self._combo_sort.currentText()}",
            color=FG, fontsize=10, y=0.985
        )
        self._install_selectors()
        canvas.draw()

    def _draw_sorted_tab(self, result):
        canvas = self._ensure_plot_canvas("sorted")
        if canvas is None:
            return
        self._set_canvas_height("sorted", 620)
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(BG)
        gs = fig.add_gridspec(2, 1, height_ratios=[0.48, 1.52], hspace=0.12)
        ax_preview = fig.add_subplot(gs[0, 0])
        bottom_gs = gs[1, 0].subgridspec(1, 2, width_ratios=[1.75, 0.75], wspace=0.16)
        ax_heatmap = fig.add_subplot(bottom_gs[0, 0])
        exemplar_gs = bottom_gs[0, 1].subgridspec(6, 1, hspace=0.24)
        decile_axes = [fig.add_subplot(exemplar_gs[i, 0]) for i in range(6)]
        self._axes["heatmap_preview"] = ax_preview
        self._axes["heatmap"] = ax_heatmap
        self._heatmap_preview_canvas = canvas
        self._draw_heatmap_preview(ax_preview, result)
        self._draw_heatmap(ax_heatmap, result)
        self._draw_decile_gallery(decile_axes, result)
        fig.suptitle(
            f"Sorted  |  {self._combo_sort.currentText()}",
            color=FG, fontsize=10, y=0.985
        )
        canvas.draw()

    def _draw_decile_gallery(self, axes, result):
        if self._filtered_indices.size == 0:
            for ax in axes:
                ax.set_facecolor(BG)
                ax.set_axis_off()
            if axes:
                axes[0].text(0.5, 0.5, "No events pass the current filters.",
                             color=FG, ha="center", va="center", fontsize=10, transform=axes[0].transAxes)
            return

        x = result["time_axis"]
        n = self._filtered_indices.size
        lo_pos = 0 if n <= 1 else int(round(0.05 * (n - 1)))
        hi_pos = max(lo_pos, int(round(0.95 * (n - 1))))
        sample_positions = np.unique(np.round(np.linspace(lo_pos, hi_pos, min(len(axes), n))).astype(int))
        sample_indices = self._filtered_indices[sample_positions]
        cmap = plt.get_cmap("plasma")

        for i, ax in enumerate(axes):
            ax.set_facecolor(BG)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID)
            ax.tick_params(colors=FG, labelsize=6, length=2)
            if i >= len(sample_indices):
                ax.set_axis_off()
                continue

            pos = sample_positions[i]
            ev_idx = sample_indices[i]
            color = cmap(i / max(len(sample_indices) - 1, 1))
            wave = result["waveforms"][ev_idx]
            y_lo = float(np.nanmin(wave))
            y_hi = float(np.nanmax(wave))
            span = y_hi - y_lo
            pad = 0.08 * span if span > 0 else max(1e-3, 0.05 * max(abs(y_lo), abs(y_hi), 1.0))
            ax.plot(x, wave, color=color, lw=1.25)
            if result["representation"] == "fixed":
                ax.axvline(0, color="#ffffff", lw=0.6, ls="--", alpha=0.4)
            ax.axhline(0, color=GRID, lw=0.5, alpha=0.7)
            ax.set_xlim(float(x[0]), float(x[-1]))
            ax.set_ylim(y_lo - pad, y_hi + pad)
            pct = 100.0 * (float(pos) / max(n - 1, 1))
            ax.set_title(f"{pct:.0f}%", color=color, fontsize=7, pad=2)
            if i < (len(axes) - 1):
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(result["x_label"], color=FG, fontsize=6)
            if i > 0:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(result["value_label"], color=FG, fontsize=6)

        if axes:
            axes[0].text(0.0, 1.14, "Sorted Exemplars (5-95%)",
                         color=FG, fontsize=9, transform=axes[0].transAxes,
                         ha="left", va="bottom")

    def _heatmap_preview_index(self) -> int | None:
        if self._heatmap_hover_idx is not None:
            return int(self._heatmap_hover_idx)
        return self._selected_event_index()

    def _draw_heatmap_preview(self, ax, result):
        ax.clear()
        self._style_ax(ax, title="Hover Preview", xlabel=result["x_label"], ylabel=result["value_label"])
        ev_idx = self._heatmap_preview_index()
        if ev_idx is None:
            ax.text(0.5, 0.5, "Hover over the heatmap to preview an event waveform.",
                    color=FG, ha="center", va="center", fontsize=10, transform=ax.transAxes)
            return

        x = result["time_axis"]
        raw = result["waveforms"][ev_idx]
        recon = result["reconstruction"][ev_idx]
        if self._chk_raw.isChecked():
            ax.plot(x, raw, color="#4cc9f0", lw=1.4, label="Raw")
        if self._chk_recon.isChecked():
            ax.plot(x, recon, color="#f9844a", lw=1.2, alpha=0.95, label=f"Recon ({result['recon_rank']} PCs)")
        if self._chk_diff.isChecked():
            ax.plot(x, raw - recon, color="#f72585", lw=1.0, alpha=0.85, label="Difference")
        if result["representation"] == "fixed":
            ax.axvline(0, color="#ffffff", lw=0.7, ls="--", alpha=0.5)
        ax.axhline(0, color=GRID, lw=0.6, alpha=0.8)
        ax.set_title(self._browser_title(result, ev_idx), color=FG, fontsize=8, pad=4)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=7, facecolor=BG, edgecolor=SEP, labelcolor=FG)
        ax.text(0.01, 0.98, f"Heatmap event #{ev_idx + 1}", color="#9ca3af", fontsize=7,
                ha="left", va="top", transform=ax.transAxes,
                bbox=dict(facecolor="#111827", edgecolor=SEP, alpha=0.78, pad=4))

    def _draw_heatmap(self, ax, result):
        self._style_ax(ax, title="Sorted Event Heatmap", xlabel=result["x_label"], ylabel="Sorted events")
        idx = self._filtered_indices
        if idx.size == 0:
            ax.text(0.5, 0.5, "No events pass the current filters.",
                    color=FG, ha="center", va="center", fontsize=10, transform=ax.transAxes)
            return

        matrix = np.asarray(result["waveforms"][idx], dtype=float)
        x = np.asarray(result["time_axis"], dtype=float)
        vmax = float(np.nanpercentile(np.abs(matrix), 99))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        im = ax.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            extent=[float(x[0]), float(x[-1]), 0, matrix.shape[0]],
            origin="lower",
        )
        if result["representation"] == "fixed":
            ax.axvline(0, color="#ffffff", lw=0.7, ls="--", alpha=0.5)
        selected = self._selected_event_index()
        if selected is not None:
            match = np.where(idx == selected)[0]
            if match.size:
                ax.axhline(float(match[0]), color="#ffd166", lw=1.0, alpha=0.9)
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.92, pad=0.02)
        cbar.outline.set_edgecolor(GRID)
        cbar.ax.tick_params(colors=FG, labelsize=7)
        cbar.set_label(result["value_label"], color=FG, fontsize=8)

    def _refresh_heatmap_preview(self):
        result = self._last_result
        ax = self._axes.get("heatmap_preview")
        canvas = self._heatmap_preview_canvas
        if result is None or ax is None or canvas is None:
            return
        self._draw_heatmap_preview(ax, result)
        canvas.draw_idle()

    def _on_heatmap_hover(self, event):
        result = self._last_result
        ax = self._axes.get("heatmap")
        if result is None or ax is None or event.inaxes is not ax or event.ydata is None:
            return
        idx = self._filtered_indices
        if idx.size == 0:
            return
        row = int(np.clip(np.floor(float(event.ydata)), 0, idx.size - 1))
        ev_idx = int(idx[row])
        if self._heatmap_hover_idx != ev_idx:
            self._heatmap_hover_idx = ev_idx
            if self._view_tabs.currentIndex() == 1:
                self._refresh_heatmap_preview()

    def _on_heatmap_leave(self, event):
        ax = self._axes.get("heatmap")
        if ax is None or getattr(event, "inaxes", None) is not ax:
            return
        if self._heatmap_hover_idx is not None:
            self._heatmap_hover_idx = None
            if self._last_result is not None and self._view_tabs.currentIndex() == 1:
                self._refresh_heatmap_preview()

    def _draw_mean_components(self, ax, result):
        title = "Mean + Components"
        if result.get("per_event_mean_center"):
            title += " (wave-centered)"
        self._style_ax(ax, title=title, xlabel=result["x_label"], ylabel=result["value_label"])
        x = result["time_axis"]
        mean = result["mean_waveform"]
        comps = result.get("components", np.zeros((0, len(x)), dtype=float))
        explained = result.get("explained", np.zeros(0, dtype=float))
        score_stds = result.get("score_stds", np.ones(comps.shape[0], dtype=float))
        colors = ["#4cc9f0", "#f9844a", "#06d6a0"]

        ax.axhline(0, color=GRID, lw=0.6, alpha=0.8)
        if result["representation"] == "fixed":
            ax.axvline(0, color="#ffffff", lw=0.7, ls="--", alpha=0.55)

        ax.plot(x, mean, color="#ffffff", lw=2.0, label="Mean")
        for i in range(min(comps.shape[0], 3)):
            scale = 2.0 * float(score_stds[i])
            pos = mean + (scale * comps[i])
            neg = mean - (scale * comps[i])
            lab = f"PC{i + 1} ({100.0 * explained[i]:.1f}%)"
            ax.plot(x, pos, color=colors[i], lw=1.5, alpha=0.95, label=f"+ {lab}")
            ax.plot(x, neg, color=colors[i], lw=1.0, alpha=0.60, ls="--", label=f"- {lab}")
        ax.legend(loc="upper right", fontsize=7, facecolor=BG, edgecolor=SEP, labelcolor=FG)

    def _draw_scatter(self, ax, result):
        x_key = self._combo_scatter_x.currentData()
        y_key = self._combo_scatter_y.currentData()
        x = self._scatter_axis_values(result, x_key)
        y = self._scatter_axis_values(result, y_key)
        color_spec = self._combo_color.currentData()
        self._style_ax(ax, title="Score Scatter", xlabel=str(self._combo_scatter_x.currentText()),
                       ylabel=str(self._combo_scatter_y.currentText()))
        ax.axhline(0, color=GRID, lw=0.6, alpha=0.8)
        ax.axvline(0, color=GRID, lw=0.6, alpha=0.8)

        cmap = "viridis"
        selected = self._selected_event_index()
        filtered_set = set(self._filtered_indices.tolist())
        self._scatter_pick_map = {}

        if isinstance(color_spec, tuple) and color_spec[0] == "categorical_meta":
            labels = result.get("categorical_metadata", {}).get(color_spec[1], np.asarray([], dtype=object))
            uniq = list(dict.fromkeys(labels.tolist()))
            palette = ["#4cc9f0", "#f9844a", "#06d6a0", "#a78bfa", "#ffd166", "#ff6b6b", "#90be6d", "#f72585"]
            for i, lab in enumerate(uniq):
                mask = labels == lab
                artist = ax.scatter(x[mask], y[mask], s=18, alpha=0.72, color=palette[i % len(palette)],
                                    label=str(lab), linewidths=0, picker=True)
                self._scatter_pick_map[artist] = np.where(mask)[0]
            self._scatter_artist = None
        else:
            if isinstance(color_spec, tuple) and color_spec[0] == "numeric_meta":
                color_vals = np.asarray(result.get("numeric_metadata", {}).get(color_spec[1]), dtype=float)
            elif isinstance(color_spec, tuple) and color_spec[0] == "metric":
                color_vals = self._metric_values(result, color_spec[1])
            else:
                color_vals = self._metric_values(result, "residual")

            self._scatter_artist = ax.scatter(
                x, y, c=color_vals, cmap=cmap, s=20, alpha=0.78, linewidths=0,
                picker=True, pickradius=5
            )
            self._scatter_pick_map[self._scatter_artist] = np.arange(result["active_events"], dtype=int)
            cbar = ax.figure.colorbar(self._scatter_artist, ax=ax, shrink=0.84, pad=0.02)
            cbar.outline.set_edgecolor(GRID)
            cbar.ax.tick_params(colors=FG, labelsize=7)
            cbar.ax.yaxis.label.set_color(FG)
            cbar.set_label(self._combo_color.currentText(), color=FG, fontsize=8)

        if filtered_set and len(filtered_set) < result["active_events"]:
            mask = np.asarray([i in filtered_set for i in range(result["active_events"])], dtype=bool)
            ax.scatter(x[~mask], y[~mask], s=14, facecolors="none", edgecolors="#6b7280",
                       alpha=0.22, linewidths=0.6)

        if selected is not None:
            ax.scatter([x[selected]], [y[selected]], s=90, facecolors="none",
                       edgecolors="#ffffff", linewidths=1.2, zorder=5)

        finite = np.isfinite(x) & np.isfinite(y)
        if np.any(finite):
            x_lo = float(np.min(x[finite]))
            x_hi = float(np.max(x[finite]))
            y_lo = float(np.min(y[finite]))
            y_hi = float(np.max(y[finite]))

            x_span = x_hi - x_lo
            y_span = y_hi - y_lo
            x_pad = 0.04 * x_span if x_span > 0 else max(1e-3, 0.05 * max(abs(x_lo), 1.0))
            y_pad = 0.04 * y_span if y_span > 0 else max(1e-3, 0.05 * max(abs(y_lo), 1.0))
            ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
            ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

        if self._scatter_artist is None and isinstance(color_spec, tuple) and color_spec[0] == "categorical_meta":
            ax.legend(loc="upper right", fontsize=7, facecolor=BG, edgecolor=SEP, labelcolor=FG)
        ax.text(0.01, 0.01, "Left click: inspect point\nRight drag: scatter filter",
                color="#9ca3af", fontsize=7, transform=ax.transAxes,
                ha="left", va="bottom",
                bbox=dict(facecolor="#111827", edgecolor=SEP, alpha=0.75, pad=4))

    def _draw_residual_hist(self, ax, result):
        residual = np.asarray(result["residual"], dtype=float)
        self._style_ax(ax, title="Residual Distribution", xlabel="Residual", ylabel="Events")
        ax.hist(residual, bins=min(40, max(10, int(np.sqrt(len(residual))))),
                color="#4cc9f0", alpha=0.75, edgecolor=GRID)
        if self._residual_filter is not None:
            lo, hi = self._residual_filter
            ax.axvspan(lo, hi, color="#f9844a", alpha=0.18, zorder=0)
            ax.axvline(lo, color="#f9844a", lw=1.2, ls="--")
            ax.axvline(hi, color="#f9844a", lw=1.2, ls="--")
        ax.text(0.01, 0.98, "Drag horizontally to filter browser", color="#9ca3af",
                fontsize=7, transform=ax.transAxes, ha="left", va="top",
                bbox=dict(facecolor="#111827", edgecolor=SEP, alpha=0.75, pad=4))

    def _draw_metadata_panel(self, ax, result):
        ax.set_facecolor(BG)
        ax.set_axis_off()
        lines = [
            "QC Summary",
            "",
            f"Anchor: {result['anchor_label']}",
            f"Recon PCs: {result['recon_rank']}",
            f"Wave-centered: {'yes' if result.get('per_event_mean_center') else 'no'}",
            f"Sample rate: {result['sample_rate']:.3f} Hz",
            f"Matching events: {result['total_matching_events']}",
            f"Usable events: {result['usable_events']}",
            f"Decomposed events: {result['active_events']}",
        ]

        skips = result.get("skip_counts", {})
        skip_line = ", ".join(f"{k}={v}" for k, v in skips.items() if v)
        if skip_line:
            lines.extend(["", f"Skipped: {skip_line}"])

        if self._scatter_filter is not None:
            lines.extend(["", f"Scatter filter: {len(self._scatter_filter)} events"])
        if self._residual_filter is not None:
            lines.append(f"Residual filter: [{self._residual_filter[0]:.4f}, {self._residual_filter[1]:.4f}]")

        ax.text(0.02, 0.98, "\n".join(lines), color=FG, fontsize=9,
                ha="left", va="top", transform=ax.transAxes,
                bbox=dict(facecolor="#111827", edgecolor=SEP, boxstyle="round,pad=0.45"))

    def _draw_browser(self, ax, result):
        self._style_ax(ax, title="Event Browser", xlabel=result["x_label"], ylabel=result["value_label"])
        selected = self._selected_event_index()
        if selected is None:
            ax.text(0.5, 0.5, "No events pass the current filters.",
                    color=FG, ha="center", va="center", fontsize=10, transform=ax.transAxes)
            ax.set_axis_off()
            return

        x = result["time_axis"]
        raw = result["waveforms"][selected]
        recon = result["reconstruction"][selected]
        diff = raw - recon

        if self._chk_raw.isChecked():
            ax.plot(x, raw, color="#4cc9f0", lw=1.8, label="Raw")
        if self._chk_recon.isChecked():
            ax.plot(x, recon, color="#f9844a", lw=1.6, alpha=0.95, label=f"Recon ({result['recon_rank']} PCs)")
        if self._chk_diff.isChecked():
            ax.plot(x, diff, color="#f72585", lw=1.2, alpha=0.85, label="Difference")

        if result["representation"] == "fixed":
            ax.axvline(0, color="#ffffff", lw=0.7, ls="--", alpha=0.55)
        ax.axhline(0, color=GRID, lw=0.6, alpha=0.8)

        title = self._browser_title(result, selected)
        ax.set_title(title, color=FG, fontsize=9, pad=5)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper right", fontsize=8, facecolor=BG, edgecolor=SEP, labelcolor=FG)

        text_lines = [
            f"Event #{selected + 1} in decomposition set",
            f"Start={result['starts'][selected]:.3f}s  Stop={result['stops'][selected]:.3f}s",
        ]
        numeric_meta = result.get("numeric_metadata", {})
        for col in list(numeric_meta.keys())[:4]:
            text_lines.append(f"{col}={numeric_meta[col][selected]:.3f}")
        ax.text(0.01, 0.98, "\n".join(text_lines), color="#9ca3af", fontsize=7,
                ha="left", va="top", transform=ax.transAxes,
                bbox=dict(facecolor="#111827", edgecolor=SEP, alpha=0.78, pad=4))

    def _install_selectors(self):
        try:
            if self._scatter_selector is not None:
                self._scatter_selector.set_active(False)
        except Exception:
            pass
        try:
            if self._residual_selector is not None:
                self._residual_selector.set_active(False)
        except Exception:
            pass

        ax_scatter = self._axes.get("scatter")
        ax_hist = self._axes.get("hist")
        if ax_scatter is not None:
            self._scatter_selector = RectangleSelector(
                ax_scatter,
                self._on_scatter_select,
                useblit=False,
                button=[3],
                minspanx=0.01,
                minspany=0.01,
                spancoords="data",
                interactive=False,
                props=dict(facecolor="#f9844a", edgecolor="#ffffff", alpha=0.12, fill=True),
            )
        if ax_hist is not None:
            self._residual_selector = SpanSelector(
                ax_hist,
                self._on_residual_select,
                direction="horizontal",
                useblit=False,
                props=dict(facecolor="#f9844a", alpha=0.18),
                button=[1],
                interactive=False,
            )
