
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
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QTabWidget, QVBoxLayout, QWidget,
)

from .explorer_base import BG, FG, GRID, _ExplorerTab
from ..file_dialogs import existing_directory, open_file_name
from ..lwf import format_lwf_summary, format_lwf_summary_compact, load_lwf_directory

TRACE_SUMMARY_THRESHOLD = 2000
QUANTILE_SUBSAMPLE_LIMIT = 4000
RENDER_BUTTON_STYLE = (
    "QPushButton {"
    " background-color:#1f6feb;"
    " color:#f8fafc;"
    " border:1px solid #388bfd;"
    " border-radius:6px;"
    " font-weight:600;"
    " padding:4px 12px;"
    "}"
    "QPushButton:hover { background-color:#2f81f7; border-color:#58a6ff; }"
    "QPushButton:pressed { background-color:#1a5fd0; }"
    "QPushButton:disabled { background-color:#1f2937; color:#9ca3af; border-color:#374151; }"
)
INSPECTOR_LOCAL_FEATURES = [
    ("rms", "RMS"),
    ("ptp", "Peak-to-peak"),
    ("mean_abs", "Mean |x|"),
    ("auc_abs", "Area |x|"),
    ("hjorth_activity", "Hjorth Activity"),
    ("hjorth_mobility", "Hjorth Mobility"),
    ("hjorth_complexity", "Hjorth Complexity"),
]


class _WidePopupComboBox(QComboBox):
    """Keep the collapsed combo compact while letting the popup fit long labels."""

    def popup_width_hint(self) -> int:
        width = self.width()
        model = self.model()
        if model is None:
            return width
        fm = self.fontMetrics()
        icon_width = max(0, self.iconSize().width())
        for row in range(model.rowCount()):
            text = str(model.index(row, self.modelColumn()).data(Qt.DisplayRole) or "")
            width = max(width, fm.horizontalAdvance(text) + icon_width + 40)
        return width

    def showPopup(self):
        view = self.view()
        if view is not None:
            width = self.popup_width_hint()
            view.setMinimumWidth(width)
            view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        super().showPopup()


def _apply_transform(mat, transform_mode):
    if transform_mode == "rectified":
        return np.abs(mat)
    return mat


def _apply_outlier_policy(seg_or_mat, outlier_mode, outlier_sd_thresh):
    if outlier_mode == "none" or not np.isfinite(outlier_sd_thresh) or outlier_sd_thresh <= 0:
        return seg_or_mat

    data = np.asarray(seg_or_mat, dtype=float)
    if data.size == 0:
        return data

    mean = np.nanmean(data, axis=0, keepdims=True)
    std = np.nanstd(data, axis=0, ddof=1, keepdims=True)
    std = np.where(np.isfinite(std) & (std > 0), std, np.nan)
    lo = mean - (float(outlier_sd_thresh) * std)
    hi = mean + (float(outlier_sd_thresh) * std)

    if outlier_mode == "winsorize":
        return np.clip(data, lo, hi)
    if outlier_mode == "remove":
        mask = (data < lo) | (data > hi)
        return np.where(mask, np.nan, data)
    return data


def _normalize_trace(seg):
    seg = np.asarray(seg, dtype=float)
    finite = np.isfinite(seg)
    if not np.any(finite):
        return seg
    vals = seg[finite]
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = np.full(seg.shape, 0.0, dtype=float)
        out[~finite] = np.nan
        return out
    out = seg.copy()
    out[finite] = ((vals - lo) / (hi - lo)) * 2.0 - 1.0
    return out


def _preprocess_traces(segs, *, n_pre, baseline_mode, outlier_mode, outlier_sd_thresh, transform_mode):
    segs = np.asarray(segs, dtype=float)
    if segs.size == 0:
        return segs

    segs = _apply_outlier_policy(segs, outlier_mode, outlier_sd_thresh)

    if baseline_mode in {"subtract", "subtract_normalize"} and n_pre > 0:
        base = segs[:, :n_pre]
        with np.errstate(invalid="ignore"):
            base_mean = np.nanmean(base, axis=1, keepdims=True)
        segs = segs - base_mean

    if baseline_mode in {"normalize", "subtract_normalize"}:
        segs = np.vstack([_normalize_trace(seg) for seg in segs])

    segs = _apply_transform(segs, transform_mode)
    return segs


def _safe_nanvar(x):
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.nan
    return float(np.nanvar(finite))


def _compute_local_trace_features(seg, sr=None):
    seg = np.asarray(seg, dtype=float)
    finite = seg[np.isfinite(seg)]
    features = {}
    if finite.size == 0:
        for key, _ in INSPECTOR_LOCAL_FEATURES:
            features[key] = np.nan
        return features

    features["rms"] = float(np.sqrt(np.mean(finite * finite)))
    features["ptp"] = float(np.max(finite) - np.min(finite))
    features["mean_abs"] = float(np.mean(np.abs(finite)))
    if np.isfinite(sr) and sr and sr > 0:
        features["auc_abs"] = float(np.sum(np.abs(finite)) / float(sr))
    else:
        features["auc_abs"] = float(np.sum(np.abs(finite)))

    activity = _safe_nanvar(seg)
    d1 = np.diff(seg)
    mobility_num = _safe_nanvar(d1)
    mobility = np.sqrt(mobility_num / activity) if np.isfinite(activity) and activity > 0 and np.isfinite(mobility_num) else np.nan
    d2 = np.diff(d1)
    mobility_d1_num = _safe_nanvar(d2)
    mobility_d1 = np.sqrt(mobility_d1_num / mobility_num) if np.isfinite(mobility_num) and mobility_num > 0 and np.isfinite(mobility_d1_num) else np.nan
    complexity = mobility_d1 / mobility if np.isfinite(mobility) and mobility > 0 and np.isfinite(mobility_d1) else np.nan

    features["hjorth_activity"] = activity
    features["hjorth_mobility"] = mobility
    features["hjorth_complexity"] = complexity
    return features


def _feature_label(feature_key: str) -> str:
    for key, label in INSPECTOR_LOCAL_FEATURES:
        if key == feature_key:
            return label
    return str(feature_key)


def _ellipsis(text: str, limit: int = 48) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _join_preview(values, *, limit_items: int = 3, limit_chars: int = 48) -> str:
    items = [str(v) for v in values if str(v)]
    if not items:
        return ""
    preview = ", ".join(items[:limit_items])
    if len(items) > limit_items:
        preview += f" +{len(items) - limit_items} more"
    return _ellipsis(preview, limit_chars)


def _format_waveform_suptitle(parts, *, prefix: str = "Peri-event waveform", line_limit: int = 118) -> str:
    lines = [prefix]
    current = ""
    for part in [str(p) for p in parts if str(p)]:
        candidate = part if not current else f"{current}  |  {part}"
        if len(candidate) > line_limit and current:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def _compact_waveform_title_parts(result, *, inspect_mode: bool, contrast_layout: str) -> list[str]:
    parts = [f"'{result['annot_class']}'", f"({result['n_events']} {result.get('trace_count_label', 'events')})"]
    contrast = result.get("contrast")
    if contrast:
        parts.append(contrast.get("title", "two-group contrast"))
        parts.append(contrast_layout)
    if inspect_mode:
        parts.append("wave inspector")
        selected_annot = result.get("_inspect_annot_label", "")
        if selected_annot:
            parts.append(f"inspect annot={selected_annot}")
    summary_mode = result.get("summary_mode", "mean_ci")
    if summary_mode == "variance":
        parts.append("variance")
    elif summary_mode == "mean_sd":
        parts.append("mean ± SD")
    elif summary_mode == "mean_quantiles":
        parts.append("quantiles")
    transform_mode = result.get("transform_mode", "raw")
    baseline_mode = result.get("baseline_mode", "subtract")
    outlier_mode = result.get("outlier_mode", "none")
    outlier_sd_thresh = result.get("outlier_sd_thresh", np.nan)
    if transform_mode == "rectified":
        parts.append("rectified")
    if baseline_mode == "normalize":
        parts.append("norm[-1,1]")
    elif baseline_mode == "subtract_normalize":
        parts.append("baseline + norm[-1,1]")
    elif baseline_mode == "subtract":
        parts.append("baseline")
    if outlier_mode != "none" and np.isfinite(outlier_sd_thresh):
        parts.append(f"{outlier_mode}@{outlier_sd_thresh:g}SD")
    return parts


def _dedupe_preserve_order(values):
    seen = set()
    ordered = []
    for value in values:
        s = str(value).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        ordered.append(s)
    return ordered


def _table_column_values(model, header_candidates):
    if model is None:
        return []
    try:
        ncols = int(model.columnCount())
    except Exception:
        return []
    headers = [str(model.headerData(c, Qt.Horizontal) or "").strip().upper() for c in range(ncols)]
    value_col = -1
    for candidate in header_candidates:
        candidate_u = str(candidate).strip().upper()
        if candidate_u in headers:
            value_col = headers.index(candidate_u)
            break
    if value_col < 0:
        value_col = 1 if ncols > 1 else 0
    out = []
    for r in range(int(model.rowCount())):
        try:
            val = model.data(model.index(r, value_col), Qt.DisplayRole)
        except Exception:
            val = None
        s = "" if val is None else str(val).strip()
        if s:
            out.append(s)
    return out


def _collect_feature_names(trace_meta, channels):
    names = []
    seen = set()
    for key, _ in INSPECTOR_LOCAL_FEATURES:
        names.append(key)
        seen.add(key)
    for ch in channels:
        for meta in trace_meta.get(ch, []):
            for key in meta.get("features", {}):
                if key not in seen:
                    seen.add(key)
                    names.append(key)
    return names


def _metric_window_mask(t_vals, meta, source, custom_pre, custom_post):
    t_vals = np.asarray(t_vals, dtype=float)
    if t_vals.size == 0:
        return np.zeros(0, dtype=bool)
    if source == "custom":
        return (t_vals >= -float(custom_pre)) & (t_vals <= float(custom_post))
    if source == "annot":
        anchor = float(meta.get("anchor_sec", np.nan))
        annot_start = float(meta.get("annot_start_sec", np.nan))
        annot_stop = float(meta.get("annot_stop_sec", np.nan))
        if np.isfinite(anchor) and np.isfinite(annot_start) and np.isfinite(annot_stop):
            rel_start = annot_start - anchor
            rel_stop = annot_stop - anchor
            if rel_stop > rel_start:
                return (t_vals >= rel_start) & (t_vals <= rel_stop)
    return np.ones(t_vals.shape, dtype=bool)


def _metric_window_label(source, custom_pre, custom_post):
    if source == "custom":
        return f"metrics [-{custom_pre:g}, +{custom_post:g}]s"
    if source == "annot":
        return "metrics annot interval"
    return "metrics visible window"


def _nearest_sample_index(sample_times: np.ndarray, target: float) -> int:
    insert_idx = int(np.searchsorted(sample_times, target, side="left"))
    center_idx = min(max(insert_idx, 0), len(sample_times) - 1)
    if 0 < insert_idx < len(sample_times):
        left_idx = insert_idx - 1
        right_idx = insert_idx
        if abs(target - sample_times[left_idx]) <= abs(sample_times[right_idx] - target):
            center_idx = left_idx
        else:
            center_idx = right_idx
    return center_idx


def _compute_summary_stats(traces_out, channels, summary_mode):
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
        segs = traces_out.get(ch, [])
        if not segs:
            continue
        mat = np.vstack(segs)
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
        "mean": mean_out,
        "ci_lo": ci_lo_out,
        "ci_hi": ci_hi_out,
        "std": std_out,
        "var": var_out,
        "median": median_out,
        "q05": q05_out,
        "q25": q25_out,
        "q75": q75_out,
        "q95": q95_out,
        "quantile_n": quantile_n_out,
    }


def _build_trace_result(
    traces_out,
    trace_meta_out,
    t_grid_out,
    channels,
    sr_out,
    summary_mode,
    *,
    annot_class,
    n_events,
    transform_mode,
    units=None,
    trace_count_label="events",
    extra_title_bits=None,
    source_mode="record",
    baseline_mode="subtract",
    outlier_mode="none",
    outlier_sd_thresh=np.nan,
    view_pre_secs=np.nan,
    view_post_secs=np.nan,
):
    stats = _compute_summary_stats(traces_out, channels, summary_mode)
    result = {
        "traces": traces_out,
        "t_grid": t_grid_out,
        "transform_mode": transform_mode,
        "summary_mode": summary_mode,
        "n_events": n_events,
        "sr": sr_out,
        "annot_class": annot_class,
        "channels": channels,
        "trace_meta": trace_meta_out,
        "feature_names": _collect_feature_names(trace_meta_out, channels),
        "trace_count_label": trace_count_label,
        "units": dict(units or {}),
        "extra_title_bits": list(extra_title_bits or []),
        "source_mode": source_mode,
        "baseline_mode": baseline_mode,
        "outlier_mode": outlier_mode,
        "outlier_sd_thresh": outlier_sd_thresh,
        "view_pre_secs": float(view_pre_secs),
        "view_post_secs": float(view_post_secs),
    }
    result.update(stats)
    return result


def _compute_two_group_pointwise_stats(group0, group1):
    mat0 = np.vstack(group0)
    mat1 = np.vstack(group1)
    valid0 = np.sum(np.isfinite(mat0), axis=0)
    valid1 = np.sum(np.isfinite(mat1), axis=0)
    mean0 = np.full(mat0.shape[1], np.nan, dtype=float)
    mean1 = np.full(mat1.shape[1], np.nan, dtype=float)
    cols0 = valid0 > 0
    cols1 = valid1 > 0
    if np.any(cols0):
        mean0[cols0] = np.nansum(mat0[:, cols0], axis=0) / valid0[cols0]
    if np.any(cols1):
        mean1[cols1] = np.nansum(mat1[:, cols1], axis=0) / valid1[cols1]
    diff = mean1 - mean0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore")
        tstat, pval = stats.ttest_ind(mat1, mat0, axis=0, equal_var=False, nan_policy="omit")
    with np.errstate(divide="ignore", invalid="ignore"):
        neglogp = -np.log10(pval)
    neglogp[~np.isfinite(neglogp)] = np.nan
    return {
        "mean_diff": diff,
        "pval": np.asarray(pval, dtype=float),
        "neglogp": np.asarray(neglogp, dtype=float),
    }


# ---------------------------------------------------------------------------
# Pure computation (background thread)
# ---------------------------------------------------------------------------

def _extract_traces(p, ns, annot_class, channels, pre_secs, post_secs, align_to, baseline_mode,
                    outlier_mode, outlier_sd_thresh, transform_mode, summary_mode):
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
        extracted = p.extract_event_waveforms_with_features(
            [annot_class],
            channels,
            float(pre_secs),
            float(post_secs),
            align=align_to,
            require="full",
            catch24=False,
            basic_stats=True,
        )
    except Exception as e:
        raise RuntimeError(f"Could not extract waveforms for '{annot_class}': {e}")

    events = list(extracted.get("events", []))
    if not events:
        raw_count = None
        try:
            raw_events = p.fetch_annots([annot_class])
            if raw_events is not None:
                raw_count = int(len(raw_events))
        except Exception:
            raw_count = None
        if raw_count:
            raise RuntimeError(
                "No usable waveform windows were extracted for annotation "
                f"'{annot_class}', although {raw_count} raw event(s) exist. "
                "The waveform extractor currently requires a full pre/post window "
                "around each event (`require=\"full\"`), so events can be dropped "
                "if the selected window, alignment, or channels leave no valid segments."
            )
        raise RuntimeError(f"No events found for annotation '{annot_class}'.")

    traces_out: dict[str, list] = {ch: [] for ch in channels}
    trace_meta_out: dict[str, list] = {ch: [] for ch in channels}
    sr_out: dict[str, float] = {}
    t_grid_out: dict[str, np.ndarray] = {}
    unit_out: dict[str, str] = {}
    n_events = int(extracted.get("total_events", len(events)))

    for ev in events:
        blocks = ev.get("blocks", {})
        for ch in channels:
            block = blocks.get(ch)
            if block is None:
                continue
            sr = float(block.get("sr", np.nan))
            rel = np.asarray(block.get("rel_time", []), dtype=float)
            vals = np.asarray(block.get("values", []), dtype=float)
            if rel.size == 0 or vals.size == 0 or rel.shape[0] != vals.shape[0]:
                continue
            if ch not in t_grid_out:
                t_grid_out[ch] = rel
                sr_out[ch] = sr
                unit_out[ch] = str(block.get("unit", "") or "")
            traces_out[ch].append(vals)
            trace_meta_out[ch].append(
                {
                    "annot": str(ev.get("annot", annot_class)),
                    "instance": str(ev.get("instance", "")),
                    "annot_ch": str(ev.get("annot_ch", "")),
                    "anchor_sec": float(ev.get("anchor_sec", np.nan)),
                    "annot_start_sec": float(ev.get("annot_start_sec", np.nan)),
                    "annot_stop_sec": float(ev.get("annot_stop_sec", np.nan)),
                    "features": dict(block.get("features", {}) or {}),
                    "feature_qc": int(block.get("feature_qc", -1)),
                }
            )

    for ch in channels:
        if ch not in t_grid_out or not traces_out[ch]:
            continue
        sr = float(sr_out.get(ch, np.nan))
        if not np.isfinite(sr) or sr <= 0:
            continue
        n_pre = int(round(float(pre_secs) * sr))
        mat = np.vstack(traces_out[ch])
        mat = _preprocess_traces(
            mat,
            n_pre=n_pre,
            baseline_mode=baseline_mode,
            outlier_mode=outlier_mode,
            outlier_sd_thresh=outlier_sd_thresh,
            transform_mode=transform_mode,
        )
        traces_out[ch] = [row for row in mat]
        updated_meta = []
        for row, meta in zip(mat, trace_meta_out[ch]):
            meta2 = dict(meta)
            features = dict(meta2.get("features", {}))
            features.update(_compute_local_trace_features(row, sr=sr))
            meta2["features"] = features
            updated_meta.append(meta2)
        trace_meta_out[ch] = updated_meta

    return _build_trace_result(
        traces_out,
        trace_meta_out,
        t_grid_out,
        channels,
        sr_out,
        summary_mode,
        annot_class=annot_class,
        n_events=n_events,
        transform_mode=transform_mode,
        baseline_mode=baseline_mode,
        outlier_mode=outlier_mode,
        outlier_sd_thresh=outlier_sd_thresh,
        units=unit_out,
        source_mode="record",
        view_pre_secs=pre_secs,
        view_post_secs=post_secs,
    )


def _format_lwf_panel_label(ch_value, annot_value, tag_value, strategies):
    parts = []
    if strategies.get("CH", "stratify") == "stratify":
        parts.append(str(ch_value))
    if strategies.get("ANNOT", "pool") == "stratify":
        parts.append(f"Annot={annot_value}")
    if strategies.get("TAG", "pool") == "stratify":
        parts.append(f"Tag={tag_value}")
    return " | ".join(parts) if parts else "Pooled"


def _extract_lwf_traces(dataset, filters, strategies, pre_secs, post_secs, align_to, baseline_mode,
                        outlier_mode, outlier_sd_thresh, transform_mode, summary_mode, average_by_id):
    files_df = dataset.files
    waves_df = dataset.waves
    channels_df = dataset.channels

    selected_channels = [str(x) for x in filters.get("CH", [])]
    selected_annots = {str(x) for x in filters.get("ANNOT", [])}
    selected_tags = {str(x) for x in filters.get("TAG", [])}
    selected_ids = {str(x) for x in filters.get("ID", [])}

    if not selected_channels:
        raise RuntimeError("No channels selected.")
    if not selected_annots:
        raise RuntimeError("No annotations selected.")
    if not selected_tags:
        raise RuntimeError("No tags selected.")
    if not selected_ids:
        raise RuntimeError("No IDs selected.")

    file_meta = files_df[["FILE", "LWF_ID", "TAG"]].drop_duplicates()
    valid_files = set(
        file_meta.loc[
            file_meta["LWF_ID"].astype(str).isin(selected_ids)
            & file_meta["TAG"].astype(str).isin(selected_tags),
            "FILE",
        ].tolist()
    )
    if not valid_files:
        raise RuntimeError("No .lwf files matched the selected ID/TAG filters.")

    selected_waves = waves_df.loc[
        waves_df["FILE"].isin(valid_files)
        & waves_df["ANNOT"].astype(str).isin(selected_annots)
    ]
    if selected_waves.empty:
        raise RuntimeError("No waveform events matched the selected filters.")

    file_id_map = dict(zip(file_meta["FILE"], file_meta["LWF_ID"]))
    sr_map = {}
    unit_map = {}
    for ch in selected_channels:
        rows = channels_df.loc[channels_df["CH"].astype(str) == ch]
        if rows.empty:
            continue
        sr_map[ch] = float(rows["SR"].iloc[0])
        unit = str(rows["UNIT"].iloc[0]).strip()
        unit_map[ch] = "" if unit == "nan" else unit

    ch_mode = str(strategies.get("CH", "stratify"))
    annot_mode = str(strategies.get("ANNOT", "pool"))
    tag_mode = str(strategies.get("TAG", "pool"))
    contrast = dict(strategies.get("CONTRAST") or {"axis": "none"})
    if ch_mode == "pool":
        pooled_srs = {sr_map.get(ch) for ch in selected_channels if np.isfinite(sr_map.get(ch, np.nan))}
        if len(pooled_srs) > 1:
            raise RuntimeError(
                "Cannot pool selected channels because they do not share the same sample rate."
            )
        pooled_units = {unit_map.get(ch, "") for ch in selected_channels}
        pooled_unit = pooled_units.pop() if len(pooled_units) == 1 else ""
    else:
        pooled_unit = ""

    traces_out = {}
    trace_meta = {}
    trace_ids = {}
    trace_group_labels = {}
    t_grid_out = {}
    used_sr = {}
    panel_units = {}
    contrast_groups = {}
    raw_trace_count = 0
    contrast_axis = str(contrast.get("axis", "none"))
    if contrast_axis == "id_group":
        cov_df = contrast["covariates"]
        cov_name = str(contrast["covariate"])
        cov_map = dict(zip(cov_df["LWF_ID"].astype(str), cov_df[cov_name]))
        group_levels = ["0", "1"]
        contrast_title = f"ID group={cov_name}"
    elif contrast_axis in {"CH", "ANNOT", "TAG"}:
        group_levels = [str(x) for x in contrast.get("levels", [])]
        contrast_title = f"{contrast_axis}: {group_levels[0]} vs {group_levels[1]}"
    else:
        group_levels = []
        contrast_title = ""

    def _row_center_sec(row):
        if align_to == "start":
            return float(row.ANNOT_START_SEC)
        if align_to == "stop":
            return float(row.ANNOT_STOP_SEC)
        return 0.5 * (float(row.ANNOT_START_SEC) + float(row.ANNOT_STOP_SEC))

    for ch in selected_channels:
        sr = sr_map.get(ch)
        if not np.isfinite(sr) or sr <= 0:
            continue
        n_pre = int(round(float(pre_secs) * sr))
        n_post = int(round(float(post_secs) * sr))
        offsets = np.arange(-n_pre, n_post + 1, dtype=np.int64)
        expected_rel = offsets.astype(float) / float(sr)
        gap_tol = max(1e-6, (1.0 / float(sr)) * 0.25)

        for row in selected_waves.itertuples(index=False):
            row_tag = str(row.TAG)
            if row_tag not in selected_tags:
                continue
            blocks = row.BLOCKS
            block = blocks.get(ch)
            if block is None or block.n <= 0:
                continue

            center_sec = _row_center_sec(row)

            panel_strategies = {"CH": ch_mode, "ANNOT": annot_mode, "TAG": tag_mode}
            if contrast_axis in panel_strategies:
                panel_strategies[contrast_axis] = "pool"
            panel_label = _format_lwf_panel_label(
                ch if ch_mode == "stratify" else ",".join(selected_channels),
                str(row.ANNOT),
                row_tag,
                panel_strategies,
            )
            if panel_label not in traces_out:
                traces_out[panel_label] = []
                trace_meta[panel_label] = []
                trace_ids[panel_label] = []
                trace_group_labels[panel_label] = []
                t_grid_out[panel_label] = expected_rel
                used_sr[panel_label] = sr
                panel_units[panel_label] = unit_map.get(ch, "") if ch_mode == "stratify" else pooled_unit
                if contrast_axis != "none":
                    contrast_groups[panel_label] = {gl: [] for gl in group_levels}

            if contrast_axis == "id_group":
                lwf_id = str(file_id_map.get(row.FILE, ""))
                group_value = cov_map.get(lwf_id, np.nan)
                if pd.isna(group_value):
                    continue
                group_label = str(int(group_value))
                if group_label not in contrast_groups[panel_label]:
                    continue
            elif contrast_axis == "CH":
                group_label = ch
                if group_label not in group_levels:
                    continue
            elif contrast_axis == "ANNOT":
                group_label = str(row.ANNOT)
                if group_label not in group_levels:
                    continue
            elif contrast_axis == "TAG":
                group_label = row_tag
                if group_label not in group_levels:
                    continue
            else:
                group_label = None

            sample_times = float(block.data_start_sec) + (np.arange(block.n, dtype=float) / float(sr))
            if float(row.ANNOT_START_SEC) == float(row.ANNOT_STOP_SEC) and sample_times.size:
                # Point-event annotations emitted at half-sample boundaries can sit
                # between stored samples for low-SR PP signals. Snap to the nearest
                # stored sample so LWF re-windowing matches the dumped waveform lattice.
                center_sec = float(sample_times[_nearest_sample_index(sample_times, center_sec)])
            center_idx = _nearest_sample_index(sample_times, center_sec)
            req_start_idx = int(center_idx + offsets[0])
            req_stop_idx = int(center_idx + offsets[-1])
            src_start_idx = max(0, req_start_idx)
            src_stop_idx = min(block.n - 1, req_stop_idx)
            if src_start_idx > src_stop_idx:
                continue
            dst_start_idx = int(src_start_idx - req_start_idx)
            dst_stop_idx = int(dst_start_idx + (src_stop_idx - src_start_idx))

            seg = np.full(expected_rel.shape[0], np.nan, dtype=float)
            src_values = np.asarray(block.values[src_start_idx:src_stop_idx + 1], dtype=float)
            rel = sample_times[src_start_idx:src_stop_idx + 1] - center_sec
            expected_slice = expected_rel[dst_start_idx:dst_stop_idx + 1]
            if src_values.shape[0] != expected_slice.shape[0]:
                continue
            if not np.all(np.abs(rel - expected_slice) <= gap_tol):
                continue
            seg[dst_start_idx:dst_stop_idx + 1] = src_values

            lwf_id = str(file_id_map.get(row.FILE, ""))
            traces_out[panel_label].append(seg)
            trace_meta[panel_label].append(
                {
                    "annot": str(row.ANNOT),
                    "instance": str(row.INSTANCE),
                    "annot_ch": str(row.ANNOT_CH),
                    "anchor_sec": float(center_sec),
                    "annot_start_sec": float(row.ANNOT_START_SEC),
                    "annot_stop_sec": float(row.ANNOT_STOP_SEC),
                    "lwf_id": lwf_id,
                    "tag": row_tag,
                    "features": dict(getattr(block, "features", {}) or {}),
                    "feature_qc": int(getattr(block, "feature_qc", -1)),
                }
            )
            trace_ids[panel_label].append(lwf_id)
            trace_group_labels[panel_label].append(group_label)
            if contrast_axis != "none":
                contrast_groups[panel_label][group_label].append((seg, lwf_id))
            raw_trace_count += 1

    for panel in list(traces_out.keys()):
        segs = traces_out.get(panel, [])
        if not segs:
            continue
        n_pre = int(round(float(pre_secs) * float(used_sr[panel])))
        proc = _preprocess_traces(
            np.vstack(segs),
            n_pre=n_pre,
            baseline_mode=baseline_mode,
            outlier_mode=outlier_mode,
            outlier_sd_thresh=outlier_sd_thresh,
            transform_mode=transform_mode,
        )
        traces_out[panel] = [row for row in proc]
        updated_meta = []
        for row, meta in zip(proc, trace_meta[panel]):
            meta2 = dict(meta)
            features = dict(meta2.get("features", {}))
            features.update(_compute_local_trace_features(row, sr=used_sr[panel]))
            meta2["features"] = features
            updated_meta.append(meta2)
        trace_meta[panel] = updated_meta
        if contrast_axis != "none":
            panel_groups = {gl: [] for gl in group_levels}
            for seg, lwf_id, group_label in zip(proc, trace_ids[panel], trace_group_labels[panel]):
                if group_label in panel_groups:
                    panel_groups[group_label].append((seg, lwf_id))
            contrast_groups[panel] = panel_groups

    panel_keys = list(traces_out.keys())
    contrast_group_stats = {}
    if average_by_id:
        averaged = {panel: [] for panel in panel_keys}
        averaged_meta = {panel: [] for panel in panel_keys}
        for panel in panel_keys:
            if not traces_out[panel]:
                continue
            by_id = {}
            by_meta = {}
            for seg, lwf_id, meta in zip(traces_out[panel], trace_ids[panel], trace_meta[panel]):
                by_id.setdefault(lwf_id, []).append(seg)
                by_meta.setdefault(lwf_id, []).append(meta)
            for lwf_id in sorted(by_id):
                mean_seg = np.vstack(by_id[lwf_id]).mean(axis=0)
                averaged[panel].append(mean_seg)
                meta = dict(by_meta[lwf_id][0]) if by_meta.get(lwf_id) else {"lwf_id": lwf_id}
                meta["anchor_sec"] = np.nan
                meta["annot_start_sec"] = np.nan
                meta["annot_stop_sec"] = np.nan
                meta["features"] = _compute_local_trace_features(mean_seg, sr=used_sr[panel])
                averaged_meta[panel].append(meta)
        traces_out = averaged
        trace_meta = averaged_meta
        if contrast_axis != "none":
            reduced_groups = {}
            for panel in panel_keys:
                reduced_groups[panel] = {gl: [] for gl in group_levels}
                for gl in group_levels:
                    by_id = {}
                    for seg, lwf_id in contrast_groups[panel][gl]:
                        by_id.setdefault(lwf_id, []).append(seg)
                    for lwf_id in sorted(by_id):
                        reduced_groups[panel][gl].append(np.vstack(by_id[lwf_id]).mean(axis=0))
            contrast_groups = reduced_groups
        n_events = sum(len(v) for v in traces_out.values())
        trace_count_label = "ID means"
        extra_title_bits = [f"{raw_trace_count} waves"]
    else:
        if contrast_axis != "none":
            reduced_groups = {}
            for panel in panel_keys:
                reduced_groups[panel] = {gl: [seg for seg, _ in contrast_groups[panel][gl]] for gl in group_levels}
            contrast_groups = reduced_groups
        n_events = sum(len(v) for v in traces_out.values())
        trace_count_label = "waves"
        extra_title_bits = []

    if n_events == 0:
        raise RuntimeError("No waveform segments matched the requested window and filters.")

    panel_keys = [panel for panel, segs in traces_out.items() if segs]
    traces_out = {panel: traces_out[panel] for panel in panel_keys}
    trace_meta = {panel: trace_meta[panel] for panel in panel_keys}
    t_grid_out = {panel: t_grid_out[panel] for panel in panel_keys}
    used_sr = {panel: used_sr[panel] for panel in panel_keys}
    panel_units = {panel: panel_units[panel] for panel in panel_keys}

    title_bits = []
    if annot_mode == "pool":
        annot_label = _join_preview(sorted(selected_annots), limit_items=3, limit_chars=52)
        title_bits.append(f"annots={annot_label}")
    else:
        annot_label = "stratified annots"
    if tag_mode == "pool":
        pooled_tags_label = _join_preview(sorted(selected_tags), limit_items=3, limit_chars=40)
        title_bits.append(f"tags={pooled_tags_label}")
    if ch_mode == "pool":
        title_bits.append(f"channels={len(selected_channels)} pooled")
    else:
        title_bits.append(f"channels={len(panel_keys)} panels")
    extra_title_bits = title_bits + extra_title_bits
    result = _build_trace_result(
        traces_out,
        trace_meta,
        t_grid_out,
        panel_keys,
        used_sr,
        summary_mode,
        annot_class=annot_label,
        n_events=n_events,
        transform_mode=transform_mode,
        baseline_mode=baseline_mode,
        outlier_mode=outlier_mode,
        outlier_sd_thresh=outlier_sd_thresh,
        units=panel_units,
        trace_count_label=trace_count_label,
        extra_title_bits=extra_title_bits,
        source_mode="lwf",
        view_pre_secs=pre_secs,
        view_post_secs=post_secs,
    )
    if contrast_axis != "none":
        panel_keys = [panel for panel in panel_keys if all(len(contrast_groups[panel][gl]) > 0 for gl in group_levels)]
        if not panel_keys:
            raise RuntimeError("The selected contrast did not leave two non-empty groups to compare.")
        contrast_stats = {}
        contrast_group_summaries = {}
        for panel in panel_keys:
            contrast_stats[panel] = _compute_two_group_pointwise_stats(
                contrast_groups[panel][group_levels[0]],
                contrast_groups[panel][group_levels[1]],
            )
            contrast_group_summaries[panel] = {}
            for gl in group_levels:
                group_result = _build_trace_result(
                    {panel: contrast_groups[panel][gl]},
                    {panel: []},
                    {panel: t_grid_out[panel]},
                    [panel],
                    {panel: used_sr[panel]},
                    summary_mode,
                    annot_class=annot_label,
                    n_events=len(contrast_groups[panel][gl]),
                    transform_mode=transform_mode,
                    units={panel: panel_units[panel]},
                    trace_count_label=trace_count_label,
                    extra_title_bits=[],
                    source_mode="lwf",
                    baseline_mode=baseline_mode,
                    outlier_mode=outlier_mode,
                    outlier_sd_thresh=outlier_sd_thresh,
                    view_pre_secs=pre_secs,
                    view_post_secs=post_secs,
                )
                contrast_group_summaries[panel][gl] = group_result
        result["channels"] = panel_keys
        result["traces"] = {panel: traces_out[panel] for panel in panel_keys}
        result["t_grid"] = {panel: t_grid_out[panel] for panel in panel_keys}
        result["sr"] = {panel: used_sr[panel] for panel in panel_keys}
        result["units"] = {panel: panel_units[panel] for panel in panel_keys}
        result["contrast"] = {
            "axis": contrast_axis,
            "labels": group_levels,
            "title": contrast_title,
            "layout": str(contrast.get("layout", "stacked")),
            "groups": contrast_group_summaries,
            "stats": contrast_stats,
            "unit_note": "Tests are descriptive only; without per-ID averaging, waves from the same ID are not independent.",
        }
        if average_by_id:
            result["contrast"]["unit_label"] = "ID means"
        else:
            result["contrast"]["unit_label"] = "waves"
    return result


def _load_lwf_folder(directory: str, recursive: bool = False):
    dataset = load_lwf_directory(directory, recursive=recursive)
    return dataset, format_lwf_summary(dataset), format_lwf_summary_compact(dataset)


def _make_select_button(label: str, slot):
    btn = QPushButton(label)
    btn.setFixedWidth(44)
    btn.clicked.connect(slot)
    return btn


def _safe_window_floor(seconds: float) -> float:
    if not np.isfinite(seconds) or seconds <= 0:
        return 0.0
    return float(np.floor(seconds * 10.0) / 10.0)


def _read_binary_covariates(path: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, sep=None, engine="python")
    if df.shape[1] < 2:
        raise ValueError("Covariate file must have at least two columns: ID and one binary variable.")
    id_col = str(df.columns[0])
    out = pd.DataFrame()
    out["LWF_ID"] = df.iloc[:, 0].astype(str).str.strip()
    if out["LWF_ID"].eq("").all():
        raise ValueError("First covariate column must contain IDs.")

    valid_cols = []
    for col in df.columns[1:]:
        series = df[col]
        norm = series.astype(str).str.strip()
        norm = norm.mask(norm.isin(["", "NA", "NaN", "nan"]))
        numeric = pd.to_numeric(norm, errors="coerce")
        bad = numeric[~numeric.isna() & ~numeric.isin([0, 1])]
        if not bad.empty:
            continue
        out[str(col)] = numeric
        valid_cols.append(str(col))
    if not valid_cols:
        raise ValueError("No usable binary covariate columns found. Expected only 0/1/NA values.")
    out = out.dropna(subset=["LWF_ID"]).drop_duplicates(subset=["LWF_ID"], keep="last")
    return out, valid_cols


# ---------------------------------------------------------------------------
# Tab widget
# ---------------------------------------------------------------------------

class WaveformTab(_ExplorerTab):
    """Peri-event waveform tab (single attached record)."""

    _sig_ok  = QtCore.Signal(object)
    _sig_err = QtCore.Signal(str)
    _sig_lwf_ok = QtCore.Signal(object, str)
    _sig_lwf_err = QtCore.Signal(str)

    def __init__(self, ctrl, parent=None):
        super().__init__(ctrl, parent)
        self._last_result = None
        self._lwf_data = None
        self._lwf_covariates = None
        self._lwf_covariate_path = ""
        self._mode = "record"
        self._pending_units = {}
        self._sig_ok.connect(self._on_ok,  Qt.QueuedConnection)
        self._sig_err.connect(self._on_err, Qt.QueuedConnection)
        self._sig_lwf_ok.connect(self._on_lwf_ok, Qt.QueuedConnection)
        self._sig_lwf_err.connect(self._on_lwf_err, Qt.QueuedConnection)
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
        row1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        btn_refresh = QPushButton("↻"); btn_refresh.setFixedWidth(30)
        btn_refresh.setToolTip("Reload channels/annotations from current record")

        combo_ann = _WidePopupComboBox(); combo_ann.setMinimumWidth(120)
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

        # row 2: plot window / alignment / primary action
        row2 = QWidget(); rl2 = QGridLayout(row2)
        rl2.setContentsMargins(0,0,0,0); rl2.setSpacing(6)
        row2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        spin_pre = QDoubleSpinBox(); spin_pre.setRange(0, 300); spin_pre.setValue(2)
        spin_pre.setSuffix(" s"); spin_pre.setDecimals(1); spin_pre.setFixedWidth(72)
        spin_pre.setToolTip("Pre-event window (seconds)")

        spin_post = QDoubleSpinBox(); spin_post.setRange(0, 300); spin_post.setValue(2)
        spin_post.setSuffix(" s"); spin_post.setDecimals(1); spin_post.setFixedWidth(72)
        spin_post.setToolTip("Post-event window (seconds)")

        combo_align = QComboBox(); combo_align.setFixedWidth(90)
        for key, lbl in [("start","Start"), ("mid","Midpoint"), ("stop","Stop")]:
            combo_align.addItem(lbl, key)
        combo_align.setToolTip("Align traces to event start / midpoint / stop")

        combo_baseline = QComboBox(); combo_baseline.setFixedWidth(150)
        combo_baseline.addItem("None", "none")
        combo_baseline.addItem("Baseline subtract", "subtract")
        combo_baseline.addItem("Normalize [-1,1]", "normalize")
        combo_baseline.addItem("Baseline + normalize", "subtract_normalize")
        combo_baseline.setCurrentIndex(combo_baseline.findData("subtract"))
        combo_baseline.setToolTip(
            "Center/scale each trace after any outlier handling. "
            "Normalization rescales each waveform to [-1, 1]."
        )

        combo_outlier = QComboBox(); combo_outlier.setFixedWidth(110)
        combo_outlier.addItem("No outliers", "none")
        combo_outlier.addItem("Winsorize", "winsorize")
        combo_outlier.addItem("Remove", "remove")
        combo_outlier.setToolTip(
            "Handle pointwise waveform outliers across traces before summary stats."
        )

        spin_outlier_sd = QDoubleSpinBox()
        spin_outlier_sd.setRange(0.1, 20.0)
        spin_outlier_sd.setDecimals(1)
        spin_outlier_sd.setSingleStep(0.5)
        spin_outlier_sd.setValue(3.0)
        spin_outlier_sd.setSuffix(" SD")
        spin_outlier_sd.setFixedWidth(86)
        spin_outlier_sd.setToolTip("Outlier threshold in standard deviations")

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

        chk_show_line = QCheckBox("Line")
        chk_show_line.setChecked(True)
        chk_show_line.setToolTip("Show the summary line (mean / median / variance)")

        chk_show_band = QCheckBox("Band")
        chk_show_band.setChecked(True)
        chk_show_band.setToolTip("Show the summary band (CI / SD / quantiles)")

        chk_summary_only = QCheckBox("Summary only")
        chk_summary_only.setToolTip(
            "Hide individual traces and show only the selected summary. "
            "Automatically locked on when too many traces are present."
        )

        btn_render = QPushButton("Render"); btn_render.setFixedWidth(96)
        btn_render.setMinimumHeight(30)
        btn_render.setStyleSheet(RENDER_BUTTON_STYLE)
        btn_render.setToolTip("Extract signal windows and draw traces")

        rl2.addWidget(QLabel("View pre:"), 0, 0); rl2.addWidget(spin_pre, 0, 1)
        rl2.addWidget(QLabel("View post:"), 0, 2); rl2.addWidget(spin_post, 0, 3)
        rl2.addWidget(QLabel("Align:"), 0, 4); rl2.addWidget(combo_align, 0, 5)
        rl2.setColumnStretch(6, 1)

        # row 3: display / preprocessing
        row3 = QWidget(); rl3 = QVBoxLayout(row3)
        rl3.setContentsMargins(0,0,0,0); rl3.setSpacing(4)
        row3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        display_row_a = QHBoxLayout()
        display_row_a.setContentsMargins(0, 0, 0, 0)
        display_row_a.setSpacing(6)
        display_row_a.addWidget(QLabel("Transform:"))
        display_row_a.addWidget(combo_transform)
        display_row_a.addSpacing(10)
        display_row_a.addWidget(QLabel("Summary:"))
        display_row_a.addWidget(combo_summary)
        display_row_a.addSpacing(8)
        display_row_a.addWidget(chk_show_line)
        display_row_a.addWidget(chk_show_band)
        display_row_a.addSpacing(16)
        display_row_a.addWidget(chk_summary_only)
        display_row_a.addStretch(1)
        rl3.addLayout(display_row_a)

        display_row_b = QHBoxLayout()
        display_row_b.setContentsMargins(0, 0, 0, 0)
        display_row_b.setSpacing(6)
        display_row_b.addWidget(QLabel("Baseline:"))
        display_row_b.addWidget(combo_baseline)
        display_row_b.addSpacing(12)
        display_row_b.addWidget(QLabel("Outliers:"))
        display_row_b.addWidget(combo_outlier)
        display_row_b.addWidget(spin_outlier_sd)
        display_row_b.addStretch(1)
        rl3.addLayout(display_row_b)

        # shared y-axis controls (applies to both current-record and .lwf renders)
        shared_axes_row = QWidget()
        shared_axes_layout = QHBoxLayout(shared_axes_row)
        shared_axes_layout.setContentsMargins(0, 0, 0, 0)
        shared_axes_layout.setSpacing(6)
        shared_axes_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

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

        shared_axes_layout.addWidget(QLabel("Y min:"))
        shared_axes_layout.addWidget(spin_ymin)
        shared_axes_layout.addWidget(chk_auto_ymin)
        shared_axes_layout.addSpacing(10)
        shared_axes_layout.addWidget(QLabel("Y max:"))
        shared_axes_layout.addWidget(spin_ymax)
        shared_axes_layout.addWidget(chk_auto_ymax)
        shared_axes_layout.addStretch(1)

        record_row4 = QWidget()
        record_row4_layout = QHBoxLayout(record_row4)
        record_row4_layout.setContentsMargins(0, 0, 0, 0)
        record_row4_layout.setSpacing(6)
        record_row4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        record_row4_layout.addStretch(1)
        record_row4_layout.addWidget(btn_render)

        record_tab = QWidget()
        record_layout = QVBoxLayout(record_tab)
        record_layout.setContentsMargins(0, 0, 0, 0)
        record_layout.setSpacing(2)
        record_layout.addWidget(row1)
        record_layout.addWidget(row2)
        record_layout.addWidget(row3)
        record_layout.addWidget(record_row4)
        record_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        lwf_tab = QWidget()
        lwf_layout = QVBoxLayout(lwf_tab)
        lwf_layout.setContentsMargins(0, 0, 0, 0)
        lwf_layout.setSpacing(2)

        lwf_row1 = QWidget(); lwf_rl1 = QGridLayout(lwf_row1)
        lwf_rl1.setContentsMargins(0, 0, 0, 0); lwf_rl1.setSpacing(6)
        lwf_row1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_load_lwf = QPushButton("Load .lwf…")
        btn_load_lwf.setToolTip("Load a folder of precomputed .lwf waveform shards")
        chk_lwf_recursive = QCheckBox("Recursive")
        chk_lwf_recursive.setToolTip("Recursively scan the selected folder for .lwf files")
        btn_load_cov = QPushButton("Load covariates…")
        btn_load_cov.setToolTip("Load an ID-level covariate file with first column = ID and binary 0/1/NA columns")
        lab_cov = QLabel("No covariates loaded.")
        lab_cov.setStyleSheet("QLabel { color:#8b949e; }")
        lab_cov.setMinimumWidth(180)
        combo_contrast = QComboBox()
        combo_contrast.setFixedWidth(130)
        combo_contrast.addItem("None", "none")
        combo_contrast.addItem("ID covariate", "id_group")
        combo_contrast.addItem("CH", "CH")
        combo_contrast.addItem("ANNOT", "ANNOT")
        combo_contrast.addItem("TAG", "TAG")
        combo_contrast.setToolTip("Select the single two-group contrast axis for pairwise comparison")
        lab_cov_choice = QLabel("Covariate:")
        combo_covariate = QComboBox()
        combo_covariate.setMinimumWidth(150)
        combo_covariate.setEnabled(False)
        combo_covariate.setToolTip("Choose the binary ID covariate to define the 0 vs 1 contrast")
        combo_contrast_layout = QComboBox()
        combo_contrast_layout.setFixedWidth(92)
        combo_contrast_layout.addItem("Stacked", "stacked")
        combo_contrast_layout.addItem("Overlay", "overlay")
        combo_contrast_layout.setToolTip("Show the two contrast groups as separate plots or overlaid in one plot")
        lwf_rl1.addWidget(btn_load_lwf, 0, 0)
        lwf_rl1.addWidget(chk_lwf_recursive, 0, 1)
        lwf_rl1.addWidget(btn_load_cov, 0, 2)
        lwf_rl1.addWidget(lab_cov, 0, 3)
        lwf_rl1.addWidget(QLabel("Contrast:"), 0, 4)
        lwf_rl1.addWidget(combo_contrast, 0, 5)
        lwf_rl1.addWidget(lab_cov_choice, 0, 6)
        lwf_rl1.addWidget(combo_covariate, 0, 7)
        lwf_rl1.addWidget(combo_contrast_layout, 0, 8)
        lwf_rl1.setColumnStretch(3, 1)

        lwf_summary = QLabel("No .lwf dataset loaded.")
        lwf_summary.setWordWrap(True)
        lwf_summary.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        lwf_summary.setStyleSheet(
            "QLabel { color:#c9d1d9; padding:3px 6px; border:1px solid #30363d; "
            "border-radius:6px; background:#111827; }"
        )
        lwf_summary.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        lwf_summary.setMinimumHeight(42)

        from .soappops import MultiSelectComboBox

        lwf_row2 = QWidget(); lwf_rl2 = QGridLayout(lwf_row2)
        lwf_rl2.setContentsMargins(0, 0, 0, 0); lwf_rl2.setSpacing(6)
        lwf_row2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        combo_lwf_ch = MultiSelectComboBox()
        combo_lwf_ch.setMinimumWidth(130)
        combo_lwf_ch.setMaximumWidth(220)
        combo_lwf_ch.lineEdit().setPlaceholderText("Select channels")
        btn_lwf_ch_all = _make_select_button("All", lambda: combo_lwf_ch.check_all())
        btn_lwf_ch_none = _make_select_button("None", lambda: combo_lwf_ch.clear_all())
        combo_lwf_ch_mode = QComboBox()
        combo_lwf_ch_mode.addItem("Stratify", "stratify")
        combo_lwf_ch_mode.addItem("Pool", "pool")
        combo_lwf_ch_mode.setFixedWidth(88)
        combo_lwf_ch_mode.setToolTip("Choose whether selected channels make separate panels or are pooled together")

        combo_lwf_annot = MultiSelectComboBox()
        combo_lwf_annot.setMinimumWidth(130)
        combo_lwf_annot.setMaximumWidth(220)
        combo_lwf_annot.lineEdit().setPlaceholderText("Select annotations")
        btn_lwf_annot_all = _make_select_button("All", lambda: combo_lwf_annot.check_all())
        btn_lwf_annot_none = _make_select_button("None", lambda: combo_lwf_annot.clear_all())
        combo_lwf_annot_mode = QComboBox()
        combo_lwf_annot_mode.addItem("Pool", "pool")
        combo_lwf_annot_mode.addItem("Stratify", "stratify")
        combo_lwf_annot_mode.setFixedWidth(88)
        combo_lwf_annot_mode.setToolTip("Choose whether selected annotations are pooled or split into separate panels")

        combo_lwf_tag = MultiSelectComboBox()
        combo_lwf_tag.setMinimumWidth(120)
        combo_lwf_tag.setMaximumWidth(200)
        combo_lwf_tag.lineEdit().setPlaceholderText("Select tags")
        btn_lwf_tag_all = _make_select_button("All", lambda: combo_lwf_tag.check_all())
        btn_lwf_tag_none = _make_select_button("None", lambda: combo_lwf_tag.clear_all())
        combo_lwf_tag_mode = QComboBox()
        combo_lwf_tag_mode.addItem("Pool", "pool")
        combo_lwf_tag_mode.addItem("Stratify", "stratify")
        combo_lwf_tag_mode.setFixedWidth(88)
        combo_lwf_tag_mode.setToolTip("Choose whether selected tags are pooled or split into separate panels")

        combo_lwf_id = MultiSelectComboBox()
        combo_lwf_id.setMinimumWidth(150)
        combo_lwf_id.setMaximumWidth(240)
        combo_lwf_id.lineEdit().setPlaceholderText("Select IDs")
        btn_lwf_id_all = _make_select_button("All", lambda: combo_lwf_id.check_all())
        btn_lwf_id_none = _make_select_button("None", lambda: combo_lwf_id.clear_all())

        lab_lwf_ch = QLabel("CH:")
        lab_lwf_ch.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lwf_rl2.addWidget(lab_lwf_ch, 0, 0)
        lwf_rl2.addWidget(combo_lwf_ch, 0, 1)
        lwf_rl2.addWidget(btn_lwf_ch_all, 0, 2)
        lwf_rl2.addWidget(btn_lwf_ch_none, 0, 3)
        lwf_rl2.addWidget(combo_lwf_ch_mode, 0, 4)
        lab_lwf_annot = QLabel("Annot:")
        lab_lwf_annot.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lwf_rl2.addWidget(lab_lwf_annot, 0, 5)
        lwf_rl2.addWidget(combo_lwf_annot, 0, 6)
        lwf_rl2.addWidget(btn_lwf_annot_all, 0, 7)
        lwf_rl2.addWidget(btn_lwf_annot_none, 0, 8)
        lwf_rl2.addWidget(combo_lwf_annot_mode, 0, 9)
        lab_lwf_tag = QLabel("Tag:")
        lab_lwf_tag.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lwf_rl2.addWidget(lab_lwf_tag, 1, 0)
        lwf_rl2.addWidget(combo_lwf_tag, 1, 1)
        lwf_rl2.addWidget(btn_lwf_tag_all, 1, 2)
        lwf_rl2.addWidget(btn_lwf_tag_none, 1, 3)
        lwf_rl2.addWidget(combo_lwf_tag_mode, 1, 4)
        lab_lwf_id = QLabel("ID:")
        lab_lwf_id.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lwf_rl2.addWidget(lab_lwf_id, 1, 5)
        lwf_rl2.addWidget(combo_lwf_id, 1, 6)
        lwf_rl2.addWidget(btn_lwf_id_all, 1, 7)
        lwf_rl2.addWidget(btn_lwf_id_none, 1, 8)
        lwf_rl2.setColumnStretch(1, 1)
        lwf_rl2.setColumnStretch(6, 1)

        lwf_row4 = QWidget(); lwf_rl4 = QGridLayout(lwf_row4)
        lwf_rl4.setContentsMargins(0, 0, 0, 0); lwf_rl4.setSpacing(6)
        lwf_row4.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        spin_lwf_pre = QDoubleSpinBox(); spin_lwf_pre.setRange(0, 300); spin_lwf_pre.setValue(2)
        spin_lwf_pre.setSuffix(" s"); spin_lwf_pre.setDecimals(1); spin_lwf_pre.setFixedWidth(72)
        spin_lwf_pre.setToolTip("Pre-event window (seconds)")

        spin_lwf_post = QDoubleSpinBox(); spin_lwf_post.setRange(0, 300); spin_lwf_post.setValue(2)
        spin_lwf_post.setSuffix(" s"); spin_lwf_post.setDecimals(1); spin_lwf_post.setFixedWidth(72)
        spin_lwf_post.setToolTip("Post-event window (seconds)")

        combo_lwf_align = QComboBox(); combo_lwf_align.setFixedWidth(90)
        for key, lbl in [("start","Start"), ("mid","Midpoint"), ("stop","Stop")]:
            combo_lwf_align.addItem(lbl, key)
        combo_lwf_align.setToolTip("Align traces to event start / midpoint / stop")

        combo_lwf_transform = QComboBox(); combo_lwf_transform.setFixedWidth(105)
        combo_lwf_transform.addItem("Raw", "raw")
        combo_lwf_transform.addItem("Rectified", "rectified")
        combo_lwf_transform.setToolTip("Apply a simple per-sample transform before summarizing")

        combo_lwf_summary = QComboBox(); combo_lwf_summary.setFixedWidth(125)
        combo_lwf_summary.addItem("Mean ± CI", "mean_ci")
        combo_lwf_summary.addItem("Mean ± SD", "mean_sd")
        combo_lwf_summary.addItem("Quantiles", "mean_quantiles")
        combo_lwf_summary.addItem("Variance", "variance")
        combo_lwf_summary.setToolTip("Summary statistic to plot across event-locked traces")

        chk_lwf_show_line = QCheckBox("Line")
        chk_lwf_show_line.setChecked(True)
        chk_lwf_show_line.setToolTip("Show the summary line (mean / median / variance)")

        chk_lwf_show_band = QCheckBox("Band")
        chk_lwf_show_band.setChecked(True)
        chk_lwf_show_band.setToolTip("Show the summary band (CI / SD / quantiles)")

        combo_lwf_baseline = QComboBox(); combo_lwf_baseline.setFixedWidth(150)
        combo_lwf_baseline.addItem("None", "none")
        combo_lwf_baseline.addItem("Baseline subtract", "subtract")
        combo_lwf_baseline.addItem("Normalize [-1,1]", "normalize")
        combo_lwf_baseline.addItem("Baseline + normalize", "subtract_normalize")
        combo_lwf_baseline.setCurrentIndex(combo_lwf_baseline.findData("subtract"))
        combo_lwf_baseline.setToolTip(
            "Center/scale each trace after any outlier handling. "
            "Normalization rescales each waveform to [-1, 1]."
        )

        combo_lwf_outlier = QComboBox(); combo_lwf_outlier.setFixedWidth(110)
        combo_lwf_outlier.addItem("No outliers", "none")
        combo_lwf_outlier.addItem("Winsorize", "winsorize")
        combo_lwf_outlier.addItem("Remove", "remove")
        combo_lwf_outlier.setToolTip(
            "Handle pointwise waveform outliers across traces before summary stats."
        )

        spin_lwf_outlier_sd = QDoubleSpinBox()
        spin_lwf_outlier_sd.setRange(0.1, 20.0)
        spin_lwf_outlier_sd.setDecimals(1)
        spin_lwf_outlier_sd.setSingleStep(0.5)
        spin_lwf_outlier_sd.setValue(3.0)
        spin_lwf_outlier_sd.setSuffix(" SD")
        spin_lwf_outlier_sd.setFixedWidth(86)
        spin_lwf_outlier_sd.setToolTip("Outlier threshold in standard deviations")

        chk_lwf_mean_by_id = QCheckBox("Average per ID")
        chk_lwf_mean_by_id.setToolTip("Average single-wave traces within each individual before plotting and summarizing")

        chk_lwf_summary_only = QCheckBox("Summary only")
        chk_lwf_summary_only.setChecked(True)
        chk_lwf_summary_only.setToolTip(
            "Hide individual traces and show only the selected summary. "
            "Automatically locked on when too many traces are present."
        )

        btn_render_lwf = QPushButton("Render"); btn_render_lwf.setFixedWidth(96)
        btn_render_lwf.setMinimumHeight(30)
        btn_render_lwf.setStyleSheet(RENDER_BUTTON_STYLE)
        btn_render_lwf.setToolTip("Render a cohort waveform plot from the loaded .lwf dataset")

        lwf_rl4.addWidget(QLabel("View pre:"), 0, 0); lwf_rl4.addWidget(spin_lwf_pre, 0, 1)
        lwf_rl4.addWidget(QLabel("View post:"), 0, 2); lwf_rl4.addWidget(spin_lwf_post, 0, 3)
        lwf_rl4.addWidget(QLabel("Align:"), 0, 4); lwf_rl4.addWidget(combo_lwf_align, 0, 5)
        lwf_rl4.addWidget(chk_lwf_mean_by_id, 0, 6)
        lwf_rl4.setColumnStretch(7, 1)

        lwf_row5 = QWidget(); lwf_rl5 = QVBoxLayout(lwf_row5)
        lwf_rl5.setContentsMargins(0, 0, 0, 0); lwf_rl5.setSpacing(4)
        lwf_row5.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lwf_display_row_a = QHBoxLayout()
        lwf_display_row_a.setContentsMargins(0, 0, 0, 0)
        lwf_display_row_a.setSpacing(6)
        lwf_display_row_a.addWidget(QLabel("Transform:"))
        lwf_display_row_a.addWidget(combo_lwf_transform)
        lwf_display_row_a.addSpacing(10)
        lwf_display_row_a.addWidget(QLabel("Summary:"))
        lwf_display_row_a.addWidget(combo_lwf_summary)
        lwf_display_row_a.addSpacing(8)
        lwf_display_row_a.addWidget(chk_lwf_show_line)
        lwf_display_row_a.addWidget(chk_lwf_show_band)
        lwf_display_row_a.addWidget(chk_lwf_summary_only)
        lwf_display_row_a.addStretch(1)
        lwf_rl5.addLayout(lwf_display_row_a)

        lwf_display_row_b = QHBoxLayout()
        lwf_display_row_b.setContentsMargins(0, 0, 0, 0)
        lwf_display_row_b.setSpacing(6)
        lwf_display_row_b.addWidget(QLabel("Baseline:"))
        lwf_display_row_b.addWidget(combo_lwf_baseline)
        lwf_display_row_b.addSpacing(12)
        lwf_display_row_b.addWidget(QLabel("Outliers:"))
        lwf_display_row_b.addWidget(combo_lwf_outlier)
        lwf_display_row_b.addWidget(spin_lwf_outlier_sd)
        lwf_display_row_b.addStretch(1)
        lwf_display_row_b.addWidget(btn_render_lwf)
        lwf_rl5.addLayout(lwf_display_row_b)

        lwf_layout.addWidget(lwf_row1)
        lwf_layout.addWidget(lwf_row2)
        lwf_layout.addWidget(lwf_row4)
        lwf_layout.addWidget(lwf_row5)
        lwf_layout.addWidget(lwf_summary)
        lwf_tab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        mode_tabs = QTabWidget()
        mode_tabs.addTab(record_tab, "Current Record")
        mode_tabs.addTab(lwf_tab, "Loaded .lwf")
        mode_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        inspector_row = QWidget()
        inspector_layout = QVBoxLayout(inspector_row)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.setSpacing(4)
        inspector_row.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        inspector_row_a = QHBoxLayout()
        inspector_row_a.setContentsMargins(0, 0, 0, 0)
        inspector_row_a.setSpacing(6)

        inspector_row_b = QHBoxLayout()
        inspector_row_b.setContentsMargins(0, 0, 0, 0)
        inspector_row_b.setSpacing(6)

        combo_view = QComboBox()
        combo_view.setFixedWidth(120)
        combo_view.addItem("Summary", "summary")
        combo_view.addItem("Wave Inspector", "inspect")
        combo_view.setToolTip("Switch between the population summary and a single-wave inspection view")

        combo_feature = QComboBox()
        combo_feature.setMinimumWidth(150)
        combo_feature.setToolTip("Rank waves by this feature before selecting one with the slider")

        lab_inspect_annot = QLabel("Inspect annot:")
        combo_inspect_annot = _WidePopupComboBox()
        combo_inspect_annot.setMinimumWidth(140)
        combo_inspect_annot.setToolTip("Restrict wave inspection to one annotation class at a time")

        chk_stable_auto_y = QCheckBox("Stable auto Y")
        chk_stable_auto_y.setChecked(True)
        chk_stable_auto_y.setToolTip("Keep auto-scaled y-limits fixed while scrolling through inspected waves")

        combo_metric_source = QComboBox()
        combo_metric_source.setFixedWidth(160)
        combo_metric_source.addItem("Visible window", "visible")
        combo_metric_source.addItem("Custom window", "custom")
        combo_metric_source.addItem("Annotation interval", "annot")
        combo_metric_source.setToolTip("Choose which segment is used to compute waveform-ranking metrics")

        lab_metric_pre = QLabel("Metric pre:")
        spin_metric_pre = QDoubleSpinBox()
        spin_metric_pre.setRange(0, 300)
        spin_metric_pre.setValue(2)
        spin_metric_pre.setSuffix(" s")
        spin_metric_pre.setDecimals(1)
        spin_metric_pre.setFixedWidth(72)
        spin_metric_pre.setToolTip("Custom pre-event interval used only for ranking metrics")

        lab_metric_post = QLabel("Metric post:")
        spin_metric_post = QDoubleSpinBox()
        spin_metric_post.setRange(0, 300)
        spin_metric_post.setValue(2)
        spin_metric_post.setSuffix(" s")
        spin_metric_post.setDecimals(1)
        spin_metric_post.setFixedWidth(72)
        spin_metric_post.setToolTip("Custom post-event interval used only for ranking metrics")

        slider_wave = QSlider(Qt.Horizontal)
        slider_wave.setMinimum(1)
        slider_wave.setMaximum(1)
        slider_wave.setValue(1)
        slider_wave.setEnabled(False)
        slider_wave.setToolTip("Select the wave rank within each panel after ordering by the chosen feature")

        lab_wave = QLabel("Wave 1 / 1")
        lab_wave.setMinimumWidth(92)
        lab_wave.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lab_wave.setStyleSheet("QLabel { color:#8b949e; }")

        inspector_row_a.addWidget(QLabel("View:"))
        inspector_row_a.addWidget(combo_view)
        inspector_row_a.addSpacing(8)
        inspector_row_a.addWidget(QLabel("Metrics:"))
        inspector_row_a.addWidget(combo_metric_source)
        inspector_row_a.addWidget(lab_metric_pre)
        inspector_row_a.addWidget(spin_metric_pre)
        inspector_row_a.addWidget(lab_metric_post)
        inspector_row_a.addWidget(spin_metric_post)
        inspector_row_a.addSpacing(8)
        inspector_row_a.addWidget(QLabel("Order by:"))
        inspector_row_a.addWidget(combo_feature)
        inspector_row_a.addWidget(lab_inspect_annot)
        inspector_row_a.addWidget(combo_inspect_annot)
        inspector_row_a.addSpacing(8)
        inspector_row_a.addWidget(chk_stable_auto_y)
        inspector_row_a.addStretch(1)

        inspector_row_b.addWidget(QLabel("Wave:"))
        inspector_row_b.addWidget(slider_wave, 1)
        inspector_row_b.addWidget(lab_wave)

        inspector_layout.addLayout(inspector_row_a)
        inspector_layout.addLayout(inspector_row_b)

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

        outer.addWidget(mode_tabs)
        outer.addWidget(shared_axes_row)
        outer.addWidget(inspector_row)
        outer.addWidget(canvas_scroll, 1)

        # store
        self._root        = root
        self._combo_ann   = combo_ann
        self._combo_ch    = combo_ch
        self._spin_pre    = spin_pre
        self._spin_post   = spin_post
        self._combo_align = combo_align
        self._combo_transform = combo_transform
        self._combo_summary = combo_summary
        self._chk_show_line = chk_show_line
        self._chk_show_band = chk_show_band
        self._combo_baseline = combo_baseline
        self._combo_outlier = combo_outlier
        self._spin_outlier_sd = spin_outlier_sd
        self._tabs_mode = mode_tabs
        self._combo_view = combo_view
        self._combo_metric_source = combo_metric_source
        self._lab_metric_pre = lab_metric_pre
        self._lab_metric_post = lab_metric_post
        self._spin_metric_pre = spin_metric_pre
        self._spin_metric_post = spin_metric_post
        self._combo_feature = combo_feature
        self._lab_inspect_annot = lab_inspect_annot
        self._combo_inspect_annot = combo_inspect_annot
        self._chk_stable_auto_y = chk_stable_auto_y
        self._slider_wave = slider_wave
        self._lab_wave = lab_wave
        self._inspector_row = inspector_row
        self._shared_axes_row = shared_axes_row
        self._tab_record = record_tab
        self._tab_lwf = lwf_tab
        self._lab_lwf_summary = lwf_summary
        self._lab_lwf_cov = lab_cov
        self._lab_lwf_cov_choice = lab_cov_choice
        self._chk_lwf_recursive = chk_lwf_recursive
        self._combo_lwf_contrast = combo_contrast
        self._combo_lwf_covariate = combo_covariate
        self._combo_lwf_contrast_layout = combo_contrast_layout
        self._combo_lwf_ch = combo_lwf_ch
        self._combo_lwf_annot = combo_lwf_annot
        self._combo_lwf_tag = combo_lwf_tag
        self._combo_lwf_id = combo_lwf_id
        self._combo_lwf_ch_mode = combo_lwf_ch_mode
        self._combo_lwf_annot_mode = combo_lwf_annot_mode
        self._combo_lwf_tag_mode = combo_lwf_tag_mode
        self._spin_lwf_pre = spin_lwf_pre
        self._spin_lwf_post = spin_lwf_post
        self._combo_lwf_align = combo_lwf_align
        self._combo_lwf_transform = combo_lwf_transform
        self._combo_lwf_summary = combo_lwf_summary
        self._chk_lwf_show_line = chk_lwf_show_line
        self._chk_lwf_show_band = chk_lwf_show_band
        self._combo_lwf_baseline = combo_lwf_baseline
        self._combo_lwf_outlier = combo_lwf_outlier
        self._spin_lwf_outlier_sd = spin_lwf_outlier_sd
        self._chk_lwf_mean_by_id = chk_lwf_mean_by_id
        self._chk_lwf_summary_only = chk_lwf_summary_only
        self._chk_auto_ymin = chk_auto_ymin
        self._chk_auto_ymax = chk_auto_ymax
        self._chk_summary_only = chk_summary_only
        self._spin_ymin   = spin_ymin
        self._spin_ymax   = spin_ymax

        # wire
        btn_refresh.clicked.connect(self.refresh_controls)
        btn_load_lwf.clicked.connect(self._load_lwf_trigger)
        btn_load_cov.clicked.connect(self._load_lwf_covariates_trigger)
        btn_render.clicked.connect(self._render_trigger)
        btn_render_lwf.clicked.connect(self._render_lwf_trigger)
        mode_tabs.currentChanged.connect(self._on_mode_tab_changed)
        combo_contrast.currentIndexChanged.connect(self._on_lwf_contrast_changed)
        chk_auto_ymin.toggled.connect(self._on_y_limit_toggle)
        chk_auto_ymax.toggled.connect(self._on_y_limit_toggle)
        combo_view.currentIndexChanged.connect(self._on_inspector_control_changed)
        combo_metric_source.currentIndexChanged.connect(self._on_inspector_control_changed)
        spin_metric_pre.valueChanged.connect(self._on_inspector_control_changed)
        spin_metric_post.valueChanged.connect(self._on_inspector_control_changed)
        combo_feature.currentIndexChanged.connect(self._on_inspector_control_changed)
        combo_inspect_annot.currentIndexChanged.connect(self._on_inspector_control_changed)
        chk_stable_auto_y.toggled.connect(self._redraw_cached)
        slider_wave.valueChanged.connect(self._on_inspector_control_changed)
        chk_summary_only.toggled.connect(self._redraw_cached)
        chk_lwf_summary_only.toggled.connect(self._redraw_cached)
        chk_show_line.toggled.connect(self._redraw_cached)
        chk_show_band.toggled.connect(self._redraw_cached)
        chk_lwf_show_line.toggled.connect(self._redraw_cached)
        chk_lwf_show_band.toggled.connect(self._redraw_cached)
        spin_ymin.valueChanged.connect(self._redraw_cached)
        spin_ymax.valueChanged.connect(self._redraw_cached)
        self._save_btn = QPushButton("Export…"); self._save_btn.setFixedWidth(80)
        rl1.addWidget(self._save_btn)
        self._save_btn.clicked.connect(self._save_figure)
        self._set_mode("record")
        self._on_lwf_contrast_changed()
        self._update_inspector_controls(None)
        QTimer.singleShot(0, self._sync_mode_tab_height)

    def _set_mode(self, mode: str):
        self._mode = mode
        tabs = getattr(self, "_tabs_mode", None)
        if mode == "lwf":
            if tabs is not None:
                tabs.setCurrentWidget(self._tab_lwf)
        else:
            if tabs is not None:
                tabs.setCurrentWidget(self._tab_record)

    def _on_mode_tab_changed(self, _index: int):
        tabs = getattr(self, "_tabs_mode", None)
        if tabs is None:
            return
        current = tabs.currentWidget()
        if current is self._tab_lwf:
            self._set_mode("lwf")
        else:
            self._set_mode("record")
        self._sync_mode_tab_height()
        self._refresh_mode_display()

    def _refresh_mode_display(self):
        result = self._last_result
        if self._mode == "record":
            if getattr(self.ctrl, "p", None) is None:
                self._update_inspector_controls(None)
                self._render_empty("No current record attached.")
                return
            if result is not None and result.get("source_mode") == "record":
                self._update_inspector_controls(result)
                self._draw(result)
                return
            self._update_inspector_controls(None)
            self._render_empty("No current-record waveform rendered.")
            return

        if self._mode == "lwf":
            if result is not None and result.get("source_mode") == "lwf":
                self._update_inspector_controls(result)
                self._draw(result)
                return
            if self._lwf_data is not None:
                self._update_inspector_controls(None)
                self._render_empty(self._lab_lwf_summary.text() or "No .lwf waveform rendered.")
                return
            self._update_inspector_controls(None)
            self._render_empty("No .lwf dataset loaded.")

    def _sync_mode_tab_height(self):
        tabs = getattr(self, "_tabs_mode", None)
        if tabs is None:
            return
        current = tabs.currentWidget()
        if current is None:
            return
        current.adjustSize()
        page_height = max(current.sizeHint().height(), current.minimumSizeHint().height())
        tabbar = tabs.tabBar()
        tabbar_height = tabbar.sizeHint().height() if tabbar is not None else 0
        frame = tabs.style().pixelMetric(QtWidgets.QStyle.PM_DefaultFrameWidth, None, tabs)
        total = page_height + tabbar_height + (frame * 2) + 4
        tabs.setMinimumHeight(total)
        tabs.setMaximumHeight(total)

    def _set_lwf_filter_items(self, combo, values):
        labels = [str(v) for v in values if str(v)]
        combo.set_items(labels, checked_labels=labels)

    def _on_lwf_contrast_changed(self, *_):
        contrast = str(getattr(self, "_combo_lwf_contrast", QComboBox()).currentData() or "none")
        combo_cov = getattr(self, "_combo_lwf_covariate", None)
        lab_cov = getattr(self, "_lab_lwf_cov_choice", None)
        combo_layout = getattr(self, "_combo_lwf_contrast_layout", None)
        has_covariates = bool(combo_cov is not None and combo_cov.count() > 0)
        if combo_cov is not None:
            combo_cov.setEnabled(contrast == "id_group" and has_covariates)
            combo_cov.setVisible(has_covariates)
        if lab_cov is not None:
            lab_cov.setVisible(has_covariates)
        if combo_layout is not None:
            combo_layout.setEnabled(contrast != "none")
        if contrast in {"CH", "ANNOT", "TAG"}:
            axis_combos = {
                "CH": getattr(self, "_combo_lwf_ch_mode", None),
                "ANNOT": getattr(self, "_combo_lwf_annot_mode", None),
                "TAG": getattr(self, "_combo_lwf_tag_mode", None),
            }
            for axis, combo in axis_combos.items():
                if combo is None:
                    continue
                target = "stratify" if axis == contrast else "pool"
                idx = combo.findData(target)
                if idx >= 0 and combo.currentIndex() != idx:
                    combo.setCurrentIndex(idx)
        elif contrast == "id_group":
            for name in ("_combo_lwf_ch_mode", "_combo_lwf_annot_mode", "_combo_lwf_tag_mode"):
                combo = getattr(self, name, None)
                if combo is None:
                    continue
                idx = combo.findData("pool")
                if idx >= 0 and combo.currentIndex() != idx:
                    combo.setCurrentIndex(idx)

    def _load_lwf_covariates_trigger(self):
        path, _ = open_file_name(
            self._root,
            "Select ID Covariate File",
            file_filter="Delimited text (*.txt *.tsv *.csv);;All files (*)",
        )
        if not path:
            return
        try:
            cov_df, cols = _read_binary_covariates(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self._root, "Waveform .lwf", str(e))
            return
        self._lwf_covariates = cov_df
        self._lwf_covariate_path = path
        self._combo_lwf_covariate.clear()
        self._combo_lwf_covariate.addItems(cols)
        preview = ", ".join(cols[:3])
        if len(cols) > 3:
            preview += f" +{len(cols) - 3} more"
        self._lab_lwf_cov.setText(f"Covariates: {path} ({len(cols)} binary fields: {preview})")
        idx = self._combo_lwf_contrast.findData("id_group")
        if idx >= 0:
            self._combo_lwf_contrast.setCurrentIndex(idx)
        self._on_lwf_contrast_changed()

    def _lwf_common_window(self, dataset, align_to: str = "mid"):
        waves_df = dataset.waves
        channels_df = dataset.channels
        if waves_df.empty or channels_df.empty:
            return 0.0, 0.0

        common_pre = None
        common_post = None
        valid = 0
        for row in waves_df.itertuples(index=False):
            if align_to == "start":
                center_sec = float(row.ANNOT_START_SEC)
            elif align_to == "stop":
                center_sec = float(row.ANNOT_STOP_SEC)
            else:
                center_sec = 0.5 * (float(row.ANNOT_START_SEC) + float(row.ANNOT_STOP_SEC))
            for block in row.BLOCKS.values():
                if block is None or block.n <= 0:
                    continue
                left = float(center_sec - float(block.data_start_sec))
                right = float(float(block.data_stop_sec) - center_sec)
                if not np.isfinite(left) or not np.isfinite(right):
                    continue
                valid += 1
                common_pre = left if common_pre is None else min(common_pre, left)
                common_post = right if common_post is None else min(common_post, right)

        if valid == 0:
            return 0.0, 0.0
        return max(0.0, float(common_pre or 0.0)), max(0.0, float(common_post or 0.0))

    def _populate_lwf_controls(self, dataset):
        waves_df = dataset.waves
        channels_df = dataset.channels
        files_df = dataset.files
        ch_values = sorted(channels_df["CH"].dropna().astype(str).unique().tolist()) if not channels_df.empty else []
        annot_values = sorted(waves_df["ANNOT"].dropna().astype(str).unique().tolist()) if not waves_df.empty else []
        tag_values = sorted(files_df["TAG"].dropna().astype(str).unique().tolist()) if not files_df.empty else []
        id_values = sorted(files_df["LWF_ID"].dropna().astype(str).unique().tolist()) if not files_df.empty else []
        self._set_lwf_filter_items(self._combo_lwf_ch, ch_values)
        self._set_lwf_filter_items(self._combo_lwf_annot, annot_values)
        self._set_lwf_filter_items(self._combo_lwf_tag, tag_values)
        self._set_lwf_filter_items(self._combo_lwf_id, id_values)
        common_pre, common_post = self._lwf_common_window(
            dataset,
            align_to=str(self._combo_lwf_align.currentData() or "mid"),
        )
        self._spin_lwf_pre.setValue(_safe_window_floor(common_pre))
        self._spin_lwf_post.setValue(_safe_window_floor(common_post))

    def _validate_lwf_contrast(self, filters, strategies):
        contrast = str(self._combo_lwf_contrast.currentData() or "none")
        if contrast == "none":
            return {"axis": "none"}

        if contrast == "id_group":
            if self._lwf_covariates is None or self._combo_lwf_covariate.count() == 0:
                raise RuntimeError("Load a binary ID covariate file before using ID-group contrast.")
            cov = str(self._combo_lwf_covariate.currentText()).strip()
            if not cov:
                raise RuntimeError("Select a binary ID covariate for the contrast.")
            return {"axis": "id_group", "covariate": cov}

        if strategies.get(contrast) != "stratify":
            raise RuntimeError(f"Set {contrast} to Stratify to use it as the contrast axis.")
        other_axes = {"CH", "ANNOT", "TAG"} - {contrast}
        for axis in other_axes:
            if strategies.get(axis) == "stratify":
                raise RuntimeError("Only one of CH / ANNOT / TAG may be stratified when contrast testing is enabled.")
        levels = [str(x) for x in filters.get(contrast, [])]
        if len(levels) != 2:
            raise RuntimeError(f"{contrast} contrast requires exactly two selected levels.")
        return {"axis": contrast, "levels": levels}

    def _set_canvas_height(self, nrows: int | None = None):
        """Let stacked waveform plots grow vertically and scroll instead of clipping."""
        canvas = self._ensure_canvas()
        if canvas is None:
            return
        nrows = max(1, int(nrows or 1))
        # Keep single-panel plots compact enough to fit the dock while still
        # allocating extra height for stacked multi-panel layouts.
        min_height = 90 + (nrows * 210) + ((nrows - 1) * 18)
        canvas.setMinimumHeight(min_height)
        canvas.setMaximumHeight(min_height)
        if self._canvas_host is not None:
            self._canvas_host.setMinimumHeight(min_height)
            self._canvas_host.setMaximumHeight(min_height)
        self._sync_canvas_width()

    # ------------------------------------------------------------------
    # Control refresh (call when switching to this tab)
    # ------------------------------------------------------------------

    def refresh_controls(self):
        """Repopulate annotation/channel combos and reset display mode."""
        if self._mode != "lwf":
            self._set_mode("record")
        self._refresh_ann_ch()

    def _refresh_ann_ch(self):
        """Update annotation and channel combo contents without changing display mode.

        Called on tab switch, sig_results_changed, and after record attach.
        """
        p = getattr(self.ctrl, "p", None)
        if p is None:
            return
        excluded = {"SleepStage", "N1", "N2", "N3", "R", "W", "L", "?"}
        view = getattr(getattr(self.ctrl, "ui", None), "tbl_desc_annots", None)
        model = view.model() if view is not None else None
        annots = []
        if model is not None:
            src = getattr(model, "sourceModel", lambda: None)()
            src = src if src is not None else model
            headers = [str(src.headerData(c, Qt.Horizontal) or "") for c in range(src.columnCount())]
            try:
                annot_col = headers.index("Annotations")
            except ValueError:
                annot_col = None
            if annot_col is not None:
                for r in range(src.rowCount()):
                    val = str(src.index(r, annot_col).data(Qt.DisplayRole) or "").strip()
                    if val:
                        annots.append(val)
        if not annots:
            try:
                annots = [str(c) for c in (p.edf.annots() or []) if str(c)]
            except Exception:
                annots = []

        all_annots = [a for a in _dedupe_preserve_order(annots) if a and a not in excluded]
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
        self._set_mode("record")
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
        baseline_mode = str(self._combo_baseline.currentData() or "subtract")
        outlier_mode = str(self._combo_outlier.currentData() or "none")
        outlier_sd_thresh = float(self._spin_outlier_sd.value())
        transform_mode = self._combo_transform.currentData()
        summary_mode = self._combo_summary.currentData()
        ns       = float(getattr(self.ctrl, "ns", 0.0))
        self._pending_units = self._get_channel_units(chs)

        fut = self.ctrl._exec.submit(
            _extract_traces, p, ns, ann, chs, pre, post, align_to, baseline_mode,
            outlier_mode, outlier_sd_thresh, transform_mode, summary_mode)
        def _done(_f=fut):
            try:
                self._sig_ok.emit(_f.result())
            except Exception as e:
                if isinstance(e, RuntimeError):
                    self._sig_err.emit(str(e))
                else:
                    self._sig_err.emit(traceback.format_exc())
        fut.add_done_callback(_done)

    def _on_ok(self, result):
        try:
            units = dict(result.get("units", {}))
            units.update(self._pending_units)
            result["units"] = units
            self._last_result = result
            self._update_inspector_controls(result)
            self._draw(result)
        finally:
            self._pending_units = {}
            self._end_work()

    def _on_err(self, tb_str):
        try:
            self._pending_units = {}
            msg = tb_str[:800]
            if "No waveform segments matched the requested window and filters." in tb_str:
                msg = (
                    "No waveform segments matched the requested window and filters.\n\n"
                    "The selected CH / ANNOT / TAG / ID combination may be valid, but the "
                    "requested Pre/Post window or alignment may exceed what was stored in "
                    "the .lwf files. Try a smaller window or load files generated with a wider dump window."
                )
            QtWidgets.QMessageBox.critical(
                self._root, "Waveform error", msg)
        finally:
            self._end_work()

    def _load_lwf_trigger(self):
        folder = existing_directory(self._root, "Select .lwf Folder")
        if not folder:
            return
        if not self._start_work("Loading .lwf waveforms…"):
            return
        recursive = bool(self._chk_lwf_recursive.isChecked())
        fut = self.ctrl._exec.submit(_load_lwf_folder, folder, recursive)

        def _done(_f=fut):
            try:
                dataset, summary_full, summary_compact = _f.result()
                self._sig_lwf_ok.emit(dataset, summary_full + "\n\n__COMPACT__\n" + summary_compact)
            except Exception as e:
                if isinstance(e, ValueError):
                    self._sig_lwf_err.emit(str(e))
                else:
                    self._sig_lwf_err.emit(traceback.format_exc())

        fut.add_done_callback(_done)

    def _on_lwf_ok(self, dataset, summary: str):
        try:
            if "\n\n__COMPACT__\n" in summary:
                summary_full, summary_compact = summary.split("\n\n__COMPACT__\n", 1)
            else:
                summary_full, summary_compact = summary, summary
            self._lwf_data = dataset
            self._populate_lwf_controls(dataset)
            self._set_mode("lwf")
            self._lab_lwf_summary.setText(summary_compact)
            self._update_inspector_controls(None)
            self._sync_mode_tab_height()
            self._render_empty(summary_compact)
            print(summary_full)
            QtWidgets.QMessageBox.information(self._root, "Waveform .lwf", summary_full)
        finally:
            self._end_work()

    def _render_lwf_trigger(self):
        self._set_mode("lwf")
        if self._lwf_data is None:
            QtWidgets.QMessageBox.warning(
                self._root, "Waveform .lwf",
                "Load a .lwf dataset before rendering cohort waveforms."
            )
            return
        filters = {
            "CH": self._combo_lwf_ch.checked_items(),
            "ANNOT": self._combo_lwf_annot.checked_items(),
            "TAG": self._combo_lwf_tag.checked_items(),
            "ID": self._combo_lwf_id.checked_items(),
        }
        strategies = {
            "CH": str(self._combo_lwf_ch_mode.currentData() or "stratify"),
            "ANNOT": str(self._combo_lwf_annot_mode.currentData() or "pool"),
            "TAG": str(self._combo_lwf_tag_mode.currentData() or "pool"),
        }
        if any(len(v) == 0 for v in filters.values()):
            QtWidgets.QMessageBox.warning(
                self._root, "Waveform .lwf",
                "Select at least one CH, ANNOT, TAG, and ID."
            )
            return
        try:
            contrast = self._validate_lwf_contrast(filters, strategies)
        except RuntimeError as e:
            QtWidgets.QMessageBox.warning(self._root, "Waveform .lwf", str(e))
            return
        contrast["layout"] = str(self._combo_lwf_contrast_layout.currentData() or "stacked")
        strategies["CONTRAST"] = contrast
        if contrast.get("axis") == "id_group":
            strategies["CONTRAST"]["covariates"] = self._lwf_covariates
        _, _, y_limits_valid = self._get_y_limits()
        if not y_limits_valid:
            QtWidgets.QMessageBox.warning(
                self._root, "Waveform",
                "Manual Y-axis minimum must be smaller than maximum."
            )
            return
        if not self._start_work("Rendering cohort waveforms…"):
            return

        fut = self.ctrl._exec.submit(
            _extract_lwf_traces,
            self._lwf_data,
            filters,
            strategies,
            float(self._spin_lwf_pre.value()),
            float(self._spin_lwf_post.value()),
            self._combo_lwf_align.currentData(),
            str(self._combo_lwf_baseline.currentData() or "subtract"),
            str(self._combo_lwf_outlier.currentData() or "none"),
            float(self._spin_lwf_outlier_sd.value()),
            self._combo_lwf_transform.currentData(),
            self._combo_lwf_summary.currentData(),
            bool(self._chk_lwf_mean_by_id.isChecked()),
        )

        def _done(_f=fut):
            try:
                self._sig_ok.emit(_f.result())
            except Exception as e:
                if isinstance(e, RuntimeError):
                    self._sig_err.emit(str(e))
                else:
                    self._sig_err.emit(traceback.format_exc())

        fut.add_done_callback(_done)

    def _on_lwf_err(self, tb_str):
        try:
            msg = "Could not load .lwf waveform data."
            if "uses .lwf version" in tb_str and "expects version 3" in tb_str:
                msg = (
                    "These .lwf files were written with an unsupported format version.\n\n"
                    "Please regenerate them with the current Luna WAVEFORMS command "
                    "and then load the folder again."
                )
            elif "No .lwf files found" in tb_str:
                msg = "No .lwf files were found in the selected folder."
            elif "Not a directory:" in tb_str:
                msg = "The selected path is not a directory."
            elif "Invalid .lwf magic" in tb_str:
                msg = "The selected folder contains files that are not valid .lwf waveform shards."
            QtWidgets.QMessageBox.critical(self._root, "Waveform .lwf", msg)
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

    def _on_inspector_control_changed(self, *_):
        self._update_inspector_controls(self._last_result)
        self._redraw_cached()

    def _inspector_mode(self):
        combo = getattr(self, "_combo_view", None)
        if combo is None:
            return "summary"
        return str(combo.currentData() or "summary")

    def _inspector_selected_annot(self):
        combo = getattr(self, "_combo_inspect_annot", None)
        if combo is None:
            return ""
        value = str(combo.currentData() or "")
        return "" if value in {"", "__all__"} else value

    def _stable_auto_y_enabled(self):
        chk = getattr(self, "_chk_stable_auto_y", None)
        return bool(chk is not None and chk.isChecked())

    def _set_inspector_row_visible(self, visible: bool):
        row = getattr(self, "_inspector_row", None)
        if row is not None:
            row.setVisible(bool(visible))
        shared_axes_row = getattr(self, "_shared_axes_row", None)
        if shared_axes_row is not None:
            shared_axes_row.setVisible(bool(visible))

    def _inspector_available_for_result(self, result) -> bool:
        if result is None or result.get("contrast"):
            return False
        metric_payload = self._metric_features_for_result(result)
        feature_maps = metric_payload.get("feature_maps", {})
        panel_count = sum(
            1
            for ch in result.get("channels", [])
            if len(feature_maps.get(ch, [])) > 0
        )
        return panel_count == 1

    def _metric_window_config(self, result=None):
        combo = getattr(self, "_combo_metric_source", None)
        spin_pre = getattr(self, "_spin_metric_pre", None)
        spin_post = getattr(self, "_spin_metric_post", None)
        source = "visible" if combo is None else str(combo.currentData() or "visible")
        custom_pre = float(spin_pre.value()) if spin_pre is not None else 0.0
        custom_post = float(spin_post.value()) if spin_post is not None else 0.0
        if result is not None and source == "visible":
            custom_pre = float(result.get("view_pre_secs", custom_pre))
            custom_post = float(result.get("view_post_secs", custom_post))
        return source, custom_pre, custom_post

    def _sync_metric_window_controls(self, result=None):
        combo = getattr(self, "_combo_metric_source", None)
        spin_pre = getattr(self, "_spin_metric_pre", None)
        spin_post = getattr(self, "_spin_metric_post", None)
        lab_pre = getattr(self, "_lab_metric_pre", None)
        lab_post = getattr(self, "_lab_metric_post", None)
        if combo is None or spin_pre is None or spin_post is None:
            return
        source, _, _ = self._metric_window_config(result)
        show_custom = source == "custom"
        for widget in (spin_pre, spin_post, lab_pre, lab_post):
            if widget is not None:
                widget.setVisible(show_custom)
                widget.setEnabled(show_custom)

    def _metric_features_for_result(self, result):
        source, custom_pre, custom_post = self._metric_window_config(result)
        cache = result.setdefault("_metric_feature_cache", {})
        cache_key = (source, round(float(custom_pre), 4), round(float(custom_post), 4))
        if cache_key in cache:
            return cache[cache_key]

        trace_meta = result.get("trace_meta", {})
        traces = result.get("traces", {})
        t_grid = result.get("t_grid", {})
        sr_map = result.get("sr", {})
        channels = list(result.get("channels", []))

        if source == "visible":
            feature_maps = {
                ch: [dict(meta.get("features", {}) or {}) for meta in trace_meta.get(ch, [])]
                for ch in channels
            }
            feature_names = list(result.get("feature_names", []))
        else:
            feature_maps = {}
            feature_names = [key for key, _ in INSPECTOR_LOCAL_FEATURES]
            for ch in channels:
                sr = float(sr_map.get(ch, np.nan))
                ch_t = np.asarray(t_grid.get(ch, []), dtype=float)
                ch_maps = []
                for trace, meta in zip(traces.get(ch, []), trace_meta.get(ch, [])):
                    mask = _metric_window_mask(ch_t, meta, source, custom_pre, custom_post)
                    seg = np.asarray(trace, dtype=float)[mask] if mask.size else np.asarray([], dtype=float)
                    ch_maps.append(_compute_local_trace_features(seg, sr=sr))
                feature_maps[ch] = ch_maps

        payload = {
            "source": source,
            "custom_pre": custom_pre,
            "custom_post": custom_post,
            "feature_names": feature_names,
            "feature_maps": feature_maps,
            "label": _metric_window_label(source, custom_pre, custom_post),
        }
        cache[cache_key] = payload
        return payload

    def _inspector_panel_rows(self, result, panel_label):
        trace_list = list(result.get("traces", {}).get(panel_label, []))
        trace_meta = list(result.get("trace_meta", {}).get(panel_label, []))
        metric_payload = self._metric_features_for_result(result)
        panel_features = list(metric_payload.get("feature_maps", {}).get(panel_label, []))
        selected_annot = self._inspector_selected_annot()
        rows = []
        for idx, (trace, meta, features) in enumerate(zip(trace_list, trace_meta, panel_features)):
            if selected_annot and str(meta.get("annot", "")) != selected_annot:
                continue
            rows.append((idx, np.asarray(trace, dtype=float), meta, dict(features or {})))
        return rows

    def _inspector_auto_limits_for_panel(self, result, panel_label):
        if not self._stable_auto_y_enabled():
            return None, None
        rows = self._inspector_panel_rows(result, panel_label)
        if not rows:
            return None, None
        mins = []
        maxs = []
        for _, trace, _, _ in rows:
            finite = trace[np.isfinite(trace)]
            if finite.size == 0:
                continue
            mins.append(float(np.min(finite)))
            maxs.append(float(np.max(finite)))
        if not mins or not maxs:
            return None, None
        return min(mins), max(maxs)

    def _update_inspector_controls(self, result):
        combo_feature = getattr(self, "_combo_feature", None)
        combo_annot = getattr(self, "_combo_inspect_annot", None)
        lab_annot = getattr(self, "_lab_inspect_annot", None)
        slider = getattr(self, "_slider_wave", None)
        lab_wave = getattr(self, "_lab_wave", None)
        combo_view = getattr(self, "_combo_view", None)
        combo_metric = getattr(self, "_combo_metric_source", None)
        if combo_feature is None or combo_annot is None or slider is None or lab_wave is None or combo_view is None:
            return

        combo_feature.blockSignals(True)
        combo_annot.blockSignals(True)
        slider.blockSignals(True)
        self._sync_metric_window_controls(result)

        if result is None:
            self._set_inspector_row_visible(False)
            combo_feature.clear()
            combo_feature.setEnabled(False)
            combo_annot.clear()
            combo_annot.setEnabled(False)
            combo_annot.setVisible(False)
            if lab_annot is not None:
                lab_annot.setVisible(False)
            slider.setMinimum(1)
            slider.setMaximum(1)
            slider.setValue(1)
            slider.setEnabled(False)
            lab_wave.setText("Wave 1 / 1")
            combo_view.setEnabled(False)
            if combo_metric is not None:
                combo_metric.setEnabled(False)
            combo_feature.blockSignals(False)
            combo_annot.blockSignals(False)
            slider.blockSignals(False)
            return

        self._set_inspector_row_visible(True)
        inspector_available = self._inspector_available_for_result(result)
        if not inspector_available:
            idx = combo_view.findData("summary")
            if idx >= 0 and combo_view.currentIndex() != idx:
                combo_view.setCurrentIndex(idx)
        combo_view.setEnabled(inspector_available)
        combo_view.setToolTip(
            "Switch between the population summary and a single-wave inspection view"
            if inspector_available else
            "Wave Inspector is only available when rendering a single panel/channel."
        )
        if combo_metric is not None:
            combo_metric.setEnabled(inspector_available)
        if result.get("contrast"):
            combo_view.setEnabled(False)
            combo_feature.clear()
            combo_feature.setEnabled(False)
            combo_annot.clear()
            combo_annot.setEnabled(False)
            combo_annot.setVisible(False)
            if lab_annot is not None:
                lab_annot.setVisible(False)
            slider.setMinimum(1)
            slider.setMaximum(1)
            slider.setValue(1)
            slider.setEnabled(False)
            lab_wave.setText("Wave 1 / 1")
            if combo_metric is not None:
                combo_metric.setEnabled(False)
            combo_feature.blockSignals(False)
            combo_annot.blockSignals(False)
            slider.blockSignals(False)
            return

        inspect_mode = inspector_available and combo_view.currentData() == "inspect"
        annot_values = sorted(
            {
                str(meta.get("annot", "")).strip()
                for metas in result.get("trace_meta", {}).values()
                for meta in metas
                if str(meta.get("annot", "")).strip()
            }
        )
        show_annot_filter = inspect_mode and result.get("source_mode") == "lwf" and len(annot_values) > 1
        current_annot = self._inspector_selected_annot()
        combo_annot.clear()
        if show_annot_filter:
            for annot in annot_values:
                combo_annot.addItem(annot, annot)
            idx = combo_annot.findData(current_annot)
            combo_annot.setCurrentIndex(idx if idx >= 0 else 0)
        combo_annot.setEnabled(show_annot_filter)
        combo_annot.setVisible(show_annot_filter)
        if lab_annot is not None:
            lab_annot.setVisible(show_annot_filter)

        metric_payload = self._metric_features_for_result(result)
        feature_names = list(metric_payload.get("feature_names", []))
        current_key = str(combo_feature.currentData() or "")
        combo_feature.clear()
        for key in feature_names:
            combo_feature.addItem(_feature_label(key), key)
        if combo_feature.count():
            idx = combo_feature.findData(current_key)
            combo_feature.setCurrentIndex(idx if idx >= 0 else 0)
        selected_key = str(combo_feature.currentData() or "")

        feature_maps = metric_payload.get("feature_maps", {})
        channels = [ch for ch in result.get("channels", []) if len(feature_maps.get(ch, [])) > 0]
        selected_annot = self._inspector_selected_annot()
        counts = [
            sum(
                1
                for meta, features in zip(result.get("trace_meta", {}).get(ch, []), feature_maps.get(ch, []))
                if (not selected_annot or str(meta.get("annot", "")) == selected_annot)
                and np.isfinite(features.get(selected_key, np.nan))
            )
            for ch in channels
        ]
        positive_counts = [count for count in counts if count > 0]
        common_n = min(positive_counts) if positive_counts else 0
        slider.setMinimum(1)
        slider.setMaximum(max(1, common_n))
        if slider.value() > max(1, common_n):
            slider.setValue(max(1, common_n))
        combo_feature.setEnabled(inspect_mode and combo_feature.count() > 0 and common_n > 0)
        slider.setEnabled(inspect_mode and common_n > 0)
        if inspector_available:
            lab_wave.setText(f"Wave {slider.value()} / {max(1, common_n)}")
        else:
            lab_wave.setText("Single-panel only")

        combo_feature.blockSignals(False)
        combo_annot.blockSignals(False)
        slider.blockSignals(False)

    def _set_summary_only_locked(self, locked: bool):
        chk = getattr(self, "_chk_summary_only", None)
        if chk is None:
            return
        chk.blockSignals(True)
        if locked:
            chk.setChecked(True)
            chk.setEnabled(False)
            chk.setToolTip(
                f"Summary-only mode is required when a channel has more than "
                f"{TRACE_SUMMARY_THRESHOLD} traces."
            )
        else:
            chk.setEnabled(True)
            chk.setToolTip(
                "Hide individual traces and show only the selected summary. "
                "Automatically locked on when too many traces are present."
            )
        chk.blockSignals(False)

    def _summary_only_checkbox_for_result(self, result):
        if result.get("source_mode") == "lwf":
            return getattr(self, "_chk_lwf_summary_only", None)
        return getattr(self, "_chk_summary_only", None)

    def _summary_component_state_for_result(self, result):
        if result.get("source_mode") == "lwf":
            line_chk = getattr(self, "_chk_lwf_show_line", None)
            band_chk = getattr(self, "_chk_lwf_show_band", None)
        else:
            line_chk = getattr(self, "_chk_show_line", None)
            band_chk = getattr(self, "_chk_show_band", None)
        return (
            True if line_chk is None else bool(line_chk.isChecked()),
            True if band_chk is None else bool(band_chk.isChecked()),
        )

    def _set_summary_only_locked_for_result(self, result, locked: bool):
        chk = self._summary_only_checkbox_for_result(result)
        if chk is None:
            return
        chk.blockSignals(True)
        if locked:
            chk.setChecked(True)
            chk.setEnabled(False)
            chk.setToolTip(
                f"Summary-only mode is required when a channel has more than "
                f"{TRACE_SUMMARY_THRESHOLD} traces."
            )
        else:
            chk.setEnabled(True)
            chk.setToolTip(
                "Hide individual traces and show only the selected summary. "
                "Automatically locked on when too many traces are present."
            )
        chk.blockSignals(False)

    def _draw_waveform_axis(self, ax, label, t_vals, trace_list, summary_payload, *,
                             data_key=None,
                             units, color, summary_mode, summary_only_requested,
                             show_summary_line, show_summary_band, show_xlabel,
                             y_min, y_max):
        data_key = label if data_key is None else data_key
        ax.set_facecolor(BG)
        n_traces = len(trace_list)
        summary_only = summary_only_requested or n_traces > TRACE_SUMMARY_THRESHOLD
        if not summary_only:
            if n_traces <= 10:
                trace_alpha = 0.75
                trace_width = 1.2
            elif n_traces <= 50:
                trace_alpha = 0.45
                trace_width = 0.9
            elif n_traces <= 250:
                trace_alpha = 0.22
                trace_width = 0.55
            else:
                trace_alpha = 0.15
                trace_width = 0.4
            for seg in trace_list:
                ax.plot(t_vals, seg, color=color, linewidth=trace_width, alpha=trace_alpha)

        mean_d = summary_payload["mean"]
        ci_lo = summary_payload["ci_lo"]
        ci_hi = summary_payload["ci_hi"]
        std_d = summary_payload.get("std", {})
        var_d = summary_payload.get("var", {})
        median_d = summary_payload.get("median", {})
        q05_d = summary_payload.get("q05", {})
        q25_d = summary_payload.get("q25", {})
        q75_d = summary_payload.get("q75", {})
        q95_d = summary_payload.get("q95", {})

        if summary_mode == "variance":
            vals = var_d.get(data_key)
            if vals is not None and show_summary_line:
                ax.plot(t_vals, vals, color=color, linewidth=2.0, alpha=0.95)
        elif summary_mode == "mean_quantiles":
            if show_summary_band and q05_d.get(data_key) is not None and q95_d.get(data_key) is not None:
                ax.fill_between(t_vals, q05_d[data_key], q95_d[data_key], color=color, alpha=0.12, linewidth=0)
            if show_summary_band and q25_d.get(data_key) is not None and q75_d.get(data_key) is not None:
                ax.fill_between(t_vals, q25_d[data_key], q75_d[data_key], color=color, alpha=0.24, linewidth=0)
            if show_summary_line and median_d.get(data_key) is not None:
                ax.plot(t_vals, median_d[data_key], color=color, linewidth=2.0, alpha=0.95)
            if summary_only and show_summary_line and data_key in mean_d:
                ax.plot(t_vals, mean_d[data_key], color="#ffffff", linewidth=0.8, alpha=0.65)
        elif summary_mode == "mean_sd":
            if show_summary_band and std_d.get(data_key) is not None:
                ax.fill_between(t_vals, mean_d[data_key] - std_d[data_key], mean_d[data_key] + std_d[data_key], color=color, alpha=0.22)
            if show_summary_line and data_key in mean_d:
                ax.plot(t_vals, mean_d[data_key], color=color, linewidth=1.8)
        else:
            if show_summary_band and data_key in ci_lo and data_key in ci_hi:
                ax.fill_between(t_vals, ci_lo[data_key], ci_hi[data_key], color=color, alpha=0.25)
            if show_summary_line and data_key in mean_d:
                ax.plot(t_vals, mean_d[data_key], color=color, linewidth=1.8)

        ax.axvline(0, color="#ffffff", lw=0.7, ls="--", alpha=0.55)
        ax.axhline(0, color=GRID, lw=0.4, alpha=0.7)
        if len(t_vals) >= 2:
            ax.set_xlim(float(t_vals[0]), float(t_vals[-1]))
        ylabel = units or ""
        if summary_mode == "variance" and ylabel:
            ylabel = f"{ylabel}^2"
        self._style_ax(ax, title=label, ylabel=ylabel)
        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)
        if show_xlabel:
            ax.set_xlabel("Time relative to event (s)", color=FG, fontsize=8)
        else:
            ax.set_xticklabels([])

    def _draw_contrast_axis(self, ax, label, t_vals, stats_payload, *, show_xlabel):
        ax.set_facecolor(BG)
        ax.plot(t_vals, stats_payload["mean_diff"], color="#f9844a", linewidth=1.8, label="Mean diff")
        ax.axvline(0, color="#ffffff", lw=0.7, ls="--", alpha=0.55)
        ax.axhline(0, color="#e5e7eb", lw=1.1, ls="--", alpha=0.95, zorder=0)
        ax2 = ax.twinx()
        ax2.plot(t_vals, stats_payload["neglogp"], color="#90be6d", linewidth=1.0, alpha=0.8, label="-log10(p)")
        if len(t_vals) >= 2:
            ax.set_xlim(float(t_vals[0]), float(t_vals[-1]))
        self._style_ax(ax, title=f"{label} | difference / stats", ylabel="Mean diff")
        ax2.tick_params(axis="y", colors=FG, labelsize=7)
        ax2.spines["right"].set_color(GRID)
        ax2.set_ylabel("-log10(p)", color=FG, fontsize=8)
        if show_xlabel:
            ax.set_xlabel("Time relative to event (s)", color=FG, fontsize=8)
        else:
            ax.set_xticklabels([])

    def _draw_overlay_contrast_axis(self, ax, label, t_vals, group_labels, group_payloads, *,
                                    units, summary_mode, summary_only_requested,
                                    show_summary_line, show_summary_band, show_xlabel,
                                    y_min, y_max):
        colors = ["#4cc9f0", "#f9844a"]
        ax.set_facecolor(BG)
        for idx, gl in enumerate(group_labels):
            payload = group_payloads[gl]
            self._draw_waveform_axis(
                ax,
                f"{label} | {gl}",
                t_vals,
                payload["traces"][label],
                payload,
                data_key=label,
                units=units,
                color=colors[idx % len(colors)],
                summary_mode=summary_mode,
                summary_only_requested=summary_only_requested,
                show_summary_line=show_summary_line,
                show_summary_band=show_summary_band,
                show_xlabel=show_xlabel,
                y_min=y_min,
                y_max=y_max,
            )
        self._style_ax(ax, title=f"{label} | {group_labels[0]} vs {group_labels[1]}", ylabel=units or "")

    def _inspector_selection_for_panel(self, result, panel_label):
        rows = self._inspector_panel_rows(result, panel_label)
        if not rows:
            return None
        metric_payload = self._metric_features_for_result(result)
        feature_key = str(getattr(self, "_combo_feature", QComboBox()).currentData() or "")
        if not feature_key:
            return None
        ranked = []
        for idx, trace, meta, features in rows:
            value = features.get(feature_key, np.nan)
            ranked.append((idx, value, meta, trace))
        ranked = [item for item in ranked if np.isfinite(item[1])]
        if not ranked:
            return None
        ranked.sort(key=lambda item: item[1])
        slider = getattr(self, "_slider_wave", None)
        rank = 0 if slider is None else max(0, min(len(ranked) - 1, int(slider.value()) - 1))
        idx, value, meta, trace = ranked[rank]
        return {
            "index": idx,
            "rank": rank + 1,
            "count": len(ranked),
            "value": float(value),
            "meta": meta,
            "trace": np.asarray(trace, dtype=float),
            "feature_key": feature_key,
            "metric_label": str(metric_payload.get("label", "")),
        }

    def _draw_inspector_axis(self, ax, label, t_vals, selection, *, units, color, show_xlabel, y_min, y_max):
        ax.set_facecolor(BG)
        trace = selection["trace"]
        meta = selection["meta"]
        feature_key = selection["feature_key"]
        title = (
            f"{label} | {_feature_label(feature_key)}={selection['value']:.3g} | "
            f"wave {selection['rank']}/{selection['count']}"
        )
        if selection.get("metric_label"):
            title += f" | {selection['metric_label']}"
        if meta.get("lwf_id"):
            title += f" | ID={meta['lwf_id']}"
        elif meta.get("instance"):
            title += f" | inst={meta['instance']}"
        ax.plot(t_vals, trace, color=color, linewidth=2.0, alpha=0.98)
        ax.fill_between(t_vals, 0.0, trace, color=color, alpha=0.10)
        ax.axvline(0, color="#ffffff", lw=0.7, ls="--", alpha=0.55)
        ax.axhline(0, color=GRID, lw=0.4, alpha=0.7)
        if len(t_vals) >= 2:
            ax.set_xlim(float(t_vals[0]), float(t_vals[-1]))
        self._style_ax(ax, title=title, ylabel=units or "")
        if y_min is not None or y_max is not None:
            ax.set_ylim(bottom=y_min, top=y_max)
        if show_xlabel:
            ax.set_xlabel("Time relative to event (s)", color=FG, fontsize=8)
        else:
            ax.set_xticklabels([])

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
        baseline_mode = result.get("baseline_mode", "subtract")
        outlier_mode = result.get("outlier_mode", "none")
        outlier_sd_thresh = result.get("outlier_sd_thresh", np.nan)
        n_ev      = result["n_events"]
        ann_cls   = result["annot_class"]
        units     = result.get("units", {})
        trace_count_label = result.get("trace_count_label", "events")
        extra_title_bits = result.get("extra_title_bits", [])
        summary_only_chk = self._summary_only_checkbox_for_result(result)
        show_summary_line, show_summary_band = self._summary_component_state_for_result(result)
        inspect_mode = self._inspector_mode() == "inspect" and not result.get("contrast")

        contrast = result.get("contrast")
        chs_with_data = [ch for ch in channels if ch in mean_d]
        forced_summary_only = any(
            len(traces.get(ch, [])) > TRACE_SUMMARY_THRESHOLD
            for ch in chs_with_data
        )
        self._set_summary_only_locked_for_result(result, forced_summary_only)
        summary_only_requested = (
            forced_summary_only
            or summary_mode == "variance"
            or (summary_only_chk.isChecked() if summary_only_chk is not None else False)
        )

        if not chs_with_data:
            self._set_canvas_height()
            self._render_empty("No signal data extracted.\n"
                               "Check that channels are loaded and event times are valid.")
            return

        contrast_layout = str(contrast.get("layout", "stacked")) if contrast else "stacked"
        n = len(chs_with_data) * ((2 if contrast_layout == "overlay" else 3) if contrast else 1)
        canvas = self._ensure_canvas()
        self._set_canvas_height(n)
        fig = canvas.figure; fig.clear(); fig.patch.set_facecolor(BG)

        axes = fig.subplots(n, 1, squeeze=False)
        fig.subplots_adjust(hspace=0.42, left=0.10, right=0.94, top=0.82, bottom=0.16)
        selected_annot = self._inspector_selected_annot() if inspect_mode else ""
        result["_inspect_annot_label"] = selected_annot
        title_bits = _compact_waveform_title_parts(result, inspect_mode=inspect_mode, contrast_layout=contrast_layout)
        if not inspect_mode:
            title_bits.extend([str(bit) for bit in extra_title_bits if bit])
        fig.suptitle(
            _format_waveform_suptitle(title_bits),
            color=FG,
            fontsize=10,
            y=0.97,
        )
        y_min, y_max, _ = self._get_y_limits()

        colors = ["#4cc9f0", "#f9844a", "#06d6a0", "#a78bfa",
                  "#ffd166", "#f72585", "#90be6d", "#ff6b6b"]

        if contrast:
            group_labels = contrast["labels"]
            for idx, ch in enumerate(chs_with_data):
                ch_t_grid = t_grid.get(ch) if isinstance(t_grid, dict) else t_grid
                if ch_t_grid is None or len(ch_t_grid) == 0:
                    continue
                g0 = contrast["groups"][ch][group_labels[0]]
                g1 = contrast["groups"][ch][group_labels[1]]
                if contrast_layout == "overlay":
                    base = idx * 2
                    self._draw_overlay_contrast_axis(
                        axes[base][0], ch, ch_t_grid, group_labels,
                        {group_labels[0]: g0, group_labels[1]: g1},
                        units=units.get(ch, ""), summary_mode=summary_mode,
                        summary_only_requested=summary_only_requested, show_summary_line=show_summary_line,
                        show_summary_band=show_summary_band, show_xlabel=False, y_min=y_min, y_max=y_max,
                    )
                    self._draw_contrast_axis(
                        axes[base + 1][0], ch, ch_t_grid, contrast["stats"][ch],
                        show_xlabel=(idx == len(chs_with_data) - 1),
                    )
                else:
                    base = idx * 3
                    self._draw_waveform_axis(
                        axes[base][0], f"{ch} | {group_labels[0]}", ch_t_grid, g0["traces"][ch], g0,
                        data_key=ch,
                        units=units.get(ch, ""), color=colors[idx % len(colors)], summary_mode=summary_mode,
                        summary_only_requested=summary_only_requested, show_summary_line=show_summary_line,
                        show_summary_band=show_summary_band, show_xlabel=False, y_min=y_min, y_max=y_max,
                    )
                    self._draw_waveform_axis(
                        axes[base + 1][0], f"{ch} | {group_labels[1]}", ch_t_grid, g1["traces"][ch], g1,
                        data_key=ch,
                        units=units.get(ch, ""), color=colors[(idx + 1) % len(colors)], summary_mode=summary_mode,
                        summary_only_requested=summary_only_requested, show_summary_line=show_summary_line,
                        show_summary_band=show_summary_band, show_xlabel=False, y_min=y_min, y_max=y_max,
                    )
                    self._draw_contrast_axis(
                        axes[base + 2][0], ch, ch_t_grid, contrast["stats"][ch],
                        show_xlabel=(idx == len(chs_with_data) - 1),
                    )
            note = contrast.get("unit_note", "")
            if note:
                fig.text(0.5, 0.01, note, ha="center", va="bottom", color="#8b949e", fontsize=7)
        else:
            for ch_idx, ch in enumerate(chs_with_data):
                ch_t_grid = t_grid.get(ch) if isinstance(t_grid, dict) else t_grid
                if ch_t_grid is None or len(ch_t_grid) == 0:
                    continue
                if inspect_mode:
                    selection = self._inspector_selection_for_panel(result, ch)
                    panel_y_min = y_min
                    panel_y_max = y_max
                    if panel_y_min is None and panel_y_max is None:
                        panel_y_min, panel_y_max = self._inspector_auto_limits_for_panel(result, ch)
                    if selection is None:
                        ax = axes[ch_idx][0]
                        ax.set_facecolor(BG)
                        msg = "No finite traces for the selected inspector filter."
                        self._style_ax(ax, title=ch, ylabel=units.get(ch, ""))
                        ax.text(
                            0.5, 0.5, msg,
                            color=FG, ha="center", va="center", fontsize=9,
                            transform=ax.transAxes, wrap=True, multialignment="center",
                        )
                        ax.axvline(0, color="#ffffff", lw=0.7, ls="--", alpha=0.55)
                        ax.axhline(0, color=GRID, lw=0.4, alpha=0.7)
                        if len(ch_t_grid) >= 2:
                            ax.set_xlim(float(ch_t_grid[0]), float(ch_t_grid[-1]))
                        if panel_y_min is not None or panel_y_max is not None:
                            ax.set_ylim(bottom=panel_y_min, top=panel_y_max)
                        if ch_idx == len(chs_with_data) - 1:
                            ax.set_xlabel("Time relative to event (s)", color=FG, fontsize=8)
                        else:
                            ax.set_xticklabels([])
                    else:
                        self._draw_inspector_axis(
                            axes[ch_idx][0], ch, ch_t_grid, selection,
                            units=units.get(ch, ""), color=colors[ch_idx % len(colors)],
                            show_xlabel=(ch_idx == len(chs_with_data) - 1),
                            y_min=panel_y_min, y_max=panel_y_max,
                        )
                else:
                    self._draw_waveform_axis(
                        axes[ch_idx][0], ch, ch_t_grid, traces.get(ch, []), result,
                        units=units.get(ch, ""), color=colors[ch_idx % len(colors)], summary_mode=summary_mode,
                        summary_only_requested=summary_only_requested, show_summary_line=show_summary_line,
                        show_summary_band=show_summary_band, show_xlabel=(ch_idx == len(chs_with_data) - 1),
                        y_min=y_min, y_max=y_max,
                    )

        canvas.draw()
