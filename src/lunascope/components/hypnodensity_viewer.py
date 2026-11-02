"""Standalone cohort viewer for Luna HDSTATS projections.

This module deliberately has no PCA implementation.  ``HDSTATS`` remains the
only producer of HDSTATS features and PC scores; fitted models and optional
descriptors are read and validated by Luna through lunapi.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import textwrap
import multiprocessing
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import lunapi as lp
from matplotlib.patches import Rectangle
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from ..file_dialogs import open_file_name, save_file_name
from ..runtime_paths import app_cache_root
from .mplcanvas import MplCanvas
from .plts import hypno_density
from .slist import DataFrameModel, NumericSortFilterProxy

PP_CHANNELS = ("PP_N1", "PP_N2", "PP_N3", "PP_R", "PP_W")
# Keep thumbnails/hard calls aligned with Luna's established stgcol() palette:
# PP_N1, PP_N2, PP_N3, PP_R, PP_W.  In particular, wake is green, not yellow.
PP_COLORS = ("#00BEFA", "#0050C8", "#000050", "#FA1432", "#31AD52")
DISPLAY_MODES = {
    "five": "Five class",
    "nrem_rem_wake": "NR / R / W",
    "wake_sleep": "W / S",
}
CACHE_VERSION = 2
_MODEL_SETTING_KEY = "hypnodensity_viewer/projection_model"
HDPC_DOWNLOAD_URL = "https://github.com/remnrem/luna-resources/releases/latest/download/luna-resources-hdpc.zip"
HDPC_MODEL_NAME = "hdstats-pc-core.model.json"
HDPC_DESCRIPTOR_NAME = "hdstats-pc-core.desc.json"


def file_signature(path: str) -> dict:
    """Return a content-aware signature suitable for cache invalidation."""
    p = Path(path)
    stat = p.stat()
    digest = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(p.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": digest.hexdigest()}


def source_signature(path: str) -> dict:
    """Cheap provenance signature for an EDF referenced by an HDSTATS cache."""
    p = Path(path)
    try:
        stat = p.stat()
        return {"path": str(p.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        return {"path": str(p), "missing": True}


def trace_archive_path(manifest_path: str) -> Path:
    path = Path(manifest_path)
    return path.with_suffix("").with_name(path.with_suffix("").name + ".traces.npz")


def _json_default(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def compact_feature_text(feature: str, label: str, description: str) -> tuple[str, str]:
    """Separate a feature's short name from stage/transition qualifiers."""
    label = label or hdstats_feature_label(feature)
    stage = re.search(r"(?:^|[.|])NSS[_=/](\d+)(?:[.|]|$)", feature)
    notes = [f"{stage.group(1)}-stage"] if stage else []
    for qualifier in re.findall(r"\(([^()]*)\)", description):
        for part in qualifier.split(";"):
            part = part.strip()
            if part and part.casefold() not in label.casefold() and part not in notes:
                notes.append(part)
    for qualifier in re.findall(r"\[(\d+-stage)\]", label):
        if qualifier not in notes:
            notes.insert(0, qualifier)
    label = re.sub(r"\s*\[\d+-stage\]", "", label).strip()
    return label, "; ".join(notes)


def hdstats_model_loadings(model: dict) -> pd.DataFrame:
    """Flatten Luna's validated model/descriptor response for the viewer."""
    panels = {str(panel.get("id", "")): panel for panel in model.get("panels", [])}
    feature_metadata = {}
    # Feature labels/descriptions are part of the fitted model's embedded spec,
    # not the optional post-fit descriptor. Keep this lookup tolerant of the
    # normalized API response shape so older models still show their IDs.
    spec = model.get("spec", {})
    for item in spec.get("feature_descriptions", []) if isinstance(spec, dict) else []:
        feature_id = str(item.get("id", ""))
        if feature_id:
            feature_metadata[feature_id] = item
    annotations = {(str(item.get("entity_type", "")).upper(), str(item.get("entity_id", ""))): str(item.get("text", ""))
                   for item in model.get("descriptions", [])}
    # A descriptor renames a component's "id" to something human-chosen (e.g.
    # "AMIN_PC1"), but live pc-proj scoring (model.json alone, no descriptor)
    # always reports the fit-time auto id for (panel, panel_pc) in whichever
    # style the model was fit with. Keep that raw id so live scores can be
    # joined back to this row and have the descriptor's orientation applied.
    component_id_style = str(spec.get("component_id_style", "legacy")) if isinstance(spec, dict) else "legacy"
    rows = []
    for component in model.get("components", []):
        panel_id = str(component.get("panel", ""))
        panel_pc = int(component.get("panel_pc", 0))
        pc_index = panel_pc - 1
        panel = panels.get(panel_id, {})
        singular_values = panel.get("singular_values", [])
        singular_value = singular_values[pc_index] if 0 <= pc_index < len(singular_values) else np.nan
        pc_id = str(component.get("id") or component.get("key") or f"{panel_id}/{panel_pc}")
        raw_pc_id = f"{panel_id}_{panel_pc}" if component_id_style == "r_safe" else f"{panel_id}/{panel_pc}"
        orientation = int(component.get("orientation", 1) or 1)
        pc_label = str(component.get("label", ""))
        pc_description = annotations.get(("PC", pc_id), str(component.get("description", "")))
        if pc_label == pc_id:
            pc_label = ""
        for item in component.get("loadings", []):
            feature_id = str(item.get("feature", ""))
            metadata = feature_metadata.get(feature_id, {})
            feature_description = annotations.get(("FEATURE", feature_id), str(metadata.get("description", "")))
            rows.append({"PC": pc_id, "PC label": pc_label,
                         "PC description": pc_description, "PC orientation": orientation,
                         "PC raw ID": raw_pc_id, "Panel": panel_id,
                         "Feature": feature_id,
                         "Feature label": str(metadata.get("label", "")),
                         "Feature description": feature_description,
                         "Loading": float(item.get("loading", np.nan)),
                         "Singular value": singular_value})
    return pd.DataFrame(rows, columns=["PC", "PC label", "PC description", "PC orientation", "PC raw ID",
                                       "Panel", "Feature", "Feature label", "Feature description",
                                       "Loading", "Singular value"])


def _default_hdstats_descriptor_path(model_path: str) -> Path:
    path = Path(model_path)
    name = path.name
    for suffix in (".model.json", ".model", ".txt"):
        if name.endswith(suffix):
            return path.with_name(name[:-len(suffix)] + ".desc.json")
    return path.with_name(name + ".desc.json")


def parse_hdstats_model(path: str, descriptor_path: str | None = None) -> pd.DataFrame:
    """Read fitted loadings through Luna's native model/descriptor reader."""
    explicit_descriptor = bool(descriptor_path)
    descriptor = Path(descriptor_path).expanduser() if explicit_descriptor else _default_hdstats_descriptor_path(path)
    if explicit_descriptor and not descriptor.is_file():
        raise FileNotFoundError(f"HDSTATS-PC descriptor does not exist: {descriptor}")
    api = lp.proj(verbose=False)
    if not hasattr(api, "read_hdstats_pc_model") or not hasattr(api, "read_hdstats_pc_descriptor"):
        raise RuntimeError("The installed lunapi is missing HDSTATS-PC model readers; rebuild and reinstall luna-api.")
    if descriptor.is_file():
        model = api.read_hdstats_pc_descriptor(str(Path(path).expanduser()), str(descriptor))
    else:
        model = api.read_hdstats_pc_model(str(Path(path).expanduser()))
    return hdstats_model_loadings(model)


def flatten_hdstats_tables(instance, sample_id: str) -> tuple[dict, dict]:
    """Collect all numeric HDSTATS outputs and authoritative PC/SCORE output."""
    values, pcs = {}, {}
    strata = instance.strata()
    if strata is None:
        return values, pcs
    for item in strata.itertuples(index=False):
        if str(item.Command) != "HDSTATS":
            continue
        frame = instance.table(item.Command, item.Strata)
        if frame is None or frame.empty:
            continue
        for ridx, row in frame.iterrows():
            levels = [f"{col}={row[col]}" for col in frame.columns
                      if col not in ("ID", "SCORE") and isinstance(row[col], str)]
            suffix = "|".join([str(item.Strata), *levels])
            for col, value in row.items():
                if col == "ID":
                    continue
                number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if not pd.notna(number):
                    continue
                key = f"{col}__{suffix}"
                values[key] = float(number)
                if "PC" in str(item.Strata).split("_") and col == "SCORE":
                    # PC component ID is normally a factor column or index.
                    pc_name = next((str(row[c]) for c in frame.columns if c not in ("ID", "SCORE") and isinstance(row[c], str)), str(ridx))
                    pcs[pc_name] = float(number)
    return values, pcs


def hdstats_feature_group(name: str) -> str:
    """Place the verbose projection-side value tables behind explicit groups."""
    if name.startswith("VAL__"):
        return "HDSTATS values (VAL)"
    if name.startswith("RAW__"):
        return "HDSTATS raw values (RAW)"
    return "HDSTATS features"


def hdstats_feature_label(name: str) -> str:
    """A display label only; the original HDSTATS key remains authoritative."""
    prefix, separator, detail = str(name).partition("__")
    title = {"VAL": "Value", "RAW": "Raw value"}.get(prefix, prefix.replace("_", " "))
    if not separator:
        return title
    factor_names = {
        "NSS": "state model", "SS": "stage", "PC": "component",
        "FTR": "feature", "TRANS": "transition", "QC": "quality control",
    }
    pieces = []
    for item in detail.split("|"):
        field, equals, value = item.partition("=")
        if not equals and "/" in field:
            field, value = field.split("/", 1)
            equals = "="
        field_label = factor_names.get(field, field.replace("_", " "))
        value = value.replace("__", " · ").replace("_", " ")
        # The transition labels produced by HDSTATS (e.g. WtoN1) are much
        # easier to scan when rendered as a directional relation.
        for stage in ("N3", "N2", "N1", "REM", "R", "W"):
            value = value.replace(f"to{stage}", f" → {stage}")
        if field == "NSS" and value in ("3", "5"):
            value = f"{value}-state"
        pieces.append(f"{field_label}: {value}" if equals else field_label)
    return f"{title} — " + " · ".join(pieces)


def apply_pc_orientation(scores: dict, raw_lookup: dict, pc_metadata: dict) -> dict:
    """Re-key live pc-proj scores to the descriptor's PC id and sign-flip.

    ``HDSTATS pc-proj=model.json`` always scores with orientation=+1 and
    reports each component's fit-time auto id (``raw_lookup`` keys, from
    ``hdstats_model_loadings``'s "PC raw ID"); the descriptor's label and
    sign convention are display-time-only in Luna itself, so lunascope must
    apply them here before the score ever reaches the session/cache.
    """
    oriented = {}
    for raw_name, value in scores.items():
        pc_id = raw_lookup.get(raw_name, raw_name)
        orientation = int(pc_metadata.get(pc_id, {}).get("PC orientation", 1) or 1)
        oriented[pc_id] = float(value) * orientation
    return oriented


def matching_hdstats_feature(model_feature: str, feature_columns) -> str | None:
    """Find the emitted HDSTATS table column corresponding to a model feature.

    Projection files use their compact feature spelling (for example
    ``H__NSS/5``), whereas Luna's result tables add strata/factor context.
    This is only a display/linking matcher: it never takes part in projection.
    """
    def compact(value):
        value = re.sub(r"^(?:VAL|RAW)__", "", str(value))
        return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()
    wanted = compact(model_feature)
    exact = [str(column) for column in feature_columns if compact(column) == wanted]
    if exact:
        return exact[0]
    embedded = [str(column) for column in feature_columns if wanted and wanted in compact(column)]
    return embedded[0] if len(embedded) == 1 else None


def display_hypnodensity(trace, mode: str = "five") -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Return a visual-only regrouping of the five authoritative PP traces.

    NREM retains all of N1, N2 and N3 (rather than pretending PP_N2 is a
    NREM posterior); its blue is deliberately the established PP_N2 colour.
    """
    values = np.asarray(trace, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(PP_CHANNELS):
        raise ValueError("Expected five PP hypnodensity columns")
    if mode == "nrem_rem_wake":
        return (np.column_stack((values[:, :3].sum(axis=1), values[:, 3], values[:, 4])),
                ("NR", "R", "W"), (PP_COLORS[1], PP_COLORS[3], PP_COLORS[4]))
    if mode == "wake_sleep":
        return (np.column_stack((values[:, 4], values[:, :4].sum(axis=1))),
                ("W", "S"), (PP_COLORS[4], "#9333ea"))
    return values, ("N1", "N2", "N3", "REM", "Wake"), PP_COLORS


def thumbnail_time_window(trace, time_axis, analysis: dict | None, mode: str) -> slice:
    """Return the thumbnail's view window without touching the HDSTATS result.

    ``HD_ANALYSIS_*`` is the persisted SPT-with-flanks interval selected by
    HDSTATS itself.  It gives the thumbnail a useful focused view and works
    naturally for non-uniform EDF+D time axes via ``searchsorted``.
    """
    n = len(trace)
    if mode != "spt_flank" or not analysis or time_axis is None or len(time_axis) != n:
        return slice(0, n)
    start = analysis.get("HD_ANALYSIS_START_SEC")
    stop = analysis.get("HD_ANALYSIS_STOP_SEC")
    if start is None or stop is None:
        return slice(0, n)
    axis = np.asarray(time_axis, dtype=float)
    left = int(np.searchsorted(axis, float(start), side="left"))
    right = int(np.searchsorted(axis, float(stop), side="right"))
    left, right = max(0, left), min(n, right)
    return slice(left, right) if right > left else slice(0, n)


def thumbnail_duration_seconds(trace, time_axis, window: slice) -> float:
    """Clock-time duration of a thumbnail window, including one sample step."""
    if time_axis is None or len(time_axis) != len(trace) or window.stop - window.start < 1:
        return float(max(0, window.stop - window.start))
    axis = np.asarray(time_axis, dtype=float)[window]
    step = float(np.nanmedian(np.diff(axis))) if len(axis) > 1 else 0.0
    return max(0.0, float(axis[-1] - axis[0]) + step)


def hdstats_analysis_metadata(instance, spt_label: str) -> dict:
    """Read the analysis bounds/soft-SPT emitted by this HDSTATS invocation."""
    out = {}
    strata = instance.strata()
    if strata is not None:
        for item in strata.itertuples(index=False):
            if str(item.Command) != "HDSTATS" or "QC" not in str(item.Strata).split("_"):
                continue
            frame = instance.table(item.Command, item.Strata)
            if frame is None or frame.empty:
                continue
            row = frame.iloc[0]
            for key in ("HD_ANALYSIS", "HD_ANALYSIS_START_SEC", "HD_ANALYSIS_STOP_SEC", "HD_ANALYSIS_FALLBACK"):
                if key in row and pd.notna(row[key]): out[key] = row[key]
    try:
        events = instance.fetch_annots([f"{spt_label}_sleep_onset", f"{spt_label}_final_wake"])
        for event in events.itertuples(index=False):
            if event.Class == f"{spt_label}_sleep_onset": out["soft_onset_sec"] = float(event.Start)
            elif event.Class == f"{spt_label}_final_wake": out["soft_final_wake_sec"] = float(event.Start)
    except Exception:
        pass
    return out


def hard_call_runs(trace, invalid=None):
    """Return contiguous [start, stop) arg-max runs; invalid samples use -1."""
    values = np.asarray(trace, dtype=float)
    if values.ndim != 2 or not len(values):
        return []
    calls = np.argmax(np.nan_to_num(values, nan=-np.inf), axis=1).astype(int)
    bad = ~np.isfinite(values).all(axis=1)
    if invalid is not None:
        bad |= np.asarray(invalid, dtype=bool)
    calls[bad] = -1
    edges = np.r_[True, calls[1:] != calls[:-1], True]
    points = np.flatnonzero(edges)
    return [(int(points[i]), int(points[i + 1]), int(calls[points[i]])) for i in range(len(points) - 1)]


def _hdstats_record_worker(task):
    """Run one record in an isolated, project-evaluation worker process."""
    from . import anal
    if anal._PROJECT_EVAL_CHILD_PROJ is None:
        anal._init_project_eval_child()
    proj = anal._PROJECT_EVAL_CHILD_PROJ
    sid = str(task["sample_row"][0]).strip()
    result = {"ordinal": task["ordinal"], "ID": sid}
    try:
        proj.clear()
        proj.eng.set_sample_list([task["sample_row"]])
        proj.clear_vars()
        proj.reinit()
        for name, value in task["param"]:
            proj.var(name, value)
        instance = proj.inst(sid)
        missing = [name for name in PP_CHANNELS if name not in set(instance.edf.channels())]
        if missing:
            raise ValueError("missing " + ", ".join(missing))
        trace, invalid, axis = HypnodensityViewerDialog._load_pp_trace(instance)
        spt_label = "__hdviewer_spt"
        instance.eval_lunascope(f"HDSTATS pc-proj={task['model_path']} qc=T emit-spt={spt_label}")
        features, scores = flatten_hdstats_tables(instance, sid)
        if not scores:
            raise ValueError("HDSTATS emitted no projected PC scores")
        result.update(features=features, scores=scores, trace=trace, invalid=invalid,
                      axis=axis, analysis=hdstats_analysis_metadata(instance, spt_label))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        proj.clear()
    return result


class _HDStatsProgress(QtWidgets.QDialog):
    """Remain application-modal until running workers have been stopped."""
    def __init__(self, parent, total, reused, shortcuts):
        super().__init__(parent)
        self.setWindowTitle("Calculating hypnodensity cohort")
        self.setWindowModality(Qt.ApplicationModal)
        self.cancel_requested = False
        layout = QtWidgets.QVBoxLayout(self)
        self.label = QtWidgets.QLabel("Calculating records…")
        layout.addWidget(self.label)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, total)
        self.progress.setValue(reused)
        self.progress.setFormat("%v / %m")
        layout.addWidget(self.progress)
        self.stop = QtWidgets.QPushButton("Stop")
        self.stop.clicked.connect(self.reject)
        layout.addWidget(self.stop)
        for sequence in shortcuts:
            shortcut = QtGui.QShortcut(sequence, self)
            shortcut.activated.connect(self.reject)
        self.resize(440, 130)

    def reject(self):
        self.cancel_requested = True
        self.label.setText("Stopping… completed records will be kept.")
        self.stop.setEnabled(False)

    def closeEvent(self, event):
        self.reject()
        event.ignore()


class _ThumbDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None, mode_getter=None, range_getter=None, xscale_getter=None):
        super().__init__(parent)
        self._mode_getter = mode_getter or (lambda: "five")
        self._range_getter = range_getter or (lambda: "full")
        self._xscale_getter = xscale_getter or (lambda: ("normalized", 0.0))

    def paint(self, painter, option, index):
        if index.column() != 1:
            return super().paint(painter, option, index)
        payload = index.data(Qt.UserRole)
        painter.fillRect(option.rect, QtGui.QColor("#121722"))
        if payload is None: return
        trace, invalid, time_axis, analysis = payload if isinstance(payload, tuple) and len(payload) == 4 else (payload, None, None, None)
        window = thumbnail_time_window(trace, time_axis, analysis, self._range_getter())
        xscale, reference_seconds = self._xscale_getter()
        trace = np.asarray(trace)[window]
        try:
            arr, _labels, colors = display_hypnodensity(trace, self._mode_getter())
        except ValueError:
            return
        if not len(arr): return
        full_width = max(1, option.rect.width() - 6)
        fraction = 1.0
        if xscale == "absolute" and reference_seconds > 0:
            fraction = min(1.0, thumbnail_duration_seconds(trace, np.asarray(time_axis)[window] if time_axis is not None else None, slice(0, len(trace))) / reference_seconds)
        width = max(2, int(round(full_width * fraction))); n = min(width, len(arr))
        pick = np.linspace(0, len(arr)-1, n).astype(int); arr = arr[pick]
        base = option.rect.bottom() - 3
        height = option.rect.height() - 6
        # PP channels sum to one.  Building each polygon between its actual
        # cumulative lower/upper edges prevents paint from spilling into
        # adjacent virtual rows (the old incremental pen position did so).
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
        denom = arr.sum(axis=1, keepdims=True)
        arr = np.divide(arr, denom, out=np.zeros_like(arr), where=denom > 0)
        lower = np.zeros(n, dtype=float)
        for col, color in enumerate(colors):
            if col >= arr.shape[1]: break
            upper = lower + arr[:, col]
            path = QtGui.QPainterPath()
            path.moveTo(option.rect.left() + 3, base - upper[0] * height)
            for i in range(1, n): path.lineTo(option.rect.left() + 3 + i, base - upper[i] * height)
            for i in range(n - 1, -1, -1): path.lineTo(option.rect.left() + 3 + i, base - lower[i] * height)
            path.closeSubpath()
            painter.fillPath(path, QtGui.QColor(color))
            lower = upper


class _CohortBrowserModel(DataFrameModel):
    """Keep trace arrays out of display formatting while exposing them to paint()."""
    def _build_row_text(self):
        # Filtering must never stringify full resident hypnodensity arrays.
        fields = self._df.drop(columns=["Hypnodensity"], errors="ignore").fillna("").astype(str)
        return fields.agg("\t".join, axis=1).tolist()

    def set_filter_terms(self, terms) -> None:
        terms = [str(term).casefold() for term in (terms or []) if str(term).strip()]
        if not terms:
            self._filter_mask = None
            return
        text = pd.Series(self.row_text(), dtype="string").str.casefold()
        mask = pd.Series(False, index=text.index)
        for term in terms:
            mask |= text.str.contains(term, regex=False, na=False)
        self._filter_mask = mask.to_numpy(dtype=bool, copy=False)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and 0 <= section < len(self._df.columns):
            name = self._df.columns[section]
            if role == Qt.DisplayRole:
                return getattr(self, "header_labels", {}).get(name, name)
            if role == Qt.ToolTipRole:
                return getattr(self, "header_tooltips", {}).get(name, name)
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and index.column() == 0 and str(self._df.iat[index.row(), 0]) in getattr(self, "flagged_ids", ()):
            if role == Qt.DecorationRole:
                icon = getattr(self, "_flag_icon", None)
                if icon is None:
                    pixmap = QtGui.QPixmap(18, 18); pixmap.fill(Qt.transparent)
                    painter = QtGui.QPainter(pixmap)
                    painter.setPen(QtGui.QColor("#fbbf24"))
                    font = painter.font(); font.setPointSize(14); painter.setFont(font)
                    painter.drawText(pixmap.rect(), Qt.AlignCenter, "★")
                    painter.end()
                    icon = self._flag_icon = QtGui.QIcon(pixmap)
                return icon
            if role == Qt.BackgroundRole:
                return QtGui.QBrush(QtGui.QColor("#695526"))
            if role == Qt.FontRole:
                font = QtGui.QFont(); font.setBold(True); return font
            if role == Qt.ToolTipRole:
                return "Flagged individual — right-click to edit or remove"
        if role == Qt.UserRole and index.isValid() and index.column() == 1:
            return self._df.iat[index.row(), index.column()]
        if index.isValid() and index.column() == 1 and role in (Qt.DisplayRole, Qt.EditRole):
            return ""
        return super().data(index, role)


class _LoadingTree(QtWidgets.QTreeWidget):
    """Tree that distinguishes keyboard selection from a click on the active row."""
    reselected = QtCore.Signal(object)

    def mousePressEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        already_current = item is not None and item is self.currentItem()
        super().mousePressEvent(event)
        if already_current:
            self.reselected.emit(item)


@dataclass
class CohortSession:
    model_path: str = ""
    model_signature: dict = field(default_factory=dict)
    results_model_signature: dict = field(default_factory=dict)
    cache_path: str = ""
    rows: list = field(default_factory=list)
    traces: dict = field(default_factory=dict)
    invalid_masks: dict = field(default_factory=dict)
    time_axes: dict = field(default_factory=dict)
    analysis: dict = field(default_factory=dict)
    browser_values: list = field(default_factory=list)
    flagged: dict = field(default_factory=dict)
    source_signatures: dict = field(default_factory=dict)
    features: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["ID"]))
    pcs: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["ID"]))
    tabular: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["ID"]))
    exclusions: list = field(default_factory=list)


class HypnodensityViewerDialog(QtWidgets.QDialog):
    """A self-contained HDSTATS cohort session window."""
    def __init__(self, controller, parent=None):
        super().__init__(parent or controller.ui); self.ctrl = controller; self.session = CohortSession()
        self.setWindowTitle("Hypnodensity viewer"); self.resize(1360, 900)
        self._build()
        saved_model = QtCore.QSettings("Lunascope", "Lunascope").value(_MODEL_SETTING_KEY, "", type=str)
        if saved_model:
            self.model_edit.setText(saved_model)
            QtCore.QTimer.singleShot(0, self._read_model)
        self.refresh_sample_list()

    def _build(self):
        root = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout(); root.addLayout(top)
        self.model_edit = QtWidgets.QLineEdit(); self.model_edit.setPlaceholderText("HDSTATS-PC fitted model (.model.json)")
        browse = QtWidgets.QPushButton("Projection model…"); browse.clicked.connect(self._choose_model)
        run = QtWidgets.QPushButton("Calculate / resume"); run.clicked.connect(self.calculate)
        cache = QtWidgets.QPushButton("Save cache…"); cache.clicked.connect(lambda: self.save_cache())
        load_cache = QtWidgets.QPushButton("Open cache…"); load_cache.clicked.connect(self.open_cache)
        top.addWidget(QtWidgets.QLabel("pc-proj:")); top.addWidget(self.model_edit, 1)
        for w in (browse, run, cache, load_cache): top.addWidget(w)
        download_hdpc = QtWidgets.QPushButton("Download HDPC")
        download_hdpc.setToolTip("Download the current core HDSTATS-PC model and descriptor")
        download_hdpc.clicked.connect(self.ctrl._download_hdpc_resources)
        top.addWidget(download_hdpc)
        descriptor_row = QtWidgets.QHBoxLayout(); root.addLayout(descriptor_row)
        self.descriptor_edit = QtWidgets.QLineEdit()
        self.descriptor_edit.setPlaceholderText("Optional descriptor (.desc.json; auto-detected beside model if blank)")
        descriptor_browse = QtWidgets.QPushButton("Descriptor…")
        descriptor_browse.clicked.connect(self._choose_descriptor)
        descriptor_row.addWidget(QtWidgets.QLabel("Description:")); descriptor_row.addWidget(self.descriptor_edit, 1)
        descriptor_row.addWidget(descriptor_browse)
        self.status = QtWidgets.QLabel(); root.addWidget(self.status)
        self.pages = QtWidgets.QTabWidget(); root.addWidget(self.pages, 1)
        model_page = QtWidgets.QWidget(); model_layout = QtWidgets.QVBoxLayout(model_page)
        model_controls = QtWidgets.QHBoxLayout()
        model_controls.addWidget(QtWidgets.QLabel("Click a PC or feature to show its model loadings. Click it again to clear the selection."))
        model_controls.addStretch(1)
        self.show_feature_variables = QtWidgets.QCheckBox("Show feature variables")
        self.show_feature_variables.toggled.connect(
            lambda checked: self.feature_list.setColumnHidden(3, not checked))
        model_controls.addWidget(self.show_feature_variables)
        model_controls.addWidget(QtWidgets.QLabel("Minimum |loading|:"))
        self.loading_threshold = QtWidgets.QDoubleSpinBox()
        self.loading_threshold.setRange(0.0, 100.0)
        self.loading_threshold.setDecimals(4)
        self.loading_threshold.setSingleStep(0.01)
        self.loading_threshold.setValue(0.1)
        self.loading_threshold.setToolTip("Do not show model loadings whose absolute value is below this value.")
        self.loading_threshold.valueChanged.connect(self._loading_threshold_changed)
        model_controls.addWidget(self.loading_threshold)
        model_layout.addLayout(model_controls)
        model_split = QtWidgets.QSplitter(Qt.Horizontal); model_layout.addWidget(model_split, 1)
        pc_box = QtWidgets.QWidget(); pc_layout = QtWidgets.QVBoxLayout(pc_box)
        self.pc_list_label = QtWidgets.QLabel("Model PCs"); pc_layout.addWidget(self.pc_list_label)
        self.pc_list = _LoadingTree(); self.pc_list.setHeaderLabels(["PC", "Loading", "Description"]); self.pc_list.setRootIsDecorated(False); self.pc_list.currentItemChanged.connect(lambda item, _previous: self._pc_selected(item)); self.pc_list.reselected.connect(self._pc_reselected); self.pc_list.header().setSectionsClickable(True); self.pc_list.header().setSortIndicatorShown(True); self.pc_list.header().sectionClicked.connect(lambda column: self._sort_loading_tree(self.pc_list, column)); pc_layout.addWidget(self.pc_list)
        # Interactive, not ResizeToContents: the latter recomputes column
        # width on every single item insert (not once at the end), which is
        # O(n^2)-ish for a few hundred rows. The explicit resizeColumnToContents()
        # calls after each rebuild/filter already give the same fitted width,
        # just computed once.
        self.pc_list.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)
        self.pc_list.header().setSectionResizeMode(1, QtWidgets.QHeaderView.Interactive)
        self.pc_list.header().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        feature_box = QtWidgets.QWidget(); feature_layout = QtWidgets.QVBoxLayout(feature_box)
        self.feature_list_label = QtWidgets.QLabel("Model features"); feature_layout.addWidget(self.feature_list_label)
        self.feature_list = _LoadingTree(); self.feature_list.setHeaderLabels(["Feature", "Loading", "Note", "Variable"]); self.feature_list.setRootIsDecorated(False); self.feature_list.currentItemChanged.connect(lambda item, _previous: self._feature_selected(item)); self.feature_list.reselected.connect(self._feature_reselected); self.feature_list.header().setSectionsClickable(True); self.feature_list.header().setSortIndicatorShown(True); self.feature_list.header().sectionClicked.connect(lambda column: self._sort_loading_tree(self.feature_list, column)); feature_layout.addWidget(self.feature_list)
        self.feature_list.header().setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)
        self.feature_list.header().setSectionResizeMode(1, QtWidgets.QHeaderView.Interactive)
        self.feature_list.header().setStretchLastSection(False)
        self.feature_list.header().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.feature_list.header().setSectionResizeMode(3, QtWidgets.QHeaderView.Interactive)
        self.feature_list.setColumnWidth(3, 280)
        self.feature_list.setColumnHidden(3, True)
        model_split.addWidget(pc_box); model_split.addWidget(feature_box); model_split.setStretchFactor(0, 1); model_split.setStretchFactor(1, 1); model_split.setSizes([600, 600])
        self.pages.addTab(model_page, "Model PCs / features")

        browser_page = QtWidgets.QWidget(); browser_layout = QtWidgets.QVBoxLayout(browser_page)
        browser_layout.addWidget(QtWidgets.QLabel("Cohort browser · ★ flagged · F flag/unflag · O open in sample list"))
        data_controls = QtWidgets.QHBoxLayout()
        join = QtWidgets.QPushButton("Load TSV…"); join.clicked.connect(self.join_tsv)
        exclusions = QtWidgets.QPushButton("Exclusions…"); exclusions.clicked.connect(self.show_exclusions)
        self.flagged_button = QtWidgets.QPushButton("Flagged (0)…"); self.flagged_button.clicked.connect(self.show_flagged)
        self.browser_values_button = QtWidgets.QPushButton("Browser values…"); self.browser_values_button.clicked.connect(self.choose_browser_values)
        data_controls.addWidget(join); data_controls.addWidget(exclusions); data_controls.addWidget(self.flagged_button); data_controls.addWidget(self.browser_values_button); data_controls.addStretch(1)
        browser_layout.addLayout(data_controls)
        browser_controls = QtWidgets.QHBoxLayout()
        self.browser_filter = QtWidgets.QLineEdit(); self.browser_filter.setPlaceholderText("Filter ID or shown values…"); self.browser_filter.setMaximumWidth(205); self.browser_filter.textChanged.connect(self._apply_browser_filter); browser_controls.addWidget(self.browser_filter)
        self.browser_flagged_only = QtWidgets.QCheckBox("Flagged only")
        self.browser_flagged_only.toggled.connect(lambda _checked: self._refresh_views())
        browser_controls.addWidget(self.browser_flagged_only)
        self.browser_complete = QtWidgets.QCheckBox("Hide rows missing shown values")
        self.browser_complete.setToolTip("Hide a record when any currently displayed PC, feature, phenotype, or covariate value is missing.")
        self.browser_complete.toggled.connect(lambda _checked: self._refresh_views()); browser_controls.addWidget(self.browser_complete)
        browser_controls.addSpacing(12); browser_controls.addWidget(QtWidgets.QLabel("Display:"))
        self._display_mode = "five"; self.display_mode_buttons = QtWidgets.QButtonGroup(self)
        for key, label in DISPLAY_MODES.items():
            button = QtWidgets.QRadioButton(label); button.setChecked(key == self._display_mode)
            if key == "nrem_rem_wake":
                button.setToolTip("NR is PP_N1 + PP_N2 + PP_N3; it uses the PP_N2 blue.")
            elif key == "wake_sleep":
                button.setToolTip("S is PP_N1 + PP_N2 + PP_N3 + PP_R (purple).")
            button.toggled.connect(lambda checked, value=key: self._display_mode_changed(value, checked))
            self.display_mode_buttons.addButton(button); browser_controls.addWidget(button)
        browser_controls.addSpacing(12); browser_controls.addWidget(QtWidgets.QLabel("Thumbnails:"))
        self._thumbnail_range = "full"
        self.thumbnail_range_toggle = QtWidgets.QCheckBox("SPT + flank")
        self.thumbnail_range_toggle.setToolTip("Unchecked: full recording. Checked: HDSTATS soft SPT with its selected flanks.")
        self.thumbnail_range_toggle.toggled.connect(lambda checked: self._thumbnail_range_changed("spt_flank" if checked else "full", True))
        browser_controls.addWidget(self.thumbnail_range_toggle)
        browser_controls.addSpacing(12); browser_controls.addWidget(QtWidgets.QLabel("Width:"))
        self._thumbnail_xscale = "normalized"; self._thumbnail_reference_seconds_value = 0.0
        self.thumbnail_xscale_toggle = QtWidgets.QCheckBox("Absolute time")
        self.thumbnail_xscale_toggle.setToolTip("Unchecked: normalized width. Checked: displayed clock-time duration, with a robust cohort cap for extreme recordings.")
        self.thumbnail_xscale_toggle.toggled.connect(lambda checked: self._thumbnail_xscale_changed("absolute" if checked else "normalized", True))
        browser_controls.addWidget(self.thumbnail_xscale_toggle)
        browser_controls.addStretch(1)
        self.browser = QtWidgets.QTableView(); self.browser.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows); self.browser.setSortingEnabled(True); self._thumb_delegate = _ThumbDelegate(self.browser, lambda: self._display_mode, lambda: self._thumbnail_range, self._thumbnail_xscale_state); self.browser.setItemDelegateForColumn(1, self._thumb_delegate); self.browser.clicked.connect(self._selected)
        self.browser.setContextMenuPolicy(Qt.CustomContextMenu)
        self.browser.customContextMenuRequested.connect(self._browser_context_menu)
        self._browser_shortcuts = []
        for key, callback in (("F", self._toggle_selected_flag), ("O", self._open_selected_in_sample_list)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self.browser)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self._browser_shortcuts.append(shortcut)
        browser_layout.addLayout(browser_controls)
        self.browser.sortByColumn(-1, Qt.AscendingOrder)
        self.browser.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.browser_split = QtWidgets.QSplitter(Qt.Horizontal)
        self.browser_split.setChildrenCollapsible(False)
        self.browser_split.setHandleWidth(9)
        self.browser_split.addWidget(self.browser)
        score_panel = QtWidgets.QWidget()
        score_layout = QtWidgets.QVBoxLayout(score_panel)
        score_layout.setContentsMargins(8, 0, 0, 0)
        self.individual_pc_title = QtWidgets.QLabel("PC scores — select an individual")
        self.individual_pc_title.setTextFormat(Qt.PlainText)
        self.individual_pc_title.setWordWrap(True)
        score_layout.addWidget(self.individual_pc_title)
        score_controls = QtWidgets.QHBoxLayout()
        score_controls.addWidget(QtWidgets.QLabel("Minimum |score|:"))
        self.individual_score_threshold = QtWidgets.QDoubleSpinBox()
        self.individual_score_threshold.setRange(0.0, 100.0)
        self.individual_score_threshold.setDecimals(2)
        self.individual_score_threshold.setSingleStep(0.1)
        self.individual_score_threshold.setValue(0.5)
        self.individual_score_threshold.setToolTip(
            "Show positive and negative scores with magnitude at least this value. "
            "Set to 0 to show all PCs, including unavailable scores. "
            "Independent of the model-loading threshold.")
        self.individual_score_threshold.valueChanged.connect(self._individual_score_threshold_changed)
        score_controls.addWidget(self.individual_score_threshold)
        score_controls.addStretch(1)
        score_layout.addLayout(score_controls)
        self.individual_pcs = _LoadingTree()
        self.individual_pcs.setHeaderLabels(["PC", "Score", "Description"])
        self.individual_pcs.setRootIsDecorated(False)
        self.individual_pcs.header().setSectionsClickable(True)
        self.individual_pcs.header().setSortIndicatorShown(True)
        self.individual_pcs.header().sectionClicked.connect(
            lambda column: self._sort_loading_tree(self.individual_pcs, column))
        self.individual_pcs.setColumnWidth(0, 165)
        self.individual_pcs.setColumnWidth(1, 70)
        self.individual_pcs.header().setStretchLastSection(False)
        self.individual_pcs.header().setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        self.individual_pcs.header().setMinimumSectionSize(45)
        self.individual_pcs.setColumnWidth(2, 210)
        self.individual_pcs.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.individual_pcs.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.individual_pcs.header().setToolTip(
            "Drag column boundaries to resize; double-click a boundary to fit its contents. "
            "Scroll horizontally to see columns to the right.")
        self.individual_pcs.setContextMenuPolicy(Qt.CustomContextMenu)
        self.individual_pcs.customContextMenuRequested.connect(self._individual_pc_context_menu)
        score_layout.addWidget(self.individual_pcs, 1)
        self.browser_split.addWidget(score_panel)
        self.browser_split.handle(1).setToolTip("Drag to resize the browser and PC scores panels.")
        self.browser_split.setStretchFactor(0, 2)
        self.browser_split.setStretchFactor(1, 1)
        self.browser_split.setSizes([880, 440])
        browser_layout.addWidget(self.browser_split, 4)
        self.canvas = MplCanvas(self); self.canvas.setMinimumHeight(145); self.canvas.setMaximumHeight(210); browser_layout.addWidget(self.canvas, 1)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setToolTip("Click for keyboard focus. Up/down: zoom; left/right: pan. Wheel: zoom; left-drag: pan; right-drag: zoom to range; double-click: reset.")
        self._plot_id = None; self._plot_full_xlim = None; self._plot_drag = None
        self._plot_window_patch = None; self._plot_duration_text = None
        self.canvas.mpl_connect("button_press_event", self._plot_mouse_press)
        self.canvas.mpl_connect("button_release_event", self._plot_mouse_release)
        self.canvas.mpl_connect("motion_notify_event", self._plot_mouse_motion)
        self.canvas.mpl_connect("scroll_event", self._plot_scroll)
        self.canvas.mpl_connect("key_press_event", self._plot_key)
        self.pages.addTab(browser_page, "Cohort browser")

    def _display_mode_changed(self, mode, checked):
        if not checked:
            return
        self._display_mode = mode
        self.browser.viewport().update()
        current = self.browser.currentIndex()
        if current.isValid():
            self._selected(current)
            return
        # The plot may have been opened programmatically (or the model reset
        # after a sort); retain its subject when a display mode is changed.
        sid = getattr(self, "_selected_id", None)
        if sid is not None and self.browser.model() is not None:
            source = self.browser.model().sourceModel()
            rows = np.flatnonzero(source._df["ID"].astype(str).eq(str(sid)).to_numpy())
            if len(rows):
                self._selected(self.browser.model().mapFromSource(source.index(int(rows[0]), 0)))

    def _thumbnail_range_changed(self, mode, checked):
        if checked:
            self._thumbnail_range = mode
            self._update_thumbnail_reference_seconds()
            self.browser.viewport().update()

    def _thumbnail_xscale_state(self):
        return self._thumbnail_xscale, self._thumbnail_reference_seconds_value

    def _thumbnail_xscale_changed(self, mode, checked):
        if checked:
            self._thumbnail_xscale = mode
            self.browser.viewport().update()

    def _update_thumbnail_reference_seconds(self):
        durations = []
        for sample_id, trace in self.session.traces.items():
            axis = self.session.time_axes.get(sample_id)
            window = thumbnail_time_window(trace, axis, self.session.analysis.get(sample_id), self._thumbnail_range)
            duration = thumbnail_duration_seconds(trace, axis, window)
            if np.isfinite(duration) and duration > 0:
                durations.append(duration)
        self._thumbnail_reference_seconds_value = float(np.quantile(durations, .90)) if durations else 0.0

    def refresh_sample_list(self):
        rows = self.ctrl._sample_rows_from_source_model() if hasattr(self.ctrl, "_sample_rows_from_source_model") else []
        # Five float32 series; use standard 5-second estimate until EDF metadata is loaded.
        estimate = len(rows) * 5760 * len(PP_CHANNELS) * 4
        self.status.setText(f"{len(rows)} sample-list rows; estimated resident PP memory: {estimate/1024**2:.1f} MB")

    @staticmethod
    def _estimated_pp_bytes(rows):
        return len(rows) * 5760 * len(PP_CHANNELS) * np.dtype(np.float32).itemsize

    def _apply_browser_filter(self, text):
        model = self.browser.model() if hasattr(self, "browser") else None
        if model is None:
            return
        source = model.sourceModel()
        if isinstance(source, DataFrameModel):
            source.set_filter_terms([text])
            model.invalidateFilter()
            if not self.browser.currentIndex().isValid():
                self._clear_individual_selection()

    def _choose_model(self):
        path, _ = open_file_name(self, "Choose HDSTATS-PC model", "", "HDSTATS models (*.model.json *.model *.txt);;All files (*)")
        if path:
            self.model_edit.setText(path)
            descriptor = _default_hdstats_descriptor_path(path)
            self.descriptor_edit.setText(str(descriptor) if descriptor.is_file() else "")
            self._read_model()

    def _choose_descriptor(self):
        path, _ = open_file_name(self, "Choose HDSTATS-PC descriptor", "", "HDSTATS descriptors (*.desc.json);;All files (*)")
        if path:
            self.descriptor_edit.setText(path)
            self._read_model()

    def _read_model(self):
        path = self.model_edit.text().strip()
        if not path: return False
        try:
            descriptor_path = self.descriptor_edit.text().strip() or None
            if descriptor_path is None:
                adjacent = _default_hdstats_descriptor_path(path)
                if adjacent.is_file():
                    descriptor_path = str(adjacent)
                    self.descriptor_edit.setText(descriptor_path)
            self._model_loadings = parse_hdstats_model(path, descriptor_path); self.session.model_path = path; self.session.model_signature = file_signature(path)
            QtCore.QSettings("Lunascope", "Lunascope").setValue(_MODEL_SETTING_KEY, path)
            self._populate_model_explorer()
            return True
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Projection model", str(exc)); return False

    @staticmethod
    def _loading_brush(value, maximum):
        if not np.isfinite(value) or maximum <= 0:
            return QtGui.QBrush(QtGui.QColor("white"))
        strength = min(1.0, abs(float(value)) / maximum)
        base = (52, 116, 181) if value >= 0 else (215, 48, 39)
        rgb = tuple(round(255 + strength * (channel - 255)) for channel in base)
        return QtGui.QBrush(QtGui.QColor(*rgb))

    def _populate_model_explorer(self):
        df = getattr(self, "_model_loadings", pd.DataFrame())
        if df.empty:
            self._model_selection_updating = True
            self.pc_list.clear(); self.feature_list.clear()
            self._model_selection_updating = False
            self._model_pc_raw_lookup = {}
            self._model_feature_metadata = {}
            return
        self._model_features = df["Feature"].drop_duplicates().astype(str).tolist()
        self._model_pcs = df["PC"].drop_duplicates().astype(str).tolist()
        self._model_pc_metadata = (df.drop_duplicates("PC").set_index("PC").to_dict("index")
                                   if "PC" in df else {})
        # Live pc-proj scores are keyed by each component's fit-time auto id
        # ("PC raw ID"), not by the descriptor's human-chosen "PC" id; this
        # lookup lets calculate() re-key and sign-flip them to match.
        self._model_pc_raw_lookup = ({str(row["PC raw ID"]): pc for pc, row in self._model_pc_metadata.items()}
                                     if "PC raw ID" in df else {})
        # A feature's label/description is the same on every (PC, Feature)
        # row that mentions it, so precompute it once rather than re-scanning
        # the whole loadings table per feature on every PC click (profiled at
        # tens of ms per click on a real model -- the actual source of the
        # reported UI lag).
        self._model_feature_metadata = (df.drop_duplicates("Feature").set_index("Feature").to_dict("index")
                                        if "Feature" in df else {})
        self._reset_model_explorer()

    @staticmethod
    def _add_loading_item(tree, name, value=np.nan, description=""):
        text = f"{float(value):+.4f}" if np.isfinite(value) else ""
        item = QtWidgets.QTreeWidgetItem([name, text, str(description)]); item.setData(0, Qt.UserRole, name)
        if np.isfinite(value):
            item.setData(1, Qt.UserRole, float(value))
        tree.addTopLevelItem(item)
        return item

    @staticmethod
    def _sort_tree_items(tree, column, ascending):
        """Apply a given column/direction without Qt calling a Python item comparator from C++."""
        items = [tree.takeTopLevelItem(0) for _ in range(tree.topLevelItemCount())]
        if column == 1:
            finite = [item for item in items if item.data(1, Qt.UserRole) is not None]
            blank = [item for item in items if item.data(1, Qt.UserRole) is None]
            finite.sort(key=lambda item: float(item.data(1, Qt.UserRole)), reverse=not ascending)
            items = finite + blank
        else:
            items.sort(key=lambda item: item.text(column).casefold(), reverse=not ascending)
        tree.addTopLevelItems(items)
        tree.header().setSortIndicator(column, Qt.AscendingOrder if ascending else Qt.DescendingOrder)

    @classmethod
    def _sort_loading_tree(cls, tree, column):
        """Header click: toggle direction on the same column, else ascending."""
        previous_column, previous_ascending = getattr(tree, "_loading_sort", (None, False))
        ascending = previous_column != column or not previous_ascending
        tree._loading_sort = (column, ascending)
        cls._sort_tree_items(tree, column, ascending)

    @classmethod
    def _reapply_tree_sort(cls, tree):
        """Keep a user-chosen sort (e.g. by Loading/"weight") across rebuilds.

        Selecting a PC rebuilds the feature list (and vice versa); without
        this, each selection silently reset the list back to its unsorted
        default order even if the user had just sorted it by a column.
        """
        sort = getattr(tree, "_loading_sort", None)
        if sort is not None:
            cls._sort_tree_items(tree, sort[0], sort[1])

    def _reset_model_explorer(self):
        self._active_pc = self._active_feature = None
        self._model_selection_updating = True
        try:
            self._rebuild_full_pc_list(hide_values=False)
            self._rebuild_full_feature_list(hide_values=False)
        finally:
            self._model_selection_updating = False

    def _rebuild_full_pc_list(self, select=None, hide_values=True):
        """Show every PC, undoing any weight-filtered subset left by a prior
        feature selection (that subset has nothing to do with a newly
        clicked PC, so it must not linger once pc_list becomes the seed)."""
        self.pc_list.setUpdatesEnabled(False)
        try:
            self.pc_list.clear()
            for pc in self._model_pcs:
                metadata = getattr(self, "_model_pc_metadata", {}).get(pc, {})
                label = str(metadata.get("PC label", "") or "")
                item = self._add_loading_item(self.pc_list, pc,
                                              description=label or metadata.get("PC description", ""))
                item.setData(0, Qt.UserRole, pc)
                if metadata.get("PC description"):
                    item.setToolTip(0, str(metadata["PC description"]))
                    item.setToolTip(2, str(metadata["PC description"]))
                if select is not None and pc == select:
                    self.pc_list.setCurrentItem(item)
            self.pc_list.setColumnHidden(1, hide_values)
            self._reapply_tree_sort(self.pc_list)
            self.pc_list.resizeColumnToContents(0); self.pc_list.resizeColumnToContents(1)
        finally:
            self.pc_list.setUpdatesEnabled(True)
        self.pc_list_label.setText("Model PCs")

    def _rebuild_full_feature_list(self, select=None, hide_values=True):
        """Show every feature; see _rebuild_full_pc_list for why this must
        happen rather than leaving a stale weight-filtered subset in place."""
        self.feature_list.setUpdatesEnabled(False)
        try:
            self.feature_list.clear()
            for feature in self._model_features:
                metadata = getattr(self, "_model_feature_metadata", {}).get(feature, {})
                label = str(metadata.get("Feature label", "") or "")
                description = str(metadata.get("Feature description", "") or "")
                label, note = compact_feature_text(feature, label, description)
                item = self._add_loading_item(self.feature_list, label, description=note)
                item.setText(3, feature)
                item.setData(0, Qt.UserRole, feature)
                item.setToolTip(0, feature)
                if description:
                    item.setToolTip(2, description)
                if select is not None and feature == select:
                    self.feature_list.setCurrentItem(item)
            self.feature_list.setColumnHidden(1, hide_values)
            self._reapply_tree_sort(self.feature_list)
            self.feature_list.resizeColumnToContents(0); self.feature_list.resizeColumnToContents(1)
        finally:
            self.feature_list.setUpdatesEnabled(True)
        self.feature_list_label.setText("Model features")

    @staticmethod
    def _set_filtered_list_title(label, title, shown, total):
        label.setText(f"{title} ({shown} of {total} above threshold)" if shown != total else title)

    @staticmethod
    def _clear_loading_values(tree):
        """Remove contextual values from the source list without rebuilding it.

        Rebuilding a list while its click event is active makes Qt reset the
        scroll position and can select a different row.  Keeping its items in
        place also lets keyboard navigation continue naturally.
        """
        for index in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(index)
            item.setText(1, "")
            item.setData(1, Qt.UserRole, None)
            for column in (0, 1):
                item.setBackground(column, QtGui.QBrush())
                item.setForeground(column, QtGui.QBrush())
        tree.setColumnHidden(1, True)

    def _loading_threshold_changed(self, _value):
        if getattr(self, "_model_selection_updating", False):
            return
        self._model_selection_updating = True
        try:
            if getattr(self, "_active_pc", None):
                self._set_feature_loadings(self._active_pc)
            elif getattr(self, "_active_feature", None):
                self._set_pc_loadings(self._active_feature)
        finally:
            self._model_selection_updating = False

    def _set_feature_loadings(self, pc):
        df = getattr(self, "_model_loadings", pd.DataFrame())
        values = df[df["PC"].eq(pc)].set_index("Feature")["Loading"] if pc else pd.Series(dtype=float)
        maximum = float(values.abs().max()) if not values.empty else 0.0
        shown = 0
        self.feature_list.setUpdatesEnabled(False)
        try:
            self.feature_list.clear()
            for feature in self._model_features:
                value = float(values.get(feature, np.nan))
                if (not np.isfinite(value) or np.isclose(value, 0.0)
                        or abs(value) < self.loading_threshold.value()):
                    continue
                shown += 1
                metadata = getattr(self, "_model_feature_metadata", {}).get(feature, {})
                label = str(metadata.get("Feature label", "") or "")
                description = str(metadata.get("Feature description", "") or "")
                label, note = compact_feature_text(feature, label, description)
                item = self._add_loading_item(self.feature_list, label, value, note)
                item.setText(3, feature)
                item.setData(0, Qt.UserRole, feature); item.setToolTip(0, feature)
                if description:
                    item.setToolTip(2, description)
                for column in (0, 1):
                    item.setBackground(column, self._loading_brush(value, maximum))
                    item.setForeground(column, QtGui.QBrush(QtGui.QColor("#111827")))
            self.feature_list.setColumnHidden(1, False)
            self._reapply_tree_sort(self.feature_list)
            self.feature_list.resizeColumnToContents(0); self.feature_list.resizeColumnToContents(1)
        finally:
            self.feature_list.setUpdatesEnabled(True)
        self._set_filtered_list_title(self.feature_list_label, "Model features", shown, len(self._model_features))

    def _set_pc_loadings(self, feature):
        df = getattr(self, "_model_loadings", pd.DataFrame())
        values = df[df["Feature"].eq(feature)].set_index("PC")["Loading"] if feature else pd.Series(dtype=float)
        maximum = float(values.abs().max()) if not values.empty else 0.0
        shown = 0
        self.pc_list.setUpdatesEnabled(False)
        try:
            self.pc_list.clear()
            for pc in self._model_pcs:
                value = float(values.get(pc, np.nan))
                if (not np.isfinite(value) or np.isclose(value, 0.0)
                        or abs(value) < self.loading_threshold.value()):
                    continue
                shown += 1
                metadata = getattr(self, "_model_pc_metadata", {}).get(pc, {})
                label = str(metadata.get("PC label", "") or "")
                item = self._add_loading_item(self.pc_list, pc, value,
                                              label or metadata.get("PC description", ""))
                item.setData(0, Qt.UserRole, pc)
                item.setToolTip(0, str(metadata.get("PC description") or pc))
                item.setToolTip(2, str(metadata.get("PC description", "")))
                for column in (0, 1):
                    item.setBackground(column, self._loading_brush(value, maximum))
                    item.setForeground(column, QtGui.QBrush(QtGui.QColor("#111827")))
            self.pc_list.setColumnHidden(1, False)
            self._reapply_tree_sort(self.pc_list)
            self.pc_list.resizeColumnToContents(0); self.pc_list.resizeColumnToContents(1)
        finally:
            self.pc_list.setUpdatesEnabled(True)
        self._set_filtered_list_title(self.pc_list_label, "Model PCs", shown, len(self._model_pcs))

    def _pc_selected(self, item, *_unused):
        if item is not None and not getattr(self, "_model_selection_updating", False):
            pc = str(item.data(0, Qt.UserRole))
            self._active_pc, self._active_feature = pc, None
            if self.pc_list.topLevelItemCount() != len(self._model_pcs):
                # pc_list is becoming the seed/domain list for this click; a
                # prior feature selection may have left it narrowed to that
                # feature's weight-filtered subset, which has no bearing on
                # the PC just clicked, so show all PCs again. Defer this:
                # rebuilding the tree synchronously, mid-click, swaps items
                # out from under the still-hovering mouse, and Qt's hover
                # highlight then sticks on whatever row ends up there instead
                # of following the actual (correctly set) selection.
                QtCore.QTimer.singleShot(0, lambda pc=pc: self._restore_pc_seed_then_filter(pc))
                return
            self._model_selection_updating = True
            try:
                # Do not otherwise rebuild the clicked tree: doing so changes
                # the current row under Qt while a mouse/keyboard event is
                # still in flight.
                self._clear_loading_values(self.pc_list)
                self._set_feature_loadings(pc)
            finally:
                self._model_selection_updating = False

    def _restore_pc_seed_then_filter(self, pc):
        self._model_selection_updating = True
        try:
            self._rebuild_full_pc_list(select=pc)
            self._set_feature_loadings(pc)
        finally:
            self._model_selection_updating = False

    def _feature_selected(self, item, *_unused):
        if item is not None and not getattr(self, "_model_selection_updating", False):
            feature = str(item.data(0, Qt.UserRole))
            self._active_pc, self._active_feature = None, feature
            if self.feature_list.topLevelItemCount() != len(self._model_features):
                QtCore.QTimer.singleShot(0, lambda feature=feature: self._restore_feature_seed_then_filter(feature))
                return
            self._model_selection_updating = True
            try:
                self._clear_loading_values(self.feature_list)
                self._set_pc_loadings(feature)
            finally:
                self._model_selection_updating = False

    def _restore_feature_seed_then_filter(self, feature):
        self._model_selection_updating = True
        try:
            self._rebuild_full_feature_list(select=feature)
            self._set_pc_loadings(feature)
        finally:
            self._model_selection_updating = False

    def _pc_reselected(self, item):
        if item is not None and str(item.data(0, Qt.UserRole)) == getattr(self, "_active_pc", None):
            self._reset_model_explorer()

    def _feature_reselected(self, item):
        if item is not None and str(item.data(0, Qt.UserRole)) == getattr(self, "_active_feature", None):
            self._reset_model_explorer()

    def calculate(self):
        if getattr(self.ctrl, "_busy", False):
            return
        if not self._read_model(): return
        if hasattr(self.ctrl, "_sync_proj_sample_list_from_view"):
            self.ctrl._sync_proj_sample_list_from_view()
        rows = self.ctrl._sample_rows_from_source_model()
        if not rows:
            return
        estimate = self._estimated_pp_bytes(rows)
        # A conservative advanced-setting threshold; ordinary users need only
        # see a clear estimate, not an implementation term such as “budget”.
        budget = QtCore.QSettings("Lunascope", "Lunascope").value("hypnodensity_viewer/memory_warning_mb", 1536, type=int) * 1024**2
        if estimate > budget:
            QtWidgets.QMessageBox.warning(
                self, "Large hypnodensity cohort",
                f"Lunascope will keep about {estimate / 1024**2:.0f} MB of hypnodensity data open while you browse this cohort.\n\n"
                "This can slow the application on some computers. Calculation will continue; no data will be discarded.")
        # A saved session can be resumed safely: an unchanged source plus the
        # exact same projection-model signature is authoritative cached work.
        reusable = self.session.results_model_signature == self.session.model_signature
        old_features = self.session.features.set_index("ID", drop=False) if not self.session.features.empty else pd.DataFrame()
        old_pcs = self.session.pcs.set_index("ID", drop=False) if not self.session.pcs.empty else pd.DataFrame()
        old_exclusions = {str(item.get("ID")): item for item in self.session.exclusions}
        old_sources = dict(self.session.source_signatures)
        old_traces, old_masks = self.session.traces, self.session.invalid_masks
        old_axes, old_analysis = self.session.time_axes, self.session.analysis
        features, pcs, exclusions, traces, invalid_masks, time_axes, analysis, sources = [], [], [], {}, {}, {}, {}, {}
        reused = 0
        tasks = []
        param = self.ctrl._parse_tab_pairs(self.ctrl.ui.txt_param)
        for ordinal, row in enumerate(rows, 1):
            sample_id = str(row[0]).strip()
            source = source_signature(str(row[1])) if len(row) > 1 else {}
            sources[sample_id] = source
            cached = reusable and old_sources.get(sample_id) == source
            if cached and sample_id in old_exclusions:
                exclusions.append(old_exclusions[sample_id]); reused += 1
                self.status.setText(f"Processed {ordinal}/{len(rows)}; {len(features)} eligible; {len(exclusions)} excluded ({reused} reused)")
                continue
            if cached and sample_id in old_features.index and sample_id in old_pcs.index and sample_id in old_traces:
                features.append(old_features.loc[sample_id].to_dict()); pcs.append(old_pcs.loc[sample_id].to_dict())
                traces[sample_id] = old_traces[sample_id]; invalid_masks[sample_id] = old_masks.get(sample_id, np.zeros(len(traces[sample_id]), dtype=bool))
                time_axes[sample_id] = old_axes.get(sample_id, np.arange(len(traces[sample_id]), dtype=float)); analysis[sample_id] = old_analysis.get(sample_id, {})
                reused += 1
                self.status.setText(f"Processed {ordinal}/{len(rows)}; {len(features)} eligible; {len(exclusions)} excluded ({reused} reused)")
                continue
            tasks.append({"ordinal": ordinal, "sample_row": list(row),
                          "model_path": self.session.model_path, "param": param})
        cancelled = False
        if tasks:
            config = self.ctrl._project_eval_config_dialog(len(tasks))
            if config is None:
                return
            results, cancelled = self._run_hdstats_tasks(tasks, config["workers"], len(rows), reused)
            for result in results:
                sid = result["ID"]
                if result.get("error"):
                    exclusions.append({"ID": sid, "reason": result["error"]})
                    continue
                features.append({"ID": sid, **result["features"]})
                pcs.append({"ID": sid, **self._apply_pc_orientation(result["scores"])})
                traces[sid], invalid_masks[sid], time_axes[sid] = result["trace"], result["invalid"], result["axis"]
                analysis[sid] = result["analysis"]
        # Completion order, cached records and worker count never change SL order.
        order = {str(row[0]).strip(): ordinal for ordinal, row in reversed(list(enumerate(rows)))}
        for records in (features, pcs, exclusions):
            records.sort(key=lambda record: order[record["ID"]])
        traces = dict(sorted(traces.items(), key=lambda pair: order[pair[0]]))
        self.session.rows, self.session.traces, self.session.invalid_masks, self.session.time_axes, self.session.analysis, self.session.exclusions, self.session.source_signatures = rows, traces, invalid_masks, time_axes, analysis, exclusions, sources
        self.session.features = pd.DataFrame(features).drop_duplicates("ID") if features else pd.DataFrame(columns=["ID"])
        self.session.pcs = pd.DataFrame(pcs).drop_duplicates("ID") if pcs else pd.DataFrame(columns=["ID"])
        self.session.results_model_signature = dict(self.session.model_signature)
        self._refresh_views()
        # Once a user has chosen a cache, later resume runs update that cache
        # without another file-dialog round trip.
        if self.session.cache_path:
            self.save_cache(self.session.cache_path, quiet=True)
        if cancelled:
            self.status.setText(f"Stopped: {len(features)} eligible / {len(exclusions)} excluded; "
                                f"{len(rows) - len(features) - len(exclusions)} unfinished. Calculate / resume to continue.")

    def _run_hdstats_tasks(self, tasks, workers, total, reused):
        from .anal import _clamp_project_eval_workers, _init_project_eval_child, _terminate_process_pool
        workers = _clamp_project_eval_workers(workers, len(tasks))
        progress = _HDStatsProgress(self, total, reused, self.ctrl._project_eval_cancel_shortcuts())
        completed, pending = [], {}
        remaining = iter(tasks)
        executor = None
        timer = QtCore.QTimer(progress)
        errors = []

        def submit_next():
            task = next(remaining, None)
            if task is not None:
                pending[executor.submit(_hdstats_record_worker, task)] = task

        def poll():
            try:
                for future in list(pending):
                    if not future.done():
                        continue
                    task = pending.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"ordinal": task["ordinal"], "ID": str(task["sample_row"][0]).strip(),
                                  "error": f"{type(exc).__name__}: {exc}"}
                    completed.append(result)
                    progress.progress.setValue(reused + len(completed))
                    if not progress.cancel_requested:
                        submit_next()
                if progress.cancel_requested or not pending:
                    timer.stop()
                    progress.accept()
            except Exception as exc:
                errors.append(str(exc))
                progress.cancel_requested = True
                timer.stop()
                progress.accept()

        self.ctrl._busy = True
        try:
            self.ctrl._buttons(False)
            self.ctrl.lock_ui("Calculating hypnodensity cohort…")
            executor = ProcessPoolExecutor(max_workers=workers,
                                           mp_context=multiprocessing.get_context("spawn"),
                                           initializer=_init_project_eval_child)
            for _ in range(workers):
                submit_next()
            timer.timeout.connect(poll)
            timer.start(100)
            progress.exec()
        except Exception as exc:
            errors.append(str(exc))
            progress.cancel_requested = True
        finally:
            timer.stop()
            try:
                if executor is not None:
                    if progress.cancel_requested:
                        for future in pending:
                            future.cancel()
                        if not _terminate_process_pool(executor):
                            executor.shutdown(wait=True, cancel_futures=True)
                    else:
                        executor.shutdown(wait=True, cancel_futures=True)
            finally:
                self.ctrl.unlock_ui()
                self.ctrl._busy = False
                self.ctrl._buttons(True)
                progress.deleteLater()
        if errors:
            QtWidgets.QMessageBox.warning(self, "Hypnodensity calculation", "\n".join(errors))
        completed.sort(key=lambda result: result["ordinal"])
        return completed, progress.cancel_requested

    @staticmethod
    def _load_pp_trace(p):
        """Read PP channels separately; EDF ``data()`` needlessly requires a matrix SR."""
        available = set(p.edf.channels())
        invalid_channel = "PP_INVALID" if "PP_INVALID" in available else None
        srv = lp.segsrv(p)
        srv.populate(chs=[*PP_CHANNELS, *([invalid_channel] if invalid_channel else [])], anns=[])
        duration = float(srv.num_seconds_clocktime_original())
        srv.window(0.0, duration)
        columns = [np.asarray(srv.get_signal(channel), dtype=np.float32) for channel in PP_CHANNELS]
        lengths = {col.size for col in columns}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("incompatible PP time axes (different channel lengths)")
        trace = np.column_stack(columns)
        invalid = np.zeros(trace.shape[0], dtype=bool)
        if invalid_channel:
            invalid_values = np.asarray(srv.get_signal(invalid_channel), dtype=float)
            if invalid_values.size != trace.shape[0]:
                raise ValueError("PP_INVALID length differs from PP channels")
            invalid = ~np.isfinite(invalid_values) | (invalid_values >= 0.5)
        return trace, invalid, np.asarray(srv.get_timetrack(PP_CHANNELS[0]), dtype=float)

    def _browser_measures(self):
        """Expose model features once, using raw measurements, never PC contributions."""
        source = self.session.features
        result = {"ID": source["ID"]}
        raw_columns = {}
        for column in source.columns:
            if column.startswith("RAW__"):
                match = re.search(r"(?:^|\|)FTR=([^|]+)", column)
                if match:
                    raw_columns.setdefault(match.group(1), []).append(column)
        for feature in getattr(self, "_model_features", []):
            columns = raw_columns.get(feature, [])
            if feature in source.columns:
                result[feature] = source[feature]
            elif columns:
                result[feature] = source[columns].bfill(axis=1).iloc[:, 0]
            else:
                direct = matching_hdstats_feature(
                    feature, [c for c in source.columns if not c.startswith(("VAL__", "RAW__"))])
                result[feature] = source[direct] if direct else np.nan
        return pd.DataFrame(result, index=source.index)

    def _merged(self):
        out = self._browser_measures().merge(self.session.pcs, on="ID", how="outer")
        if not self.session.tabular.empty: out = out.merge(self.session.tabular, on="ID", how="left")
        if self.session.rows:
            order = {str(row[0]).strip(): ordinal for ordinal, row in reversed(list(enumerate(self.session.rows)))}
            out = out.iloc[np.argsort(out["ID"].astype(str).map(order).fillna(len(order)).to_numpy(), kind="stable")]
        return out

    def _apply_pc_orientation(self, scores: dict) -> dict:
        return apply_pc_orientation(scores, getattr(self, "_model_pc_raw_lookup", {}),
                                     getattr(self, "_model_pc_metadata", {}))

    def _display_label(self, name: str) -> str:
        metadata = getattr(self, "_model_feature_metadata", {}).get(name)
        if metadata is not None:
            label, note = compact_feature_text(name, str(metadata.get("Feature label", "")),
                                                str(metadata.get("Feature description", "")))
            return f"{label} ({note})" if note else label
        if name in self.session.pcs.columns:
            label = str(getattr(self, "_model_pc_metadata", {}).get(name, {}).get("PC label") or "")
            return label or name
        return name

    def _refresh_views(self):
        previous_id = getattr(self, "_selected_id", None)
        df = self._merged(); choices = [c for c in df.columns if c != "ID"]
        self._update_thumbnail_reference_seconds()
        tabular = [c for c in self.session.tabular.columns if c != "ID"]
        allowed = set(choices)
        selected = [c for c in self.session.browser_values if c in allowed][:2]
        self.session.browser_values = selected
        if self.browser_flagged_only.isChecked() and self.session.flagged:
            df = df[df.ID.astype(str).isin(self.session.flagged)].copy()
        elif self.browser_flagged_only.isChecked():
            df = df.iloc[0:0].copy()
        if self.browser_complete.isChecked() and selected:
            df = df[df.loc[:, selected].notna().all(axis=1)].copy()
        durations = [self._duration_hours(i) for i in df.ID]
        show = pd.DataFrame({"ID": df["ID"], "Hypnodensity": [
            (self.session.traces.get(i), self.session.invalid_masks.get(i),
             self.session.time_axes.get(i), self.session.analysis.get(i))
            for i in df.ID
        ], "Hours": durations})
        self._browser_display_columns = {}
        header_labels, header_tooltips = {}, {}
        for col in selected:
            label = self._display_label(col)
            # Preserve a unique header even if two source fields simplify to
            # the same human-readable wording.
            unique_label, suffix = label, 2
            while unique_label in show.columns:
                unique_label = f"{label} ({suffix})"; suffix += 1
            self._browser_display_columns[col] = unique_label
            show[unique_label] = df[col]
            if col in self.session.pcs.columns:
                panel, separator, rest = col.partition("_")
                header_labels[unique_label] = panel + "\n" + "\n".join(textwrap.wrap(rest.replace("_", " "), 18)) if separator else col
            else:
                header_labels[unique_label] = "\n".join(textwrap.wrap(label, 18))
            header_tooltips[unique_label] = f"{label}\n{col}"
        table_model = _CohortBrowserModel(show, coerce_numeric=False, build_row_text=False)
        table_model.flagged_ids = self.session.flagged
        self._update_flagged_count()
        table_model.header_labels = header_labels
        table_model.header_tooltips = header_tooltips
        proxy = NumericSortFilterProxy(self.browser); proxy.setSourceModel(table_model); self.browser.setModel(proxy)
        header = self.browser.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        header.setMinimumSectionSize(40)
        self.browser.setColumnWidth(0, 135); self.browser.setColumnWidth(1, 300); self.browser.setColumnWidth(2, 60)
        metrics = header.fontMetrics()
        for col in range(3, show.shape[1]):
            label = header_labels.get(show.columns[col], str(show.columns[col]))
            header_width = max(metrics.horizontalAdvance(line) for line in label.split("\n")) + 28
            self.browser.setColumnWidth(col, max(150, min(header_width, 280)))
        self.browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.browser.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        header.setToolTip("Drag a column boundary to resize it; double-click a boundary to fit its contents. Scroll right to reach the last boundary.")
        lines = max((label.count("\n") + 1 for label in header_labels.values()), default=1)
        header.setFixedHeight(max(32, lines * header.fontMetrics().lineSpacing() + 12))
        self.browser.verticalHeader().setDefaultSectionSize(38)
        self.browser.selectionModel().currentChanged.connect(self._selected)
        self._apply_browser_filter(self.browser_filter.text())
        self._clear_individual_selection()
        if previous_id is not None:
            matches = np.flatnonzero(show["ID"].astype(str).eq(previous_id).to_numpy())
            if len(matches):
                index = proxy.mapFromSource(proxy.sourceModel().index(int(matches[0]), 0))
                if index.isValid():
                    self.browser.setCurrentIndex(index)
        detail = ""
        if self.session.exclusions:
            detail = " First exclusion: " + str(self.session.exclusions[0].get("reason", ""))
        self.status.setText(f"{len(self.session.features)} eligible / {len(self.session.exclusions)} excluded; {len(self.session.traces)} traces resident." + detail)

    def _set_sort_column(self, name):
        src = self.browser.model().sourceModel()
        display_name = getattr(self, "_browser_display_columns", {}).get(name, name)
        if display_name not in src._df.columns:
            self.session.browser_values = [name, *[c for c in self.session.browser_values if c != name]][:2]
            self._refresh_views(); src = self.browser.model().sourceModel(); display_name = self._browser_display_columns.get(name, name)
        self.pages.setCurrentIndex(1)
        col = list(src._df.columns).index(display_name); self.browser.sortByColumn(col, Qt.AscendingOrder)

    def _duration_hours(self, sample_id):
        time_axis = self.session.time_axes.get(str(sample_id))
        if time_axis is None or len(time_axis) < 2: return np.nan
        step = float(np.nanmedian(np.diff(time_axis)))
        return (float(time_axis[-1]) - float(time_axis[0]) + step) / 3600.0

    def choose_browser_values(self):
        groups = {
            "Measures / features": list(getattr(self, "_model_features", [])),
            "PC scores": [c for c in self.session.pcs.columns if c != "ID"],
            "Phenotypes / covariates": [c for c in self.session.tabular.columns if c != "ID"],
        }
        dialog = QtWidgets.QDialog(self); dialog.setWindowTitle("Browser values (choose up to two)"); layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(QtWidgets.QLabel("Choose up to two values to show alongside ID, thumbnail, and duration."))
        tree = QtWidgets.QTreeWidget(); tree.setHeaderLabels(["Measure", "Note / description"]); tree.header().setStretchLastSection(True); tree.setColumnWidth(0, 360)
        selected = set(self.session.browser_values)
        for title, names in groups.items():
            if not names:
                continue
            parent = QtWidgets.QTreeWidgetItem([title]); parent.setFlags(parent.flags() & ~Qt.ItemIsUserCheckable); tree.addTopLevelItem(parent)
            for name in names:
                metadata = getattr(self, "_model_feature_metadata", {}).get(name, {})
                if title == "Measures / features":
                    label, note = compact_feature_text(name, str(metadata.get("Feature label", "")),
                                                        str(metadata.get("Feature description", "")))
                elif title == "PC scores":
                    label, note = name, self._display_label(name)
                else:
                    label, note = name, ""
                child = QtWidgets.QTreeWidgetItem([label, note]);
                child.setToolTip(0, name)
                child.setToolTip(1, str(metadata.get("Feature description", "") or note))
                child.setData(0, Qt.UserRole, name); child.setCheckState(0, Qt.Checked if name in selected else Qt.Unchecked); parent.addChild(child)
            parent.setExpanded(title == "Measures / features")
        layout.addWidget(tree); buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel); layout.addWidget(buttons)
        clear = buttons.addButton("Clear", QtWidgets.QDialogButtonBox.ResetRole)
        def clear_selection():
            for i in range(tree.topLevelItemCount()):
                parent = tree.topLevelItem(i)
                for j in range(parent.childCount()):
                    parent.child(j).setCheckState(0, Qt.Unchecked)
        clear.clicked.connect(clear_selection)
        screen = self.screen().availableGeometry()
        dialog.resize(min(900, screen.width() - 60), max(400, screen.height() - 100))
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            values = []
            for i in range(tree.topLevelItemCount()):
                parent = tree.topLevelItem(i)
                for j in range(parent.childCount()):
                    child = parent.child(j)
                    if child.checkState(0) == Qt.Checked: values.append(child.data(0, Qt.UserRole))
            if len(values) > 2:
                QtWidgets.QMessageBox.warning(self, "Browser values", "Choose at most two values."); return
            self.session.browser_values = values; self._refresh_views()

    def _clear_individual_selection(self):
        self._selected_id = None
        self._plot_id = None; self._plot_full_xlim = None
        self._plot_window_patch = None; self._plot_duration_text = None
        self.individual_pc_title.setText("PC scores — select an individual")
        self.individual_pcs.clear()
        self.canvas.ax.clear()
        self.canvas.draw_idle()

    def _individual_score_threshold_changed(self, _value):
        sid = getattr(self, "_selected_id", None)
        if sid is not None:
            self._show_individual_pcs(sid)

    def _individual_pc_context_menu(self, pos):
        item = self.individual_pcs.itemAt(pos)
        if item is None: return
        pc = str(item.data(0, Qt.UserRole))
        menu = QtWidgets.QMenu(self.individual_pcs)
        action = menu.addAction("Show as first thumbnail value", lambda: self._add_individual_pc_to_browser(pc))
        action.setEnabled(pc not in self.session.browser_values and pc in self._merged().columns)
        menu.exec(self.individual_pcs.viewport().mapToGlobal(pos))

    def _add_individual_pc_to_browser(self, pc):
        if pc in self.session.browser_values or pc not in self._merged().columns:
            return
        self.session.browser_values = [pc, *self.session.browser_values][:2]
        self._refresh_views()

    def _show_individual_pcs(self, sid):
        self.individual_pc_title.setText(f"PC scores — {sid}")
        scores = self.session.pcs
        matches = scores[scores["ID"].astype(str).eq(sid)] if "ID" in scores else pd.DataFrame()
        row = matches.iloc[0] if not matches.empty else pd.Series(dtype=float)
        pcs = list(dict.fromkeys([*getattr(self, "_model_pcs", []),
                                  *(c for c in scores.columns if c != "ID")]))
        self.individual_pcs.setUpdatesEnabled(False)
        try:
            self.individual_pcs.clear()
            for pc in pcs:
                metadata = getattr(self, "_model_pc_metadata", {}).get(pc, {})
                value = pd.to_numeric(row.get(pc, np.nan), errors="coerce")
                threshold = self.individual_score_threshold.value()
                if threshold > 0 and (not np.isfinite(value) or abs(value) < threshold):
                    continue
                item = self._add_loading_item(self.individual_pcs, pc, value,
                                              metadata.get("PC label", ""))
                tooltip = str(metadata.get("PC description", "") or metadata.get("PC label", "") or pc)
                item.setToolTip(0, tooltip)
                item.setToolTip(2, tooltip)
                if np.isfinite(value):
                    item.setText(1, f"{value:+.2f}")
                    item.setToolTip(1, f"{value:+.6g} (model score; descriptor orientation applied)")
                    item.setBackground(1, self._loading_brush(value, 3.0))
                    item.setForeground(1, QtGui.QBrush(QtGui.QColor("#111827")))
                else:
                    item.setText(1, "—")
                    item.setToolTip(1, "No numeric score in calculated/cached results; not hidden by the loading threshold")
            self._reapply_tree_sort(self.individual_pcs)
        finally:
            self.individual_pcs.setUpdatesEnabled(True)

    def _set_plot_window(self, left, right):
        if self._plot_full_xlim is None: return
        full_left, full_right = self._plot_full_xlim
        full_width = full_right - full_left
        width = min(full_width, max(2.0, float(right) - float(left)))
        left = min(max(float(left), full_left), full_right - width)
        right = left + width
        self.canvas.ax.set_xlim(left, right)
        if self._plot_window_patch is not None:
            self._plot_window_patch.set_x((left - full_left) / full_width)
            self._plot_window_patch.set_width(width / full_width)
            self._plot_window_patch.set_visible(width < full_width - 0.01)
        if self._plot_duration_text is not None:
            if width < full_width - 0.01:
                sid = self._plot_id
                axis = self.session.time_axes.get(sid)
                if axis is not None and len(axis) > 1:
                    a = np.asarray(axis, dtype=float)
                    step = float(np.nanmedian(np.diff(a)))
                    seconds = float(np.interp(right - .5, np.arange(len(a)), a) -
                                    np.interp(left - .5, np.arange(len(a)), a) + step)
                    trt = float(a[-1] - a[0] + step)
                    window_text = f"{seconds / 3600:.2f} h"
                    trt_text = f"{trt / 3600:.2f} h"
                else:
                    window_text = trt_text = "—"
                meta = self.session.analysis.get(sid, {})
                onset, final = meta.get("soft_onset_sec"), meta.get("soft_final_wake_sec")
                spt = f"{(float(final) - float(onset)) / 3600:.2f} h" if onset is not None and final is not None else "—"
                self._plot_duration_text.set_text(f"Window {window_text}  ·  SPT {spt}  ·  TRT {trt_text}")
                self._plot_duration_text.set_visible(True)
            else:
                self._plot_duration_text.set_visible(False)
        self.canvas.draw_idle()

    def _plot_zoom(self, factor, center=None):
        if self._plot_full_xlim is None: return
        left, right = self.canvas.ax.get_xlim()
        if center is None or not np.isfinite(center): center = (left + right) / 2
        new_width = (right - left) * factor
        ratio = (center - left) / (right - left)
        self._set_plot_window(center - ratio * new_width, center + (1 - ratio) * new_width)

    def _plot_key(self, event):
        if self._plot_full_xlim is None: return
        key = str(event.key).lower()
        if key == "up": self._plot_zoom(.8)
        elif key == "down": self._plot_zoom(1.25)
        elif key in ("left", "right"):
            left, right = self.canvas.ax.get_xlim()
            shift = (right - left) * (-.2 if key == "left" else .2)
            self._set_plot_window(left + shift, right + shift)

    def _plot_scroll(self, event):
        if self._plot_full_xlim is not None and event.inaxes is self.canvas.ax:
            self._plot_zoom(.8 if event.step > 0 else 1.25, event.xdata)

    def _plot_mouse_press(self, event):
        if self._plot_full_xlim is None or event.inaxes is not self.canvas.ax: return
        self.canvas.setFocus(Qt.MouseFocusReason)
        if event.dblclick:
            self._plot_drag = None
            self._set_plot_window(*self._plot_full_xlim)
            return
        if event.button in (1, 3) and event.xdata is not None:
            self._plot_drag = (event.button, event.xdata, self.canvas.ax.get_xlim())

    def _plot_mouse_motion(self, event):
        if self._plot_drag is None or self._plot_drag[0] != 1 or event.xdata is None:
            return
        _, start, (left, right) = self._plot_drag
        # Convert cursor movement through pixel space so dragging stays stable
        # while the x limits change under the cursor.
        span = right - left
        if event.x is not None:
            initial_pixel = self.canvas.ax.transAxes.transform(((start - left) / span, 0))[0]
            shift = (initial_pixel - event.x) / self.canvas.ax.bbox.width * span
            self._set_plot_window(left + shift, right + shift)

    def _plot_mouse_release(self, event):
        drag = self._plot_drag; self._plot_drag = None
        if drag is None or drag[0] != 3 or event.xdata is None: return
        start = drag[1]
        if abs(event.xdata - start) >= 2:
            self._set_plot_window(min(start, event.xdata), max(start, event.xdata))

    def _selected(self, index, *_unused):
        if not index.isValid() or self.browser.model() is None:
            self._clear_individual_selection()
            return
        source = self.browser.model().mapToSource(index)
        if not source.isValid():
            self._clear_individual_selection()
            return
        sid = str(self.browser.model().sourceModel()._df.iloc[source.row()]["ID"])
        self._selected_id = sid
        self._show_individual_pcs(sid)
        trace = self.session.traces.get(sid)
        if trace is None:
            self._plot_id = None; self._plot_full_xlim = None
            self.canvas.ax.clear()
            self.canvas.draw_idle()
            return
        old_window = self.canvas.ax.get_xlim() if self._plot_id == sid and self._plot_full_xlim is not None else None
        self._plot_id = sid; self._plot_full_xlim = (.5, len(trace) + .5)
        values, labels, colors = display_hypnodensity(trace, self._display_mode)
        frame = pd.DataFrame(values, columns=labels); hypno_density(frame, self.canvas.ax, colors=colors)
        self.canvas.ax.set_axis_off()
        invalid = self.session.invalid_masks.get(sid)
        full_width = float(len(trace))
        def overview_span(x0, x1, y0, height, color, **kwargs):
            patch = Rectangle(((x0 - .5) / full_width, y0), (x1 - x0) / full_width, height,
                              transform=self.canvas.ax.transAxes, facecolor=color, clip_on=False, **kwargs)
            self.canvas.ax.add_patch(patch)
            return patch
        for start, stop, call in hard_call_runs(trace, invalid):
            color = "#9ca3af" if call < 0 else PP_COLORS[call]
            overview_span(start + .5, stop + .5, 1.12 / 1.18, .05 / 1.18, color, lw=0)
        self.canvas.ax.set_ylim(0, 1.18)
        meta = self.session.analysis.get(sid, {})
        time_axis = self.session.time_axes.get(sid)
        def position(seconds):
            if time_axis is None or len(time_axis) != len(trace): return float(seconds) + .5
            return float(np.searchsorted(time_axis, float(seconds), side="left")) + .5
        onset, final = meta.get("soft_onset_sec"), meta.get("soft_final_wake_sec")
        if onset is not None and final is not None:
            x0, x1 = position(onset), position(final)
            # Orange deliberately contrasts with the N1 cyan.  The narrow
            # SPT band sits just below the hard-call strip rather than being
            # visually centred in its gap.
            soft_color = "#f97316"
            overview_span(x0, x1, 1.035 / 1.18, .045 / 1.18, "#4b5563", alpha=.70, lw=0, zorder=3)
            for x, label, seconds in ((x0, "onset", onset), (x1, "final wake", final)):
                fraction = (x - .5) / full_width
                self.canvas.ax.plot([fraction, fraction], [1.010 / 1.18, 1.095 / 1.18],
                                    transform=self.canvas.ax.transAxes, color=soft_color, lw=3.2, zorder=5)
                self.canvas.ax.text(fraction, 1.095 / 1.18, f"{label}\n{float(seconds) / 3600:.2f} h", rotation=90,
                                    va="bottom", ha="right", color=soft_color, fontsize=6.5,
                                    transform=self.canvas.ax.transAxes, clip_on=True, zorder=6)
        start, stop = meta.get("HD_ANALYSIS_START_SEC"), meta.get("HD_ANALYSIS_STOP_SEC")
        if start is not None and stop is not None:
            for x in (position(start), position(stop)):
                # Analysis-flank markers mirror the soft-SPT marker exactly:
                # they annotate the strip, never cut through the densities.
                fraction = (x - .5) / full_width
                self.canvas.ax.plot([fraction, fraction], [1.010 / 1.18, 1.095 / 1.18],
                                    transform=self.canvas.ax.transAxes, color="#c084fc", lw=3.2, ls=":", zorder=4)
        if invalid is not None and np.any(invalid):
            mask = np.asarray(invalid, dtype=bool); edges = np.diff(np.r_[False, mask, False].astype(int)); starts = np.flatnonzero(edges == 1); stops = np.flatnonzero(edges == -1)
            for start, stop in zip(starts, stops): self.canvas.ax.axvspan(start + 1, stop + 1, color="#9ca3af", alpha=.32, lw=0)
        self._plot_window_patch = overview_span(.5, len(trace) + .5, 1.105 / 1.18, .065 / 1.18,
                                                (0.984, .749, .141, .22), edgecolor="#fbbf24", lw=1.8, zorder=7)
        self._plot_duration_text = self.canvas.ax.text(.99, .03, "", transform=self.canvas.ax.transAxes,
            va="bottom", ha="right", color="white", fontsize=8,
            bbox={"facecolor": "#111827", "edgecolor": "none", "alpha": .82, "pad": 3}, zorder=9)
        self.canvas.ax.set_title(f"{sid} — {DISPLAY_MODES[self._display_mode]}")
        self._set_plot_window(*(old_window or self._plot_full_xlim))

    def join_tsv(self):
        path, _ = open_file_name(self, "Load phenotype/covariate TSV", "", "Tab-delimited files (*.tsv *.txt);;All files (*)")
        if not path: return
        tab = pd.read_csv(path, sep="\t", dtype=None); options = list(tab.columns)
        ident, ok = QtWidgets.QInputDialog.getItem(self, "TSV ID column", "Map to sample-list ID:", options, 0, False)
        if not ok: return
        dup = int(tab[ident].duplicated().sum()); tab[ident] = tab[ident].astype(str); tab = tab.drop_duplicates(ident, keep="first").rename(columns={ident: "ID"})
        # One explicit TSV keeps variable names and missingness unambiguous. Loading another file intentionally replaces this one.
        self.session.tabular = tab
        ids = set(self._merged().ID.astype(str)); tids = set(tab.ID); missing = sorted(ids-tids); orphan = sorted(tids-ids)
        matched = tab[tab.ID.isin(ids)]
        missing_counts = matched.drop(columns="ID").isna().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
        summary = [f"{len(matched)} individuals matched · {len(tab.columns) - 1} measures"]
        if orphan:
            summary.append(f"{len(orphan)} unmatched TSV IDs ignored.")
        if missing:
            summary.append(f"{len(missing)} cohort IDs have no TSV data.")
        if dup:
            summary.append(f"{dup} duplicate rows ignored (first row kept).")
        if not missing_counts.empty:
            summary.append(f"{len(missing_counts)} measures contain missing values.")
        details = []
        if not missing_counts.empty:
            details.append(f"Missing values among {len(matched)} matched individuals:\n" +
                           "\n".join(f"{name}: {count}" for name, count in missing_counts.items()))
        if missing:
            details.append("Cohort IDs without TSV data:\n" + "\n".join(missing))
        if orphan:
            details.append("Ignored TSV IDs:\n" + "\n".join(orphan))
        message = QtWidgets.QMessageBox(self)
        message.setWindowTitle("TSV loaded")
        message.setTextFormat(Qt.PlainText)
        message.setText(Path(path).name)
        message.setInformativeText("\n".join(summary))
        message.setStandardButtons(QtWidgets.QMessageBox.Ok)
        if details:
            message.setDetailedText("\n\n".join(details))
        message.exec()
        self._refresh_views()

    def show_exclusions(self):
        dialog = QtWidgets.QDialog(self); dialog.setWindowTitle("HDSTATS exclusions")
        layout = QtWidgets.QVBoxLayout(dialog)
        data = pd.DataFrame(self.session.exclusions, columns=["ID", "reason"])
        layout.addWidget(QtWidgets.QLabel(f"{len(data)} excluded sample(s)"))
        table = QtWidgets.QTableView(); table.setModel(DataFrameModel(data, coerce_numeric=False)); table.resizeColumnsToContents()
        layout.addWidget(table)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close); buttons.rejected.connect(dialog.reject); buttons.accepted.connect(dialog.accept); layout.addWidget(buttons)
        dialog.resize(900, 440); dialog.exec()

    def _browser_context_menu(self, pos):
        index = self.browser.indexAt(pos)
        if not index.isValid(): return
        sid = str(self.browser.model().index(index.row(), 0).data())
        menu = QtWidgets.QMenu(self.browser)
        menu.addAction("Open in sample list (O)", lambda: self._open_in_sample_list(sid))
        menu.addSeparator()
        if sid in self.session.flagged:
            menu.addAction("Edit flag / note…", lambda: self.show_flagged(sid))
            menu.addAction("Remove from flagged (F)", lambda: self._remove_flag(sid))
        else:
            menu.addAction("Flag individual (F)", lambda: self._flag_individual(index))
        menu.exec(self.browser.viewport().mapToGlobal(pos))

    def _selected_browser_index(self):
        index = self.browser.currentIndex()
        return index if index.isValid() else None

    def _toggle_selected_flag(self):
        index = self._selected_browser_index()
        if index is None: return
        sid = str(self.browser.model().index(index.row(), 0).data())
        if sid in self.session.flagged: self._remove_flag(sid)
        else: self._flag_individual(index)

    def _open_selected_in_sample_list(self):
        index = self._selected_browser_index()
        if index is not None:
            self._open_in_sample_list(str(self.browser.model().index(index.row(), 0).data()))

    def _update_flagged_count(self):
        self.flagged_button.setText(f"Flagged ({len(self.session.flagged)})…")

    def _flag_individual(self, index):
        model = self.browser.model()
        sid = str(model.index(index.row(), 0).data())
        values = {}
        for col in [2, *range(3, model.columnCount())]:
            key = str(model.headerData(col, Qt.Horizontal, Qt.DisplayRole)).replace("\n", " ")
            value = model.index(index.row(), col).data()
            values[key] = "" if value is None else str(value)
        self.session.flagged[sid] = {"note": "", "values": values}
        self._update_flagged_count()
        if self.browser_flagged_only.isChecked(): self._refresh_views()
        else: self.browser.viewport().update()

    def _remove_flag(self, sid):
        self.session.flagged.pop(sid, None)
        self._update_flagged_count()
        if self.browser_flagged_only.isChecked(): self._refresh_views()
        else: self.browser.viewport().update()

    def _open_in_sample_list(self, sid):
        if not self.ctrl._restore_attached_subject(sid):
            QtWidgets.QMessageBox.information(self, "Sample list", f"{sid} is not in the current sample list.")
            return False
        self.ctrl.ui.dock_slist.show()
        self.ctrl.ui.tbl_slist.scrollTo(self.ctrl.ui.tbl_slist.currentIndex())
        self.ctrl.ui.raise_()
        self.ctrl.ui.activateWindow()
        return True

    def show_flagged(self, focus_id=None):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Flagged individuals")
        layout = QtWidgets.QVBoxLayout(dialog)
        summary = QtWidgets.QLabel()
        layout.addWidget(summary)
        table = QtWidgets.QTableWidget()
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(table)
        keys = list(dict.fromkeys(key for entry in self.session.flagged.values()
                                  for key in entry.get("values", {})))
        columns = ["ID", "Note", *keys]
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.horizontalHeader().setStretchLastSection(False)
        table.setColumnWidth(0, 170)
        table.setColumnWidth(1, 260)
        for col in range(2, len(columns)): table.setColumnWidth(col, 160)
        table.setAlternatingRowColors(True)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        def populate():
            table.blockSignals(True)
            table.setRowCount(len(self.session.flagged))
            for row, (sid, entry) in enumerate(self.session.flagged.items()):
                cells = [sid, entry.get("note", ""), *(entry.get("values", {}).get(key, "") for key in keys)]
                for col, value in enumerate(cells):
                    item = QtWidgets.QTableWidgetItem(str(value))
                    if col != 1: item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    table.setItem(row, col, item)
                if sid == focus_id: table.selectRow(row)
            summary.setText(f"{len(self.session.flagged)} flagged individuals · double-click a note to edit")
            table.blockSignals(False)

        def note_changed(item):
            if item.column() == 1:
                sid = table.item(item.row(), 0).text()
                if sid in self.session.flagged: self.session.flagged[sid]["note"] = item.text()

        def selected_ids():
            return [table.item(row, 0).text() for row in sorted({i.row() for i in table.selectionModel().selectedRows()})]

        def remove_selected():
            for sid in selected_ids(): self._remove_flag(sid)
            populate()

        def open_selected():
            ids = selected_ids()
            if not ids: return
            if self._open_in_sample_list(ids[0]): dialog.accept()

        def clear_all():
            if not self.session.flagged: return
            if QtWidgets.QMessageBox.question(dialog, "Clear flagged individuals", "Remove all flagged individuals and their notes?") == QtWidgets.QMessageBox.Yes:
                self.session.flagged.clear(); self._update_flagged_count()
                if self.browser_flagged_only.isChecked(): self._refresh_views()
                else: self.browser.viewport().update()
                populate()

        def copy_table():
            lines = ["\t".join(columns)]
            for sid, entry in self.session.flagged.items():
                cells = [sid, entry.get("note", ""), *(entry.get("values", {}).get(key, "") for key in keys)]
                lines.append("\t".join(str(cell).replace("\t", " ").replace("\n", " ") for cell in cells))
            QtWidgets.QApplication.clipboard().setText("\n".join(lines))

        table.itemChanged.connect(note_changed)
        table.cellDoubleClicked.connect(lambda row, col: open_selected() if col == 0 else None)
        actions = QtWidgets.QHBoxLayout()
        for label, callback in (("Open in sample list", open_selected),
                                ("Remove selected", remove_selected), ("Clear list", clear_all),
                                ("Copy table", copy_table),
                                ("Copy IDs", lambda: QtWidgets.QApplication.clipboard().setText("\n".join(self.session.flagged)))):
            button = QtWidgets.QPushButton(label); button.clicked.connect(callback); actions.addWidget(button)
        actions.addStretch()
        close = QtWidgets.QPushButton("Close"); close.clicked.connect(dialog.accept); actions.addWidget(close)
        layout.addLayout(actions)
        populate()
        dialog.resize(850, 550)
        dialog.exec()

    def save_cache(self, path=None, quiet=False):
        if path is None:
            path, _ = save_file_name(self, "Save HDSTATS cohort cache", "", "HDSTATS cache (*.hdstats.json)")
        if not path:
            return False
        self.session.cache_path = path
        archive = trace_archive_path(path)
        trace_ids = sorted(self.session.traces)
        archive_arrays = {}
        for index, sample_id in enumerate(trace_ids):
            prefix = f"r{index}"
            archive_arrays[f"{prefix}_trace"] = np.asarray(self.session.traces[sample_id], dtype=np.float32)
            archive_arrays[f"{prefix}_invalid"] = np.asarray(self.session.invalid_masks.get(sample_id, []), dtype=bool)
            archive_arrays[f"{prefix}_time"] = np.asarray(self.session.time_axes.get(sample_id, []), dtype=np.float64)
        try:
            np.savez_compressed(archive, **archive_arrays)
        except Exception as exc:
            if not quiet: QtWidgets.QMessageBox.critical(self, "Save HDSTATS cache", f"Could not save trace archive:\n{exc}")
            return False
        payload = {"version": CACHE_VERSION, "model": self.session.model_signature, "results_model": self.session.results_model_signature, "rows": self.session.rows,
                   "model_path": self.session.model_path, "model_loadings": getattr(self, "_model_loadings", pd.DataFrame()).to_dict("records"),
                   "features": self.session.features.to_dict("records"), "pcs": self.session.pcs.to_dict("records"),
                   "tabular": self.session.tabular.to_dict("records"), "exclusions": self.session.exclusions, "browser_values": self.session.browser_values,
                   "flagged": self.session.flagged,
                   "feature_columns": self.session.features.columns.tolist(), "pc_columns": self.session.pcs.columns.tolist(), "tabular_columns": self.session.tabular.columns.tolist(),
                   "required_channels": PP_CHANNELS, "source_signatures": self.session.source_signatures,
                   "analysis": self.session.analysis, "trace_archive": archive.name,
                   "trace_ids": trace_ids}
        try:
            Path(path).write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        except Exception as exc:
            try: archive.unlink(missing_ok=True)
            except OSError: pass
            if not quiet: QtWidgets.QMessageBox.critical(self, "Save HDSTATS cache", f"Could not save cache manifest:\n{exc}")
            return False
        self.status.setText(f"Saved self-contained cache: {Path(path).name} + {archive.name}")
        return True

    def open_cache(self):
        path, _ = open_file_name(self, "Open HDSTATS cohort cache", "", "HDSTATS cache (*.hdstats.json)")
        if not path: return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            archive = Path(path).with_name(payload.get("trace_archive") or trace_archive_path(path).name)
            if not archive.is_file(): raise FileNotFoundError(f"Trace archive is missing: {archive.name}")
            traces, masks, axes = {}, {}, {}
            with np.load(archive, allow_pickle=False) as loaded:
                for index, sample_id in enumerate(payload.get("trace_ids", [])):
                    prefix = f"r{index}"
                    trace = np.asarray(loaded[f"{prefix}_trace"], dtype=np.float32)
                    invalid = np.asarray(loaded[f"{prefix}_invalid"], dtype=bool)
                    time_axis = np.asarray(loaded[f"{prefix}_time"], dtype=float)
                    if trace.ndim != 2 or trace.shape[1] != len(PP_CHANNELS) or invalid.size != len(trace) or time_axis.size != len(trace):
                        raise ValueError(f"Invalid stored trace payload for {sample_id}")
                    traces[str(sample_id)], masks[str(sample_id)], axes[str(sample_id)] = trace, invalid, time_axis
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Open HDSTATS cache", f"Could not open self-contained cache:\n{exc}")
            return
        self.session = CohortSession()
        self.session.rows = payload.get("rows", []); self.session.traces = traces; self.session.invalid_masks = masks; self.session.time_axes = axes
        self.session.features = pd.DataFrame(payload.get("features", []), columns=payload.get("feature_columns") or None); self.session.pcs = pd.DataFrame(payload.get("pcs", []), columns=payload.get("pc_columns") or None); self.session.tabular = pd.DataFrame(payload.get("tabular", []), columns=payload.get("tabular_columns") or None); self.session.exclusions = payload.get("exclusions", [])
        self.session.model_signature = payload.get("model", {}); self.session.results_model_signature = payload.get("results_model", self.session.model_signature); self.session.cache_path = path; self.session.model_path = payload.get("model_path", ""); self.session.browser_values = payload.get("browser_values", []); self.session.flagged = payload.get("flagged", {}); self.session.source_signatures = payload.get("source_signatures", {}); self.session.analysis = payload.get("analysis", {})
        saved_loadings = pd.DataFrame(payload.get("model_loadings", []))
        if not saved_loadings.empty:
            self._model_loadings = saved_loadings
            self._populate_model_explorer()
        else:
            self._model_loadings = pd.DataFrame()
            self._populate_model_explorer()
        self.model_edit.setText(self.session.model_path)
        self.status.setText(f"Loaded self-contained cache: {len(traces)} traces available without source EDFs.")
        self._refresh_views()

class HypnodensityViewerMixin:
    def _init_hypnodensity_viewer(self):
        self._hypnodensity_viewer = None

    def show_hypnodensity_viewer(self):
        if self._hypnodensity_viewer is None:
            self._hypnodensity_viewer = HypnodensityViewerDialog(self, self.ui)
        self._hypnodensity_viewer.show(); self._hypnodensity_viewer.raise_(); self._hypnodensity_viewer.activateWindow()

    def _download_hdpc_resources(self):
        cache_dir = app_cache_root() / "hdpc"
        archive = cache_dir / "luna-resources-hdpc.zip"
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(HDPC_DOWNLOAD_URL, timeout=180) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
            with zipfile.ZipFile(archive) as bundle:
                contents = {}
                for name in (HDPC_MODEL_NAME, HDPC_DESCRIPTOR_NAME):
                    matches = [member for member in bundle.namelist() if Path(member).name == name]
                    if len(matches) != 1:
                        raise FileNotFoundError(f"Expected one {name} in the HDPC resource archive; found {len(matches)}.")
                    contents[name] = bundle.read(matches[0])
            model_data = json.loads(contents[HDPC_MODEL_NAME])
            desc_data = json.loads(contents[HDPC_DESCRIPTOR_NAME])
            if not isinstance(model_data, dict) or not isinstance(desc_data, dict):
                raise ValueError("The downloaded core model or descriptor is not a JSON object.")
            if desc_data.get("model", {}).get("model_id") != HDPC_MODEL_NAME:
                raise ValueError("The HDPC descriptor does not identify the core model.")
            staged = {}
            with tempfile.TemporaryDirectory(prefix="hdpc-", dir=str(cache_dir)) as temp_dir:
                for name, data in contents.items():
                    staged_path = Path(temp_dir) / name
                    staged_path.write_bytes(data)
                    staged[name] = staged_path
                targets = {name: cache_dir / name for name in contents}
                for name, target in targets.items():
                    staged[name].replace(target)
            settings = QtCore.QSettings("Lunascope", "Lunascope")
            settings.setValue(_MODEL_SETTING_KEY, str(cache_dir / HDPC_MODEL_NAME))
            viewer = self._hypnodensity_viewer
            if viewer is not None:
                viewer.model_edit.setText(str(cache_dir / HDPC_MODEL_NAME))
                viewer.descriptor_edit.setText(str(cache_dir / HDPC_DESCRIPTOR_NAME))
                viewer._read_model()
            QtWidgets.QMessageBox.information(self.ui, "HDPC resources", f"Downloaded HDPC model and descriptor to:\n{cache_dir}")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self.ui, "HDPC download error", f"Could not download HDPC resources.\n\n{type(exc).__name__}: {exc}")
