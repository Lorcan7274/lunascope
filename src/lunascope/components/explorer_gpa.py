
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
#  Luna / Lunascope  —  Explorer: GPA association analysis tab
#  --------------------------------------------------------------------

"""Two-phase GPA workflow tab.

Build sub-tab  — point at source files, assign column roles, generate a
                 --gpa-prep JSON spec, build the binary .dat matrix.
Analyze sub-tab — load a .dat manifest, pick X/Y/Z variables, run GPA,
                 explore results in a volcano plot + sortable table.
"""

import io
import json
import os
import shutil
import tempfile
import traceback
import zipfile

import numpy as np
import pandas as pd

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer, QSortFilterProxyModel, QRegularExpression
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QSplitter, QStackedWidget, QTableView, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from .explorer_base import BG, FG, GRID, SEP, _ExplorerTab
from ..file_dialogs import open_file_name, save_file_name
from .tbl_funcs import copy_selection, save_table_as_tsv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROLES        = ["ID", "FAC", "VAR", "fixed", "exclude"]
_ROLE_COLORS  = {
    "ID":      "#4cc9f0",
    "FAC":     "#a78bfa",
    "VAR":     "#06d6a0",
    "fixed":   "#ffd166",
    "exclude": "#444455",
}
_PALETTE = [
    "#4cc9f0", "#f9844a", "#06d6a0", "#a78bfa",
    "#ffd166", "#f72585", "#90be6d", "#ff6b6b",
    "#43aa8b", "#577590", "#c77dff", "#fb8500",
]

# ---------------------------------------------------------------------------
# Source-file reading helpers
# ---------------------------------------------------------------------------

def _sniff_tsv(path, nrows=6):
    """Read first *nrows* of a tab-delimited file; return DataFrame or None."""
    try:
        return pd.read_csv(path, sep="\t", nrows=nrows, dtype=str)
    except Exception:
        return None


def _parse_gpa_manifest_text(text):
    """Parse GPA manifest TSV text into a DataFrame."""
    text = (text or "").strip()
    if not text:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text), sep="\t", dtype=str)


def _gpa_manifest_sidecar_path(dat_path):
    """Return the sidecar path used to cache the full prep-time manifest."""
    root, _ = os.path.splitext(dat_path)
    return root + ".manifest"


def _read_gpa_manifest_sidecar(dat_path):
    """Read a cached prep-time manifest sidecar if present."""
    sidecar = _gpa_manifest_sidecar_path(dat_path)
    if not os.path.exists(sidecar):
        return None
    try:
        with open(sidecar, "r", encoding="utf-8") as fh:
            return _parse_gpa_manifest_text(fh.read())
    except Exception:
        return None


def _write_gpa_manifest_sidecar(dat_path, manifest_text):
    """Persist the prep-time manifest next to the .dat for later reload."""
    if not manifest_text or not manifest_text.strip():
        return False
    sidecar = _gpa_manifest_sidecar_path(dat_path)
    with open(sidecar, "w", encoding="utf-8") as fh:
        fh.write(manifest_text)
        if not manifest_text.endswith("\n"):
            fh.write("\n")
    return True


def _coerce_numeric_series(series):
    """Return a numeric version of *series* with non-numeric values as NaN."""
    return pd.to_numeric(series, errors="coerce")


class _GpaResultsSortProxy(QSortFilterProxyModel):
    """Sort GPA result columns numerically whenever numeric sort data exists."""

    def lessThan(self, left, right):
        model = self.sourceModel()
        lv = model.data(left, Qt.UserRole)
        rv = model.data(right, Qt.UserRole)
        try:
            lnum = float(lv)
        except (TypeError, ValueError):
            lnum = None
        try:
            rnum = float(rv)
        except (TypeError, ValueError):
            rnum = None
        if lnum is not None or rnum is not None:
            if lnum is None and rnum is None:
                return super().lessThan(left, right)
            if lnum is None:
                return False
            if rnum is None:
                return True
            return lnum < rnum
        return super().lessThan(left, right)


def _summarize_observed_n(result_tables):
    """Return a compact observed-N summary from GPA result tables."""
    count_cols = ("NOBS", "N", "OBS")
    vals = []
    labels = []
    for key, df in (result_tables or {}).items():
        if df is None or df.empty:
            continue
        for col in count_cols:
            if col not in df.columns:
                continue
            nums = _coerce_numeric_series(df[col]).dropna()
            if nums.empty:
                continue
            labels.append(f"{key}/{col}")
            vals.extend(int(v) for v in nums.tolist())
    if not vals:
        return None
    lo = min(vals)
    hi = max(vals)
    if lo == hi:
        return f"observed N={lo}"
    return f"observed N={lo}–{hi}"


def _split_selected_vars(value):
    """Split a GPA variable string/list into a clean list of variable names."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    return [str(v).strip() for v in parts if str(v).strip()]


def _rtables_to_dfs(raw):
    """Convert Luna GPA raw table output to {name: DataFrame}."""
    out = {}
    for cmd, strata_map in (raw or {}).items():
        for stratum, payload in strata_map.items():
            cols, data = payload
            key = f"{cmd}: {stratum}"
            df = pd.DataFrame(data).T
            df.columns = cols
            out[key] = df
    return out


def _strata_from_label(label, col_set):
    """Return list of strata column names encoded in a Luna strata label.

    Luna separates compound strata with '_X_' (e.g. 'E_X_F') or plain '_'
    (e.g. 'POST_PRE').  We check the full label first, then split on '_X_',
    then split on '_' — keeping only parts that actually appear as columns.
    """
    if label in col_set:
        return [label]
    parts = label.split("_X_")
    if all(p in col_set for p in parts):
        return parts
    parts = label.split("_")
    matched = [p for p in parts if p in col_set]
    return matched if matched else []


def _read_source(path, nrows=6, proj=None):
    """Return list of (label, DataFrame, strata_cols) tuples from a source file.

    strata_cols is a list of known FAC column names (possibly empty) derived
    from Luna metadata embedded in the file (_manifest.tsv in ZIPs, tree in
    pickles, strata() for .db).  None means no metadata was available.

    Supports .txt/.tsv (single table), .zip (one entry per member TSV),
    .pkl/.pickle (single DataFrame), and .db (Luna destrat SQLite via proj).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in (".txt", ".tsv", ".csv"):
        df = _sniff_tsv(path, nrows)
        return [(os.path.basename(path), df, None)] if df is not None else []

    if ext == ".zip":
        # Read _manifest.tsv first if present — maps table key → strata label
        strata_map = {}
        try:
            with zipfile.ZipFile(path, "r") as zf:
                mf_names = [n for n in zf.namelist()
                            if os.path.basename(n).startswith("_manifest")
                            and n.lower().endswith((".tsv", ".txt", ".csv"))]
                if mf_names:
                    with zf.open(mf_names[0]) as fh:
                        mf = pd.read_csv(fh, sep="\t", dtype=str)
                    if {"key", "strata"}.issubset(mf.columns):
                        for _, row in mf.iterrows():
                            strata_map[str(row["key"])] = str(row["strata"])
        except Exception:
            pass

        results = []
        try:
            with zipfile.ZipFile(path, "r") as zf:
                for name in sorted(zf.namelist()):
                    if name.endswith(("/", "\\")):
                        continue
                    if not any(name.lower().endswith(e) for e in (".txt", ".tsv", ".csv")):
                        continue
                    if os.path.basename(name).startswith("_manifest"):
                        continue
                    try:
                        with zf.open(name) as fh:
                            df = pd.read_csv(fh, sep="\t", nrows=nrows, dtype=str)
                        key = os.path.splitext(os.path.basename(name))[0]
                        strata_label = strata_map.get(key)
                        col_set = set(df.columns)
                        strata_cols = (_strata_from_label(strata_label, col_set)
                                       if strata_label else None)
                        results.append((name, df, strata_cols))
                    except Exception:
                        pass
        except Exception:
            pass
        return results

    if ext in (".pkl", ".pickle"):
        try:
            obj = pd.read_pickle(path)
            if isinstance(obj, pd.DataFrame):
                return [(os.path.basename(path), obj.head(nrows).astype(str), None)]
            if isinstance(obj, dict):
                # Build strata map from tree: [(cmd, strata_label), ...]
                tree = obj.get("tree", [])
                tree_map = {}
                for cmd, strata_label in (tree if isinstance(tree, list) else []):
                    key = f"{cmd}_{strata_label}"
                    tree_map[key] = strata_label
                results_dict = obj.get("results", obj)
                results = []
                for k, v in results_dict.items():
                    if not isinstance(v, pd.DataFrame) or v.empty:
                        continue
                    df = v.head(nrows).astype(str)
                    strata_label = tree_map.get(str(k))
                    col_set = set(df.columns)
                    strata_cols = (_strata_from_label(strata_label, col_set)
                                   if strata_label else None)
                    results.append((str(k), df, strata_cols))
                return results
        except Exception:
            pass
        return []

    if ext == ".db" and proj is not None:
        results = []
        try:
            proj.import_db(path)
            tbls = proj.strata()
            if tbls is None or tbls.empty:
                return []
            for row in tbls.itertuples(index=False):
                try:
                    df = proj.table(row.Command, row.Strata)
                    if df is None or df.empty:
                        continue
                    key = f"{row.Command}_{row.Strata}"
                    df_prev = df.head(nrows).astype(str)
                    col_set = set(df_prev.columns)
                    strata_cols = _strata_from_label(row.Strata, col_set)
                    results.append((key, df_prev, strata_cols))
                except Exception:
                    pass
        except Exception:
            pass
        return results

    return []


def _auto_roles(df, strata_cols=None):
    """Return {col: role} using known strata_cols when available, else heuristic.

    strata_cols=None means no metadata (TSV without manifest); use heuristics.
    strata_cols=[]   means metadata present but no FAC columns in this table;
                     still trust it — everything non-ID defaults to VAR.
    """
    has_meta = strata_cols is not None
    known_fac = set(strata_cols) if has_meta else set()
    roles = {}
    for col in df.columns:
        if col.upper() == "ID":
            roles[col] = "ID"
        elif col in known_fac:
            roles[col] = "FAC"
        elif has_meta:
            roles[col] = "VAR"
        else:
            vals = df[col].dropna()
            vals = vals[vals != "."]
            n_uniq = vals.nunique()
            numeric_frac = pd.to_numeric(vals, errors="coerce").notna().mean()
            if numeric_frac >= 0.8:
                roles[col] = "VAR"
            elif n_uniq <= max(10, len(df) * 0.25):
                roles[col] = "FAC"
            else:
                roles[col] = "VAR"
    return roles


# ---------------------------------------------------------------------------
# Reusable variable-picker widget
# ---------------------------------------------------------------------------

class _VarPicker(QWidget):
    """Searchable checklist for picking GPA variables by group and base name."""

    selectionChanged = QtCore.Signal()

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self._manifest = None
        self._pair_rows = pd.DataFrame()
        self._selected_pairs = set()
        self._updating_list = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Header row
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(f"<b>{label}</b>")
        lbl.setStyleSheet(f"color:{FG}; font-size:11px;")
        self._grp_combo = QComboBox()
        self._grp_combo.setFixedWidth(110)
        self._grp_combo.setToolTip("Filter by group")
        self._grp_combo.addItem("(all groups)", None)
        btn_none = QPushButton("✕")
        btn_none.setFixedWidth(22)
        btn_none.setToolTip("Clear all")
        btn_all = QPushButton("✓")
        btn_all.setFixedWidth(22)
        btn_all.setToolTip("Select all visible")
        hdr.addWidget(lbl)
        hdr.addWidget(self._grp_combo, 1)
        hdr.addWidget(btn_all)
        hdr.addWidget(btn_none)

        # Search
        self._search = QLineEdit()
        self._search.setPlaceholderText("search base names…")
        self._search.setClearButtonEnabled(True)

        # List
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.NoSelection)
        self._list.setUniformItemSizes(True)
        self._list.setSpacing(0)
        self._list.setStyleSheet(
            "QListWidget { background:#0d1117; border:1px solid #21262d; font-size:11px; }"
            "QListWidget::item { padding:1px 2px; }"
        )
        self._list.itemChanged.connect(self._on_item_changed)

        layout.addLayout(hdr)
        layout.addWidget(self._search)
        layout.addWidget(self._list, 1)

        btn_all.clicked.connect(self._select_all_visible)
        btn_none.clicked.connect(self._clear_all)
        self._grp_combo.currentIndexChanged.connect(self._refilter)
        self._search.textChanged.connect(self._refilter)

    # ------------------------------------------------------------------

    def populate(self, manifest_df):
        """Populate list from a manifest DataFrame."""
        self._manifest = manifest_df
        self._pair_rows = self._build_pair_rows(manifest_df)
        valid_pairs = set(zip(self._pair_rows["GRP"], self._pair_rows["BASE"]))
        self._selected_pairs = {p for p in self._selected_pairs if p in valid_pairs}
        self._grp_combo.blockSignals(True)
        self._grp_combo.clear()
        self._grp_combo.addItem("(all groups)", None)
        if not self._pair_rows.empty:
            groups = [g for g in self._pair_rows["GRP"].unique() if g and g != "."]
            for g in sorted(groups):
                self._grp_combo.addItem(g, g)
        self._grp_combo.blockSignals(False)
        self._refilter()

    def _build_pair_rows(self, manifest_df):
        if manifest_df is None or manifest_df.empty or "VAR" not in manifest_df.columns:
            return pd.DataFrame(columns=["GRP", "BASE", "LONGS", "N_LONG", "NI_MIN", "NI_MAX"])
        df = manifest_df.copy()
        df = df[df["VAR"] != "ID"].copy()
        if "BASE" not in df.columns:
            df["BASE"] = df["VAR"]
        if "GRP" not in df.columns:
            df["GRP"] = "."
        df["VAR"] = df["VAR"].astype(str)
        df["BASE"] = df["BASE"].astype(str)
        df["GRP"] = df["GRP"].astype(str)
        if "NI" in df.columns:
            df["_NI_NUM"] = pd.to_numeric(df["NI"], errors="coerce")
        else:
            df["_NI_NUM"] = np.nan
        rows = []
        for (grp, base), sub in df.groupby(["GRP", "BASE"], dropna=False):
            longs = sorted({str(v) for v in sub["VAR"].tolist()})
            ni_vals = sub["_NI_NUM"].dropna().tolist()
            rows.append({
                "GRP": grp,
                "BASE": base,
                "LONGS": longs,
                "N_LONG": len(longs),
                "NI_MIN": int(min(ni_vals)) if ni_vals else None,
                "NI_MAX": int(max(ni_vals)) if ni_vals else None,
            })
        return pd.DataFrame(rows)

    def _visible_rows(self):
        if self._pair_rows is None or self._pair_rows.empty:
            return []
        grp_filter = self._grp_combo.currentData()
        text_filter = self._search.text().strip().lower()
        rows = []
        if grp_filter:
            sub = self._pair_rows[self._pair_rows["GRP"] == grp_filter].sort_values("BASE")
            for row in sub.itertuples(index=False):
                rows.append({
                    "base": row.BASE,
                    "pairs": [(row.GRP, row.BASE)],
                    "n_long": int(row.N_LONG),
                    "n_groups": 1,
                    "ni_min": row.NI_MIN,
                    "ni_max": row.NI_MAX,
                    "tip": f"group={row.GRP}   base={row.BASE}   long-vars={row.N_LONG}",
                })
        else:
            for base, sub in self._pair_rows.groupby("BASE", dropna=False):
                pairs = [(r.GRP, r.BASE) for r in sub.itertuples(index=False)]
                n_long = int(sub["N_LONG"].sum())
                ni_vals = [v for v in sub["NI_MIN"].tolist() if v is not None]
                ni_vals += [v for v in sub["NI_MAX"].tolist() if v is not None]
                rows.append({
                    "base": base,
                    "pairs": pairs,
                    "n_long": n_long,
                    "n_groups": len(sub),
                    "ni_min": min(ni_vals) if ni_vals else None,
                    "ni_max": max(ni_vals) if ni_vals else None,
                    "tip": (
                        f"base={base}   groups={len(sub)}   long-vars={n_long}\n"
                        + ", ".join(sorted(sub["GRP"].astype(str).tolist())[:8])
                    ),
                })
            rows.sort(key=lambda x: str(x["base"]).lower())
        if text_filter:
            rows = [r for r in rows if text_filter in str(r["base"]).lower()]
        return rows

    @staticmethod
    def _format_n(ni_min, ni_max):
        if ni_min is None or ni_max is None:
            return ""
        return f"N={ni_min}" if ni_min == ni_max else f"N={ni_min}-{ni_max}"

    def _refilter(self, *_):
        self._updating_list = True
        try:
            self._list.clear()
            for row in self._visible_rows():
                n_txt = self._format_n(row["ni_min"], row["ni_max"])
                extra = f"{row['n_long']} long var" + ("" if row["n_long"] == 1 else "s")
                if row["n_groups"] > 1:
                    extra += f" / {row['n_groups']} groups"
                label = str(row["base"])
                if n_txt:
                    label += f"  [{n_txt}]"
                label += f"  ·  {extra}"
                item = QListWidgetItem(label)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                pairs = row["pairs"]
                selected_n = sum(1 for p in pairs if p in self._selected_pairs)
                if selected_n == 0:
                    state = Qt.Unchecked
                elif selected_n == len(pairs):
                    state = Qt.Checked
                else:
                    state = Qt.PartiallyChecked
                item.setCheckState(state)
                item.setData(Qt.UserRole, pairs)
                item.setData(Qt.UserRole + 1, row["base"])
                item.setToolTip(row["tip"])
                self._list.addItem(item)
        finally:
            self._updating_list = False

    def _on_item_changed(self, item):
        if self._updating_list:
            return
        pairs = item.data(Qt.UserRole) or []
        if item.checkState() == Qt.Checked:
            for pair in pairs:
                self._selected_pairs.add(tuple(pair))
        elif item.checkState() == Qt.Unchecked:
            for pair in pairs:
                self._selected_pairs.discard(tuple(pair))
        self.selectionChanged.emit()

    def _select_all_visible(self):
        self._updating_list = True
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.Checked)
        self._updating_list = False
        for row in self._visible_rows():
            for pair in row["pairs"]:
                self._selected_pairs.add(tuple(pair))
        self.selectionChanged.emit()

    def _clear_all(self):
        self._updating_list = True
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.Unchecked)
        self._updating_list = False
        visible_pairs = set()
        for row in self._visible_rows():
            for pair in row["pairs"]:
                visible_pairs.add(tuple(pair))
        self._selected_pairs = {p for p in self._selected_pairs if p not in visible_pairs}
        self.selectionChanged.emit()

    def selected(self):
        """Return selected base names."""
        return sorted({base for _grp, base in self._selected_pairs})

    def selected_str(self):
        """Return comma-joined string of checked variables, or ''."""
        return ",".join(self.selected())

    def selected_pairs(self):
        return sorted(self._selected_pairs)

    def selected_long_names(self):
        if self._pair_rows is None or self._pair_rows.empty:
            return []
        wanted = set(self._selected_pairs)
        longs = []
        for row in self._pair_rows.itertuples(index=False):
            if (row.GRP, row.BASE) in wanted:
                longs.extend(list(row.LONGS))
        return sorted(dict.fromkeys(longs))

    def set_selected(self, names):
        """Check entries whose base names are in *names* across all groups."""
        wanted = set(names)
        self._selected_pairs = set()
        for row in self._pair_rows.itertuples(index=False):
            if row.BASE in wanted:
                self._selected_pairs.add((row.GRP, row.BASE))
        self._refilter()
        self.selectionChanged.emit()


# ---------------------------------------------------------------------------
# Main GPA tab
# ---------------------------------------------------------------------------

class GPATab(_ExplorerTab):
    """GPA / Association Explorer tab."""

    _sig_ok         = QtCore.Signal(object)
    _sig_err        = QtCore.Signal(str)
    _sig_progress   = QtCore.Signal(str)
    _sig_scatter_ok = QtCore.Signal(object)   # (ids, xs, ys, xvar, yvar)
    _sig_scatter_err= QtCore.Signal(str)

    def __init__(self, ctrl, parent=None):
        super().__init__(ctrl, parent)

        # ---- state -------------------------------------------------------
        self._manifest_df: pd.DataFrame | None = None
        self._results_dfs: dict = {}
        # {path: {col: {role, value, group}}}
        self._col_assignments: dict = {}
        # entries currently displayed in the column table
        self._col_table_path: str | None = None
        # rendered sub-table (X_Y, VAR, or X depending on GPA mode)
        self._active_result_df: pd.DataFrame | None = None
        # manifest and results proxy models
        self._manifest_proxy: QSortFilterProxyModel | None = None
        self._results_proxy: QSortFilterProxyModel | None = None
        # scatter: True while showing a per-row scatter instead of volcano
        self._scatter_mode = False
        # generation counter — lets us discard stale scatter callbacks
        self._scatter_gen  = 0
        # last scatter vars (for toggle redraw)
        self._scatter_xvar: str = ""
        self._scatter_yvar: str = ""
        # Z variables from the most recent gpa_run (for partial scatter)
        self._last_gpa_z: list = []
        self._last_gpa_request: dict = {}

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(250)
        self._render_timer.timeout.connect(self._render_result)

        self._sig_ok.connect(self._on_ok,               Qt.QueuedConnection)
        self._sig_err.connect(self._on_err,              Qt.QueuedConnection)
        self._sig_progress.connect(self._on_progress,   Qt.QueuedConnection)
        self._sig_scatter_ok.connect(self._on_scatter_ok,  Qt.QueuedConnection)
        self._sig_scatter_err.connect(self._on_scatter_err, Qt.QueuedConnection)

        self._build_widget()

    # ======================================================================
    # Widget construction
    # ======================================================================

    def _build_widget(self):
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(0)

        # ---- outer horizontal splitter (left controls | right panels) ----
        outer = QSplitter(Qt.Horizontal)
        outer.setHandleWidth(5)
        root_layout.addWidget(outer)

        # ---- left: control panel (scrollable) ----------------------------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(200)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_inner = QWidget()
        left_layout = QVBoxLayout(left_inner)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_scroll.setWidget(left_inner)

        self._inner_tabs = QTabWidget()
        self._inner_tabs.setTabPosition(QTabWidget.North)
        self._inner_tabs.setDocumentMode(True)
        left_layout.addWidget(self._inner_tabs)

        build_w    = self._build_build_tab()
        analyze_w  = self._build_analyze_tab()
        self._inner_tabs.addTab(build_w,   "Build")
        self._inner_tabs.addTab(analyze_w, "Analyze")

        # ---- right: stacked panel — Build→manifest, Analyze→results ----
        right_stack = QStackedWidget()
        manifest_frame = self._build_manifest_panel()
        results_frame  = self._build_results_panel()
        right_stack.addWidget(manifest_frame)   # index 0
        right_stack.addWidget(results_frame)    # index 1
        right_stack.setCurrentIndex(0)
        self._inner_tabs.currentChanged.connect(
            lambda idx: right_stack.setCurrentIndex(0 if idx == 0 else 1))

        outer.addWidget(left_scroll)
        outer.addWidget(right_stack)
        outer.setSizes([320, 900])
        outer.setCollapsible(0, False)
        outer.setCollapsible(1, False)

        self._root = root

    # ------------------------------------------------------------------
    # Build tab
    # ------------------------------------------------------------------

    def _build_build_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        # Working directory
        wd_row = QHBoxLayout()
        wd_row.setSpacing(4)
        wd_row.addWidget(QLabel("Working dir:"))
        self._wd_edit = QLineEdit()
        self._wd_edit.setPlaceholderText("(directory for extracted files and .dat output)")
        btn_wd = QPushButton("…"); btn_wd.setFixedWidth(26)
        btn_wd.clicked.connect(self._browse_wd)
        wd_row.addWidget(self._wd_edit, 1)
        wd_row.addWidget(btn_wd)
        lay.addLayout(wd_row)

        # Source files
        src_hdr = QHBoxLayout()
        src_lbl = QLabel("<b>Source files</b>")
        src_lbl.setStyleSheet(f"color:{FG};")
        src_hdr.addWidget(src_lbl)
        src_hdr.addStretch(1)
        btn_add = QPushButton("+ Add"); btn_add.setFixedWidth(58)
        btn_rm  = QPushButton("− Remove"); btn_rm.setFixedWidth(72)
        src_hdr.addWidget(btn_add); src_hdr.addWidget(btn_rm)
        lay.addLayout(src_hdr)

        self._files_list = QListWidget()
        self._files_list.setFixedHeight(110)
        self._files_list.setToolTip("TSV, ZIP, PKL, or Luna .db files")
        lay.addWidget(self._files_list)

        btn_add.clicked.connect(self._add_source_file)
        btn_rm.clicked.connect(self._remove_source_file)
        self._files_list.currentRowChanged.connect(self._on_file_selected)

        # Column assignment (hidden until file selected)
        self._col_frame = QFrame()
        self._col_frame.setFrameShape(QFrame.StyledPanel)
        col_lay = QVBoxLayout(self._col_frame)
        col_lay.setContentsMargins(4, 4, 4, 4)
        col_lay.setSpacing(4)

        col_hdr = QHBoxLayout()
        self._col_file_lbl = QLabel("Columns")
        self._col_file_lbl.setStyleSheet(f"color:{FG}; font-size:11px;")
        col_hdr.addWidget(self._col_file_lbl, 1)
        col_hdr.addWidget(QLabel("Group:"))
        self._col_group_edit = QLineEdit()
        self._col_group_edit.setFixedWidth(80)
        self._col_group_edit.setPlaceholderText("grp1")
        self._col_group_edit.textChanged.connect(self._on_group_name_changed)
        col_hdr.addWidget(self._col_group_edit)
        col_lay.addLayout(col_hdr)

        self._col_table = QTableWidget(0, 5)
        self._col_table.setHorizontalHeaderLabels(
            ["Column", "Role", "Value", "Group", "Preview"])
        self._col_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents)
        self._col_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Fixed)
        self._col_table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.Stretch)
        self._col_table.setColumnWidth(1, 72)
        self._col_table.setColumnWidth(2, 60)
        self._col_table.setColumnWidth(3, 60)
        self._col_table.setMinimumHeight(130)
        self._col_table.setMaximumHeight(200)
        self._col_table.verticalHeader().setVisible(False)
        self._col_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._col_table.setEditTriggers(QAbstractItemView.DoubleClicked)
        self._col_table.itemChanged.connect(self._on_col_table_changed)
        col_lay.addWidget(self._col_table)

        self._col_frame.setVisible(False)
        lay.addWidget(self._col_frame)

        # JSON / Specs  -------------------------------------------------
        json_toggle = QPushButton("▶  Specs JSON")
        json_toggle.setCheckable(True)
        json_toggle.setStyleSheet("text-align:left; padding-left:4px;")
        lay.addWidget(json_toggle)

        self._json_frame = QFrame()
        self._json_frame.setFrameShape(QFrame.StyledPanel)
        json_lay = QVBoxLayout(self._json_frame)
        json_lay.setContentsMargins(4, 4, 4, 4)
        json_lay.setSpacing(4)

        json_btns = QHBoxLayout()
        btn_sync_to   = QPushButton("UI → JSON"); btn_sync_to.setFixedWidth(84)
        btn_sync_from = QPushButton("JSON → UI"); btn_sync_from.setFixedWidth(84)
        btn_save_json = QPushButton("Save…");     btn_save_json.setFixedWidth(60)
        btn_load_json = QPushButton("Load…");     btn_load_json.setFixedWidth(60)
        json_btns.addWidget(btn_sync_to); json_btns.addWidget(btn_sync_from)
        json_btns.addStretch(1)
        json_btns.addWidget(btn_save_json); json_btns.addWidget(btn_load_json)
        json_lay.addLayout(json_btns)

        self._json_edit = QPlainTextEdit()
        self._json_edit.setMinimumHeight(120)
        self._json_edit.setMaximumHeight(200)
        self._json_edit.setFont(QtGui.QFont("Courier New", 10))
        self._json_edit.setPlaceholderText('{"inputs": [...]}')
        json_lay.addWidget(self._json_edit)

        self._json_frame.setVisible(False)
        lay.addWidget(self._json_frame)

        json_toggle.toggled.connect(
            lambda on: (self._json_frame.setVisible(on),
                        json_toggle.setText(("▼" if on else "▶") + "  Specs JSON")))
        btn_sync_to.clicked.connect(self._sync_ui_to_json)
        btn_sync_from.clicked.connect(self._sync_json_to_ui)
        btn_save_json.clicked.connect(self._save_json)
        btn_load_json.clicked.connect(self._load_json)

        # Output .dat + build
        dat_row = QHBoxLayout()
        dat_row.setSpacing(4)
        dat_row.addWidget(QLabel(".dat output:"))
        self._build_dat_edit = QLineEdit()
        self._build_dat_edit.setPlaceholderText("out.dat")
        btn_dat = QPushButton("…"); btn_dat.setFixedWidth(26)
        btn_dat.clicked.connect(lambda: self._browse_dat_save(self._build_dat_edit))
        dat_row.addWidget(self._build_dat_edit, 1); dat_row.addWidget(btn_dat)
        lay.addLayout(dat_row)

        build_btns = QHBoxLayout()
        self._build_btn = QPushButton("Build Dataset")
        self._build_btn.setStyleSheet(
            "QPushButton { background:#166534; color:#fff; padding:4px 10px; border-radius:4px; }"
            "QPushButton:hover { background:#15803d; }"
        )
        self._keep_tsv_chk = QCheckBox("Keep temp TSVs")
        self._keep_tsv_chk.setToolTip(
            "When checked, TSV files extracted from .db/.pkl/.zip sources are\n"
            "kept in the working directory after the build instead of deleted.")
        self._build_status = QLabel("")
        self._build_status.setStyleSheet(f"color:#888; font-size:11px;")
        self._build_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        build_btns.addWidget(self._build_btn)
        build_btns.addWidget(self._keep_tsv_chk)
        build_btns.addWidget(self._build_status, 1)
        lay.addLayout(build_btns)

        self._build_btn.clicked.connect(self._run_prep)

        lay.addStretch(1)
        return w

    # ------------------------------------------------------------------
    # Analyze tab
    # ------------------------------------------------------------------

    def _build_analyze_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        # .dat file
        dat_row = QHBoxLayout(); dat_row.setSpacing(4)
        self._dat_edit = QLineEdit()
        self._dat_edit.setPlaceholderText("path/to/out.dat")
        btn_dat_open = QPushButton("…"); btn_dat_open.setFixedWidth(26)
        btn_dat_open.clicked.connect(lambda: self._browse_dat_open(self._dat_edit))
        self._load_manifest_btn = QPushButton("Load .dat")
        self._load_manifest_btn.setFixedWidth(80)
        self._load_manifest_btn.clicked.connect(self._run_load_manifest)
        dat_row.addWidget(QLabel(".dat:")); dat_row.addWidget(self._dat_edit, 1)
        dat_row.addWidget(btn_dat_open); dat_row.addWidget(self._load_manifest_btn)
        lay.addLayout(dat_row)

        self._desc_lbl = QLabel("")
        self._desc_lbl.setStyleSheet(f"color:#888; font-size:10px;")
        lay.addWidget(self._desc_lbl)

        # Variable pickers
        self._picker_x = _VarPicker("X  predictors")
        self._picker_y = _VarPicker("Y  outcomes")
        self._picker_z = _VarPicker("Z  covariates")
        for p in (self._picker_x, self._picker_y, self._picker_z):
            p.setMinimumHeight(140)
            p.setMaximumHeight(160)
            lay.addWidget(p)
        self._picker_x.selectionChanged.connect(self._update_selection_desc)
        self._picker_y.selectionChanged.connect(self._update_selection_desc)
        self._picker_z.selectionChanged.connect(self._update_selection_desc)

        # Options (collapsible)
        opts_toggle = QPushButton("▶  Options")
        opts_toggle.setCheckable(True)
        opts_toggle.setStyleSheet("text-align:left; padding-left:4px;")
        lay.addWidget(opts_toggle)

        self._opts_frame = QFrame()
        self._opts_frame.setFrameShape(QFrame.StyledPanel)
        opts_lay = QVBoxLayout(self._opts_frame)
        opts_lay.setContentsMargins(6, 4, 6, 4)
        opts_lay.setSpacing(4)

        def _row(label, widget):
            r = QHBoxLayout()
            lbl = QLabel(label); lbl.setFixedWidth(100)
            lbl.setStyleSheet(f"color:{FG}; font-size:11px;")
            r.addWidget(lbl); r.addWidget(widget, 1)
            return r

        # Mode
        self._mode_combo = QComboBox()
        for key, lbl in [
            ("assoc", "Association  (X → Y)"),
            ("stats", "Descriptive stats  (X)"),
            ("comp",  "Comparison  (X by FAC)"),
        ]:
            self._mode_combo.addItem(lbl, key)
        opts_lay.addLayout(_row("Mode:", self._mode_combo))

        # FAC filter — required for comp mode; always available
        fac_row = QHBoxLayout()
        fac_row.addWidget(QLabel("facs:"))
        self._facs_edit = QLineEdit(); self._facs_edit.setFixedWidth(90)
        self._facs_edit.setPlaceholderText("FAC1,FAC2")
        self._facs_edit.setToolTip(
            "Factor variable(s) to include (required for Comparison mode)")
        fac_row.addWidget(self._facs_edit)
        fac_row.addWidget(QLabel("xfacs:"))
        self._xfacs_edit = QLineEdit(); self._xfacs_edit.setFixedWidth(90)
        self._xfacs_edit.setPlaceholderText("exclude…")
        fac_row.addWidget(self._xfacs_edit)
        fac_row.addStretch(1)
        self._fac_row_widget = QWidget(); self._fac_row_widget.setLayout(fac_row)
        opts_lay.addWidget(self._fac_row_widget)

        def _update_fac_highlight():
            is_comp = self._mode_combo.currentData() == "comp"
            self._facs_edit.setStyleSheet(
                "border: 1px solid #f9844a;" if is_comp else "")
        self._mode_combo.currentIndexChanged.connect(_update_fac_highlight)

        # nreps
        self._nreps_spin = QSpinBox()
        self._nreps_spin.setRange(0, 100000); self._nreps_spin.setValue(0)
        self._nreps_spin.setSpecialValueText("0 (asymptotic)")
        opts_lay.addLayout(_row("Permutations:", self._nreps_spin))

        # Corrections
        corr_row = QHBoxLayout()
        self._chk_fdr    = QCheckBox("FDR"); self._chk_fdr.setChecked(True)
        self._chk_bonf   = QCheckBox("Bonf")
        self._chk_holm   = QCheckBox("Holm")
        self._chk_fdr_by = QCheckBox("FDR-BY")
        self._chk_adj_all = QCheckBox("adj-all-X")
        for c in (self._chk_fdr, self._chk_bonf, self._chk_holm,
                  self._chk_fdr_by, self._chk_adj_all):
            corr_row.addWidget(c)
        corr_row.addStretch(1)
        opts_lay.addLayout(_row("Corrections:", QWidget()))
        opts_lay.addLayout(corr_row)

        # P thresholds
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("p ≤")); self._p_spin = QDoubleSpinBox()
        self._p_spin.setRange(0, 1); self._p_spin.setValue(1.0)
        self._p_spin.setSingleStep(0.01); self._p_spin.setDecimals(3)
        self._p_spin.setFixedWidth(70)
        thresh_row.addWidget(self._p_spin)
        thresh_row.addWidget(QLabel("padj ≤")); self._padj_spin = QDoubleSpinBox()
        self._padj_spin.setRange(0, 1); self._padj_spin.setValue(1.0)
        self._padj_spin.setSingleStep(0.01); self._padj_spin.setDecimals(3)
        self._padj_spin.setFixedWidth(70)
        thresh_row.addWidget(self._padj_spin)
        thresh_row.addStretch(1)
        opts_lay.addLayout(thresh_row)

        # Missing data / QC
        qc_row = QHBoxLayout()
        qc_row.addWidget(QLabel("n-prop:"))
        self._nprop_edit = QLineEdit(); self._nprop_edit.setFixedWidth(50)
        self._nprop_edit.setPlaceholderText("0.05")
        qc_row.addWidget(self._nprop_edit)
        qc_row.addWidget(QLabel("n-req:"))
        self._nreq_edit = QLineEdit(); self._nreq_edit.setFixedWidth(45)
        self._nreq_edit.setPlaceholderText("off")
        qc_row.addWidget(self._nreq_edit)
        qc_row.addStretch(1)
        opts_lay.addLayout(qc_row)

        qc_row2 = QHBoxLayout()
        qc_row2.addWidget(QLabel("knn:"))
        self._knn_edit = QLineEdit(); self._knn_edit.setFixedWidth(40)
        self._knn_edit.setPlaceholderText("off")
        qc_row2.addWidget(self._knn_edit)
        qc_row2.addWidget(QLabel("winsor:"))
        self._winsor_edit = QLineEdit(); self._winsor_edit.setFixedWidth(45)
        self._winsor_edit.setPlaceholderText("off")
        qc_row2.addWidget(self._winsor_edit)
        qc_row2.addStretch(1)
        opts_lay.addLayout(qc_row2)

        # Subset / IDs
        self._subset_edit = QLineEdit(); self._subset_edit.setPlaceholderText("e.g. +MALE")
        opts_lay.addLayout(_row("Subset:", self._subset_edit))
        self._incids_edit = QLineEdit(); self._incids_edit.setPlaceholderText("id1,id2,…")
        opts_lay.addLayout(_row("inc-ids:", self._incids_edit))
        self._exids_edit  = QLineEdit(); self._exids_edit.setPlaceholderText("id1,id2,…")
        opts_lay.addLayout(_row("ex-ids:", self._exids_edit))

        self._opts_frame.setVisible(False)
        lay.addWidget(self._opts_frame)
        opts_toggle.toggled.connect(
            lambda on: (self._opts_frame.setVisible(on),
                        opts_toggle.setText(("▼" if on else "▶") + "  Options")))

        # Run
        run_row = QHBoxLayout()
        self._run_btn = QPushButton("Run GPA")
        self._run_btn.setStyleSheet(
            "QPushButton { background:#1e3a5f; color:#fff; padding:4px 12px; border-radius:4px; }"
            "QPushButton:hover { background:#1d4ed8; }"
        )
        self._analyze_status = QLabel("")
        self._analyze_status.setStyleSheet(f"color:#888; font-size:11px;")
        self._analyze_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._analyze_status, 1)
        lay.addLayout(run_row)

        self._run_btn.clicked.connect(self._run_gpa)

        lay.addStretch(1)
        return w

    # ------------------------------------------------------------------
    # Manifest panel (right top)
    # ------------------------------------------------------------------

    def _build_manifest_panel(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.NoFrame)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(3)

        hdr = QHBoxLayout()
        lbl = QLabel("Manifest")
        lbl.setStyleSheet(f"color:{FG}; font-size:11px; font-weight:600;")
        self._manifest_filter = QLineEdit()
        self._manifest_filter.setPlaceholderText("filter…")
        self._manifest_filter.setClearButtonEnabled(True)
        self._manifest_filter.setFixedWidth(150)
        btn_export_m = QPushButton("Export…"); btn_export_m.setFixedWidth(70)
        self._manifest_desc = QLabel("")
        self._manifest_desc.setStyleSheet(f"color:#888; font-size:10px;")
        hdr.addWidget(lbl)
        hdr.addWidget(self._manifest_desc, 1)
        hdr.addWidget(self._manifest_filter)
        hdr.addWidget(btn_export_m)
        lay.addLayout(hdr)

        self._manifest_view = QTableView()
        self._manifest_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._manifest_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._manifest_view.setSortingEnabled(True)
        self._manifest_view.horizontalHeader().setStretchLastSection(True)
        self._manifest_view.verticalHeader().setVisible(False)
        self._manifest_view.setAlternatingRowColors(True)
        lay.addWidget(self._manifest_view)

        btn_export_m.clicked.connect(
            lambda: save_table_as_tsv(self._manifest_view, self))

        return frame

    # ------------------------------------------------------------------
    # Results panel (right bottom)
    # ------------------------------------------------------------------

    def _build_results_panel(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.NoFrame)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(3)

        hdr = QHBoxLayout()
        lbl = QLabel("Results")
        lbl.setStyleSheet(f"color:{FG}; font-size:11px; font-weight:600;")
        self._results_filter = QLineEdit()
        self._results_filter.setPlaceholderText("filter…")
        self._results_filter.setClearButtonEnabled(True)
        self._results_filter.setFixedWidth(150)
        self._results_table_combo = QComboBox()
        self._results_table_combo.setMinimumWidth(140)
        btn_export_r = QPushButton("Export…"); btn_export_r.setFixedWidth(70)
        hdr.addWidget(lbl)
        hdr.addWidget(self._results_table_combo, 1)
        hdr.addWidget(self._results_filter)
        hdr.addWidget(btn_export_r)
        lay.addLayout(hdr)

        # Splitter: results table above, canvas below
        rsplit = QSplitter(Qt.Vertical)
        rsplit.setHandleWidth(4)

        self._results_view = QTableView()
        self._results_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._results_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._results_view.setSortingEnabled(True)
        self._results_view.horizontalHeader().setStretchLastSection(True)
        self._results_view.verticalHeader().setVisible(False)
        self._results_view.setAlternatingRowColors(True)

        # Canvas frame — raw/partial toggle bar sits above the matplotlib canvas
        canvas_outer = QFrame()
        canvas_outer.setFrameShape(QFrame.NoFrame)
        canvas_outer_lay = QVBoxLayout(canvas_outer)
        canvas_outer_lay.setContentsMargins(0, 0, 0, 0)
        canvas_outer_lay.setSpacing(0)

        # Toggle row (hidden until a scatter row is selected)
        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(4, 2, 4, 0)
        self._scatter_raw_btn     = QPushButton("Raw")
        self._scatter_partial_btn = QPushButton("Partial")
        for btn in (self._scatter_raw_btn, self._scatter_partial_btn):
            btn.setCheckable(True)
            btn.setFixedWidth(64)
            btn.setStyleSheet(
                "QPushButton { padding:2px 6px; font-size:10px; border:1px solid #333; }"
                "QPushButton:checked { background:#1e3a5f; color:#fff; border-color:#4cc9f0; }")
        self._scatter_raw_btn.setChecked(True)
        toggle_row.addWidget(QLabel("Scatter:"))
        toggle_row.addWidget(self._scatter_raw_btn)
        toggle_row.addWidget(self._scatter_partial_btn)
        toggle_row.addStretch(1)
        self._scatter_toggle_widget = QWidget()
        self._scatter_toggle_widget.setLayout(toggle_row)
        self._scatter_toggle_widget.setVisible(False)
        canvas_outer_lay.addWidget(self._scatter_toggle_widget)

        canvas_host = QFrame()
        canvas_host.setFrameShape(QFrame.NoFrame)
        canvas_host.setLayout(QVBoxLayout())
        canvas_host.layout().setContentsMargins(0, 0, 0, 0)
        canvas_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas_host = canvas_host
        canvas_outer_lay.addWidget(canvas_host, 1)

        rsplit.addWidget(self._results_view)
        rsplit.addWidget(canvas_outer)
        rsplit.setSizes([200, 300])

        lay.addWidget(rsplit, 1)

        btn_export_r.clicked.connect(
            lambda: save_table_as_tsv(self._results_view, self))
        self._results_table_combo.currentIndexChanged.connect(
            self._on_results_table_changed)
        self._results_filter.textChanged.connect(self._apply_results_filter)
        self._results_view.clicked.connect(self._on_result_row_clicked)
        self._scatter_raw_btn.clicked.connect(
            lambda: self._on_scatter_toggle(partial=False))
        self._scatter_partial_btn.clicked.connect(
            lambda: self._on_scatter_toggle(partial=True))

        return frame

    # ======================================================================
    # Build tab — source files
    # ======================================================================

    def _browse_wd(self):
        from ..file_dialogs import existing_directory
        d = existing_directory(self._root, "Working Directory")
        if d:
            self._wd_edit.setText(d)

    def _add_source_file(self):
        fns, _ = QFileDialog.getOpenFileNames(
            self._root, "Add Source Files", "",
            "Supported files (*.txt *.tsv *.csv *.zip *.pkl *.pickle *.db);;"
            "TSV / text (*.txt *.tsv *.csv);;"
            "ZIP archive (*.zip);;"
            "Pickle (*.pkl *.pickle);;"
            "Luna DB (*.db);;"
            "All files (*)",
            options=QFileDialog.DontUseNativeDialog)
        if not fns:
            return

        proj = getattr(getattr(self, "ctrl", None), "proj", None)
        failed = []
        for fn in fns:
            if not self._add_one_source_file(fn, proj):
                failed.append(fn)
        if failed:
            QtWidgets.QMessageBox.warning(
                self._root, "GPA",
                "Could not read columns from:\n" + "\n".join(failed))

        self._files_list.setCurrentRow(self._files_list.count() - 1)
        self._sync_ui_to_json()

    def _add_one_source_file(self, fn, proj):
        """Add a single source file to the list; returns False if unreadable."""
        # Set working dir from first file added
        if not self._wd_edit.text().strip():
            self._wd_edit.setText(os.path.dirname(fn))

        ext = os.path.splitext(fn)[1].lower()
        type_labels = {
            ".zip": "[ZIP]", ".pkl": "[PKL]", ".pickle": "[PKL]",
            ".db": "[DB]",
        }
        prefix = type_labels.get(ext, "[TSV]")

        entries = _read_source(fn, nrows=6, proj=proj)
        if not entries:
            return False

        if ext == ".zip":
            for name, df, strata_cols in entries:
                item = QListWidgetItem(f"[ZIP] {name}")
                item.setData(Qt.UserRole, fn)
                item.setData(Qt.UserRole + 1, name)
                item.setToolTip(fn)
                self._files_list.addItem(item)
                key = fn + "::" + name
                self._col_assignments[key] = {
                    "_roles": _auto_roles(df, strata_cols),
                    "_group": os.path.splitext(name)[0],
                    "_df_preview": df,
                }
        elif len(entries) == 1:
            _, df, strata_cols = entries[0]
            item = QListWidgetItem(f"{prefix} {os.path.basename(fn)}")
            item.setData(Qt.UserRole, fn)
            item.setData(Qt.UserRole + 1, None)
            item.setToolTip(fn)
            self._files_list.addItem(item)
            self._col_assignments[fn] = {
                "_roles": _auto_roles(df, strata_cols),
                "_group": os.path.splitext(os.path.basename(fn))[0],
                "_df_preview": df,
            }
        else:
            for name, df, strata_cols in entries:
                item = QListWidgetItem(f"[DB] {name}")
                item.setData(Qt.UserRole, fn)
                item.setData(Qt.UserRole + 1, name)
                item.setToolTip(fn)
                self._files_list.addItem(item)
                key = fn + "::" + name
                self._col_assignments[key] = {
                    "_roles": _auto_roles(df, strata_cols),
                    "_group": name,
                    "_df_preview": df,
                }
        return True

    def _remove_source_file(self):
        row = self._files_list.currentRow()
        if row < 0:
            return
        item = self._files_list.item(row)
        key = self._item_key(item)
        self._col_assignments.pop(key, None)
        self._files_list.takeItem(row)
        self._col_frame.setVisible(False)
        self._col_table_path = None
        self._sync_ui_to_json()

    def _item_key(self, item):
        path = item.data(Qt.UserRole)
        member = item.data(Qt.UserRole + 1)
        return (path + "::" + member) if member else path

    def _on_file_selected(self, row):
        if row < 0:
            self._col_frame.setVisible(False)
            return
        item = self._files_list.item(row)
        key = self._item_key(item)
        asgn = self._col_assignments.get(key)
        if asgn is None:
            return
        self._col_table_path = key
        df = asgn.get("_df_preview")
        group = asgn.get("_group", "grp")
        short = item.text()
        self._col_file_lbl.setText(f"Columns: {short}")
        self._col_group_edit.blockSignals(True)
        self._col_group_edit.setText(group)
        self._col_group_edit.blockSignals(False)
        self._populate_col_table(df, asgn["_roles"])
        self._col_frame.setVisible(True)

    def _populate_col_table(self, df, roles):
        self._col_table.blockSignals(True)
        self._col_table.setRowCount(0)
        if df is None:
            self._col_table.blockSignals(False)
            return
        for col in df.columns:
            row = self._col_table.rowCount()
            self._col_table.insertRow(row)
            # Column name (non-editable)
            name_item = QTableWidgetItem(col)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self._col_table.setItem(row, 0, name_item)
            # Role combo
            combo = QComboBox()
            for r in _ROLES:
                combo.addItem(r, r)
            role = roles.get(col, "exclude")
            combo.setCurrentIndex(_ROLES.index(role) if role in _ROLES else 0)
            combo.currentIndexChanged.connect(
                lambda _, r=row, c=col: self._on_role_changed(r, c))
            self._col_table.setCellWidget(row, 1, combo)
            # Value (for fixed role)
            val_item = QTableWidgetItem("")
            self._col_table.setItem(row, 2, val_item)
            # Group (inherits from file group)
            grp_item = QTableWidgetItem("")  # empty = inherit
            self._col_table.setItem(row, 3, grp_item)
            # Preview
            preview = ", ".join(df[col].dropna().astype(str).head(3).tolist())
            prev_item = QTableWidgetItem(preview)
            prev_item.setFlags(prev_item.flags() & ~Qt.ItemIsEditable)
            prev_item.setForeground(QtGui.QColor("#666688"))
            self._col_table.setItem(row, 4, prev_item)
            # Color the name by role
            self._color_row(row, role)
        self._col_table.blockSignals(False)

    def _color_row(self, row, role):
        col_hex = _ROLE_COLORS.get(role, "#555555")
        name_item = self._col_table.item(row, 0)
        if name_item:
            name_item.setForeground(QtGui.QColor(col_hex))

    def _on_role_changed(self, row, col):
        combo = self._col_table.cellWidget(row, 1)
        if combo is None:
            return
        role = combo.currentData()
        self._color_row(row, role)
        # Save back to assignments
        self._save_col_assignments()
        self._sync_ui_to_json()

    def _on_col_table_changed(self, item):
        self._save_col_assignments()
        self._sync_ui_to_json()

    def _on_group_name_changed(self, text):
        if self._col_table_path is None:
            return
        asgn = self._col_assignments.get(self._col_table_path, {})
        asgn["_group"] = text
        self._col_assignments[self._col_table_path] = asgn
        self._sync_ui_to_json()

    def _save_col_assignments(self):
        if self._col_table_path is None:
            return
        asgn = self._col_assignments.get(self._col_table_path, {})
        roles = {}
        for row in range(self._col_table.rowCount()):
            name_item = self._col_table.item(row, 0)
            if name_item is None:
                continue
            col = name_item.text()
            combo = self._col_table.cellWidget(row, 1)
            role = combo.currentData() if combo else "exclude"
            roles[col] = role
        asgn["_roles"] = roles
        self._col_assignments[self._col_table_path] = asgn

    # ======================================================================
    # Build tab — JSON sync
    # ======================================================================

    def _specs_from_ui(self, wd_override=None):
        """Build a specs list from current UI assignments."""
        specs = []
        for i in range(self._files_list.count()):
            item = self._files_list.item(i)
            key = self._item_key(item)
            asgn = self._col_assignments.get(key, {})
            roles = asgn.get("_roles", {})
            group = asgn.get("_group", "grp")
            path = item.data(Qt.UserRole)
            member = item.data(Qt.UserRole + 1)

            vars_list, facs_list, fixed_list = [], [], []
            for col, role in roles.items():
                if col.startswith("_"):
                    continue
                if role == "VAR":
                    vars_list.append(col)
                elif role == "FAC":
                    facs_list.append(col)
                elif role == "fixed":
                    # Value comes from column 2
                    val = self._col_value(i, key, col)
                    if val:
                        fixed_list.append({col: val})

            if not vars_list and not facs_list:
                continue

            # Skip entries with no ID column — Luna gpa_prep requires one
            id_cols = [c for c, r in roles.items() if r == "ID" and not c.startswith("_")]
            if not id_cols:
                continue

            # Resolve file path
            if member:
                wd = wd_override or self._wd_edit.text().strip() or os.path.dirname(path)
                # For ZIP: the extracted file will be in wd/member
                ext = os.path.splitext(path)[1].lower()
                if ext == ".zip":
                    file_path = os.path.join(wd, os.path.basename(member))
                elif ext in (".pkl", ".pickle", ".db"):
                    file_path = os.path.join(wd, member + ".tsv")
                else:
                    file_path = path
            else:
                file_path = path

            entry = {"group": group, "file": file_path, "vars": vars_list}
            if facs_list:
                entry["facs"] = facs_list
            if fixed_list:
                entry["fixed"] = fixed_list
            specs.append(entry)
        return specs

    def _col_value(self, file_list_row, key, col):
        """Get the "Value" cell for a given file row and column name."""
        # Re-check col table if it matches current path
        if self._col_table_path == key:
            for row in range(self._col_table.rowCount()):
                name_item = self._col_table.item(row, 0)
                if name_item and name_item.text() == col:
                    val_item = self._col_table.item(row, 2)
                    return val_item.text().strip() if val_item else ""
        return ""

    def _sync_ui_to_json(self):
        specs = self._specs_from_ui()
        text = json.dumps({"inputs": specs}, indent=2)
        self._json_edit.blockSignals(True)
        self._json_edit.setPlainText(text)
        self._json_edit.blockSignals(False)

    def _sync_json_to_ui(self):
        text = self._json_edit.toPlainText().strip()
        if not text:
            return
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            QtWidgets.QMessageBox.warning(self._root, "GPA", f"JSON parse error:\n{e}")
            return
        inputs = data.get("inputs", [])
        # For each entry, find or add the file in the list and update roles
        for entry in inputs:
            path = entry.get("file", "")
            if not path or not os.path.exists(path):
                continue
            # Normalise to absolute so relative paths match existing list entries
            path = os.path.abspath(path)
            vars_list = entry.get("vars", [])
            if isinstance(vars_list, str):
                vars_list = [vars_list]
            facs_list = entry.get("facs", [])
            if isinstance(facs_list, str):
                facs_list = [facs_list]
            fixed = entry.get("fixed", [])
            group = entry.get("group", "grp")
            # Find existing entry or add
            existing_key = None
            for i in range(self._files_list.count()):
                it = self._files_list.item(i)
                stored = it.data(Qt.UserRole) or ""
                if (os.path.abspath(stored) == path
                        and it.data(Qt.UserRole + 1) is None):
                    existing_key = self._item_key(it)
                    break
            if existing_key is None:
                # Add the file
                df = _sniff_tsv(path)
                if df is None:
                    continue
                item = QListWidgetItem(f"[TSV] {os.path.basename(path)}")
                item.setData(Qt.UserRole, path)
                item.setData(Qt.UserRole + 1, None)
                item.setToolTip(path)
                self._files_list.addItem(item)
                existing_key = path
            # Build roles from spec
            all_cols = set()
            asgn = self._col_assignments.get(existing_key, {})
            df = asgn.get("_df_preview") or _sniff_tsv(path)
            if df is not None:
                all_cols = set(df.columns)
            roles = {}
            for col in all_cols:
                if col.upper() == "ID":
                    roles[col] = "ID"
                elif col in vars_list or col in [str(v) for v in vars_list]:
                    roles[col] = "VAR"
                elif col in facs_list:
                    roles[col] = "FAC"
                elif any(col in f for f in fixed):
                    roles[col] = "fixed"
                else:
                    roles[col] = "exclude"
            asgn["_roles"] = roles
            asgn["_group"] = group
            if df is not None:
                asgn["_df_preview"] = df
            self._col_assignments[existing_key] = asgn
        # Refresh display if the current file was updated
        cur = self._files_list.currentRow()
        if cur >= 0:
            self._on_file_selected(cur)

    def _save_json(self):
        fn, _ = save_file_name(
            self._root, "Save Specs JSON", "specs", "JSON (*.json)")
        if not fn:
            return
        if not fn.endswith(".json"):
            fn += ".json"
        try:
            with open(fn, "w", encoding="utf-8") as fh:
                fh.write(self._json_edit.toPlainText())
        except OSError as e:
            QtWidgets.QMessageBox.warning(self._root, "GPA", f"Could not save:\n{e}")

    def _load_json(self):
        fn, _ = open_file_name(
            self._root, "Load Specs JSON", "", "JSON (*.json);;All files (*)")
        if not fn:
            return
        try:
            text = open(fn, encoding="utf-8").read()
            self._json_edit.setPlainText(text)
            self._sync_json_to_ui()
        except OSError as e:
            QtWidgets.QMessageBox.warning(self._root, "GPA", f"Could not load:\n{e}")

    # ======================================================================
    # Build tab — run --gpa-prep
    # ======================================================================

    def _browse_dat_save(self, edit):
        fn, _ = save_file_name(
            self._root, "Output .dat file", "out", "GPA binary (*.dat);;All files (*)")
        if fn:
            if not fn.endswith(".dat"):
                fn += ".dat"
            edit.setText(fn)

    def _browse_dat_open(self, edit):
        fn, _ = open_file_name(
            self._root, "Open .dat file", "", "GPA binary (*.dat);;All files (*)")
        if fn:
            edit.setText(fn)

    def _run_prep(self):
        dat_path = self._build_dat_edit.text().strip()
        if not dat_path:
            QtWidgets.QMessageBox.warning(self._root, "GPA",
                                          "Set an output .dat path first.")
            return

        self._sync_ui_to_json()
        if not self._files_list.count():
            QtWidgets.QMessageBox.warning(self._root, "GPA",
                                          "No source files configured.")
            return

        # Use a temp dir for ZIP extraction when no working dir is set;
        # worker cleans it up after gpa_prep finishes.
        wd = self._wd_edit.text().strip()
        tmp_dir = None
        if not wd:
            tmp_dir = tempfile.mkdtemp(prefix="lunascope_gpa_")
            wd = tmp_dir
        else:
            os.makedirs(wd, exist_ok=True)

        specs_list = self._specs_from_ui(wd_override=wd)
        if not specs_list:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            QtWidgets.QMessageBox.warning(self._root, "GPA",
                                          "No source files configured.")
            return

        # Collect ZIP/PKL/DB members to extract in the worker thread
        zip_members = []
        pkl_members = []
        db_members  = []
        for i in range(self._files_list.count()):
            item = self._files_list.item(i)
            path   = item.data(Qt.UserRole)
            member = item.data(Qt.UserRole + 1)
            if not member:
                continue
            ext_i = os.path.splitext(path)[1].lower()
            if ext_i == ".zip":
                dest = os.path.join(wd, os.path.basename(member))
                zip_members.append((path, member, dest))
            elif ext_i in (".pkl", ".pickle"):
                dest = os.path.join(wd, member + ".tsv")
                pkl_members.append((path, member, dest))
            elif ext_i == ".db":
                dest = os.path.join(wd, member + ".tsv")
                db_members.append((path, member, dest))

        if not self._start_work("Running --gpa-prep…"):
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            return
        self._build_status.setText("Running…")

        keep_tsv = self._keep_tsv_chk.isChecked()
        fut = self.ctrl._exec.submit(
            self._prep_worker, dat_path, specs_list,
            zip_members, pkl_members, db_members, tmp_dir, keep_tsv)
        def _done(_f=fut):
            try:
                self._sig_ok.emit({"type": "prep", "result": _f.result()})
            except Exception:
                self._sig_err.emit(traceback.format_exc())
        fut.add_done_callback(_done)

    @staticmethod
    def _prep_worker(dat_path, specs_list, zip_members, pkl_members, db_members,
                     tmp_dir=None, keep_tsv=False):
        import lunapi.lunapi0 as _l0
        _eng = _l0.inaugurate()

        # The C++ engine is a singleton shared with the main Lunascope session.
        # When a study is loaded, its sample list is set on the singleton, and
        # run_gpa/gpa-prep intersects TSV subject IDs with that list — dropping
        # everyone not in the study and making nearly all variables invariant.
        # Save the list, clear it for the duration of this build, then restore.
        _saved_sl = _eng.get_sample_list()
        _saved_rows = [[r[0], r[1]] + list(r[2]) for r in _saved_sl]
        _eng.set_sample_list([])

        extracted = []
        try:
            for zip_path, member, dest in zip_members:
                try:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        with zf.open(member) as src, open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                    extracted.append(dest)
                except Exception:
                    pass

            for pkl_path, sub_key, dest in pkl_members:
                try:
                    obj = pd.read_pickle(pkl_path)
                    if isinstance(obj, dict):
                        tables = obj.get("results", obj)
                        df = tables.get(sub_key)
                    else:
                        df = obj if sub_key == os.path.basename(pkl_path) else None
                    if df is not None:
                        df.to_csv(dest, sep="\t", index=False, na_rep=".")
                        extracted.append(dest)
                except Exception:
                    pass

            # Group db members by file so we import each db once.
            # Use the C++ engine directly to avoid _lp.proj() constructor
            # side effects (verbose print, silence state change).
            if db_members:
                from collections import defaultdict
                db_by_path = defaultdict(list)
                for db_path, key, dest in db_members:
                    db_by_path[db_path].append((key, dest))
                for db_path, items in db_by_path.items():
                    try:
                        _eng.import_db(db_path)
                        raw_strata = _eng.strata()
                        if not raw_strata:
                            continue
                        tbls = pd.DataFrame(raw_strata)
                        tbls.columns = ["Command", "Strata"]
                        key_to_dest = {k: d for k, d in items}
                        for row in tbls.itertuples(index=False):
                            k = f"{row.Command}_{row.Strata}"
                            if k not in key_to_dest:
                                continue
                            raw = _eng.table(row.Command, row.Strata)
                            if not raw or not raw[0]:
                                continue
                            df = pd.DataFrame(raw[1]).T
                            df.columns = raw[0]
                            df.to_csv(key_to_dest[k], sep="\t", index=False, na_rep=".")
                            extracted.append(key_to_dest[k])
                    except Exception:
                        pass

            from lunapi import gpa_prep
            return gpa_prep(dat_path, specs=specs_list)
        finally:
            if _saved_rows:
                _eng.set_sample_list(_saved_rows)
            if not keep_tsv:
                for f in extracted:
                    try:
                        os.unlink(f)
                    except OSError:
                        pass
                if tmp_dir:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ======================================================================
    # Analyze tab — manifest
    # ======================================================================

    def _run_load_manifest(self):
        dat_path = self._dat_edit.text().strip()
        if not dat_path:
            QtWidgets.QMessageBox.warning(self._root, "GPA",
                                          "Set a .dat file path first.")
            return
        if not os.path.exists(dat_path):
            QtWidgets.QMessageBox.warning(self._root, "GPA",
                                          f"File not found:\n{dat_path}")
            return

        if not self._start_work("Loading manifest…"):
            return
        self._analyze_status.setText("Loading manifest…")

        fut = self.ctrl._exec.submit(self._manifest_worker, dat_path)
        def _done(_f=fut):
            try:
                self._sig_ok.emit({"type": "manifest", "result": _f.result()})
            except Exception:
                self._sig_err.emit(traceback.format_exc())
        fut.add_done_callback(_done)

    @staticmethod
    def _manifest_worker(dat_path):
        cached = _read_gpa_manifest_sidecar(dat_path)
        if cached is not None and not cached.empty:
            return cached
        from lunapi import gpa_manifest
        return gpa_manifest(dat_path)

    def _populate_manifest_table(self, df):
        model = QStandardItemModel(len(df), len(df.columns), self)
        model.setHorizontalHeaderLabels(list(df.columns))
        for r, row in df.iterrows():
            for c, val in enumerate(row):
                item = QStandardItem(str(val))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                model.setItem(int(r), c, item)
        proxy = QSortFilterProxyModel(self)
        proxy.setSourceModel(model)
        proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        proxy.setFilterKeyColumn(-1)
        self._manifest_view.setModel(proxy)
        self._manifest_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._manifest_proxy = proxy
        self._manifest_filter.textChanged.connect(
            lambda t: (proxy.setFilterRegularExpression(
                QRegularExpression(t, QRegularExpression.CaseInsensitiveOption))
            )
        )
        n_vars = len(df[df["VAR"] != "ID"]) if "VAR" in df.columns else len(df)
        n_ids  = int(df["NI"].iloc[1]) if "NI" in df.columns and len(df) > 1 else "?"
        self._manifest_desc.setText(f"{n_vars} variables  ·  N={n_ids}")

    # ======================================================================
    # Analyze tab — run GPA
    # ======================================================================

    def _run_gpa(self):
        dat_path = self._dat_edit.text().strip()
        if not dat_path or not os.path.exists(dat_path):
            QtWidgets.QMessageBox.warning(
                self._root, "GPA", "Load a manifest first (valid .dat path).")
            return

        x_bases = self._picker_x.selected()
        y_bases = self._picker_y.selected()
        z_bases = self._picker_z.selected()
        x_vars = self._picker_x.selected_long_names()
        y_vars = self._picker_y.selected_long_names()
        z_vars = self._picker_z.selected_long_names()
        mode = self._mode_combo.currentData()
        self._last_gpa_z = z_vars  # list for partial scatter

        facs = self._facs_edit.text().strip()

        if mode == "assoc" and (not x_bases or not y_bases):
            QtWidgets.QMessageBox.warning(
                self._root, "GPA",
                "Association mode requires at least one X predictor and one Y outcome.")
            return
        if mode == "comp" and not x_bases:
            QtWidgets.QMessageBox.warning(
                self._root, "GPA",
                "Comparison mode requires at least one X variable to test.")
            return
        if mode == "comp" and not facs:
            QtWidgets.QMessageBox.warning(
                self._root, "GPA",
                "Comparison mode requires a factor variable in the FAC field (Options).\n"
                "Enter the name of a binary FAC column from your dataset.")
            return
        if mode == "stats" and not x_bases:
            QtWidgets.QMessageBox.warning(
                self._root, "GPA", "Descriptive stats mode requires at least one X variable.")
            return

        overlaps = []
        for label, left, right in (
            ("X and Y", x_vars, y_vars),
            ("X and Z", x_vars, z_vars),
            ("Y and Z", y_vars, z_vars),
        ):
            dup = sorted(set(left) & set(right))
            if dup:
                overlaps.append(f"{label}: {', '.join(dup[:4])}")
        if overlaps:
            self._clear_results_display()
            self._analyze_status.setText("Invalid selection: overlapping X/Y/Z variables.")
            self._desc_lbl.setText("  ·  ".join(overlaps))
            QtWidgets.QMessageBox.warning(
                self._root, "GPA",
                "A variable cannot be selected in more than one of X, Y, and Z.\n\n"
                + "\n".join(overlaps)
            )
            return

        request = self._collect_gpa_request(x_vars, y_vars, z_vars)
        self._last_gpa_request = {
            "mode": mode,
            "x_count": len(x_bases),
            "y_count": len(y_bases),
            "z_count": len(z_bases),
            "x_long_count": len(x_vars),
            "y_long_count": len(y_vars),
            "z_long_count": len(z_vars),
            "n_prop": request["meta"].get("n_prop"),
            "n_req": request["meta"].get("n_req"),
            "x_n": self._selection_n_summary(x_vars),
            "y_n": self._selection_n_summary(y_vars),
            "z_n": self._selection_n_summary(z_vars),
        }
        if not self._start_work("Running GPA…"):
            return
        self._clear_results_display()
        bits = []
        if mode == "assoc":
            bits.append(f"{self._last_gpa_request['x_count']} X")
            bits.append(f"{self._last_gpa_request['y_count']} Y")
            if self._last_gpa_request["z_count"]:
                bits.append(f"{self._last_gpa_request['z_count']} Z")
        elif mode == "stats":
            bits.append(f"{self._last_gpa_request['x_count']} X")
        elif mode == "comp":
            bits.append(f"{self._last_gpa_request['x_count']} X")
            if facs:
                bits.append(f"FAC={facs}")
        for key in ("x_long_count", "y_long_count", "z_long_count"):
            val = self._last_gpa_request.get(key, 0)
            if val:
                bits.append(f"{key[0].upper()}long={val}")
        for key in ("x_n", "y_n", "z_n"):
            val = self._last_gpa_request.get(key)
            if val:
                bits.append(f"{key[0].upper()} {val}")
        if request["meta"].get("n_prop") is not None:
            bits.append(f"n-prop<={request['meta']['n_prop']}")
        if request["meta"].get("n_req") is not None:
            bits.append(f"n-req>={request['meta']['n_req']}")
        self._analyze_status.setText(
            "Running… " + ", ".join(bits) if bits else "Running…")

        fut = self.ctrl._exec.submit(self._gpa_worker, dat_path, request)
        def _done(_f=fut):
            try:
                self._sig_ok.emit({"type": "gpa", "result": _f.result()})
            except Exception:
                self._sig_err.emit(traceback.format_exc())
        fut.add_done_callback(_done)

    def _collect_gpa_request(self, x_longs, y_longs, z_longs):
        mode = self._mode_combo.currentData()
        opts = {"dat": self._dat_edit.text().strip()}
        meta = {"mode": mode}
        if x_longs:
            opts["X"] = ",".join(x_longs)
        if z_longs:
            opts["Z"] = ",".join(z_longs)
        if mode == "assoc" and y_longs:
            opts["lvars"] = ",".join(y_longs)
        elif mode == "stats":
            opts["stats"] = ""
        elif mode == "comp":
            opts["comp"] = ""

        nreps = self._nreps_spin.value()
        if nreps:
            opts["nreps"] = str(nreps)
        if not self._chk_fdr.isChecked():
            opts["fdr"] = "F"
        if self._chk_bonf.isChecked():
            opts["bonf"] = ""
        if self._chk_holm.isChecked():
            opts["holm"] = ""
        if self._chk_fdr_by.isChecked():
            opts["fdr-by"] = ""
        if self._chk_adj_all.isChecked():
            opts["adj-all-X"] = ""
        p = self._p_spin.value()
        if p < 1.0:
            opts["p"] = str(p)
        padj = self._padj_spin.value()
        if padj < 1.0:
            opts["padj"] = str(padj)
        np_txt = self._nprop_edit.text().strip()
        if np_txt:
            try:
                meta["n_prop"] = float(np_txt)
                opts["n-prop"] = np_txt
            except ValueError: pass
        nr_txt = self._nreq_edit.text().strip()
        if nr_txt:
            try:
                meta["n_req"] = int(nr_txt)
                opts["n-req"] = nr_txt
            except ValueError: pass
        knn = self._knn_edit.text().strip()
        if knn:
            try: opts["knn"] = str(int(knn))
            except ValueError: pass
        win = self._winsor_edit.text().strip()
        if win:
            try: opts["winsor"] = str(float(win))
            except ValueError: pass
        facs = self._facs_edit.text().strip()
        if facs:
            opts["facs"] = facs
        xfacs = self._xfacs_edit.text().strip()
        if xfacs:
            opts["xfacs"] = xfacs
        sub = self._subset_edit.text().strip()
        if sub:
            opts["subset"] = sub
        inc = self._incids_edit.text().strip()
        if inc:
            opts["inc-ids"] = inc
        ex = self._exids_edit.text().strip()
        if ex:
            opts["ex-ids"] = ex
        return {
            "opts": opts,
            "diag": {
                "X": list(x_longs),
                "Y": list(y_longs),
                "Z": list(z_longs),
                "subset": opts.get("subset"),
                "inc-ids": opts.get("inc-ids"),
                "ex-ids": opts.get("ex-ids"),
            },
            "meta": meta,
        }

    @staticmethod
    def _gpa_worker(dat_path, request):
        import lunapi.lunapi0 as _l0
        eng = _l0.inaugurate()
        opts = dict(request.get("opts") or {})
        diag = GPATab._gpa_selection_diagnostics(dat_path, request.get("diag") or {})
        raw, stdout = eng.run_gpa(opts, False)
        tables = _rtables_to_dfs(raw)
        return {"tables": tables, "diag": diag, "stdout": stdout}

    @staticmethod
    def _gpa_selection_diagnostics(dat_path, selected_sets):
        """Inspect the selected GPA slice to explain empty runs."""
        import lunapi.lunapi0 as _l0

        sets = {
            "X": _split_selected_vars(selected_sets.get("X")),
            "Y": _split_selected_vars(selected_sets.get("Y")),
            "Z": _split_selected_vars(selected_sets.get("Z")),
        }
        selected = []
        for key in ("X", "Y", "Z"):
            for var in sets[key]:
                if var not in selected:
                    selected.append(var)

        diag = {
            "selected": sets,
            "joint_n": None,
            "row_n": None,
            "set_n": {},
            "overlap": {},
            "constant": {},
            "dump_complete": False,
        }
        diag["overlap"] = {
            "X∩Y": sorted(set(sets["X"]) & set(sets["Y"])),
            "X∩Z": sorted(set(sets["X"]) & set(sets["Z"])),
            "Y∩Z": sorted(set(sets["Y"]) & set(sets["Z"])),
        }

        if not selected:
            return diag

        opts = {"dat": dat_path, "dump": "", "lvars": ",".join(selected)}
        if selected_sets.get("subset"):
            opts["subset"] = selected_sets["subset"]
        if selected_sets.get("inc-ids"):
            opts["inc-ids"] = selected_sets["inc-ids"]
        if selected_sets.get("ex-ids"):
            opts["ex-ids"] = selected_sets["ex-ids"]
        _eng = _l0.inaugurate()
        _, stdout = _eng.run_gpa(opts, False)
        dump_df = _parse_gpa_manifest_text(stdout)
        if dump_df is None or dump_df.empty:
            return diag

        diag["row_n"] = int(len(dump_df))
        use_cols = [v for v in selected if v in dump_df.columns]
        diag["dump_complete"] = len(use_cols) == len(selected)
        if not diag["dump_complete"]:
            return diag

        sub = dump_df[use_cols].replace(".", np.nan)
        for label, vars_ in sets.items():
            vals = []
            for var in vars_:
                if var not in sub.columns:
                    continue
                vals.append(int(sub[var].notna().sum()))
            if vals:
                lo = min(vals)
                hi = max(vals)
                diag["set_n"][label] = lo if lo == hi else (lo, hi)

        complete_mask = sub.notna().all(axis=1)
        diag["joint_n"] = int(complete_mask.sum())
        if diag["joint_n"] <= 0:
            return diag

        cc = sub.loc[complete_mask]
        for label, vars_ in sets.items():
            const = []
            for var in vars_:
                if var not in cc.columns:
                    continue
                if cc[var].dropna().nunique() <= 1:
                    const.append(var)
            if const:
                diag["constant"][label] = const
        return diag

    def _format_gpa_status(self, result, diag=None):
        """Build a more informative one-line GPA run summary."""
        req = self._last_gpa_request or {}
        mode = req.get("mode", "")
        if mode == "assoc":
            lead = f"Assoc: {req.get('x_count', 0)} X, {req.get('y_count', 0)} Y"
            if req.get("z_count"):
                lead += f", {req.get('z_count', 0)} Z"
        elif mode == "stats":
            lead = f"Desc: {req.get('x_count', 0)} X"
        elif mode == "comp":
            lead = f"Comp: {req.get('x_count', 0)} X"
        else:
            lead = "GPA"

        details = []
        if diag:
            joint_n = diag.get("joint_n")
            row_n = diag.get("row_n")
            if joint_n is not None:
                if row_n is not None:
                    details.append(f"complete-case N={joint_n}/{row_n}")
                else:
                    details.append(f"complete-case N={joint_n}")
        for label, key in (("X", "x_long_count"), ("Y", "y_long_count"), ("Z", "z_long_count")):
            if req.get(key):
                details.append(f"{label}long={req[key]}")
        for label, key in (("X", "x_n"), ("Y", "y_n"), ("Z", "z_n")):
            if req.get(key):
                details.append(f"{label} {req[key]}")
        n_summary = _summarize_observed_n(result)
        if n_summary:
            details.append(n_summary)
        if req.get("n_prop") is not None:
            details.append(f"n-prop<={req['n_prop']}")
        if req.get("n_req") is not None:
            details.append(f"n-req>={req['n_req']}")
        n_tables = len(result or {})
        nrows = sum(len(v) for v in (result or {}).values())
        details.append(f"{n_tables} table(s)")
        details.append(f"{nrows} row(s)")
        return lead + "  ·  " + "  ·  ".join(details)

    def _format_gpa_no_results_status(self, diag, stdout=""):
        """Explain an empty GPA run in terms of the selected matrix slice."""
        parts = ["No results"]
        if diag:
            joint_n = diag.get("joint_n")
            row_n = diag.get("row_n")
            if joint_n is not None:
                parts.append(
                    f"complete-case N={joint_n}/{row_n}" if row_n is not None else f"complete-case N={joint_n}"
                )
            elif diag.get("row_n") is not None and not diag.get("dump_complete", False):
                parts.append("selection diagnostic unavailable")
            overlap = diag.get("overlap") or {}
            overlap_bits = []
            for key in ("X∩Y", "X∩Z", "Y∩Z"):
                vals = overlap.get(key) or []
                if vals:
                    overlap_bits.append(f"{key}={','.join(vals[:2])}")
            if overlap_bits:
                parts.append("overlap " + "; ".join(overlap_bits))
            constant = diag.get("constant") or {}
            const_bits = []
            for key in ("X", "Y", "Z"):
                vals = constant.get(key) or []
                if vals:
                    const_bits.append(f"{key} const={','.join(vals[:2])}")
            if const_bits:
                parts.append("; ".join(const_bits))
        req = self._last_gpa_request or {}
        qualifiers = []
        if req.get("n_prop") is not None:
            qualifiers.append(f"n-prop<={req['n_prop']}")
        if req.get("n_req") is not None:
            qualifiers.append(f"n-req>={req['n_req']}")
        if qualifiers:
            parts.append(", ".join(qualifiers))
        if stdout:
            lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
            if lines:
                parts.append(lines[-1])
        return "  ·  ".join(parts)

    def _selection_n_summary(self, names):
        """Return a compact NI summary for selected variables from the manifest."""
        if self._manifest_df is None or self._manifest_df.empty or not names:
            return None
        if "VAR" not in self._manifest_df.columns or "NI" not in self._manifest_df.columns:
            return None
        sub = self._manifest_df[self._manifest_df["VAR"].isin(names)]
        if sub.empty:
            return None
        vals = _coerce_numeric_series(sub["NI"]).dropna()
        if vals.empty:
            return None
        vals = [int(v) for v in vals.tolist()]
        lo = min(vals)
        hi = max(vals)
        if lo == hi:
            return f"N={lo}"
        return f"N={lo}–{hi}"

    def _update_selection_desc(self):
        """Summarize selected variables and their manifest non-missing counts."""
        parts = []
        for label, picker in (("X", self._picker_x), ("Y", self._picker_y), ("Z", self._picker_z)):
            bases = picker.selected()
            longs = picker.selected_long_names()
            if not bases:
                continue
            n_txt = self._selection_n_summary(longs)
            part = f"{label}: {len(bases)} base"
            if len(bases) != 1:
                part += "s"
            if longs:
                part += f", {len(longs)} long"
            if n_txt:
                part += f" ({n_txt})"
            parts.append(part)
        self._desc_lbl.setText("  ·  ".join(parts) if parts else "")

    def _clear_results_display(self):
        """Clear any stale GPA results and plots before/after a run."""
        self._results_dfs = {}
        self._active_result_df = None
        self._results_table_combo.blockSignals(True)
        self._results_table_combo.clear()
        self._results_table_combo.blockSignals(False)
        self._results_view.setModel(None)
        self._results_proxy = None
        self._render_timer.stop()
        self._scatter_mode = False
        self._scatter_gen += 1
        self._scatter_toggle_widget.setVisible(False)
        self._scatter_raw_btn.setChecked(True)
        self._scatter_partial_btn.setChecked(False)
        self._scatter_partial_btn.setEnabled(True)
        if self._canvas is not None:
            fig = self._canvas.figure
            fig.clear()
            self._canvas.draw()

    def _show_gpa_diagnostics(self, diag):
        """Mirror the most useful empty-run diagnostics in the small summary line."""
        if not diag:
            return
        bits = []
        joint_n = diag.get("joint_n")
        row_n = diag.get("row_n")
        if joint_n is not None:
            bits.append(
                f"joint complete-case N={joint_n}/{row_n}" if row_n is not None else f"joint complete-case N={joint_n}"
            )
        for key in ("X∩Y", "X∩Z", "Y∩Z"):
            vals = (diag.get("overlap") or {}).get(key) or []
            if vals:
                bits.append(f"{key}: {', '.join(vals[:3])}")
        for key in ("X", "Y", "Z"):
            vals = (diag.get("constant") or {}).get(key) or []
            if vals:
                bits.append(f"{key} const: {', '.join(vals[:3])}")
        self._desc_lbl.setText("  ·  ".join(bits))

    # ======================================================================
    # Results
    # ======================================================================

    def _populate_results_tables(self, dfs):
        self._results_dfs = dfs
        self._results_table_combo.blockSignals(True)
        self._results_table_combo.clear()
        for key in sorted(dfs.keys()):
            self._results_table_combo.addItem(key, key)
        self._results_table_combo.blockSignals(False)
        if dfs:
            self._results_table_combo.setCurrentIndex(0)
            self._on_results_table_changed(0)

    def _on_results_table_changed(self, idx):
        # switching tables → exit scatter mode, return to volcano
        self._scatter_mode = False
        self._scatter_gen += 1
        self._scatter_toggle_widget.setVisible(False)
        self._scatter_raw_btn.setChecked(True)
        self._scatter_partial_btn.setChecked(False)
        self._scatter_partial_btn.setEnabled(True)
        self._results_view.clearSelection()
        key = self._results_table_combo.currentData()
        if not key:
            return
        df = self._results_dfs.get(key)
        if df is None or df.empty:
            return
        self._active_result_df = df
        self._fill_results_view(df)
        self._render_timer.start()

    def _fill_results_view(self, df):
        df_disp = df.copy()
        numeric_cols = {
            col: pd.to_numeric(df_disp[col], errors="coerce")
            for col in df_disp.columns
        }

        model = QStandardItemModel(len(df_disp), len(df_disp.columns), self)
        model.setHorizontalHeaderLabels(list(df_disp.columns))
        for r in range(len(df_disp)):
            row = df_disp.iloc[r]
            for c, col in enumerate(df_disp.columns):
                raw_val = row.iloc[c]
                num_val = numeric_cols[col].iloc[r]
                if not pd.isna(num_val):
                    s = f"{num_val:.4g}"
                else:
                    s = "" if pd.isna(raw_val) else str(raw_val)
                item = QStandardItem(s)
                if not pd.isna(num_val):
                    item.setData(float(num_val), Qt.UserRole)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                model.setItem(r, c, item)

        proxy = _GpaResultsSortProxy(self)
        proxy.setSourceModel(model)
        proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        proxy.setFilterKeyColumn(-1)
        self._results_view.setModel(proxy)
        self._results_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._results_proxy = proxy
        self._apply_results_filter(self._results_filter.text())
        self._results_view.selectionModel().currentRowChanged.connect(
            lambda cur, _prev: self._on_result_row_clicked(cur) if cur.isValid() else None)

    def _apply_results_filter(self, text):
        if self._results_proxy:
            self._results_proxy.setFilterRegularExpression(
                QRegularExpression(text, QRegularExpression.CaseInsensitiveOption))

    # ======================================================================
    # Volcano / result visualization
    # ======================================================================

    def _render_result(self):
        df = self._active_result_df
        if df is None or df.empty:
            return
        if "B" not in df.columns or "P" not in df.columns:
            return
        canvas = self._ensure_canvas()
        if canvas is None:
            return
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(BG)
        self._render_volcano(fig, df)
        try:
            fig.tight_layout(pad=1.5)
        except Exception:
            pass
        canvas.draw()

    def _render_volcano(self, fig, df):
        """Volcano plot: B on x-axis, -log10(p) on y-axis, coloured by GROUP."""
        df = df.copy()
        for col in ("B", "P", "P_FDR", "T"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        p_col = "P_FDR" if "P_FDR" in df.columns else "P"
        if p_col not in df.columns or "B" not in df.columns:
            ax = fig.add_subplot(111)
            ax.set_facecolor(BG)
            ax.text(0.5, 0.5, "Need B and P columns for volcano plot",
                    color=FG, ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_axis_off()
            return

        df = df.dropna(subset=["B", p_col])
        df["_neglogp"] = -np.log10(df[p_col].clip(lower=1e-300))

        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)

        groups = df["GROUP"].unique() if "GROUP" in df.columns else ["all"]
        for i, grp in enumerate(sorted(groups)):
            sub = df[df["GROUP"] == grp] if "GROUP" in df.columns else df
            color = _PALETTE[i % len(_PALETTE)]
            ax.scatter(sub["B"], sub["_neglogp"], color=color, s=10,
                       alpha=0.7, linewidths=0, label=str(grp))

        # Significance line at p=0.05
        ax.axhline(-np.log10(0.05), color="#555", lw=0.8, ls="--")
        ax.axvline(0, color="#444", lw=0.5)

        ax.set_xlabel("β (regression coefficient)", color=FG, fontsize=9)
        ax.set_ylabel(f"−log₁₀({p_col})", color=FG, fontsize=9)
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)

        x_lbl = self._picker_x.selected_str() or "X"
        ax.set_title(f"Volcano: {x_lbl}", color=FG, fontsize=9)

        if len(groups) <= 10:
            leg = ax.legend(fontsize=7, framealpha=0.3,
                            facecolor="#1a1a1a", edgecolor=GRID)
            for t in leg.get_texts():
                t.set_color(FG)

    # ======================================================================
    # Row-click scatter plot
    # ======================================================================

    def _on_result_row_clicked(self, index):
        """User clicked a row in the results table — request a scatter plot."""
        proxy = self._results_proxy
        if proxy is None:
            return
        model = proxy.sourceModel()
        src_row = proxy.mapToSource(index).row()
        headers = [model.headerData(c, Qt.Horizontal)
                   for c in range(model.columnCount())]
        if "X" not in headers or "Y" not in headers:
            return
        x_var = model.item(src_row, headers.index("X")).text()
        y_var = model.item(src_row, headers.index("Y")).text()
        if not x_var or not y_var:
            return
        self._scatter_xvar = x_var
        self._scatter_yvar = y_var
        # Always show toggle; disable Partial when no Z covariates available
        has_z = bool(self._last_gpa_z)
        self._scatter_toggle_widget.setVisible(True)
        self._scatter_partial_btn.setEnabled(has_z)
        if not has_z:
            self._scatter_raw_btn.setChecked(True)
            self._scatter_partial_btn.setChecked(False)
        # Honour whichever toggle is currently active
        partial = has_z and self._scatter_partial_btn.isChecked()
        self._request_scatter(x_var, y_var, partial=partial)

    def _on_scatter_toggle(self, partial: bool):
        """Raw / Partial button clicked — redraw with same X, Y."""
        self._scatter_raw_btn.setChecked(not partial)
        self._scatter_partial_btn.setChecked(partial)
        if self._scatter_xvar and self._scatter_yvar:
            self._request_scatter(self._scatter_xvar, self._scatter_yvar, partial=partial)

    def _request_scatter(self, x_var, y_var, partial=False):
        self._scatter_mode = True
        self._render_timer.stop()
        self._scatter_gen += 1
        gen = self._scatter_gen
        zvars = self._last_gpa_z if partial else []
        self._analyze_status.setText(
            f"Loading {'partial ' if partial else ''}scatter: {x_var} × {y_var}…")
        fut = self.ctrl._exec.submit(self._scatter_worker, x_var, y_var, zvars)
        def _done(_f=fut, _g=gen, _p=partial):
            try:
                ids, xs_plot, ys_plot, xs_group = _f.result()
                if _g == self._scatter_gen:
                    self._sig_scatter_ok.emit(
                        (ids, xs_plot, ys_plot, xs_group, x_var, y_var, _p))
            except Exception:
                if _g == self._scatter_gen:
                    self._sig_scatter_err.emit(traceback.format_exc())
        fut.add_done_callback(_done)

    @staticmethod
    def _scatter_worker(x_var, y_var, zvars):
        """Return (ids, xs_plot, ys_plot, xs_group).

        xs_plot / ys_plot  — values used for axes (residualized when partial)
        xs_group           — raw X values used for group detection / box labelling
        """
        from lunapi import gpa_get_xy
        ids_raw, xs_raw, ys_raw = gpa_get_xy(x_var, y_var)
        if not zvars:
            return ids_raw, xs_raw, ys_raw, xs_raw

        from lunapi import gpa_get_xy_partial
        ids_p, xs_resid, ys_resid = gpa_get_xy_partial(x_var, y_var, zvars)
        # align raw X to the (potentially smaller) partial ID set
        raw_map = dict(zip(ids_raw, xs_raw))
        xs_group = [raw_map.get(i, float("nan")) for i in ids_p]
        return ids_p, xs_resid, ys_resid, xs_group

    def _on_scatter_ok(self, payload):
        ids, xs_plot, ys_plot, xs_group, x_var, y_var, partial = payload
        label = "partial " if partial else ""
        # detect dichotomous from raw group values
        xs_g = np.array(xs_group, dtype=float)
        unique_vals = np.unique(xs_g[~np.isnan(xs_g)])
        if len(unique_vals) == 2:
            self._analyze_status.setText(f"Box plot ({label}{x_var} × {y_var})")
            self._render_boxplot(xs_g, np.array(ys_plot, dtype=float),
                                 np.array(xs_plot, dtype=float),
                                 unique_vals, x_var, y_var, partial=partial)
        else:
            self._analyze_status.setText(f"Scatter ({label}{x_var} × {y_var})")
            self._render_scatter(np.array(xs_plot, dtype=float),
                                 np.array(ys_plot, dtype=float),
                                 x_var, y_var, partial=partial)

    def _on_scatter_err(self, tb):
        last_line = tb.strip().splitlines()[-1] if tb else "unknown error"
        self._analyze_status.setText(f"Scatter error: {last_line}")

    def _render_boxplot(self, xs_group, ys_plot, xs_plot,
                        unique_vals, x_var, y_var, partial=False):
        canvas = self._ensure_canvas()
        if canvas is None:
            return
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)

        # split Y values by group; z-score Y for scale-independence
        ys_z = (ys_plot - ys_plot.mean()) / (ys_plot.std() + 1e-12)
        groups = [ys_z[xs_group == v] for v in unique_vals]
        ns     = [len(g) for g in groups]
        labels = [f"{v:.4g}\n(N={n})" for v, n in zip(unique_vals, ns)]

        bp = ax.boxplot(groups, patch_artist=True, widths=0.45,
                        medianprops=dict(color="#fff", lw=1.5),
                        whiskerprops=dict(color=FG, lw=0.8),
                        capprops=dict(color=FG, lw=0.8),
                        flierprops=dict(marker="o", color=FG, alpha=0.3,
                                        markersize=3, linestyle="none"))
        for patch, color in zip(bp["boxes"], _PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # jitter strip
        rng = np.random.default_rng(42)
        for i, (grp, color) in enumerate(zip(groups, _PALETTE), start=1):
            jitter = rng.uniform(-0.18, 0.18, size=len(grp))
            ax.scatter(np.full(len(grp), i) + jitter, grp,
                       s=5, alpha=0.4, color=color, linewidths=0, zorder=3)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels, color=FG, fontsize=8)
        ax.set_xlabel(x_var, color=FG, fontsize=8)
        y_lbl = f"residual({y_var})" if partial else y_var
        ax.set_ylabel(f"{y_lbl} (z-scored)", color=FG, fontsize=8)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)

        # Pearson r from (residualized or raw) xs_plot vs ys_plot
        xs_z = (xs_plot - xs_plot.mean()) / (xs_plot.std() + 1e-12)
        r = float(np.corrcoef(xs_z, ys_z)[0, 1]) if len(xs_z) > 1 else float("nan")
        r_label = "r_partial" if partial else "r"
        r_str = f"{r:.3f}" if not np.isnan(r) else "n/a"
        n_total = sum(ns)
        ax.set_title(f"{r_label} = {r_str}  ·  N = {n_total}", color=FG, fontsize=9)

        try:
            fig.tight_layout(pad=1.2)
        except Exception:
            pass
        canvas.draw()

    def _render_scatter(self, xs, ys, x_var, y_var, partial=False):
        canvas = self._ensure_canvas()
        if canvas is None:
            return
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)

        xs = np.array(xs, dtype=float)
        ys = np.array(ys, dtype=float)
        n  = len(xs)

        # z-score for axis comparability; partial residuals are already mean-zero
        xs_z = (xs - xs.mean()) / (xs.std() + 1e-12)
        ys_z = (ys - ys.mean()) / (ys.std() + 1e-12)

        r = float(np.corrcoef(xs_z, ys_z)[0, 1]) if n > 1 else float("nan")
        r_label = "r_partial" if partial else "r"

        ax.scatter(xs_z, ys_z, s=8, alpha=0.55, color=_PALETTE[0], linewidths=0)

        if n > 1:
            slope, intercept = np.polyfit(xs_z, ys_z, 1)
            xl = np.array([xs_z.min(), xs_z.max()])
            ax.plot(xl, slope * xl + intercept, color=_PALETTE[2], lw=1.2)

        ax.axhline(0, color="#444", lw=0.5)
        ax.axvline(0, color="#444", lw=0.5)
        x_lbl = f"residual({x_var})" if partial else x_var
        y_lbl = f"residual({y_var})" if partial else y_var
        ax.set_xlabel(x_lbl, color=FG, fontsize=8)
        ax.set_ylabel(y_lbl, color=FG, fontsize=8)
        r_str = f"{r:.3f}" if not np.isnan(r) else "n/a"
        ax.set_title(f"{r_label} = {r_str}  ·  N = {n}", color=FG, fontsize=9)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)

        try:
            fig.tight_layout(pad=1.2)
        except Exception:
            pass
        canvas.draw()

    # ======================================================================
    # Threading callbacks
    # ======================================================================

    def _on_ok(self, payload):
        self._end_work()
        t = payload.get("type")
        result = payload.get("result")

        if t == "prep":
            self._build_status.setText("Dataset built.")
            dat = self._build_dat_edit.text().strip()
            if dat:
                self._dat_edit.setText(dat)
                manifest_df = _parse_gpa_manifest_text(result)
                if not manifest_df.empty:
                    try:
                        _write_gpa_manifest_sidecar(dat, result)
                    except Exception:
                        pass
                    self._manifest_df = manifest_df
                    self._analyze_status.setText(
                        f"{len(manifest_df)} manifest row(s) loaded.")
                    self._populate_manifest_table(manifest_df)
                    for p in (self._picker_x, self._picker_y, self._picker_z):
                        p.populate(manifest_df)
                    self._update_selection_desc()
                else:
                    # Fall back to the engine-derived manifest if prep did not
                    # return one for some reason.
                    self._run_load_manifest()

        elif t == "manifest":
            self._manifest_df = result
            self._analyze_status.setText(
                f"{len(result)} variables loaded." if result is not None else "Manifest empty.")
            if result is not None and not result.empty:
                self._populate_manifest_table(result)
                for p in (self._picker_x, self._picker_y, self._picker_z):
                    p.populate(result)
                self._update_selection_desc()

        elif t == "gpa":
            diag = {}
            tables = result
            stdout = ""
            if isinstance(result, dict) and "tables" in result and "diag" in result:
                tables = result.get("tables") or {}
                diag = result.get("diag") or {}
                stdout = result.get("stdout") or ""
            if not tables:
                self._clear_results_display()
                self._show_gpa_diagnostics(diag)
                self._analyze_status.setText(self._format_gpa_no_results_status(diag, stdout))
                return
            self._analyze_status.setText(self._format_gpa_status(tables, diag))
            self._populate_results_tables(tables)
            self._inner_tabs.setCurrentIndex(1)  # stay on Analyze

    def _on_err(self, tb):
        self._end_work()
        self._clear_results_display()
        self._build_status.setText("Error.")
        self._analyze_status.setText("Error.")
        QtWidgets.QMessageBox.critical(
            self._root, "GPA Error",
            f"An error occurred:\n\n{tb[-600:]}")

    def _on_progress(self, msg):
        self._analyze_status.setText(msg)
