
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
from scipy import special, stats

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer, QSortFilterProxyModel, QRegularExpression
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPlainTextEdit, QPushButton, QRadioButton, QScrollArea, QSizePolicy,
    QSpinBox, QSplitter, QStackedWidget, QTabBar, QTableView, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
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
        return pd.read_csv(path, sep="\t", nrows=nrows, dtype=str, encoding="utf-8-sig")
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


def _with_dump_qc_disabled(opts):
    """Return GPA dump options with robust normalization disabled."""
    out = dict(opts or {})
    out["qc"] = "F"
    return out


def _safe_corrcoef(x, y, atol: float = 1e-12):
    """Return Pearson r or NaN when either input is too short or constant."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    if valid.sum() < 2:
        return float("nan")
    xv = x_arr[valid]
    yv = y_arr[valid]
    if np.std(xv) <= atol or np.std(yv) <= atol:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _binary_level_values(values, atol: float = 1e-8, rtol: float = 1e-8):
    """Return the low/high levels if *values* are effectively binary, else None."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    uniq = np.unique(arr)
    if uniq.size < 2:
        return None
    lo = float(np.min(uniq))
    hi = float(np.max(uniq))
    if np.isclose(lo, hi, atol=atol, rtol=rtol):
        return None
    mask = np.isclose(arr, lo, atol=atol, rtol=rtol) | np.isclose(arr, hi, atol=atol, rtol=rtol)
    if not np.all(mask):
        return None
    return lo, hi


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


def _unique_preserve(values):
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _normalize_gpa_role_overlap(x_vars, y_vars, z_vars):
    """Resolve GPA role overlap before building the request."""
    x_vars = _unique_preserve(x_vars or [])
    y_vars = _unique_preserve(y_vars or [])
    z_vars = _unique_preserve(z_vars or [])

    z_set = set(z_vars)
    xz_overlap = [name for name in x_vars if name in z_set]
    if xz_overlap:
        return {
            "ok": False,
            "error_lines": [f"X and Z: {', '.join(xz_overlap[:4])}"],
            "x_vars": x_vars,
            "y_vars": y_vars,
            "z_vars": z_vars,
        }

    x_set = set(x_vars)
    yx_overlap = [name for name in y_vars if name in x_set]
    yz_overlap = [name for name in y_vars if name in z_set]
    y_drop = set(yx_overlap) | set(yz_overlap)
    normalized_y = [name for name in y_vars if name not in y_drop]

    warning_lines = []
    if yx_overlap:
        warning_lines.append(
            "Dropped from Y because also selected in X: " + ", ".join(yx_overlap[:8])
        )
    if yz_overlap:
        warning_lines.append(
            "Dropped from Y because also selected in Z: " + ", ".join(yz_overlap[:8])
        )

    return {
        "ok": True,
        "warning_lines": warning_lines,
        "x_vars": x_vars,
        "y_vars": normalized_y,
        "z_vars": z_vars,
        "dropped_from_y": [name for name in y_vars if name in y_drop],
    }


def _fmt_float(value):
    try:
        if value is None or np.isnan(value):
            return ""
    except TypeError:
        pass
    return f"{float(value):.4g}"


def _manifest_var_frame(manifest_df):
    """Return one metadata row per long variable from the GPA manifest."""
    core_cols = ["VAR", "BASE", "GRP", "NI"]
    if manifest_df is None or manifest_df.empty or "VAR" not in manifest_df.columns:
        return pd.DataFrame(columns=core_cols)
    out = manifest_df.copy()
    out = out[out["VAR"] != "ID"].copy()
    out["VAR"] = out["VAR"].astype(str)
    if "BASE" not in out.columns:
        out["BASE"] = out["VAR"]
    else:
        out["BASE"] = out["BASE"].astype(str)
    if "GRP" not in out.columns:
        out["GRP"] = "."
    else:
        out["GRP"] = out["GRP"].astype(str)
    if "NI" not in out.columns:
        out["NI"] = ""
    ordered = [col for col in core_cols if col in out.columns]
    extras = [col for col in out.columns if col not in ordered]
    return out[ordered + extras].drop_duplicates(subset=["VAR"], keep="first").reset_index(drop=True)


def _rank_seed_correlations(
    df: pd.DataFrame,
    seed_var: str,
    candidate_vars,
    meta_df: pd.DataFrame | None = None,
    min_n: int = 3,
):
    """Rank pairwise-complete correlations between *seed_var* and candidates."""
    if seed_var not in df.columns:
        raise ValueError(f"Seed variable not found in data: {seed_var}")

    meta_lookup = {}
    if meta_df is not None and not meta_df.empty and "VAR" in meta_df.columns:
        for row in meta_df.itertuples(index=False):
            meta_lookup[str(row.VAR)] = {
                "TARGET_BASE": getattr(row, "BASE", str(row.VAR)),
                "TARGET_GRP": getattr(row, "GRP", "."),
                "TARGET_NI": getattr(row, "NI", ""),
            }

    x = pd.to_numeric(df[seed_var], errors="coerce").to_numpy(dtype=float)
    rows = []
    seen = set()
    for target in candidate_vars:
        target = str(target)
        if not target or target == seed_var or target in seen or target not in df.columns:
            continue
        seen.add(target)
        y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        n_obs = int(valid.sum())
        if n_obs < min_n:
            continue
        xv = x[valid]
        yv = y[valid]
        r = _safe_corrcoef(xv, yv)
        if not np.isfinite(r):
            continue
        if abs(r) >= 1.0:
            p_val = 0.0
        else:
            t_stat = r * np.sqrt(max(n_obs - 2, 1) / max(1.0 - r * r, 1e-12))
            p_val = float(2.0 * stats.t.sf(abs(t_stat), max(n_obs - 2, 1)))
        meta = meta_lookup.get(target, {})
        rows.append(
            {
                "SEED": seed_var,
                "TARGET": target,
                "TARGET_BASE": meta.get("TARGET_BASE", target),
                "TARGET_GRP": meta.get("TARGET_GRP", "."),
                "TARGET_NI": meta.get("TARGET_NI", ""),
                "N": n_obs,
                "R": r,
                "ABS_R": abs(r),
                "P": p_val,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=["SEED", "TARGET", "TARGET_BASE", "TARGET_GRP", "TARGET_NI", "N", "R", "ABS_R", "P"]
        )
    return out.sort_values(["ABS_R", "P", "TARGET"], ascending=[False, True, True]).reset_index(drop=True)


def _fit_variable_pca(
    df: pd.DataFrame,
    meta_df: pd.DataFrame | None = None,
    min_col_prop: float = 1.0,
    row_mode: str = "complete",
    standardize: bool = True,
    max_components: int = 6,
):
    """Fit a simple PCA on variables and return per-variable component coordinates."""
    work = df.copy()
    for col in work.columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    observed = work.notna().mean(axis=0)
    keep_cols = [col for col in work.columns if float(observed.get(col, 0.0)) >= float(min_col_prop)]
    low_obs_cols = [col for col in work.columns if col not in keep_cols]
    work = work[keep_cols].copy()
    if work.empty:
        return {
            "error": "No variables passed the missingness threshold.",
            "loadings": pd.DataFrame(),
            "explained_ratio": [],
            "n_rows_input": int(len(df)),
            "n_rows_used": 0,
            "n_cols_used": 0,
            "low_obs_cols": low_obs_cols,
            "dropped_constant": [],
        }

    n_rows_input = int(len(work))
    if row_mode == "median":
        work = work.apply(lambda s: s.fillna(s.median()), axis=0)
    else:
        work = work.dropna(axis=0, how="any")
    n_rows_used = int(len(work))
    if n_rows_used < 2:
        return {
            "error": "Need at least two usable rows after missing-data handling.",
            "loadings": pd.DataFrame(),
            "explained_ratio": [],
            "n_rows_input": n_rows_input,
            "n_rows_used": n_rows_used,
            "n_cols_used": 0,
            "low_obs_cols": low_obs_cols,
            "dropped_constant": [],
        }

    std = work.std(axis=0, ddof=1).replace(0, np.nan)
    keep_nonconst = std[std > 1e-12].index.tolist()
    dropped_constant = [col for col in work.columns if col not in keep_nonconst]
    work = work[keep_nonconst].copy()
    if work.shape[1] < 2:
        return {
            "error": "Need at least two non-constant variables for PCA.",
            "loadings": pd.DataFrame(),
            "explained_ratio": [],
            "n_rows_input": n_rows_input,
            "n_rows_used": n_rows_used,
            "n_cols_used": int(work.shape[1]),
            "low_obs_cols": low_obs_cols,
            "dropped_constant": dropped_constant,
        }

    x = work.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    if standardize:
        scale = np.std(x, axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
        x = x / scale

    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    denom = max(x.shape[0] - 1, 1)
    eigenvalues = (singular ** 2) / denom
    total = float(np.sum(eigenvalues))
    explained_ratio = (eigenvalues / total).tolist() if total > 0 else [0.0] * len(eigenvalues)
    n_comp = max(1, min(int(max_components), vt.shape[0]))
    coords = vt.T[:, :n_comp] * (singular[:n_comp] / np.sqrt(denom))

    loadings = pd.DataFrame({"VAR": keep_nonconst})
    for idx in range(n_comp):
        loadings[f"PC{idx + 1}"] = coords[:, idx]
    if meta_df is not None and not meta_df.empty:
        loadings = loadings.merge(meta_df, on="VAR", how="left")
    if "BASE" not in loadings.columns:
        loadings["BASE"] = loadings["VAR"]
    if "GRP" not in loadings.columns:
        loadings["GRP"] = "."
    if "NI" not in loadings.columns:
        loadings["NI"] = ""

    return {
        "error": "",
        "loadings": loadings,
        "explained_ratio": explained_ratio[:n_comp],
        "n_rows_input": n_rows_input,
        "n_rows_used": n_rows_used,
        "n_cols_used": int(work.shape[1]),
        "low_obs_cols": low_obs_cols,
        "dropped_constant": dropped_constant,
        "row_mode": row_mode,
        "standardize": bool(standardize),
    }


def _fit_observation_pca(
    df: pd.DataFrame,
    metric_cols,
    min_col_prop: float = 1.0,
    row_mode: str = "complete",
    standardize: bool = True,
    max_components: int = 6,
):
    """Fit PCA with observations as points and metrics as axes."""
    metric_cols = [str(col) for col in metric_cols if str(col).strip() and col in df.columns]
    if len(metric_cols) < 2:
        return {
            "error": "Need at least two metric columns for observation PCA.",
            "scores": pd.DataFrame(),
            "explained_ratio": [],
            "n_rows_input": int(len(df)),
            "n_rows_used": 0,
            "n_cols_used": 0,
            "low_obs_cols": [],
            "dropped_constant": [],
        }

    work = df[metric_cols].copy()
    for col in metric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    observed = work.notna().mean(axis=0)
    keep_cols = [col for col in metric_cols if float(observed.get(col, 0.0)) >= float(min_col_prop)]
    low_obs_cols = [col for col in metric_cols if col not in keep_cols]
    work = work[keep_cols].copy()
    if work.shape[1] < 2:
        return {
            "error": "Need at least two metric columns after missingness filtering.",
            "scores": pd.DataFrame(),
            "explained_ratio": [],
            "n_rows_input": int(len(df)),
            "n_rows_used": 0,
            "n_cols_used": int(work.shape[1]),
            "low_obs_cols": low_obs_cols,
            "dropped_constant": [],
        }

    n_rows_input = int(len(work))
    if row_mode == "median":
        work = work.apply(lambda s: s.fillna(s.median()), axis=0)
        keep_index = work.index
    else:
        keep_index = work.dropna(axis=0, how="any").index
        work = work.loc[keep_index].copy()
    n_rows_used = int(len(work))
    if n_rows_used < 2:
        return {
            "error": "Need at least two usable observations after missing-data handling.",
            "scores": pd.DataFrame(),
            "explained_ratio": [],
            "n_rows_input": n_rows_input,
            "n_rows_used": n_rows_used,
            "n_cols_used": 0,
            "low_obs_cols": low_obs_cols,
            "dropped_constant": [],
        }

    std = work.std(axis=0, ddof=1).replace(0, np.nan)
    keep_nonconst = std[std > 1e-12].index.tolist()
    dropped_constant = [col for col in work.columns if col not in keep_nonconst]
    work = work[keep_nonconst].copy()
    if work.shape[1] < 2:
        return {
            "error": "Need at least two non-constant metrics for observation PCA.",
            "scores": pd.DataFrame(),
            "explained_ratio": [],
            "n_rows_input": n_rows_input,
            "n_rows_used": n_rows_used,
            "n_cols_used": int(work.shape[1]),
            "low_obs_cols": low_obs_cols,
            "dropped_constant": dropped_constant,
        }

    x = work.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    if standardize:
        scale = np.std(x, axis=0, ddof=1)
        scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
        x = x / scale

    u, singular, _vt = np.linalg.svd(x, full_matrices=False)
    denom = max(x.shape[0] - 1, 1)
    eigenvalues = (singular ** 2) / denom
    total = float(np.sum(eigenvalues))
    explained_ratio = (eigenvalues / total).tolist() if total > 0 else [0.0] * len(eigenvalues)
    n_comp = max(1, min(int(max_components), len(singular)))
    coords = u[:, :n_comp] * singular[:n_comp]

    non_metric_cols = [col for col in df.columns if col not in metric_cols]
    scores = df.loc[keep_index, non_metric_cols].copy().reset_index(drop=True)
    for idx in range(n_comp):
        scores[f"PC{idx + 1}"] = coords[:, idx]

    return {
        "error": "",
        "scores": scores,
        "explained_ratio": explained_ratio[:n_comp],
        "n_rows_input": n_rows_input,
        "n_rows_used": n_rows_used,
        "n_cols_used": int(work.shape[1]),
        "low_obs_cols": low_obs_cols,
        "dropped_constant": dropped_constant,
        "row_mode": row_mode,
        "standardize": bool(standardize),
    }


def _present_columns(df: pd.DataFrame, requested):
    requested = [str(col) for col in requested if str(col).strip()]
    present = [col for col in requested if col in df.columns]
    missing = [col for col in requested if col not in df.columns]
    return present, missing


def _assoc_dump_cache_key(dat_path, requested_cols, dump_filters):
    filt = tuple(sorted((dump_filters or {}).items()))
    cols = tuple(_unique_preserve([str(col) for col in requested_cols if str(col).strip()]))
    real_path = os.path.realpath(str(dat_path))
    try:
        stat = os.stat(real_path)
        file_id = (int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        file_id = None
    return (real_path, file_id, filt, cols)


def _assoc_pca_color_fields(manifest_df):
    meta = _manifest_var_frame(manifest_df)
    if meta.empty:
        return []
    skip = {"VAR", "BASE", "GRP", "NI", "NV"}
    return [col for col in meta.columns if col not in skip]


def _full_source_table(path, member=None, proj=None):
    """Return the full DataFrame for one source-table selection."""
    ext = os.path.splitext(path)[1].lower()

    if ext in (".txt", ".tsv", ".csv"):
        sep = "\t" if ext in (".txt", ".tsv") else ","
        return pd.read_csv(path, sep=sep, encoding="utf-8-sig")

    if ext == ".zip":
        if not member:
            raise ValueError("ZIP source requires a member selection.")
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open(member) as fh:
                return pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8-sig"), sep="\t")

    if ext in (".pkl", ".pickle"):
        obj = pd.read_pickle(path)
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
        if isinstance(obj, dict):
            key = member or ""
            if "results" in obj and isinstance(obj["results"], dict) and key in obj["results"]:
                df = obj["results"][key]
            elif key in obj:
                df = obj[key]
            else:
                raise KeyError(f"Table not found in pickle source: {key}")
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Selected pickle table is not a DataFrame: {key}")
            return df.copy()
        raise ValueError("Unsupported pickle source format.")

    if ext == ".db" and proj is not None:
        if not member or "_" not in member:
            raise ValueError("DB source requires a Command_Strata member name.")
        cmd, strata = member.split("_", 1)
        df = proj.table(cmd, strata)
        if df is None:
            raise ValueError(f"Table not found in DB source: {member}")
        return df.copy()

    raise ValueError(f"Unsupported source type: {path}")



def _rank_select_predictors(df: pd.DataFrame, names, roles):
    """Keep the largest stable full-rank subset, preserving input order."""
    kept_names = []
    kept_roles = []
    kept_cols = []
    dropped = []
    current = np.ones((len(df), 1), dtype=float)
    current_rank = np.linalg.matrix_rank(current)

    for name, role in zip(names, roles):
        col = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        if len(col) == 0 or np.nanstd(col) <= 1e-12:
            dropped.append((name, "constant"))
            continue
        trial = np.column_stack([current, col])
        trial_rank = np.linalg.matrix_rank(trial)
        if trial_rank <= current_rank:
            dropped.append((name, "collinear"))
            continue
        kept_names.append(name)
        kept_roles.append(role)
        kept_cols.append(col)
        current = trial
        current_rank = trial_rank

    return kept_names, kept_roles, kept_cols, dropped


def _fit_linear_terms(y, design):
    n_obs, n_terms = design.shape
    if n_obs <= n_terms:
        raise ValueError("Need more complete cases than model parameters for linear regression.")
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = n_obs - n_terms
    rss = float(np.dot(resid, resid))
    sigma2 = rss / max(dof, 1)
    xtx_inv = np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.clip(np.diag(sigma2 * xtx_inv), 0.0, None))
    stat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pvals = 2.0 * stats.t.sf(np.abs(stat), dof)
    return {
        "coef": beta,
        "se": se,
        "stat": stat,
        "p": pvals,
        "stat_label": "T",
        "model_type": "linear",
    }


def _fit_logistic_terms(y, design, max_iter=100, tol=1e-8):
    n_obs, n_terms = design.shape
    if n_obs <= n_terms:
        raise ValueError("Need more complete cases than model parameters for logistic regression.")

    beta = np.zeros(n_terms, dtype=float)
    converged = False
    hess = None

    for _ in range(max_iter):
        eta = design @ beta
        p = np.clip(special.expit(eta), 1e-8, 1 - 1e-8)
        w = np.clip(p * (1 - p), 1e-8, None)
        z = eta + (y - p) / w
        xtw = design.T * w
        hess = xtw @ design
        rhs = xtw @ z
        beta_new = np.linalg.pinv(hess) @ rhs
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            converged = True
            break
        beta = beta_new

    if not converged:
        raise ValueError("Logistic regression did not converge.")

    cov = np.linalg.pinv(hess)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    stat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    pvals = 2.0 * stats.norm.sf(np.abs(stat))
    return {
        "coef": beta,
        "se": se,
        "stat": stat,
        "p": pvals,
        "stat_label": "Z",
        "model_type": "logistic",
    }


def _fit_joint_model_frame(df: pd.DataFrame, x_var: str, y_vars, z_vars):
    """Fit X ~ Y + Z from an already-fetched raw GPA data slice."""
    y_vars = _unique_preserve([str(v) for v in y_vars if str(v).strip()])
    z_vars = _unique_preserve([str(v) for v in z_vars if str(v).strip()])
    requested = _unique_preserve([x_var] + y_vars + z_vars)

    missing_cols = [col for col in requested if col not in df.columns]
    out = {
        "table": pd.DataFrame(columns=["TERM", "ROLE", "BETA", "SE", "STAT", "P"]),
        "warnings": [],
        "active_y": list(y_vars),
        "active_z": list(z_vars),
        "x_var": x_var,
        "n_complete": 0,
        "n_total": int(len(df)),
        "model_type": "",
        "binary_labels": None,
        "missing_cols": missing_cols,
    }
    if missing_cols:
        out["warnings"].append("Missing columns: " + ", ".join(missing_cols[:6]))

    if x_var not in df.columns:
        out["error"] = f"Missing dependent variable: {x_var}"
        return out

    use_cols = [col for col in requested if col in df.columns]
    work = df[use_cols].copy()
    for col in use_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna()
    out["n_complete"] = int(len(work))
    if work.empty:
        out["error"] = "No complete cases available for the current X/Y/Z set."
        return out

    x_vals = work[x_var].to_numpy(dtype=float)
    binary_levels = _binary_level_values(x_vals)
    is_binary = binary_levels is not None
    if is_binary:
        lo, hi = binary_levels
        y_target = np.isclose(x_vals, hi, atol=1e-8, rtol=1e-8).astype(float)
        out["binary_labels"] = (lo, hi)
    else:
        y_target = x_vals

    term_names = []
    term_roles = []
    for role, values in (("Y", y_vars), ("Z", z_vars)):
        for value in values:
            if value == x_var or value not in work.columns:
                continue
            term_names.append(value)
            term_roles.append(role)

    kept_names, kept_roles, kept_cols, dropped = _rank_select_predictors(work, term_names, term_roles)
    for name, why in dropped:
        out["warnings"].append(f"Dropped {name} ({why})")

    if not kept_names:
        out["error"] = "No usable predictors remain after filtering missing or collinear terms."
        return out

    design = np.column_stack([np.ones(len(work), dtype=float)] + kept_cols)
    try:
        if is_binary:
            fit = _fit_logistic_terms(y_target, design)
        else:
            fit = _fit_linear_terms(y_target, design)
    except ValueError as exc:
        out["error"] = str(exc)
        return out

    terms = ["Intercept"] + kept_names
    roles = ["Intercept"] + kept_roles
    stat_label = fit["stat_label"]
    out["table"] = pd.DataFrame({
        "TERM": terms,
        "ROLE": roles,
        "BETA": fit["coef"],
        "SE": fit["se"],
        "STAT": fit["stat"],
        "P": fit["p"],
    })
    out["stat_label"] = stat_label
    out["model_type"] = fit["model_type"]
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
                        mf = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8-sig"), sep="\t", dtype=str)
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
                            df = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8-sig"), sep="\t", nrows=nrows, dtype=str)
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
    seen_var = False
    for col in df.columns:
        if col.upper() == "ID":
            roles[col] = "ID"
        elif col in known_fac:
            roles[col] = "FAC"
        elif has_meta:
            roles[col] = "VAR"
            seen_var = True
        else:
            vals = df[col].dropna()
            vals = vals[vals != "."]
            numeric_frac = pd.to_numeric(vals, errors="coerce").notna().mean()
            if numeric_frac >= 0.8:
                roles[col] = "VAR"
                seen_var = True
            elif seen_var:
                # Categorical column appearing after numeric columns — exclude
                roles[col] = "exclude"
            else:
                roles[col] = "FAC"
    return roles


# ---------------------------------------------------------------------------
# Reusable variable-picker widget
# ---------------------------------------------------------------------------

class _VarPicker(QWidget):
    """Searchable checklist for picking GPA variables by group and base name."""

    selectionChanged = QtCore.Signal()

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self._label_text = str(label)
        self._manifest = None
        self._pair_rows = pd.DataFrame()
        self._selected_pairs = set()
        self._updating_list = False
        self._single_select = False
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Header row — wrapped in a widget so it can be hidden or reparented
        self._hdr_w = QWidget()
        self._hdr_w.setContentsMargins(0, 0, 0, 0)
        hdr = QHBoxLayout(self._hdr_w)
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(3)
        lbl = QLabel(f"<b>{label}</b>")
        lbl.setStyleSheet(f"color:{FG}; font-size:11px;")
        self._title_lbl = lbl
        self._summary_lbl = QLabel("")
        self._summary_lbl.setStyleSheet(f"color:#888; font-size:10px;")
        self._summary_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._summary_lbl.setMinimumWidth(0)
        self._summary_lbl.setWordWrap(True)
        self._grp_combo = QComboBox()
        self._grp_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self._grp_combo.setMinimumContentsLength(6)
        self._grp_combo.setToolTip("Filter by group")
        self._grp_combo.addItem("(all groups)", None)
        self._btn_none = QPushButton("✕")
        self._btn_none.setFixedWidth(22)
        self._btn_none.setToolTip("Clear all")
        self._btn_all = QPushButton("✓")
        self._btn_all.setFixedWidth(22)
        self._btn_all.setToolTip("Select all visible")
        hdr.addWidget(lbl)
        hdr.addWidget(self._grp_combo, 1)
        hdr.addWidget(self._btn_all)
        hdr.addWidget(self._btn_none)

        # Search
        self._search = QLineEdit()
        self._search.setPlaceholderText("search base names…")
        self._search.setClearButtonEnabled(True)
        self._search.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        # List
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.NoSelection)
        self._list.setUniformItemSizes(True)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.setTextElideMode(Qt.ElideRight)
        self._list.setWordWrap(False)
        self._list.setSpacing(0)
        self._list.setStyleSheet(
            "QListWidget { background:#0d1117; border:1px solid #21262d; font-size:11px; }"
            "QListWidget::item { padding:1px 2px; }"
        )
        self._list.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        self._list.viewport().installEventFilter(self)
        self._list.itemChanged.connect(self._on_item_changed)

        layout.addWidget(self._hdr_w)
        layout.addWidget(self._summary_lbl)
        layout.addWidget(self._search)
        layout.addWidget(self._list, 1)

        self._btn_all.clicked.connect(self._select_all_visible)
        self._btn_none.clicked.connect(self._clear_all)
        self._grp_combo.currentIndexChanged.connect(self._refilter)
        self._search.textChanged.connect(self._refilter)

    def set_summary(self, text: str):
        self._summary_lbl.setText(text or "")

    def set_title(self, text: str):
        self._label_text = str(text or "")
        self._title_lbl.setText(f"<b>{self._label_text}</b>")

    def set_single_selection(self, enabled: bool):
        self._single_select = bool(enabled)

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
        # Size the popup view to the widest item so nothing is ever elided
        fm = self._grp_combo.fontMetrics()
        max_w = max(
            (fm.horizontalAdvance(self._grp_combo.itemText(i))
             for i in range(self._grp_combo.count())),
            default=0,
        )
        self._grp_combo.view().setMinimumWidth(max_w + 36)  # 36px for padding + scrollbar
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
            if self._single_select:
                self._selected_pairs = {tuple(pair) for pair in pairs}
                self._refilter()
            else:
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
        names = [str(name) for name in names if str(name).strip()]
        wanted = set(names[:1] if self._single_select and names else names)
        self._selected_pairs = set()
        for row in self._pair_rows.itertuples(index=False):
            if row.BASE in wanted:
                self._selected_pairs.add((row.GRP, row.BASE))
        self._refilter()
        self.selectionChanged.emit()

    def eventFilter(self, obj, event):
        if obj is self._list.viewport() and event.type() == QtCore.QEvent.MouseButtonRelease:
            if event.button() == Qt.LeftButton:
                item = self._list.itemAt(event.pos())
                if item is not None:
                    self._updating_list = True
                    try:
                        item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)
                    finally:
                        self._updating_list = False
                    self._on_item_changed(item)
                    return True
        return super().eventFilter(obj, event)


# ---------------------------------------------------------------------------
# Main GPA tab
# ---------------------------------------------------------------------------

class GPATab(_ExplorerTab):
    """GPA / Association Explorer tab."""

    _sig_ok         = QtCore.Signal(object)
    _sig_err        = QtCore.Signal(object)
    _sig_progress   = QtCore.Signal(str)
    _sig_scatter_ok = QtCore.Signal(object)   # (ids, xs, ys, xvar, yvar)
    _sig_scatter_err= QtCore.Signal(str)
    _sig_obs_vals   = QtCore.Signal(object)   # (row_w, [unique_val_strs])
    _sig_obs_count  = QtCore.Signal(object)   # (ids_or_None, n_match, n_total, err_str)

    def __init__(self, ctrl, parent=None):
        super().__init__(ctrl, parent)

        # ---- state -------------------------------------------------------
        self._manifest_df: pd.DataFrame | None = None
        self._results_dfs: dict = {}
        # {path: {col: {role, value, group}}}
        self._col_assignments: dict = {}
        # entries currently displayed in the column table
        self._col_table_path: str | None = None
        # currently displayed GPA results table
        self._results_table_key: str | None = None
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
        self._joint_mode = False
        self._joint_fit_gen = 0
        self._joint_xvar: str = ""
        self._joint_yvars: list[str] = []
        self._joint_zvars: list[str] = []
        self._joint_result: dict | None = None
        self._pre_joint_status_text: str = ""

        self._assoc_corr_df: pd.DataFrame | None = None
        self._assoc_corr_proxy: QSortFilterProxyModel | None = None
        self._assoc_seed_long: str = ""
        self._assoc_scatter_seed: str = ""
        self._assoc_scatter_target: str = ""
        self._assoc_plot_mode: str = "ranked"
        self._assoc_pca_result: dict | None = None
        self._assoc_pca_df: pd.DataFrame | None = None
        self._assoc_pca_proxy: QSortFilterProxyModel | None = None
        self._assoc_pca_artist = None
        self._assoc_pca_labels: list[str] = []
        self._assoc_pca_xy: np.ndarray | None = None
        self._assoc_pca_hover = None
        self._assoc_suspend_auto_run = False
        self._assoc_ranked_hover = None
        self._assoc_ranked_bars = []
        self._assoc_ranked_labels: list[str] = []
        self._assoc_matrix_cache_key = None
        self._assoc_matrix_cache_df: pd.DataFrame | None = None
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(250)
        self._render_timer.timeout.connect(self._render_result)

        # Observations filter state
        self._obs_row_widgets: list  = []
        self._obs_inc_ids:     list | None = None   # None = all; list = subset
        self._obs_col_cache:   dict  = {}           # var_name -> pd.Series (ID-indexed)
        self._obs_count_timer  = QTimer(self)
        self._obs_count_timer.setSingleShot(True)
        self._obs_count_timer.setInterval(600)
        self._obs_count_timer.timeout.connect(self._obs_refresh_count)

        self._sig_ok.connect(self._on_ok,               Qt.QueuedConnection)
        self._sig_err.connect(self._on_err,              Qt.QueuedConnection)
        self._sig_progress.connect(self._on_progress,   Qt.QueuedConnection)
        self._sig_scatter_ok.connect(self._on_scatter_ok,  Qt.QueuedConnection)
        self._sig_scatter_err.connect(self._on_scatter_err, Qt.QueuedConnection)
        self._sig_obs_vals.connect(self._obs_on_unique_vals,  Qt.QueuedConnection)
        self._sig_obs_count.connect(self._obs_on_count_result, Qt.QueuedConnection)

        self._build_widget()

    # ======================================================================
    # Widget construction
    # ======================================================================

    def _build_widget(self):
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(0)

        # Single full-width QTabWidget — each tab owns its content entirely.
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        root_layout.addWidget(self._tabs)

        # Build tab — scrollable form
        build_scroll = QScrollArea()
        build_scroll.setWidgetResizable(True)
        build_scroll.setFrameShape(QFrame.NoFrame)
        build_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        build_scroll.setWidget(self._build_build_tab())

        # Manifest tab — full-width table
        manifest_frame = self._build_manifest_panel()

        # Select tab — left form + right var-detail panel (splitter built inside)
        select_w = self._build_select_tab()

        # Results tab — full-width results panel
        results_frame = self._build_results_panel()

        # Correl tab — internal splitter: narrow settings left, wide results right
        correl_outer = QSplitter(Qt.Horizontal)
        correl_outer.setHandleWidth(5)
        correl_left_scroll = QScrollArea()
        correl_left_scroll.setWidgetResizable(True)
        correl_left_scroll.setMinimumWidth(180)
        correl_left_scroll.setFrameShape(QFrame.NoFrame)
        correl_left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        correl_left_scroll.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
        correl_left_scroll.setWidget(self._build_assoc_explore_tab())
        assoc_frame = self._build_assoc_results_panel()
        correl_outer.addWidget(correl_left_scroll)
        correl_outer.addWidget(assoc_frame)
        correl_outer.setSizes([360, 940])
        correl_outer.setStretchFactor(0, 0)
        correl_outer.setStretchFactor(1, 1)
        correl_outer.setCollapsible(0, False)
        correl_outer.setCollapsible(1, False)
        self._assoc_outer_splitter = correl_outer

        self._tabs.addTab(build_scroll,   "Build/Load")
        self._tabs.addTab(manifest_frame, "Manifest")
        self._tabs.addTab(select_w,       "Select")
        self._tabs.addTab(results_frame,  "Results")
        self._tabs.addTab(correl_outer,   "Correl")

        QTimer.singleShot(0, self._apply_assoc_default_splitter_sizes)
        QTimer.singleShot(0, self._init_assoc_mode_ui)

        self._root = root

    def _init_assoc_mode_ui(self):
        self._on_assoc_mode_changed()

    def _apply_assoc_default_splitter_sizes(self):
        outer = getattr(self, "_assoc_outer_splitter", None)
        if outer is None:
            return
        try:
            total = max(800, outer.size().width())
        except RuntimeError:
            return
        left_w = min(380, max(300, int(total * 0.28)))
        outer.setSizes([left_w, max(400, total - left_w)])

    # ------------------------------------------------------------------
    # Build tab
    # ------------------------------------------------------------------

    def _build_build_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        # ---- mode selector ----
        mode_frame = QFrame()
        mode_frame.setFrameShape(QFrame.StyledPanel)
        mode_lay = QHBoxLayout(mode_frame)
        mode_lay.setContentsMargins(8, 4, 8, 4)
        mode_lay.setSpacing(20)
        self._mode_build_radio = QRadioButton("Build  (assemble sources → new .dat)")
        self._mode_load_radio  = QRadioButton("Load  (open existing .dat)")
        self._mode_build_radio.setChecked(True)
        self._mode_build_radio.setStyleSheet("font-weight:600;")
        self._mode_load_radio.setStyleSheet("font-weight:600;")
        mode_lay.addWidget(self._mode_build_radio)
        mode_lay.addWidget(self._mode_load_radio)
        mode_lay.addStretch(1)
        lay.addWidget(mode_frame)

        # ---- stacked pages ----
        self._build_mode_stack = QStackedWidget()
        lay.addWidget(self._build_mode_stack, 1)

        self._mode_build_radio.toggled.connect(
            lambda on: self._build_mode_stack.setCurrentIndex(0 if on else 1))

        # === page 0: Build ===
        build_page = QWidget()
        build_lay = QVBoxLayout(build_page)
        build_lay.setContentsMargins(0, 4, 0, 0)
        build_lay.setSpacing(6)

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
        build_lay.addLayout(wd_row)

        # Source files
        src_hdr = QHBoxLayout()
        src_lbl = QLabel("<b>Source files</b>")
        src_lbl.setStyleSheet(f"color:{FG};")
        src_hdr.addWidget(src_lbl)
        src_hdr.addStretch(1)
        btn_add = QPushButton("+ Add"); btn_add.setFixedWidth(58)
        btn_rm  = QPushButton("− Remove"); btn_rm.setFixedWidth(72)
        src_hdr.addWidget(btn_add); src_hdr.addWidget(btn_rm)
        build_lay.addLayout(src_hdr)

        self._files_list = QListWidget()
        self._files_list.setFixedHeight(110)
        self._files_list.setToolTip("TSV, ZIP, PKL, or Luna .db files")
        build_lay.addWidget(self._files_list)

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
        col_lay.addLayout(col_hdr)

        self._col_table = QTableWidget(0, 3)
        self._col_table.setHorizontalHeaderLabels(["Column", "Role", "Preview"])
        self._col_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents)
        self._col_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Fixed)
        self._col_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Stretch)
        self._col_table.setColumnWidth(1, 72)
        self._col_table.setMinimumHeight(200)
        self._col_table.setMaximumHeight(400)
        self._col_table.verticalHeader().setVisible(False)
        self._col_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._col_table.setEditTriggers(QAbstractItemView.DoubleClicked)
        self._col_table.itemChanged.connect(self._on_col_table_changed)
        col_lay.addWidget(self._col_table)

        self._col_frame.setVisible(False)
        build_lay.addWidget(self._col_frame)

        # JSON / Specs  -------------------------------------------------
        json_toggle = QPushButton("▶  Specs JSON")
        json_toggle.setCheckable(True)
        json_toggle.setStyleSheet("text-align:left; padding-left:4px;")
        build_lay.addWidget(json_toggle)

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
        build_lay.addWidget(self._json_frame)

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
        build_lay.addLayout(dat_row)

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
        build_lay.addLayout(build_btns)

        self._build_btn.clicked.connect(self._run_prep)
        build_lay.addStretch(1)
        self._build_mode_stack.addWidget(build_page)  # page 0

        # === page 1: Load ===
        load_page = QWidget()
        load_outer = QVBoxLayout(load_page)
        load_outer.setContentsMargins(0, 4, 0, 0)
        load_outer.setSpacing(0)

        load_inner = QFrame()
        load_inner.setFrameShape(QFrame.StyledPanel)
        load_inner_lay = QVBoxLayout(load_inner)
        load_inner_lay.setContentsMargins(16, 16, 16, 16)
        load_inner_lay.setSpacing(12)

        load_lbl = QLabel("Open a previously built <b>.dat</b> file to load its manifest "
                          "and populate the variable pickers.")
        load_lbl.setWordWrap(True)
        load_lbl.setStyleSheet(f"color:{FG}; font-size:11px;")
        load_inner_lay.addWidget(load_lbl)

        load_dat_row = QHBoxLayout()
        load_dat_row.setSpacing(4)
        load_dat_row.addWidget(QLabel(".dat file:"))
        self._dat_edit = QLineEdit()
        self._dat_edit.setPlaceholderText("path/to/out.dat")
        self._dat_edit.textChanged.connect(lambda *_: self._update_assoc_dat_label())
        btn_dat_open = QPushButton("…"); btn_dat_open.setFixedWidth(26)
        btn_dat_open.clicked.connect(lambda: self._browse_dat_open(self._dat_edit))
        self._load_manifest_btn = QPushButton("Load .dat")
        self._load_manifest_btn.setFixedWidth(90)
        self._load_manifest_btn.setStyleSheet(
            "QPushButton { background:#1e3a5f; color:#fff; padding:4px 10px; border-radius:4px; }"
            "QPushButton:hover { background:#1d4ed8; }"
        )
        self._load_manifest_btn.clicked.connect(self._run_load_manifest)
        load_dat_row.addWidget(self._dat_edit, 1)
        load_dat_row.addWidget(btn_dat_open)
        load_dat_row.addWidget(self._load_manifest_btn)
        load_inner_lay.addLayout(load_dat_row)

        load_inner_lay.addStretch(1)
        load_outer.addWidget(load_inner)
        load_outer.addStretch(1)
        self._build_mode_stack.addWidget(load_page)   # page 1

        return w

    # ------------------------------------------------------------------
    # Select tab — Y/X/Z bands + bottom run bar
    # ------------------------------------------------------------------

    _SELECT_ROLES = [
        ("Y", "Y outcomes",    "#a78bfa"),
        ("X", "X predictors",  "#4cc9f0"),
        ("Z", "Z covariates",  "#06d6a0"),
    ]

    def _build_select_tab(self):
        root = QWidget()
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(4, 4, 4, 4)
        root_lay.setSpacing(4)

        # Inner tab widget — Variables / Observations
        self._select_inner_tabs = QTabWidget()
        self._select_inner_tabs.setDocumentMode(True)
        root_lay.addWidget(self._select_inner_tabs, 1)

        # ---- Tab 0: Variables (Columns) ----
        vars_tab = QWidget()
        vars_tab_lay = QVBoxLayout(vars_tab)
        vars_tab_lay.setContentsMargins(0, 4, 0, 0)
        vars_tab_lay.setSpacing(0)

        vsplit = QSplitter(Qt.Vertical)
        vsplit.setHandleWidth(4)
        self._select_vsplit = vsplit
        self._select_var_tables:    dict = {}
        self._select_var_data:      dict = {}
        self._select_var_grp_longs: dict = {}   # role -> {grp_key: [long_var_names]}
        self._band_slots:           list = []   # strong refs so PySide6 can't GC the closures

        for role, label, color in self._SELECT_ROLES:
            band = QWidget()
            band_lay = QVBoxLayout(band)
            band_lay.setContentsMargins(0, 0, 0, 0)
            band_lay.setSpacing(0)

            # 2 px colored accent strip — clear visual demarcation between bands
            accent = QFrame()
            accent.setFixedHeight(2)
            accent.setStyleSheet(f"background-color:{color}; border:none;")
            band_lay.addWidget(accent)

            # Full-width label bar: role label | detail/count | grp_combo | ✓ | ✕
            lbl_bar = QWidget()
            lbl_bar.setStyleSheet("background: rgba(255,255,255,0.03);")
            lbl_bar_lay = QHBoxLayout(lbl_bar)
            lbl_bar_lay.setContentsMargins(6, 2, 6, 2)
            lbl_bar_lay.setSpacing(6)
            role_lbl = QLabel(f"<b>{label}</b>")
            role_lbl.setStyleSheet(f"color:{color}; font-size:11px;")
            detail_lbl = QLabel("(select a variable to see long-form details)")
            detail_lbl.setStyleSheet("color:#666; font-size:10px;")
            lbl_bar_lay.addWidget(role_lbl)
            lbl_bar_lay.addWidget(detail_lbl, 1)

            # Picker is created here so we can reparent its controls into this bar
            picker = _VarPicker("")
            picker._title_lbl.setVisible(False)
            picker._hdr_w.setVisible(False)
            picker._summary_lbl.setVisible(False)
            picker.setMinimumWidth(180)

            lbl_bar_lay.addWidget(picker._grp_combo)
            lbl_bar_lay.addWidget(picker._btn_all)
            lbl_bar_lay.addWidget(picker._btn_none)
            band_lay.addWidget(lbl_bar)

            # Horizontal splitter: picker | (grp selector | var/NI table)
            hsplit = QSplitter(Qt.Horizontal)
            hsplit.setHandleWidth(5)

            # Left sub-table: GRP selector (GRP / Vars / NI) — selectable, filters right
            t_grps = QTableWidget(0, 3)
            t_grps.setHorizontalHeaderLabels(["GRP", "Vars", "NI"])
            t_grps.verticalHeader().setVisible(False)
            t_grps.setSelectionMode(QAbstractItemView.SingleSelection)
            t_grps.setSelectionBehavior(QAbstractItemView.SelectRows)
            t_grps.setEditTriggers(QAbstractItemView.NoEditTriggers)
            t_grps.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
            t_grps.horizontalHeader().setStretchLastSection(False)
            t_grps.setColumnWidth(0, 160)
            t_grps.setColumnWidth(1, 42)
            t_grps.setColumnWidth(2, 52)
            t_grps.setAlternatingRowColors(True)
            t_grps.setStyleSheet("font-size:11px;")
            t_grps.setToolTip("Click a group to filter the VAR list; click again to clear")

            # Right sub-table: VAR / NI — read-only, filtered by grp selection
            t_vars = QTableWidget(0, 2)
            t_vars.setHorizontalHeaderLabels(["VAR", "NI"])
            t_vars.verticalHeader().setVisible(False)
            t_vars.setSelectionMode(QAbstractItemView.NoSelection)
            t_vars.setEditTriggers(QAbstractItemView.NoEditTriggers)
            t_vars.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
            t_vars.horizontalHeader().setStretchLastSection(False)
            t_vars.setColumnWidth(0, 280)
            t_vars.setColumnWidth(1, 52)
            t_vars.setAlternatingRowColors(True)
            t_vars.setStyleSheet("font-size:11px;")

            right_w = QSplitter(Qt.Horizontal)
            right_w.setHandleWidth(5)
            right_w.addWidget(t_grps)
            right_w.addWidget(t_vars)
            right_w.setSizes([260, 560])
            right_w.setStretchFactor(0, 0)
            right_w.setStretchFactor(1, 1)
            right_w.setCollapsible(0, False)
            right_w.setCollapsible(1, False)

            hsplit.addWidget(picker)
            hsplit.addWidget(right_w)
            hsplit.setSizes([380, 820])
            hsplit.setStretchFactor(0, 0)
            hsplit.setStretchFactor(1, 1)
            hsplit.setCollapsible(0, False)
            hsplit.setCollapsible(1, False)

            band_lay.addWidget(hsplit, 1)
            vsplit.addWidget(band)
            vsplit.setCollapsible(vsplit.count() - 1, False)

            # Assign pickers
            if   role == "Y": self._picker_y = picker
            elif role == "X": self._picker_x = picker
            else:             self._picker_z = picker

            picker.selectionChanged.connect(self._update_selection_desc)

            # Named closures stored in _band_slots so PySide6 cannot GC them.
            def _make_var_slot(role_key):
                def _slot():
                    self._update_select_var_tables(
                        role_key,
                        getattr(self, f"_picker_{role_key.lower()}"))
                return _slot

            def _make_grp_slot(role_key, tg, tv):
                def _slot(cur, prev):
                    self._on_grp_filter_changed(role_key, tg, tv)
                return _slot

            _var_slot = _make_var_slot(role)
            self._band_slots.append(_var_slot)
            picker.selectionChanged.connect(_var_slot)

            _grp_slot = _make_grp_slot(role, t_grps, t_vars)
            self._band_slots.append(_grp_slot)
            t_grps.currentItemChanged.connect(_grp_slot)

            self._select_var_tables[role] = (t_vars, t_grps, detail_lbl)

        vars_tab_lay.addWidget(vsplit, 1)
        self._select_inner_tabs.addTab(vars_tab, "Variables (Columns)")
        QTimer.singleShot(0, self._apply_select_vsplit_sizes)

        # ---- Tab 1: Observations (rows) ----
        obs_tab = QWidget()
        self._build_obs_filter_tab(obs_tab)
        self._select_inner_tabs.addTab(obs_tab, "Observations (rows)")

        # ---- bottom bar (outside tabs) ----
        bar = QHBoxLayout()
        bar.setSpacing(8)

        self._run_btn = QPushButton("Run GPA")
        self._run_btn.setStyleSheet(
            "QPushButton { background:#1e3a5f; color:#fff; padding:4px 12px; border-radius:4px; }"
            "QPushButton:hover { background:#1d4ed8; }"
        )
        self._run_btn.clicked.connect(self._run_gpa)

        winsor_lbl = QLabel("winsor:")
        winsor_lbl.setToolTip("Tail fraction trimmed each side (0–0.2). Leave blank to disable.")
        self._winsor_edit = QLineEdit()
        self._winsor_edit.setFixedWidth(50)
        self._winsor_edit.setPlaceholderText("off")
        self._winsor_edit.setToolTip("Tail fraction trimmed each side (0–0.2). Leave blank to disable.")

        self._dump_tsv_btn = QPushButton("Save as TSV…")
        self._dump_tsv_btn.setToolTip("Export selected variables as a flat TSV (GPA dump mode).")
        self._dump_tsv_btn.clicked.connect(self._save_gpa_dump_tsv)

        self._select_count_lbl = QLabel("")
        self._select_count_lbl.setStyleSheet("color:#aaa; font-size:10px;")
        self._select_count_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._analyze_status = QLabel("")
        self._analyze_status.setStyleSheet("color:#888; font-size:11px;")
        self._analyze_status.setWordWrap(True)
        self._analyze_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        bar.addWidget(self._run_btn)
        bar.addWidget(winsor_lbl)
        bar.addWidget(self._winsor_edit)
        bar.addSpacing(8)
        bar.addWidget(self._dump_tsv_btn)
        bar.addWidget(self._select_count_lbl, 1)
        bar.addWidget(self._analyze_status, 1)
        root_lay.addLayout(bar)

        return root

    def _apply_select_vsplit_sizes(self):
        vsplit = getattr(self, "_select_vsplit", None)
        if vsplit is None:
            return
        try:
            total = max(600, vsplit.size().height())
        except RuntimeError:
            return
        y_h = int(total * 0.50)
        x_h = int(total * 0.25)
        vsplit.setSizes([y_h, x_h, total - y_h - x_h])

    def _on_grp_filter_changed(self, role, t_grps, t_vars):
        grp_to_longs = getattr(self, "_select_var_grp_longs", {}).get(role) or {}
        ni_lookup    = self._select_var_data.get(role) or {}
        row = t_grps.currentRow()
        grp = t_grps.item(row, 0).text() if row >= 0 and t_grps.item(row, 0) else None
        if grp and grp in grp_to_longs:
            longs = grp_to_longs[grp]
        else:
            # no valid row selected — show all longs across all groups
            longs = sorted({v for vs in grp_to_longs.values() for v in vs})
        self._fill_var_table_longs(t_vars, longs, ni_lookup)

    def _fill_var_table_longs(self, t_vars, longs: list, ni_lookup: dict):
        """Populate the VAR/NI table from a list of long-var names + NI lookup dict."""
        t_vars.setRowCount(0)
        for var in longs:
            r = t_vars.rowCount()
            t_vars.insertRow(r)
            for ci, val in enumerate((str(var), ni_lookup.get(str(var), ""))):
                item = QTableWidgetItem(val)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                t_vars.setItem(r, ci, item)

    def _update_select_var_tables(self, role: str, picker):
        """Refresh the GRP selector and VAR/NI tables for *role* from the picker's selection.

        Uses the picker's own _pair_rows (GRP/BASE/LONGS) as the authoritative source
        so that the grouping is always consistent with what the picker shows, regardless
        of how the manifest's own GRP column is structured.
        """
        if not hasattr(self, "_select_var_tables") or role not in self._select_var_tables:
            return
        t_vars, t_grps, detail_lbl = self._select_var_tables[role]

        pairs = picker.selected_pairs()   # [(grp, base), ...]
        if not pairs:
            t_vars.setRowCount(0)
            t_grps.setRowCount(0)
            self._select_var_data[role] = None
            self._select_var_grp_longs[role] = {}
            detail_lbl.setText("(select a variable to see long-form details)")
            return

        # --- GRP selector table ---
        # Built from picker._pair_rows so GRP labels match what the picker shows.
        pair_set = set(pairs)
        pr = picker._pair_rows
        pr_sel = pr[pr.apply(
            lambda row: (row["GRP"], row["BASE"]) in pair_set, axis=1
        )].reset_index(drop=True)

        grp_to_longs: dict = {}
        for _, row in pr_sel.iterrows():
            grp_key = str(row["GRP"])
            grp_to_longs.setdefault(grp_key, []).extend(list(row["LONGS"]))

        # Aggregate per-GRP so one row appears per group regardless of how many bases
        # were selected within that group.
        t_grps.blockSignals(True)
        t_grps.setRowCount(0)
        for grp_key, grp_rows in pr_sel.groupby("GRP", sort=False):
            grp_key = str(grp_key)
            n_long = int(grp_rows["N_LONG"].sum())
            ni_mins = grp_rows["NI_MIN"].dropna().tolist()
            ni_maxs = grp_rows["NI_MAX"].dropna().tolist()
            all_ni = [v for v in ni_mins + ni_maxs if not (isinstance(v, float) and np.isnan(v))]
            if not all_ni:
                n_str = ""
            else:
                lo, hi = int(min(all_ni)), int(max(all_ni))
                n_str = str(lo) if lo == hi else f"{lo}–{hi}"
            r = t_grps.rowCount()
            t_grps.insertRow(r)
            for ci, val in enumerate((grp_key, str(n_long), n_str)):
                item = QTableWidgetItem(val)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                t_grps.setItem(r, ci, item)
        t_grps.clearSelection()
        t_grps.blockSignals(False)
        self._select_var_grp_longs[role] = grp_to_longs

        # --- VAR/NI table ---
        # Collect all long var names then look them up in the manifest for NI values.
        all_longs = picker.selected_long_names()
        manifest = getattr(self, "_manifest_df", None)
        ni_lookup: dict = {}
        if manifest is not None and not manifest.empty and "VAR" in manifest.columns and "NI" in manifest.columns:
            sub = manifest[manifest["VAR"].isin(all_longs)]
            for _, row in sub.iterrows():
                ni_lookup[str(row["VAR"])] = str(row["NI"])
        self._select_var_data[role] = ni_lookup  # used by _on_grp_filter_changed

        n_bases = len({b for _, b in pairs})
        detail_lbl.setText(
            f"{n_bases} base{'s' if n_bases != 1 else ''} selected  ·  "
            f"{len(all_longs)} long var{'s' if len(all_longs) != 1 else ''}")

        self._fill_var_table_longs(t_vars, all_longs, ni_lookup)

    # ------------------------------------------------------------------
    # Observations (rows) filter tab
    # ------------------------------------------------------------------

    def _build_obs_filter_tab(self, container: QWidget):
        lay = QVBoxLayout(container)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        # Toolbar
        tb = QHBoxLayout()
        tb.setSpacing(6)
        add_btn = QPushButton("+ Add filter")
        add_btn.setToolTip("Add a new individual-level filter condition")
        add_btn.clicked.connect(self._obs_add_row)
        clear_btn = QPushButton("Clear all")
        clear_btn.setToolTip("Remove all filter conditions")
        clear_btn.clicked.connect(self._obs_clear_all)
        tb.addWidget(add_btn)
        tb.addWidget(clear_btn)
        tb.addStretch()
        lay.addLayout(tb)

        # Scroll area containing the filter rows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        self._obs_rows_lay = QVBoxLayout(scroll_content)
        self._obs_rows_lay.setContentsMargins(0, 0, 0, 0)
        self._obs_rows_lay.setSpacing(3)
        self._obs_rows_lay.addStretch()
        scroll.setWidget(scroll_content)
        lay.addWidget(scroll, 1)

        # Count / status label
        self._obs_count_lbl = QLabel("No filters — all individuals included.")
        self._obs_count_lbl.setStyleSheet("color:#aaa; font-size:11px;")
        lay.addWidget(self._obs_count_lbl)

    def _obs_var_items(self):
        """Sorted list of VAR names available for filtering."""
        manifest = getattr(self, "_manifest_df", None)
        if manifest is None or manifest.empty or "VAR" not in manifest.columns:
            return []
        return sorted(v for v in manifest["VAR"].dropna().astype(str).unique() if v != "ID")

    def _obs_add_row(self, var: str = "", op: str = "==", val: str = ""):
        row_w = QWidget()
        row_lay = QHBoxLayout(row_w)
        row_lay.setContentsMargins(0, 0, 0, 0)
        row_lay.setSpacing(4)

        var_combo = QComboBox()
        var_combo.setEditable(True)
        var_combo.setInsertPolicy(QComboBox.NoInsert)
        var_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        var_combo.setFixedWidth(180)
        var_combo.setMaxVisibleItems(20)
        _var_items = self._obs_var_items()
        var_combo.addItems(_var_items)
        if _var_items:
            _fm = var_combo.fontMetrics()
            _max_w = max(_fm.horizontalAdvance(t) for t in _var_items)
            var_combo.view().setMinimumWidth(_max_w + 36)
        if var:
            var_combo.setCurrentText(var)

        op_combo = QComboBox()
        op_combo.addItems(["==", "!=", ">=", "<=", ">", "<"])
        op_combo.setCurrentText(op)
        op_combo.setFixedWidth(52)

        val_combo = QComboBox()
        val_combo.setEditable(True)
        val_combo.setInsertPolicy(QComboBox.NoInsert)
        val_combo.setFixedWidth(110)
        if val:
            val_combo.setCurrentText(val)

        # Range label: "min – max" or "N vals" — shown after value combo
        range_lbl = QLabel("")
        range_lbl.setStyleSheet("color:#666; font-size:10px;")
        range_lbl.setFixedWidth(90)
        range_lbl.setToolTip("Raw (unnormalized) range from .dat")

        # Per-criterion count: how many individuals pass THIS condition alone
        crit_lbl = QLabel("")
        crit_lbl.setStyleSheet("color:#8b9dc3; font-size:10px;")
        crit_lbl.setFixedWidth(76)
        crit_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        crit_lbl.setToolTip("Individuals passing this filter condition alone")

        rm_btn = QPushButton("×")
        rm_btn.setFixedWidth(22)
        rm_btn.setFixedHeight(22)
        rm_btn.setStyleSheet(
            "QPushButton{color:#f87171;border:none;font-weight:bold;}"
            "QPushButton:hover{color:#ef4444;}")
        rm_btn.setToolTip("Remove this filter")

        row_lay.addWidget(var_combo)
        row_lay.addWidget(op_combo)
        row_lay.addWidget(val_combo)
        row_lay.addWidget(range_lbl)
        row_lay.addWidget(crit_lbl)
        row_lay.addWidget(rm_btn)

        row_w.var_combo  = var_combo
        row_w.op_combo   = op_combo
        row_w.val_combo  = val_combo
        row_w.range_lbl  = range_lbl
        row_w.crit_lbl   = crit_lbl

        # Insert before the trailing stretch
        idx = len(self._obs_row_widgets)
        self._obs_rows_lay.insertWidget(idx, row_w)
        self._obs_row_widgets.append(row_w)

        # Connections — closures keep strong refs (prevent PySide6 GC)
        def _make_slots(rw):
            def _on_var(_text):
                self._obs_on_var_changed(rw)
            def _on_change(_text=None):
                self._obs_update_row_crit(rw)
                self._obs_schedule_refresh()
            def _on_remove():
                self._obs_remove_row(rw)
            return _on_var, _on_change, _on_remove

        _sv, _sc, _sr = _make_slots(row_w)
        if not hasattr(self, "_obs_row_slots"):
            self._obs_row_slots = []
        self._obs_row_slots.extend([_sv, _sc, _sr])

        var_combo.currentTextChanged.connect(_sv)
        op_combo.currentTextChanged.connect(_sc)
        val_combo.currentTextChanged.connect(_sc)
        rm_btn.clicked.connect(_sr)

        if var:
            self._obs_on_var_changed(row_w)
        self._obs_schedule_refresh()

    def _obs_remove_row(self, row_w):
        if row_w in self._obs_row_widgets:
            self._obs_row_widgets.remove(row_w)
        self._obs_rows_lay.removeWidget(row_w)
        row_w.deleteLater()
        self._obs_schedule_refresh()

    def _obs_clear_all(self):
        for rw in list(self._obs_row_widgets):
            self._obs_rows_lay.removeWidget(rw)
            rw.deleteLater()
        self._obs_row_widgets.clear()
        self._obs_inc_ids = None
        self._obs_count_timer.stop()
        if hasattr(self, "_obs_count_lbl"):
            self._obs_count_lbl.setText("No filters — all individuals included.")
        self._update_selection_desc()

    def _obs_on_var_changed(self, row_w):
        var = row_w.var_combo.currentText().strip()
        row_w.val_combo.blockSignals(True)
        row_w.val_combo.clear()
        row_w.val_combo.blockSignals(False)
        self._obs_schedule_refresh()
        if not var:
            return
        # Check cache first
        if var in self._obs_col_cache:
            self._obs_populate_val_combo(row_w, self._obs_col_cache[var])
            return
        dat_path = getattr(self, "_dat_edit", None)
        dat_path = dat_path.text().strip() if dat_path else ""
        if not dat_path or not os.path.exists(dat_path):
            return
        fut = self.ctrl._exec.submit(self._obs_fetch_col_worker, dat_path, var)
        def _done(_f=fut, _rw=row_w, _var=var):
            try:
                col_series = _f.result()
                self._sig_obs_vals.emit((_rw, _var, col_series))
            except Exception:
                pass
        fut.add_done_callback(_done)

    def _obs_on_unique_vals(self, payload):
        row_w, var, col_series = payload
        self._obs_col_cache[var] = col_series
        if row_w in self._obs_row_widgets:
            self._obs_populate_val_combo(row_w, col_series)
            self._obs_update_row_crit(row_w)
        self._obs_schedule_refresh()

    def _obs_update_row_crit(self, row_w):
        """Compute how many individuals pass this single criterion; update crit_lbl."""
        var = row_w.var_combo.currentText().strip()
        op  = row_w.op_combo.currentText().strip()
        val = row_w.val_combo.currentText().strip()
        if not var or not val or var not in self._obs_col_cache:
            row_w.crit_lbl.setText("")
            return
        col = self._obs_col_cache[var]
        n_total = len(col.dropna())
        try:
            val_num = float(val)
            col_num = pd.to_numeric(col, errors="coerce").dropna()
            n_total = len(col_num)
            if   op == "==": n = int((col_num == val_num).sum())
            elif op == "!=": n = int((col_num != val_num).sum())
            elif op == ">=": n = int((col_num >= val_num).sum())
            elif op == "<=": n = int((col_num <= val_num).sum())
            elif op == ">":  n = int((col_num >  val_num).sum())
            elif op == "<":  n = int((col_num <  val_num).sum())
            else: n = n_total
        except (ValueError, TypeError):
            s_col = col.dropna().astype(str)
            n_total = len(s_col)
            if   op == "==": n = int((s_col == str(val)).sum())
            elif op == "!=": n = int((s_col != str(val)).sum())
            else: n = n_total
        row_w.crit_lbl.setText(f"{n:,}/{n_total:,}")

    @staticmethod
    def _obs_populate_val_combo(row_w, col_series):
        non_null = col_series.dropna()
        unique_vals = sorted(non_null.astype(str).unique().tolist())

        # Value dropdown — pre-fill when ≤ 10 unique values
        current = row_w.val_combo.currentText()
        row_w.val_combo.blockSignals(True)
        row_w.val_combo.clear()
        if len(unique_vals) <= 10:
            row_w.val_combo.addItems(unique_vals)
        row_w.val_combo.setCurrentText(current)
        row_w.val_combo.blockSignals(False)

        # Range label — numeric range or unique-count
        try:
            num = pd.to_numeric(non_null, errors="coerce").dropna()
            if len(num) > 0:
                lo, hi = num.min(), num.max()
                def _fmt(v):
                    return str(int(v)) if float(v) == int(v) else f"{v:.4g}"
                row_w.range_lbl.setText(f"{_fmt(lo)} – {_fmt(hi)}")
                row_w.range_lbl.setToolTip(
                    f"Raw values: min={_fmt(lo)}, max={_fmt(hi)}, N={len(num):,}")
            else:
                row_w.range_lbl.setText(f"{len(unique_vals)} vals")
        except Exception:
            row_w.range_lbl.setText(f"{len(unique_vals)} vals")

    @staticmethod
    def _obs_fetch_col_worker(dat_path, var_name):
        """Return a Series of raw (unnormalized) values for *var_name* keyed by ID."""
        from lunapi import gpa_dump
        df = gpa_dump(dat_path, lvars=var_name, qc="F")
        if "ID" not in df.columns or var_name not in df.columns:
            return pd.Series(dtype=object)
        return df.set_index("ID")[var_name]

    def _obs_schedule_refresh(self):
        self._obs_count_timer.start()

    def _obs_refresh_count(self):
        filters = self._obs_collect_filters()
        if not filters:
            self._obs_inc_ids = None
            if hasattr(self, "_obs_count_lbl"):
                self._obs_count_lbl.setText("No filters — all individuals included.")
            self._update_selection_desc()
            return
        dat_path = getattr(self, "_dat_edit", None)
        dat_path = dat_path.text().strip() if dat_path else ""
        if not dat_path or not os.path.exists(dat_path):
            if hasattr(self, "_obs_count_lbl"):
                self._obs_count_lbl.setText("Load a .dat file to evaluate filters.")
            return
        if hasattr(self, "_obs_count_lbl"):
            self._obs_count_lbl.setText("Evaluating…")
        # Pass a snapshot of the cache so the worker doesn't touch Qt objects
        cache_snap = {k: v.copy() for k, v in self._obs_col_cache.items()}
        fut = self.ctrl._exec.submit(
            self._obs_eval_worker, dat_path, filters, cache_snap)
        def _done(_f=fut):
            try:
                self._sig_obs_count.emit(_f.result())
            except Exception as exc:
                self._sig_obs_count.emit((None, 0, 0, str(exc)))
        fut.add_done_callback(_done)

    def _obs_on_count_result(self, payload):
        ids, n_match, n_total, err = payload
        if err:
            self._obs_inc_ids = None
            if hasattr(self, "_obs_count_lbl"):
                self._obs_count_lbl.setText(f"Error: {err[:120]}")
            return
        # Cache any newly fetched columns
        if isinstance(ids, dict) and "cache" in ids:
            for k, v in ids["cache"].items():
                if k not in self._obs_col_cache:
                    self._obs_col_cache[k] = v
            ids = ids["ids"]
        self._obs_inc_ids = ids  # None = all; list = subset
        if ids is None:
            txt = "No filters — all individuals included."
        else:
            pct = int(100 * n_match / n_total) if n_total else 0
            txt = (f"N = {n_match:,} / {n_total:,} individuals match  ({pct}%)")
        if hasattr(self, "_obs_count_lbl"):
            self._obs_count_lbl.setText(txt)
        self._update_selection_desc()

    @staticmethod
    def _obs_eval_worker(dat_path, filters, cache):
        """Evaluate AND-combined filters; return (result_dict, n_match, n_total, err)."""
        from lunapi import gpa_dump
        needed = [f["var"] for f in filters if f["var"] not in cache]
        new_cache = {}
        try:
            if needed:
                df_new = gpa_dump(dat_path, lvars=",".join(dict.fromkeys(needed)), qc="F")
                if "ID" in df_new.columns:
                    for v in needed:
                        if v in df_new.columns:
                            new_cache[v] = df_new.set_index("ID")[v]
        except Exception as exc:
            return (None, 0, 0, str(exc))

        merged_cache = {**cache, **new_cache}

        # Build a combined DataFrame from the cache, aligned on ID
        all_ids = None
        for f in filters:
            var = f["var"]
            if var not in merged_cache:
                continue
            s = merged_cache[var]
            if all_ids is None:
                all_ids = s.index.tolist()
            else:
                all_ids = [i for i in all_ids if i in s.index]

        if all_ids is None:
            return (None, 0, 0, None)

        n_total = len(all_ids)
        mask = pd.Series(True, index=all_ids)

        for f in filters:
            var, op, val = f["var"], f["op"], f["val"]
            if not val or var not in merged_cache:
                continue
            col = merged_cache[var].reindex(all_ids)
            try:
                val_num = float(val)
                col_num = pd.to_numeric(col, errors="coerce")
                if   op == "==": mask &= (col_num == val_num)
                elif op == "!=": mask &= (col_num != val_num)
                elif op == ">=": mask &= (col_num >= val_num)
                elif op == "<=": mask &= (col_num <= val_num)
                elif op == ">":  mask &= (col_num >  val_num)
                elif op == "<":  mask &= (col_num <  val_num)
            except (ValueError, TypeError):
                s_col = col.astype(str)
                if   op == "==": mask &= (s_col == str(val))
                elif op == "!=": mask &= (s_col != str(val))

        matching = [i for i, m in mask.items() if m]
        result = {"ids": matching, "cache": new_cache}
        return (result, len(matching), n_total, None)

    def _obs_collect_filters(self):
        out = []
        for rw in self._obs_row_widgets:
            var = rw.var_combo.currentText().strip()
            op  = rw.op_combo.currentText().strip()
            val = rw.val_combo.currentText().strip()
            if var and val:
                out.append({"var": var, "op": op, "val": val})
        return out

    def _obs_clear_cache(self):
        self._obs_col_cache.clear()

    def _obs_summary_str(self):
        """Short string for the outer count bar, or '' when no filter active."""
        if not self._obs_row_widgets or self._obs_inc_ids is None:
            return ""
        n = len(self._obs_inc_ids)
        return f"obs: {n:,}"

    def _build_assoc_explore_tab(self):
        w = QWidget()
        w.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)

        self._assoc_standard_frame = QFrame()
        std_lay = QVBoxLayout(self._assoc_standard_frame)
        std_lay.setContentsMargins(0, 0, 0, 0)
        std_lay.setSpacing(6)
        self._assoc_dat_label = QLabel("")
        self._assoc_dat_label.setWordWrap(True)
        self._assoc_dat_label.setStyleSheet(f"color:#888; font-size:11px;")
        std_lay.addWidget(self._assoc_dat_label)
        self._update_assoc_dat_label()

        self._assoc_seed_picker = _VarPicker("Seed variable")
        self._assoc_seed_picker.set_single_selection(True)
        self._assoc_seed_picker.setMinimumHeight(120)
        self._assoc_seed_picker.setMaximumHeight(150)
        self._assoc_seed_picker.selectionChanged.connect(self._on_assoc_seed_picker_changed)
        std_lay.addWidget(self._assoc_seed_picker)

        self._assoc_seed_long_label = QLabel("Actual seed variable")
        std_lay.addWidget(self._assoc_seed_long_label)
        self._assoc_seed_long_list = QListWidget()
        self._assoc_seed_long_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._assoc_seed_long_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._assoc_seed_long_list.setTextElideMode(Qt.ElideRight)
        self._assoc_seed_long_list.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._assoc_seed_long_list.setMinimumHeight(72)
        self._assoc_seed_long_list.setMaximumHeight(96)
        self._assoc_seed_long_list.setStyleSheet(
            "QListWidget { background:#0d1117; border:1px solid #21262d; font-size:11px; }"
            "QListWidget::item { padding:2px 4px; }"
        )
        self._assoc_seed_long_list.currentRowChanged.connect(self._on_assoc_seed_long_row_changed)
        std_lay.addWidget(self._assoc_seed_long_list)

        self._assoc_pool_picker = _VarPicker("Target variables")
        self._assoc_pool_picker.setMinimumHeight(150)
        self._assoc_pool_picker.setMaximumHeight(190)
        self._assoc_pool_picker.set_summary(
            "Empty selection = all variables in the manifest."
        )
        std_lay.addWidget(self._assoc_pool_picker)
        lay.addWidget(self._assoc_standard_frame)

        sep_targets = QFrame()
        sep_targets.setFrameShape(QFrame.HLine)
        sep_targets.setFrameShadow(QFrame.Plain)
        sep_targets.setStyleSheet(f"color:{SEP};")
        lay.addWidget(sep_targets)

        corr_frame = QFrame()
        corr_frame.setFrameShape(QFrame.StyledPanel)
        corr_lay = QVBoxLayout(corr_frame)
        corr_lay.setContentsMargins(6, 4, 6, 4)
        corr_lay.setSpacing(4)

        corr_opts = QHBoxLayout()
        corr_opts.addWidget(QLabel("|r| ≥"))
        self._assoc_abs_spin = QDoubleSpinBox()
        self._assoc_abs_spin.setRange(0.0, 1.0)
        self._assoc_abs_spin.setDecimals(3)
        self._assoc_abs_spin.setSingleStep(0.05)
        self._assoc_abs_spin.setValue(0.0)
        self._assoc_abs_spin.setFixedWidth(70)
        corr_opts.addWidget(self._assoc_abs_spin)
        corr_opts.addWidget(QLabel("Top"))
        self._assoc_topn_spin = QSpinBox()
        self._assoc_topn_spin.setRange(10, 5000)
        self._assoc_topn_spin.setValue(200)
        self._assoc_topn_spin.setFixedWidth(72)
        corr_opts.addWidget(self._assoc_topn_spin)
        corr_opts.addStretch(1)
        corr_lay.addLayout(corr_opts)

        corr_run = QHBoxLayout()
        self._assoc_run_btn = QPushButton("Rank Correlations")
        self._assoc_run_btn.setStyleSheet(
            "QPushButton { background:#1e3a5f; color:#fff; padding:4px 12px; border-radius:4px; }"
            "QPushButton:hover { background:#1d4ed8; }"
        )
        corr_run.addWidget(self._assoc_run_btn)
        corr_run.addStretch(1)
        corr_lay.addLayout(corr_run)
        self._assoc_run_btn.clicked.connect(self._run_assoc_correlations)
        self._assoc_abs_spin.valueChanged.connect(
            lambda *_: self._apply_assoc_corr_filter(self._assoc_corr_filter.text())
        )
        self._assoc_topn_spin.valueChanged.connect(
            lambda *_: self._apply_assoc_corr_filter(self._assoc_corr_filter.text())
        )

        lay.addWidget(corr_frame)

        self._assoc_status = QLabel("")
        self._assoc_status.setWordWrap(True)
        self._assoc_status.setStyleSheet(f"color:#888; font-size:11px;")
        lay.addWidget(self._assoc_status)

        lay.addStretch(1)
        return w

    def _build_assoc_results_panel(self):
        from .mplcanvas import MplCanvas

        frame = QFrame()
        frame.setFrameShape(QFrame.NoFrame)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(3)

        corr_tab = QWidget()
        corr_lay = QVBoxLayout(corr_tab)
        corr_lay.setContentsMargins(0, 0, 0, 0)
        corr_lay.setSpacing(3)

        corr_hdr = QHBoxLayout()
        self._assoc_corr_filter = QLineEdit()
        self._assoc_corr_filter.setPlaceholderText("filter correlations…")
        self._assoc_corr_filter.setClearButtonEnabled(True)
        self._assoc_corr_filter.setFixedWidth(180)
        self._assoc_ranked_btn = QPushButton("Ranked")
        self._assoc_scatter_btn = QPushButton("Scatter")
        for btn in (self._assoc_ranked_btn, self._assoc_scatter_btn):
            btn.setCheckable(True)
            btn.setFixedWidth(72)
            btn.setStyleSheet(
                "QPushButton { padding:2px 6px; font-size:10px; border:1px solid #333; }"
                "QPushButton:checked { background:#1e3a5f; color:#fff; border-color:#4cc9f0; }"
            )
        self._assoc_ranked_btn.setChecked(True)
        btn_export_corr = QPushButton("Export…")
        btn_export_corr.setFixedWidth(70)
        corr_hdr.addWidget(QLabel("Correlations"))
        corr_hdr.addStretch(1)
        corr_hdr.addWidget(self._assoc_ranked_btn)
        corr_hdr.addWidget(self._assoc_scatter_btn)
        corr_hdr.addWidget(self._assoc_corr_filter)
        corr_hdr.addWidget(btn_export_corr)
        corr_lay.addLayout(corr_hdr)

        corr_split = QSplitter(Qt.Vertical)
        corr_split.setHandleWidth(4)
        self._assoc_corr_view = QTableView()
        self._assoc_corr_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._assoc_corr_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._assoc_corr_view.setSortingEnabled(True)
        self._assoc_corr_view.horizontalHeader().setStretchLastSection(True)
        self._assoc_corr_view.verticalHeader().setVisible(False)
        self._assoc_corr_view.setAlternatingRowColors(True)
        corr_split.addWidget(self._assoc_corr_view)

        self._assoc_corr_canvas = MplCanvas()
        self._assoc_corr_canvas.figure.patch.set_facecolor(BG)
        corr_split.addWidget(self._assoc_corr_canvas)
        corr_split.setSizes([230, 320])
        corr_lay.addWidget(corr_split, 1)
        lay.addWidget(corr_tab, 1)

        btn_export_corr.clicked.connect(lambda: save_table_as_tsv(self._assoc_corr_view, self))
        self._assoc_corr_filter.textChanged.connect(self._apply_assoc_corr_filter)
        self._assoc_corr_view.clicked.connect(self._on_assoc_corr_row_clicked)
        self._assoc_ranked_btn.clicked.connect(self._show_assoc_ranked_plot)
        self._assoc_scatter_btn.clicked.connect(self._show_assoc_scatter_plot)
        self._assoc_corr_canvas.mpl_connect("motion_notify_event", self._on_assoc_ranked_hover)

        return frame

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
        btn_export_r = QPushButton("Export…"); btn_export_r.setFixedWidth(70)
        hdr.addWidget(lbl)
        hdr.addStretch(1)
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
        self._summary_btn         = QPushButton("Summary")
        self._scatter_raw_btn     = QPushButton("Raw")
        self._scatter_partial_btn = QPushButton("Partial")
        self._joint_mode_btn      = QPushButton("Joint")
        for btn in (self._summary_btn, self._scatter_raw_btn, self._scatter_partial_btn, self._joint_mode_btn):
            btn.setCheckable(True)
            btn.setFixedWidth(64)
            btn.setStyleSheet(
                "QPushButton { padding:2px 6px; font-size:10px; border:1px solid #333; }"
                "QPushButton:checked { background:#1e3a5f; color:#fff; border-color:#4cc9f0; }")
        self._summary_btn.setChecked(True)
        self._joint_mode_btn.setEnabled(False)
        toggle_row.addWidget(QLabel("View:"))
        toggle_row.addWidget(self._summary_btn)
        toggle_row.addWidget(self._scatter_raw_btn)
        toggle_row.addWidget(self._scatter_partial_btn)
        toggle_row.addWidget(self._joint_mode_btn)
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
        self._joint_host = self._build_joint_panel()
        self._viz_stack = QStackedWidget()
        self._viz_stack.addWidget(canvas_host)
        self._viz_stack.addWidget(self._joint_host)
        canvas_outer_lay.addWidget(self._viz_stack, 1)

        rsplit.addWidget(self._results_view)
        rsplit.addWidget(canvas_outer)
        rsplit.setSizes([200, 300])

        lay.addWidget(rsplit, 1)

        btn_export_r.clicked.connect(
            lambda: save_table_as_tsv(self._results_view, self))
        self._results_filter.textChanged.connect(self._apply_results_filter)
        self._results_view.clicked.connect(self._on_result_row_clicked)
        self._summary_btn.clicked.connect(self._on_summary_clicked)
        self._scatter_raw_btn.clicked.connect(
            lambda: self._on_scatter_toggle(partial=False))
        self._scatter_partial_btn.clicked.connect(
            lambda: self._on_scatter_toggle(partial=True))
        self._joint_mode_btn.clicked.connect(self._on_joint_mode_toggled)

        return frame

    def _build_joint_panel(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.NoFrame)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        self._joint_status_lbl = QLabel("Joint model unavailable.")
        self._joint_status_lbl.setWordWrap(True)
        self._joint_status_lbl.setStyleSheet(f"color:{FG}; font-size:11px;")
        lay.addWidget(self._joint_status_lbl)

        action_row = QHBoxLayout()
        self._joint_add_btn = QPushButton("Add selected Y")
        self._joint_remove_btn = QPushButton("Remove selected Y")
        self._joint_clear_btn = QPushButton("Clear Y")
        self._joint_export_btn = QPushButton("Export…")
        for btn in (self._joint_add_btn, self._joint_remove_btn, self._joint_clear_btn, self._joint_export_btn):
            btn.setFixedHeight(24)
        action_row.addWidget(self._joint_add_btn)
        action_row.addWidget(self._joint_remove_btn)
        action_row.addWidget(self._joint_clear_btn)
        action_row.addStretch(1)
        action_row.addWidget(self._joint_export_btn)
        lay.addLayout(action_row)

        mid = QSplitter(Qt.Horizontal)
        mid.setHandleWidth(4)

        y_frame = QFrame()
        y_lay = QVBoxLayout(y_frame)
        y_lay.setContentsMargins(0, 0, 0, 0)
        y_lay.setSpacing(3)
        y_lay.addWidget(QLabel("Active Y Predictors"))
        self._joint_y_list = QListWidget()
        self._joint_y_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        y_lay.addWidget(self._joint_y_list, 1)

        coef_frame = QFrame()
        coef_lay = QVBoxLayout(coef_frame)
        coef_lay.setContentsMargins(0, 0, 0, 0)
        coef_lay.setSpacing(3)
        coef_lay.addWidget(QLabel("Coefficients"))
        self._joint_table = QTableView()
        self._joint_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._joint_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._joint_table.setAlternatingRowColors(True)
        self._joint_table.verticalHeader().setVisible(False)
        self._joint_table.horizontalHeader().setStretchLastSection(True)
        coef_lay.addWidget(self._joint_table, 1)

        mid.addWidget(y_frame)
        mid.addWidget(coef_frame)
        mid.setSizes([220, 520])
        lay.addWidget(mid, 1)

        self._joint_add_btn.clicked.connect(self._joint_add_selected_y)
        self._joint_remove_btn.clicked.connect(self._joint_remove_selected_y)
        self._joint_clear_btn.clicked.connect(self._joint_clear_y)
        self._joint_export_btn.clicked.connect(
            lambda: save_table_as_tsv(self._joint_table, self))
        self._joint_y_list.itemSelectionChanged.connect(self._update_joint_action_buttons)

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
        short = item.text()
        self._col_file_lbl.setText(f"Columns: {short}")
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
            # Preview
            preview = ", ".join(df[col].dropna().astype(str).head(3).tolist())
            prev_item = QTableWidgetItem(preview)
            prev_item.setFlags(prev_item.flags() & ~Qt.ItemIsEditable)
            prev_item.setForeground(QtGui.QColor("#666688"))
            self._col_table.setItem(row, 2, prev_item)
            # Color the name by role
            self._color_row(row, role)
        self._col_table.blockSignals(False)
        self._col_table.resizeColumnToContents(0)

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
                    src_base = os.path.splitext(os.path.basename(path))[0]
                    file_path = os.path.join(wd, f"{src_base}__{member}.tsv")
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
            self._set_dat_path(fn)

    def _set_dat_path(self, path):
        path = (path or "").strip()
        if hasattr(self, "_dat_edit"):
            self._dat_edit.setText(path)
        self._update_assoc_dat_label()

    def _update_assoc_dat_label(self):
        label = getattr(self, "_assoc_dat_label", None)
        if label is None:
            return
        dat_path = self._dat_edit.text().strip() if hasattr(self, "_dat_edit") else ""
        if dat_path:
            label.setText(f"Using current .dat: {dat_path}")
        else:
            label.setText("Using current .dat: none loaded")

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
                src_base = os.path.splitext(os.path.basename(path))[0]
                dest = os.path.join(wd, f"{src_base}__{member}.tsv")
                pkl_members.append((path, member, dest))
            elif ext_i == ".db":
                src_base = os.path.splitext(os.path.basename(path))[0]
                dest = os.path.join(wd, f"{src_base}__{member}.tsv")
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
                        df.to_csv(dest, sep="\t", index=False, na_rep=".", encoding="utf-8")
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
                            df.to_csv(
                                key_to_dest[k],
                                sep="\t",
                                index=False,
                                na_rep=".",
                                encoding="utf-8",
                            )
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
        self._run_load_manifest_for(self._dat_edit.text().strip())

    def _run_load_manifest_for(self, dat_path):
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
        if hasattr(self, "_assoc_status"):
            self._assoc_status.setText("Loading manifest…")
        self._set_dat_path(dat_path)

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
        cols = list(df.columns)
        numeric_cols = {i for i, c in enumerate(cols) if c in ("NV", "NI")}
        model = QStandardItemModel(len(df), len(cols), self)
        model.setHorizontalHeaderLabels(cols)
        for r, row in df.iterrows():
            for c, val in enumerate(row):
                item = QStandardItem(str(val))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if c in numeric_cols:
                    try:
                        item.setData(float(val), Qt.UserRole)
                    except (TypeError, ValueError):
                        pass
                model.setItem(int(r), c, item)
        proxy = _GpaResultsSortProxy(self)
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

    def _active_assoc_manifest_df(self):
        return self._manifest_df

    def _on_assoc_mode_changed(self, *_args):
        self._assoc_standard_frame.setVisible(True)
        self._assoc_seed_picker.set_title("Seed variable")
        self._assoc_seed_long_label.setText("Actual seed variable")
        self._assoc_seed_long_label.setVisible(True)
        self._assoc_seed_long_list.setVisible(True)
        self._assoc_seed_picker.populate(self._manifest_df)
        self._assoc_pool_picker.populate(self._manifest_df)
        self._refresh_assoc_seed_long_list(preferred=self._assoc_seed_long)
        self._clear_assoc_matrix_cache()
        self._assoc_corr_df = pd.DataFrame()
        self._assoc_pca_df = pd.DataFrame()
        self._assoc_pca_result = None
        self._populate_assoc_corr_table(pd.DataFrame())
        self._populate_assoc_pca_table(pd.DataFrame())
        self._populate_assoc_pca_color_combo()

    def _selected_list_values(self, widget):
        return [item.text() for item in widget.selectedItems()]

    def _assoc_candidate_long_names(self):
        selected = self._assoc_pool_picker.selected_long_names()
        if selected:
            return selected
        return self._manifest_long_names()

    def _manifest_long_names(self):
        meta = _manifest_var_frame(self._active_assoc_manifest_df())
        if meta.empty:
            return []
        return meta["VAR"].astype(str).tolist()

    def _manifest_meta_for_vars(self, long_vars=None):
        meta = _manifest_var_frame(self._active_assoc_manifest_df())
        if meta.empty or not long_vars:
            return meta
        return meta[meta["VAR"].isin(set(long_vars))].copy()

    def _populate_assoc_pca_color_combo(self):
        combo = getattr(self, "_assoc_pca_color_combo", None)
        if combo is None:
            return
        combo.blockSignals(True)
        current = str(combo.currentData() or "")
        combo.clear()
        combo.addItem("(none)", "none")
        combo.addItem("group", "group")
        combo.addItem("base var", "base")
        for col in _assoc_pca_color_fields(self._active_assoc_manifest_df()):
            combo.addItem(col, col)
        idx = 0
        for i in range(combo.count()):
            if str(combo.itemData(i) or "") == current:
                idx = i
                break
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _refresh_assoc_seed_long_list(self, preferred=""):
        self._assoc_seed_long_list.blockSignals(True)
        self._assoc_seed_long_list.clear()
        longs = self._assoc_seed_picker.selected_long_names()
        for name in longs:
            self._assoc_seed_long_list.addItem(name)
        idx = 0
        if preferred:
            for i in range(self._assoc_seed_long_list.count()):
                item = self._assoc_seed_long_list.item(i)
                if item is not None and item.text() == preferred:
                    idx = i
                    break
        if self._assoc_seed_long_list.count():
            self._assoc_seed_long_list.setCurrentRow(idx)
            item = self._assoc_seed_long_list.currentItem()
            self._assoc_seed_long = item.text() if item is not None else ""
        else:
            self._assoc_seed_long = ""
        self._assoc_seed_long_list.blockSignals(False)

    def _on_assoc_seed_picker_changed(self):
        self._refresh_assoc_seed_long_list()
        self._maybe_auto_run_assoc_corr()

    def _on_assoc_seed_long_row_changed(self, row):
        if row < 0:
            self._assoc_seed_long = ""
            return
        item = self._assoc_seed_long_list.item(row)
        self._assoc_seed_long = item.text() if item is not None else ""
        self._maybe_auto_run_assoc_corr()

    def _maybe_auto_run_assoc_corr(self):
        if self._assoc_suspend_auto_run:
            return
        if getattr(self.ctrl, "_busy", False):
            return
        dat_path = self._dat_edit.text().strip()
        if not dat_path or not os.path.exists(dat_path):
            return
        if not self._assoc_seed_long:
            return
        self._run_assoc_correlations()

    def _set_assoc_seed_variable(self, long_var: str, auto_run: bool = False):
        long_var = str(long_var or "").strip()
        manifest_df = self._active_assoc_manifest_df()
        if not long_var or manifest_df is None or manifest_df.empty:
            return
        meta = _manifest_var_frame(manifest_df)
        row = meta[meta["VAR"] == long_var]
        if row.empty:
            return
        base = str(row.iloc[0]["BASE"])
        self._assoc_suspend_auto_run = True
        try:
            self._assoc_seed_picker.set_selected([base])
            self._refresh_assoc_seed_long_list(preferred=long_var)
            self._assoc_seed_long = long_var
        finally:
            self._assoc_suspend_auto_run = False
        if auto_run:
            self._run_assoc_correlations()

    def _collect_assoc_dump_filters(self):
        return self._joint_dump_filters()

    def _clear_assoc_matrix_cache(self):
        self._assoc_matrix_cache_key = None
        self._assoc_matrix_cache_df = None

    def _assoc_matrix_cache_payload(self, dat_path, requested_cols):
        dump_filters = self._collect_assoc_dump_filters()
        cache_key = _assoc_dump_cache_key(dat_path, requested_cols, dump_filters)
        if self._assoc_matrix_cache_key == cache_key and self._assoc_matrix_cache_df is not None:
            return {
                "cache_key": cache_key,
                "df": self._assoc_matrix_cache_df,
                "from_cache": True,
                "filters": dump_filters,
            }
        return {
            "cache_key": cache_key,
            "df": None,
            "from_cache": False,
            "filters": dump_filters,
        }

    def _store_assoc_matrix_cache(self, cache_key, df):
        self._assoc_matrix_cache_key = cache_key
        self._assoc_matrix_cache_df = df.copy() if df is not None else None

    @staticmethod
    def _assoc_matrix_worker(dat_path, candidates, dump_filters):
        from lunapi import gpa_dump

        cols = _unique_preserve(list(candidates))
        dump_opts = dict(dump_filters or {})
        dump_opts["lvars"] = ",".join(cols)
        return gpa_dump(dat_path, **dump_opts)

    def _run_assoc_correlations(self):
        seed_var = str(self._assoc_seed_long or "").strip()
        if not seed_var:
            QtWidgets.QMessageBox.warning(self._root, "Assoc", "Select one seed variable first.")
            return
        candidates = self._assoc_candidate_long_names()
        if not candidates:
            QtWidgets.QMessageBox.warning(self._root, "Assoc", "No candidate variables available.")
            return
        if not self._start_work("Ranking correlations…"):
            return
        self._assoc_status.setText(f"Ranking correlations for {seed_var}…")
        meta = self._manifest_meta_for_vars(_unique_preserve([seed_var] + candidates))
        requested_cols = _unique_preserve([seed_var] + candidates)
        dat_path = self._dat_edit.text().strip()
        if not dat_path or not os.path.exists(dat_path):
            self._end_work()
            QtWidgets.QMessageBox.warning(self._root, "Assoc", "Load a valid .dat file first.")
            return
        self._set_dat_path(dat_path)
        cache = self._assoc_matrix_cache_payload(dat_path, requested_cols)
        if cache["from_cache"]:
            fut = self.ctrl._exec.submit(
                self._assoc_corr_from_df_worker,
                cache["df"],
                seed_var,
                candidates,
                meta,
            )
        else:
            fut = self.ctrl._exec.submit(
                self._assoc_corr_worker,
                dat_path, seed_var, candidates, cache["filters"], meta, cache["cache_key"],
            )
        def _done(_f=fut):
            try:
                self._sig_ok.emit({"type": "assoc_corr", "result": _f.result()})
            except Exception:
                self._sig_err.emit({"type": "assoc", "traceback": traceback.format_exc()})
        fut.add_done_callback(_done)

    @staticmethod
    def _assoc_corr_from_df_worker(raw_df, seed_var, candidates, meta_df):
        cols = _unique_preserve([seed_var] + list(candidates))
        present, missing = _present_columns(raw_df, cols)
        if seed_var not in present:
            raise ValueError(f"Seed variable was not returned from .dat: {seed_var}")
        meta_use = meta_df[meta_df["VAR"].isin(present)].copy() if meta_df is not None and not meta_df.empty else meta_df
        return {
            "seed": seed_var,
            "table": _rank_seed_correlations(raw_df, seed_var, present, meta_df=meta_use),
            "row_count": int(len(raw_df)),
            "missing": missing,
        }

    @staticmethod
    def _assoc_corr_worker(dat_path, seed_var, candidates, dump_filters, meta_df, cache_key):
        raw_df = GPATab._assoc_matrix_worker(dat_path, _unique_preserve([seed_var] + list(candidates)), dump_filters)
        out = GPATab._assoc_corr_from_df_worker(raw_df, seed_var, candidates, meta_df)
        out["cache_key"] = cache_key
        out["raw_df"] = raw_df
        return out

    def _run_assoc_pca(self):
        QtWidgets.QMessageBox.information(
            self._root, "Assoc", "PCA is currently hidden in this explorer."
        )

    @staticmethod
    def _assoc_pca_from_df_worker(raw_df, candidates, meta_df, min_prop, row_mode, standardize, point_mode="variable"):
        cols = _unique_preserve([str(col) for col in candidates if str(col).strip()])
        present, missing = _present_columns(raw_df, cols)
        if point_mode == "observation":
            result = _fit_observation_pca(
                raw_df.copy(),
                present,
                min_col_prop=min_prop,
                row_mode=row_mode,
                standardize=standardize,
            )
            result["loadings"] = result.pop("scores", pd.DataFrame())
            result["point_kind"] = "observation"
            result["point_label_col"] = "RID" if "RID" in result["loadings"].columns else ("ID" if "ID" in result["loadings"].columns else "")
        else:
            meta_use = meta_df[meta_df["VAR"].isin(present)].copy() if meta_df is not None and not meta_df.empty else meta_df
            result = _fit_variable_pca(
                raw_df[present].copy(),
                meta_df=meta_use,
                min_col_prop=min_prop,
                row_mode=row_mode,
                standardize=standardize,
            )
            result["point_kind"] = "variable"
            result["point_label_col"] = "VAR"
        result["missing_dump_cols"] = missing
        return result

    @staticmethod
    def _assoc_pca_worker(dat_path, candidates, dump_filters, meta_df, min_prop, row_mode, standardize, cache_key, point_mode="variable"):
        raw_df = GPATab._assoc_matrix_worker(dat_path, candidates, dump_filters)
        result = GPATab._assoc_pca_from_df_worker(raw_df, candidates, meta_df, min_prop, row_mode, standardize, point_mode=point_mode)
        result["cache_key"] = cache_key
        result["raw_df"] = raw_df
        return result

    # ======================================================================
    # Analyze tab — run GPA
    # ======================================================================

    def _save_gpa_dump_tsv(self):
        """Export selected X/Y/Z variables as a flat TSV via GPA dump mode."""
        dat_path = self._dat_edit.text().strip()
        if not dat_path or not os.path.exists(dat_path):
            QtWidgets.QMessageBox.warning(
                self._root, "GPA", "Load a manifest first (valid .dat path).")
            return

        x_vars = self._picker_x.selected_long_names()
        y_vars = self._picker_y.selected_long_names()
        z_vars = self._picker_z.selected_long_names()
        all_vars = list(dict.fromkeys(x_vars + y_vars + z_vars))
        if not all_vars:
            QtWidgets.QMessageBox.warning(
                self._root, "GPA", "Select at least one variable in X, Y, or Z first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self._root, "Save GPA dump as TSV", "", "TSV files (*.tsv);;All files (*)")
        if not path:
            return

        opts = _with_dump_qc_disabled({"dat": dat_path, "dump": "", "lvars": ",".join(all_vars)})
        win = self._winsor_edit.text().strip()
        if win:
            try:
                win_f = float(win)
                if not (0.0 <= win_f <= 0.2):
                    raise ValueError(f"winsor must be between 0 and 0.2 (got {win_f})")
                opts["winsor"] = str(win_f)
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self._root, "GPA – winsor", str(exc))
                return

        if not self._start_work("Dumping GPA data…"):
            return

        fut = self.ctrl._exec.submit(self._gpa_dump_worker, opts)
        def _done(_f=fut, _path=path):
            try:
                self._sig_ok.emit({"type": "dump_tsv", "result": _f.result(), "path": _path})
            except Exception:
                self._sig_err.emit(traceback.format_exc())
        fut.add_done_callback(_done)

    @staticmethod
    def _gpa_dump_worker(opts):
        import lunapi.lunapi0 as _l0
        eng = _l0.inaugurate()
        _, stdout = eng.run_gpa(opts, False)
        return stdout

    def _run_gpa(self):
        self._obs_clear_cache()   # fresh data each run
        dat_path = self._dat_edit.text().strip()
        if not dat_path or not os.path.exists(dat_path):
            QtWidgets.QMessageBox.warning(
                self._root, "GPA", "Load a manifest first (valid .dat path).")
            return

        x_vars = self._picker_x.selected_long_names()
        y_vars = self._picker_y.selected_long_names()
        z_vars = self._picker_z.selected_long_names()

        overlap_info = _normalize_gpa_role_overlap(x_vars, y_vars, z_vars)
        if not overlap_info.get("ok"):
            overlaps = overlap_info.get("error_lines") or []
            self._clear_results_display()
            self._analyze_status.setText("Invalid selection: overlapping X/Z variables.")
            self._picker_x.set_summary(overlaps[0] if overlaps else "")
            self._picker_y.set_summary("")
            self._picker_z.set_summary(overlaps[0] if overlaps else "")
            QtWidgets.QMessageBox.warning(
                self._root, "GPA",
                "A variable cannot be selected in both X and Z.\n\n"
                + "\n".join(overlaps)
            )
            return

        x_vars = overlap_info["x_vars"]
        y_vars = overlap_info["y_vars"]
        z_vars = overlap_info["z_vars"]
        x_bases = self._selection_base_count(x_vars)
        y_bases = self._selection_base_count(y_vars)
        z_bases = self._selection_base_count(z_vars)
        self._last_gpa_z = z_vars  # list for partial scatter

        if x_bases == 0 or y_bases == 0:
            QtWidgets.QMessageBox.warning(
                self._root, "GPA",
                "Association mode requires at least one X predictor and one Y outcome.")
            return

        warning_lines = overlap_info.get("warning_lines") or []
        if warning_lines:
            self._picker_y.set_summary("Dropped overlap from Y for this run.")
            self._analyze_status.setText("Dropping overlapping Y variables, then running…")
            QtWidgets.QMessageBox.warning(
                self._root, "GPA",
                "Some overlapping variables were dropped from Y for this run.\n\n"
                + "\n".join(warning_lines)
            )

        request = self._collect_gpa_request(x_vars, y_vars, z_vars)
        if request is None:
            return
        self._last_gpa_request = {
            "x_count": x_bases,
            "y_count": y_bases,
            "z_count": z_bases,
            "x_vars": list(x_vars),
            "y_vars": list(y_vars),
            "z_vars": list(z_vars),
            "x_long_count": len(x_vars),
            "y_long_count": len(y_vars),
            "z_long_count": len(z_vars),
            "n_prop": request["meta"].get("n_prop"),
            "n_req": request["meta"].get("n_req"),
            "x_n": self._selection_n_summary(x_vars),
            "y_n": self._selection_n_summary(y_vars),
            "z_n": self._selection_n_summary(z_vars),
            "opts": dict(request["opts"]),
            "selection_warning": " | ".join(warning_lines) if warning_lines else "",
        }
        if not self._start_work("Running GPA…"):
            return
        self._clear_results_display()
        bits = []
        bits.append(f"{self._last_gpa_request['x_count']} X")
        bits.append(f"{self._last_gpa_request['y_count']} Y")
        if self._last_gpa_request["z_count"]:
            bits.append(f"{self._last_gpa_request['z_count']} Z")
        for key in ("x_long_count", "y_long_count", "z_long_count"):
            val = self._last_gpa_request.get(key, 0)
            if val:
                bits.append(f"{key[0].upper()}long={val}")
        for key in ("x_n", "y_n", "z_n"):
            val = self._last_gpa_request.get(key)
            if val:
                bits.append(f"{key[0].upper()} {val}")
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
        opts = {"dat": self._dat_edit.text().strip()}
        if x_longs:
            opts["X"] = ",".join(x_longs)
        if z_longs:
            opts["Z"] = ",".join(z_longs)
        if y_longs:
            opts["lvars"] = ",".join(y_longs)
        win = self._winsor_edit.text().strip()
        if win:
            try:
                win_f = float(win)
                if not (0.0 <= win_f <= 0.2):
                    raise ValueError(f"winsor must be between 0 and 0.2 (got {win_f})")
                opts["winsor"] = str(win_f)
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self._root, "GPA – winsor", str(exc))
                return None
        # Observations filter — pass inc-ids when a subset is active
        inc_ids = getattr(self, "_obs_inc_ids", None)
        if inc_ids is not None:
            opts["inc-ids"] = ",".join(str(i) for i in inc_ids)
        return {
            "opts": opts,
            "diag": {
                "X": list(x_longs),
                "Y": list(y_longs),
                "Z": list(z_longs),
            },
            "meta": {},
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
        lead = f"Assoc: {req.get('x_count', 0)} X, {req.get('y_count', 0)} Y"
        if req.get("z_count"):
            lead += f", {req.get('z_count', 0)} Z"

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
        if req.get("selection_warning"):
            details.append(req["selection_warning"])
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

    def _selection_base_count(self, names):
        """Return the number of distinct base names represented by selected long vars."""
        if not names:
            return 0
        if self._manifest_df is None or self._manifest_df.empty:
            return len(names)
        if "VAR" not in self._manifest_df.columns:
            return len(names)
        sub = self._manifest_df[self._manifest_df["VAR"].isin(names)]
        if sub.empty:
            return len(names)
        base_col = "BASE" if "BASE" in sub.columns else "VAR"
        return len({str(v) for v in sub[base_col].tolist() if str(v).strip()})

    def _update_selection_desc(self):
        """Summarize selected variables inline beside each picker header."""
        for picker in (self._picker_x, self._picker_y, self._picker_z):
            picker.set_summary("")
        role_parts = []
        all_longs = []
        for label, picker in (("X", self._picker_x), ("Y", self._picker_y), ("Z", self._picker_z)):
            bases = picker.selected()
            longs = picker.selected_long_names()
            all_longs.extend(longs)
            if not bases:
                continue
            n_txt = self._selection_n_summary(longs)
            part = f"{len(bases)} base"
            if len(bases) != 1:
                part += "s"
            if longs:
                part += f", {len(longs)} long"
            if n_txt:
                part += f" ({n_txt})"
            picker.set_summary(part)
            role_parts.append(f"{label}: {len(bases)} var{'s' if len(bases) != 1 else ''}")
        if hasattr(self, "_select_count_lbl"):
            parts = list(role_parts)
            if role_parts:
                n_txt = self._selection_n_summary(all_longs)
                if n_txt:
                    parts.append(f"{n_txt} individuals")
            obs_s = self._obs_summary_str() if hasattr(self, "_obs_row_widgets") else ""
            if obs_s:
                parts.append(obs_s)
            self._select_count_lbl.setText("  ·  ".join(parts))

    def _clear_results_display(self):
        """Clear any stale GPA results and plots before/after a run."""
        self._results_dfs = {}
        self._results_table_key = None
        self._active_result_df = None
        self._results_view.setModel(None)
        self._results_proxy = None
        self._render_timer.stop()
        self._scatter_mode = False
        self._scatter_gen += 1
        self._scatter_toggle_widget.setVisible(False)
        self._summary_btn.setChecked(True)
        self._scatter_raw_btn.setChecked(False)
        self._scatter_partial_btn.setChecked(False)
        self._scatter_partial_btn.setEnabled(True)
        self._joint_mode_btn.setChecked(False)
        self._joint_mode_btn.setEnabled(False)
        self._joint_mode = False
        self._joint_fit_gen += 1
        self._joint_xvar = ""
        self._joint_yvars = []
        self._joint_zvars = []
        self._joint_result = None
        self._pre_joint_status_text = ""
        self._joint_y_list.clear()
        self._joint_table.setModel(None)
        self._joint_status_lbl.setText("Joint model unavailable.")
        self._update_joint_action_buttons()
        if hasattr(self, "_viz_stack"):
            self._viz_stack.setCurrentIndex(0)
        if self._canvas is not None:
            fig = self._canvas.figure
            fig.clear()
            self._canvas.draw()

    def _show_gpa_diagnostics(self, diag):
        """Mirror useful empty-run diagnostics inline with the X/Y/Z pickers."""
        for picker in (self._picker_x, self._picker_y, self._picker_z):
            picker.set_summary("")
        if not diag:
            return
        joint_n = diag.get("joint_n")
        row_n = diag.get("row_n")
        joint_txt = ""
        if joint_n is not None:
            joint_txt = (
                f"cc N={joint_n}/{row_n}" if row_n is not None else f"cc N={joint_n}"
            )
        overlap = diag.get("overlap") or {}
        constant = diag.get("constant") or {}
        self._picker_x.set_summary(
            "  ·  ".join(
                txt for txt in [
                    joint_txt,
                    f"∩Y: {', '.join((overlap.get('X∩Y') or [])[:2])}" if overlap.get("X∩Y") else "",
                    f"∩Z: {', '.join((overlap.get('X∩Z') or [])[:2])}" if overlap.get("X∩Z") else "",
                    f"const: {', '.join((constant.get('X') or [])[:2])}" if constant.get("X") else "",
                ] if txt
            )
        )
        self._picker_y.set_summary(
            "  ·  ".join(
                txt for txt in [
                    f"∩X: {', '.join((overlap.get('X∩Y') or [])[:2])}" if overlap.get("X∩Y") else "",
                    f"∩Z: {', '.join((overlap.get('Y∩Z') or [])[:2])}" if overlap.get("Y∩Z") else "",
                    f"const: {', '.join((constant.get('Y') or [])[:2])}" if constant.get("Y") else "",
                ] if txt
            )
        )
        self._picker_z.set_summary(
            "  ·  ".join(
                txt for txt in [
                    f"∩X: {', '.join((overlap.get('X∩Z') or [])[:2])}" if overlap.get("X∩Z") else "",
                    f"∩Y: {', '.join((overlap.get('Y∩Z') or [])[:2])}" if overlap.get("Y∩Z") else "",
                    f"const: {', '.join((constant.get('Z') or [])[:2])}" if constant.get("Z") else "",
                ] if txt
            )
        )

    # ======================================================================
    # Results
    # ======================================================================

    def _joint_mode_available(self):
        req = self._last_gpa_request or {}
        return (
            len(req.get("x_vars") or []) == 1
            and self._active_result_df is not None
            and not self._active_result_df.empty
            and {"X", "Y"}.issubset(self._active_result_df.columns)
        )

    def _selected_result_yvars(self):
        proxy = self._results_proxy
        if proxy is None or proxy.sourceModel() is None:
            return []
        selection = self._results_view.selectionModel()
        if selection is None:
            return []
        model = proxy.sourceModel()
        headers = [str(model.headerData(c, Qt.Horizontal) or "") for c in range(model.columnCount())]
        if "Y" not in headers:
            return []
        y_col = headers.index("Y")
        x_col = headers.index("X") if "X" in headers else -1
        want_x = self._joint_xvar or ((self._last_gpa_request or {}).get("x_vars") or [""])[0]
        out = []
        for idx in selection.selectedRows():
            src = proxy.mapToSource(idx)
            if not src.isValid():
                continue
            if x_col >= 0 and want_x:
                row_x = model.item(src.row(), x_col).text()
                if row_x != want_x:
                    continue
            y_val = model.item(src.row(), y_col).text()
            if y_val:
                out.append(y_val)
        return _unique_preserve(out)

    def _update_joint_controls_visibility(self):
        show = self._joint_mode_available() or self._joint_mode or self._scatter_mode
        self._scatter_toggle_widget.setVisible(show)
        self._joint_mode_btn.setEnabled(self._joint_mode_available())
        if not self._joint_mode_available() and not self._joint_mode:
            self._joint_mode_btn.setChecked(False)

    def _update_joint_action_buttons(self):
        has_selection = bool(self._selected_result_yvars())
        has_active = bool(self._joint_yvars)
        self._joint_add_btn.setEnabled(self._joint_mode and has_selection)
        self._joint_remove_btn.setEnabled(
            self._joint_mode and (has_selection or bool(self._joint_y_list.selectedItems()))
        )
        self._joint_clear_btn.setEnabled(self._joint_mode and has_active)
        joint_model = self._joint_table.model()
        self._joint_export_btn.setEnabled(
            self._joint_mode and joint_model is not None and joint_model.rowCount() > 0
        )

    def _set_joint_active_y_list(self):
        self._joint_y_list.clear()
        for y_val in self._joint_yvars:
            self._joint_y_list.addItem(y_val)
        self._update_joint_action_buttons()

    def _render_joint_table(self, df):
        model = QStandardItemModel(len(df), len(df.columns), self)
        model.setHorizontalHeaderLabels(list(df.columns))
        for r in range(len(df)):
            row = df.iloc[r]
            for c, col in enumerate(df.columns):
                raw_val = row.iloc[c]
                if isinstance(raw_val, (float, np.floating)):
                    text = _fmt_float(raw_val)
                else:
                    text = "" if pd.isna(raw_val) else str(raw_val)
                item = QStandardItem(text)
                if isinstance(raw_val, (int, float, np.integer, np.floating)) and not pd.isna(raw_val):
                    item.setData(float(raw_val), Qt.UserRole)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                model.setItem(r, c, item)
        proxy = _GpaResultsSortProxy(self)
        proxy.setSourceModel(model)
        self._joint_table.setModel(proxy)
        self._joint_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._joint_table.horizontalHeader().setStretchLastSection(True)

    def _joint_status_text(self, result):
        if result is None:
            return "Joint model unavailable."
        bits = []
        if result.get("model_type"):
            bits.append(result["model_type"].title())
        if result.get("x_var"):
            bits.append(f"X={result['x_var']}")
        if result.get("binary_labels"):
            bits.append("binary")
        bits.append(f"N={result.get('n_complete', 0)}")
        bits.append(f"Y={len(self._joint_yvars)}")
        bits.append(f"Z={len(self._joint_zvars)}")
        warnings = result.get("warnings") or []
        if warnings:
            bits.append("; ".join(warnings[:3]))
        if result.get("error"):
            bits.append(result["error"])
        return "  ·  ".join(bits)

    def _joint_dump_filters(self):
        opts = dict((self._last_gpa_request or {}).get("opts") or {})
        keep = {}
        if "winsor" in opts and opts["winsor"] not in (None, ""):
            keep["winsor"] = opts["winsor"]
        if "inc-ids" in opts and opts["inc-ids"] not in (None, ""):
            keep["inc-ids"] = opts["inc-ids"]
        return _with_dump_qc_disabled(keep)

    def _start_joint_fit(self):
        self._joint_fit_gen += 1
        gen = self._joint_fit_gen
        self._joint_result = None
        self._joint_status_lbl.setText(
            f"Fitting joint model… X={self._joint_xvar}  ·  Y={len(self._joint_yvars)}  ·  Z={len(self._joint_zvars)}"
        )
        self._joint_table.setModel(None)
        dat_path = self._dat_edit.text().strip()
        y_vars = list(self._joint_yvars)
        z_vars = list(self._joint_zvars)
        dump_filters = self._joint_dump_filters()
        fut = self.ctrl._exec.submit(
            self._joint_model_worker, dat_path, self._joint_xvar, y_vars, z_vars, dump_filters
        )
        def _done(_f=fut, _gen=gen):
            try:
                result = _f.result()
                if _gen == self._joint_fit_gen and self._joint_mode:
                    self._sig_ok.emit({"type": "joint", "result": result})
            except Exception as exc:
                if _gen == self._joint_fit_gen and self._joint_mode:
                    self._sig_ok.emit({
                        "type": "joint",
                        "result": {
                            "table": pd.DataFrame(columns=["TERM", "ROLE", "BETA", "SE", "STAT", "P"]),
                            "warnings": [],
                            "active_y": list(y_vars),
                            "active_z": list(z_vars),
                            "x_var": self._joint_xvar,
                            "n_complete": 0,
                            "n_total": 0,
                            "model_type": "",
                            "error": str(exc),
                        },
                    })
        fut.add_done_callback(_done)

    def _ensure_joint_mode(self, seed_from_selection=False):
        if not self._joint_mode_available():
            return False
        if not self._joint_mode:
            req = self._last_gpa_request or {}
            self._joint_mode = True
            self._joint_xvar = (req.get("x_vars") or [""])[0]
            self._joint_zvars = list(req.get("z_vars") or [])
            self._joint_yvars = []
            self._pre_joint_status_text = self._analyze_status.text()
            self._summary_btn.setChecked(False)
            self._joint_mode_btn.setChecked(True)
            self._scatter_raw_btn.setChecked(False)
            self._scatter_partial_btn.setChecked(False)
            self._viz_stack.setCurrentIndex(1)
        if seed_from_selection:
            self._joint_yvars = _unique_preserve(self._joint_yvars + self._selected_result_yvars())
        self._set_joint_active_y_list()
        self._update_joint_controls_visibility()
        self._update_joint_action_buttons()
        if self._joint_yvars or self._joint_zvars:
            self._start_joint_fit()
        else:
            self._joint_table.setModel(None)
            self._joint_status_lbl.setText(
                f"Joint mode active for X={self._joint_xvar}. Add one or more Y rows to fit X ~ Y + Z."
            )
        return True

    def _leave_joint_mode(self):
        self._joint_mode = False
        self._joint_mode_btn.setChecked(False)
        if hasattr(self, "_viz_stack"):
            self._viz_stack.setCurrentIndex(0)
        self._update_joint_controls_visibility()
        self._update_joint_action_buttons()
        if self._pre_joint_status_text:
            self._analyze_status.setText(self._pre_joint_status_text)
        if self._canvas is None:
            if self._scatter_xvar and self._scatter_yvar:
                partial = bool(self._last_gpa_z) and self._scatter_partial_btn.isChecked()
                self._request_scatter(self._scatter_xvar, self._scatter_yvar, partial=partial)
            else:
                self._render_timer.start()
        elif not self._scatter_mode:
            self._render_timer.start()

    def _on_joint_mode_toggled(self, checked):
        if checked:
            if not self._ensure_joint_mode(seed_from_selection=True):
                self._joint_mode_btn.setChecked(False)
        else:
            self._leave_joint_mode()

    def _joint_add_selected_y(self):
        if not self._ensure_joint_mode(seed_from_selection=False):
            return
        selected = self._selected_result_yvars()
        if not selected:
            self._update_joint_action_buttons()
            return
        new_y = _unique_preserve(self._joint_yvars + selected)
        if new_y == self._joint_yvars:
            self._update_joint_action_buttons()
            return
        self._joint_yvars = new_y
        self._set_joint_active_y_list()
        self._start_joint_fit()

    def _joint_remove_selected_y(self):
        if not self._joint_mode:
            return
        remove = set(self._selected_result_yvars())
        remove.update(item.text() for item in self._joint_y_list.selectedItems())
        if not remove:
            self._update_joint_action_buttons()
            return
        self._joint_yvars = [y_val for y_val in self._joint_yvars if y_val not in remove]
        self._set_joint_active_y_list()
        if self._joint_yvars or self._joint_zvars:
            self._start_joint_fit()
        else:
            self._joint_table.setModel(None)
            self._joint_result = None
            self._joint_status_lbl.setText(
                f"Joint mode active for X={self._joint_xvar}. Add one or more Y rows to fit X ~ Y + Z."
            )
            self._update_joint_action_buttons()

    def _joint_clear_y(self):
        if not self._joint_mode:
            return
        self._joint_yvars = []
        self._set_joint_active_y_list()
        if self._joint_zvars:
            self._start_joint_fit()
        else:
            self._joint_table.setModel(None)
            self._joint_result = None
            self._joint_status_lbl.setText(
                f"Joint mode active for X={self._joint_xvar}. Add one or more Y rows to fit X ~ Y + Z."
            )
        self._update_joint_action_buttons()

    def _populate_results_tables(self, dfs):
        self._results_dfs = dfs
        if dfs:
            self._show_results_table(sorted(dfs.keys())[0])

    def _show_results_table(self, key):
        # switching tables → exit scatter mode, return to volcano
        self._scatter_mode = False
        self._scatter_gen += 1
        self._scatter_toggle_widget.setVisible(False)
        self._summary_btn.setChecked(True)
        self._scatter_raw_btn.setChecked(False)
        self._scatter_partial_btn.setChecked(False)
        self._scatter_partial_btn.setEnabled(True)
        self._results_view.clearSelection()
        if self._joint_mode:
            self._leave_joint_mode()
        if not key:
            return
        self._results_table_key = key
        df = self._results_dfs.get(key)
        if df is None or df.empty:
            return
        self._active_result_df = df
        self._fill_results_view(df)
        self._update_joint_controls_visibility()
        self._render_timer.start()

    @staticmethod
    def _reorder_result_cols(cols):
        """Return cols reordered: X Y | B | P P_* | stat cols | GROUP strata."""
        _KNOWN_STATS = {"SE", "T", "Z", "STAT", "N", "NOBS", "OBS", "DF", "R2",
                        "OR", "HR", "CI_LO", "CI_HI", "CI_LOW", "CI_HIGH"}
        cols = [c for c in cols if c != "ID"]
        col_set = set(cols)

        def _pick(*names):
            return [n for n in names if n in col_set]

        front  = _pick("X", "Y")
        b_col  = _pick("B")
        p_main = _pick("P")
        p_adj  = [c for c in cols if c.startswith("P_")]
        used   = set(front + b_col + p_main + p_adj)
        stats  = [c for c in cols if c not in used and c in _KNOWN_STATS]
        used  |= set(stats)
        grp    = [c for c in cols if c not in used]
        return front + b_col + p_main + p_adj + stats + grp

    def _fill_results_view(self, df):
        df_disp = df.copy()
        if "ID" in df_disp.columns:
            df_disp = df_disp.drop(columns=["ID"])
        df_disp = df_disp[self._reorder_result_cols(list(df_disp.columns))]
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
        self._results_view.selectionModel().selectionChanged.connect(
            lambda *_: self._update_joint_action_buttons())
        self._update_joint_action_buttons()

    def _apply_results_filter(self, text):
        if self._results_proxy:
            self._results_proxy.setFilterRegularExpression(
                QRegularExpression(text, QRegularExpression.CaseInsensitiveOption))

    def _populate_assoc_corr_table(self, df):
        if not hasattr(self, "_assoc_corr_view"):
            return
        display_cols = [c for c in ["R", "ABS_R", "P", "N", "TARGET", "TARGET_BASE", "TARGET_GRP", "TARGET_NI"] if c in df.columns]
        df_disp = df[display_cols].copy() if display_cols else df.copy()
        model = QStandardItemModel(len(df_disp), len(df_disp.columns), self)
        model.setHorizontalHeaderLabels(list(df_disp.columns))
        numeric_cols = {"N", "R", "ABS_R", "P"}
        for r in range(len(df_disp)):
            row = df_disp.iloc[r]
            for c, col in enumerate(df_disp.columns):
                raw_val = row.iloc[c]
                if col in numeric_cols:
                    text = _fmt_float(raw_val)
                else:
                    text = "" if pd.isna(raw_val) else str(raw_val)
                item = QStandardItem(text)
                if col in numeric_cols and pd.notna(raw_val):
                    item.setData(float(raw_val), Qt.UserRole)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                model.setItem(r, c, item)
        proxy = _GpaResultsSortProxy(self)
        proxy.setSourceModel(model)
        proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        proxy.setFilterKeyColumn(-1)
        self._assoc_corr_view.setModel(proxy)
        hdr = self._assoc_corr_view.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Interactive)
        for col_name, width in (("R", 70), ("ABS_R", 70), ("P", 80), ("N", 55), ("TARGET", 360), ("TARGET_BASE", 120), ("TARGET_GRP", 120), ("TARGET_NI", 70)):
            if col_name in df_disp.columns:
                idx = df_disp.columns.get_loc(col_name)
                self._assoc_corr_view.setColumnWidth(idx, width)
        hdr.setStretchLastSection(True)
        self._assoc_corr_proxy = proxy
        self._apply_assoc_corr_filter(self._assoc_corr_filter.text())
        selection_model = self._assoc_corr_view.selectionModel()
        if selection_model is not None:
            selection_model.currentRowChanged.connect(
                lambda cur, _prev: self._on_assoc_corr_row_changed(cur) if cur.isValid() else None
            )

    def _apply_assoc_corr_filter(self, text):
        proxy = self._assoc_corr_proxy
        if proxy is None:
            return
        df = self._assoc_corr_df
        if df is None or df.empty:
            proxy.setFilterRegularExpression(QRegularExpression())
            return
        abs_min = float(self._assoc_abs_spin.value())
        top_n = int(self._assoc_topn_spin.value())
        mask = pd.Series(True, index=df.index)
        if "ABS_R" in df.columns:
            mask &= pd.to_numeric(df["ABS_R"], errors="coerce").fillna(-1) >= abs_min
        keep_rows = set(df.loc[mask].index[:top_n].tolist())
        pattern = str(text or "").strip().lower()
        source = proxy.sourceModel()
        for row in range(source.rowCount()):
            visible = row in keep_rows
            if visible and pattern:
                values = []
                for col in range(source.columnCount()):
                    values.append(str(source.item(row, col).text() or "").lower())
                visible = any(pattern in value for value in values)
            self._assoc_corr_view.setRowHidden(proxy.mapFromSource(source.index(row, 0)).row(), not visible)
        self._render_assoc_corr_ranked()

    def _render_assoc_corr_ranked(self):
        canvas = self._assoc_corr_canvas
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)

        df = self._filtered_assoc_corr_df()
        if df.empty:
            ax.text(0.5, 0.5, "No ranked correlations to display", color=FG,
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw()
            return

        show = min(len(df), 40)
        top = df.head(show).iloc[::-1]
        colors = [_PALETTE[0] if float(v) >= 0 else _PALETTE[1] for v in top["R"].tolist()]
        bars = ax.barh(np.arange(show), top["R"].to_numpy(dtype=float), color=colors, alpha=0.85)
        labels = [str(v) for v in top["TARGET"].tolist()]
        ax.set_yticks(np.arange(show))
        ax.set_yticklabels(labels, color=FG, fontsize=7)
        ax.axvline(0, color="#444", lw=0.6)
        self._style_ax(ax, title=f"Top |r| hits for {self._assoc_seed_long or 'seed'}", xlabel="Correlation r")
        self._assoc_ranked_bars = list(bars)
        self._assoc_ranked_labels = labels
        self._assoc_ranked_hover = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(8, 8),
            textcoords="offset points",
            color=FG,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="#111827", ec="#374151", alpha=0.95),
        )
        self._assoc_ranked_hover.set_visible(False)
        try:
            fig.tight_layout(pad=1.2)
        except Exception:
            pass
        canvas.draw()

    def _filtered_assoc_corr_df(self):
        df = self._assoc_corr_df
        if df is None or df.empty:
            return pd.DataFrame()
        abs_min = float(self._assoc_abs_spin.value())
        top_n = int(self._assoc_topn_spin.value())
        out = df.copy()
        out = out[pd.to_numeric(out["ABS_R"], errors="coerce").fillna(-1) >= abs_min]
        text = self._assoc_corr_filter.text().strip().lower()
        if text:
            mask = pd.Series(False, index=out.index)
            for col in ("TARGET", "TARGET_BASE", "TARGET_GRP"):
                if col in out.columns:
                    mask |= out[col].astype(str).str.lower().str.contains(text, regex=False)
            out = out[mask]
        return out.head(top_n).reset_index(drop=True)

    def _on_assoc_ranked_hover(self, event):
        hover = self._assoc_ranked_hover
        if hover is None or not self._assoc_ranked_bars:
            return
        ax = self._assoc_ranked_bars[0].axes if self._assoc_ranked_bars else None
        if event.inaxes != ax:
            if hover.get_visible():
                hover.set_visible(False)
                self._assoc_corr_canvas.draw_idle()
            return
        for idx, bar in enumerate(self._assoc_ranked_bars):
            contains, _info = bar.contains(event)
            if not contains:
                continue
            hover.xy = (bar.get_width(), bar.get_y() + bar.get_height() * 0.5)
            hover.set_text(self._assoc_ranked_labels[idx] if idx < len(self._assoc_ranked_labels) else "")
            hover.set_visible(True)
            self._assoc_corr_canvas.draw_idle()
            return
        if hover.get_visible():
            hover.set_visible(False)
            self._assoc_corr_canvas.draw_idle()

    def _show_assoc_ranked_plot(self):
        self._assoc_plot_mode = "ranked"
        self._assoc_ranked_btn.setChecked(True)
        self._assoc_scatter_btn.setChecked(False)
        self._render_assoc_corr_ranked()

    def _show_assoc_scatter_plot(self):
        if not (self._assoc_scatter_seed and self._assoc_scatter_target):
            self._assoc_plot_mode = "ranked"
            self._assoc_ranked_btn.setChecked(True)
            self._assoc_scatter_btn.setChecked(False)
            self._render_assoc_corr_ranked()
            return
        self._assoc_plot_mode = "scatter"
        self._assoc_ranked_btn.setChecked(False)
        self._assoc_scatter_btn.setChecked(True)
        self._request_assoc_scatter(self._assoc_scatter_seed, self._assoc_scatter_target)

    def _on_assoc_corr_row_changed(self, index):
        proxy = self._assoc_corr_proxy
        if proxy is None:
            return
        src = proxy.mapToSource(index)
        model = proxy.sourceModel()
        headers = [str(model.headerData(c, Qt.Horizontal) or "") for c in range(model.columnCount())]
        if "TARGET" not in headers:
            return
        self._assoc_scatter_seed = self._assoc_seed_long
        self._assoc_scatter_target = model.item(src.row(), headers.index("TARGET")).text()
        self._show_assoc_scatter_plot()

    def _on_assoc_corr_row_clicked(self, index):
        self._on_assoc_corr_row_changed(index)

    def _request_assoc_scatter(self, seed_var, target_var):
        dat_path = self._dat_edit.text().strip()
        candidates = self._assoc_candidate_long_names()
        cache = self._assoc_matrix_cache_payload(dat_path, candidates)
        if cache["from_cache"] and all(col in cache["df"].columns for col in [seed_var, target_var]):
            fut = self.ctrl._exec.submit(
                self._assoc_scatter_from_df_worker, cache["df"], seed_var, target_var
            )
        else:
            if not dat_path or not os.path.exists(dat_path):
                return
            fut = self.ctrl._exec.submit(
                self._scatter_worker, dat_path, seed_var, target_var, [], cache["filters"]
            )
        def _done(_f=fut):
            try:
                ids, xs, ys, xs_group = _f.result()
                self._sig_ok.emit({
                    "type": "assoc_scatter",
                    "result": (ids, xs, ys, xs_group, seed_var, target_var, False),
                })
            except Exception:
                self._sig_err.emit({"type": "assoc", "traceback": traceback.format_exc()})
        fut.add_done_callback(_done)

    @staticmethod
    def _assoc_scatter_from_df_worker(raw_df, seed_var, target_var):
        work = raw_df.copy()
        for col in [seed_var, target_var]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
        missing = [c for c in [seed_var, target_var] if c not in work.columns]
        if missing:
            raise ValueError(f"Columns not found in cached matrix: {', '.join(missing)}")
        work = work.dropna(subset=[seed_var, target_var])
        ids_raw = work["ID"].tolist() if "ID" in work.columns else list(range(len(work)))
        return ids_raw, work[seed_var].tolist(), work[target_var].tolist(), work[seed_var].tolist()

    def _render_assoc_scatter(self, xs, ys, x_var, y_var):
        canvas = self._assoc_corr_canvas
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        valid = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[valid]
        ys = ys[valid]
        if len(xs) == 0:
            ax.text(0.5, 0.5, "No complete pairwise observations", color=FG,
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw()
            return
        xs_z = (xs - xs.mean()) / (xs.std() + 1e-12)
        ys_z = (ys - ys.mean()) / (ys.std() + 1e-12)
        r = _safe_corrcoef(xs_z, ys_z)
        ax.scatter(xs_z, ys_z, s=8, alpha=0.55, color=_PALETTE[0], linewidths=0)
        if len(xs_z) > 1:
            slope, intercept = np.polyfit(xs_z, ys_z, 1)
            line_x = np.array([xs_z.min(), xs_z.max()])
            ax.plot(line_x, slope * line_x + intercept, color=_PALETTE[1], lw=1.2)
        ax.axhline(0, color="#444", lw=0.5)
        ax.axvline(0, color="#444", lw=0.5)
        self._style_ax(
            ax,
            title=f"{x_var} × {y_var}  ·  r={_fmt_float(r)}  ·  N={len(xs_z)}",
            xlabel=x_var,
            ylabel=y_var,
        )
        try:
            fig.tight_layout(pad=1.2)
        except Exception:
            pass
        canvas.draw()

    def _populate_assoc_pca_table(self, df):
        if not hasattr(self, "_assoc_pca_view"):
            return
        model = QStandardItemModel(len(df), len(df.columns), self)
        model.setHorizontalHeaderLabels(list(df.columns))
        for r in range(len(df)):
            row = df.iloc[r]
            for c, col in enumerate(df.columns):
                raw_val = row.iloc[c]
                text = _fmt_float(raw_val) if col.startswith("PC") else ("" if pd.isna(raw_val) else str(raw_val))
                item = QStandardItem(text)
                if col.startswith("PC") and pd.notna(raw_val):
                    item.setData(float(raw_val), Qt.UserRole)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                model.setItem(r, c, item)
        proxy = _GpaResultsSortProxy(self)
        proxy.setSourceModel(model)
        proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        proxy.setFilterKeyColumn(-1)
        self._assoc_pca_view.setModel(proxy)
        hdr = self._assoc_pca_view.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Interactive)
        hdr.setMinimumSectionSize(48)
        for idx, col in enumerate(df.columns):
            if col == "VAR":
                width = 320
            elif col == "RID":
                width = 220
            elif col == "ID":
                width = 90
            elif col == "BASE":
                width = 120
            elif col == "GRP":
                width = 90
            elif col == "NI":
                width = 60
            elif col.startswith("PC"):
                width = 88
            else:
                width = 110
            self._assoc_pca_view.setColumnWidth(idx, width)
        hdr.setStretchLastSection(False)
        self._assoc_pca_proxy = proxy
        self._apply_assoc_pca_filter(self._assoc_pca_filter.text())

    def _apply_assoc_pca_filter(self, text):
        if self._assoc_pca_proxy:
            self._assoc_pca_proxy.setFilterRegularExpression(
                QRegularExpression(text, QRegularExpression.CaseInsensitiveOption)
            )

    def _render_assoc_pca_plot(self, *_args):
        from matplotlib import colormaps
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D

        canvas = self._assoc_pca_canvas
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)

        df = self._assoc_pca_df
        if df is None or df.empty:
            ax.text(0.5, 0.5, "No PCA results to display", color=FG,
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw()
            return

        x_col = str(self._assoc_pca_x_combo.currentData() or "")
        y_col = str(self._assoc_pca_y_combo.currentData() or "")
        if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
            ax.text(0.5, 0.5, "Select PCA components to plot", color=FG,
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw()
            return

        xs = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
        ys = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(xs) & np.isfinite(ys)
        xs = xs[valid]
        ys = ys[valid]
        plot_df = df.loc[valid].copy()
        label_col = str((self._assoc_pca_result or {}).get("point_label_col") or "")
        if not label_col or label_col not in plot_df.columns:
            if "VAR" in plot_df.columns:
                label_col = "VAR"
            elif "RID" in plot_df.columns:
                label_col = "RID"
            elif "ID" in plot_df.columns:
                label_col = "ID"
            else:
                label_col = plot_df.columns[0]
        labels = plot_df[label_col].astype(str).tolist()
        self._assoc_pca_labels = labels
        self._assoc_pca_xy = np.column_stack([xs, ys]) if len(xs) else np.empty((0, 2))
        color_mode = str(self._assoc_pca_color_combo.currentData() or "none")
        self._assoc_pca_artist = None
        title_root = "Variable PCA map" if str((self._assoc_pca_result or {}).get("point_kind") or "variable") == "variable" else "Observation PCA map"
        if color_mode == "none":
            self._assoc_pca_artist = ax.scatter(xs, ys, s=14, alpha=0.7, color=_PALETTE[2], picker=True)
        else:
            if color_mode == "group":
                values = plot_df["GRP"] if "GRP" in plot_df.columns else pd.Series(["."] * len(plot_df))
                title_suffix = "colored by group"
            elif color_mode == "base":
                values = plot_df["BASE"] if "BASE" in plot_df.columns else plot_df["VAR"]
                title_suffix = "colored by base var"
            else:
                values = plot_df[color_mode] if color_mode in plot_df.columns else pd.Series(["."] * len(plot_df))
                title_suffix = f"colored by {color_mode}"
            raw = values.astype(str).replace({"nan": ".", "None": "."})
            missing_mask = raw.isin(["", "."])
            numeric = pd.to_numeric(raw.where(~missing_mask), errors="coerce")
            nonmissing = raw[~missing_mask]
            uniq_count = int(nonmissing.nunique()) if len(nonmissing) else 0
            if numeric.notna().sum() == (~missing_mask).sum() and uniq_count > 10:
                cmap = colormaps["turbo"]
                vals = numeric.to_numpy(dtype=float)
                finite = np.isfinite(vals)
                vmin = float(np.nanmin(vals[finite])) if np.any(finite) else 0.0
                vmax = float(np.nanmax(vals[finite])) if np.any(finite) else 1.0
                if vmin == vmax:
                    vmax = vmin + 1.0
                norm = Normalize(vmin=vmin, vmax=vmax)
                colors = np.tile(np.array([[0.85, 0.85, 0.85, 1.0]]), (len(plot_df), 1))
                for i, val in enumerate(vals):
                    if np.isfinite(val):
                        colors[i] = cmap(norm(val))
                self._assoc_pca_artist = ax.scatter(xs, ys, s=14, alpha=0.8, c=colors, picker=True)
                sm = ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
                cbar.ax.tick_params(colors=FG, labelsize=7)
                cbar.outline.set_edgecolor(GRID)
                cbar.set_label(color_mode, color=FG, fontsize=8)
                title_suffix += f" [{_fmt_float(vmin)}–{_fmt_float(vmax)}]"
            else:
                levels = sorted(nonmissing.unique().tolist())
                palette_levels = levels[:10]
                color_map = {level: _PALETTE[i % len(_PALETTE)] for i, level in enumerate(palette_levels)}
                point_colors = []
                for val, miss in zip(raw.tolist(), missing_mask.tolist()):
                    if miss:
                        point_colors.append("#d1d5db")
                    elif val in color_map:
                        point_colors.append(color_map[val])
                    else:
                        point_colors.append("#9ca3af")
                self._assoc_pca_artist = ax.scatter(xs, ys, s=14, alpha=0.8, c=point_colors, picker=True)
                handles = [
                    Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                           markerfacecolor=color_map[level], markeredgecolor=color_map[level], label=str(level))
                    for level in palette_levels
                ]
                if any(missing_mask):
                    handles.append(
                        Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                               markerfacecolor="#d1d5db", markeredgecolor="#d1d5db", label="missing")
                    )
                if len(levels) > 10:
                    handles.append(
                        Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                               markerfacecolor="#9ca3af", markeredgecolor="#9ca3af", label="other")
                    )
                leg = ax.legend(handles=handles, fontsize=7, framealpha=0.35,
                                facecolor="#111827", edgecolor=GRID, loc="best", title=color_mode)
                if leg is not None:
                    leg.get_title().set_color(FG)
                    for txt in leg.get_texts():
                        txt.set_color(FG)
            self._assoc_pca_artist = self._assoc_pca_artist or ax.scatter(xs, ys, s=14, alpha=0.7, color=_PALETTE[2], picker=True)
        ax.axhline(0, color="#444", lw=0.5)
        ax.axvline(0, color="#444", lw=0.5)
        self._style_ax(
            ax,
            title=title_root + (f"  ·  {title_suffix}" if color_mode != "none" else ""),
            xlabel=x_col,
            ylabel=y_col,
        )
        self._assoc_pca_hover = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(8, 8),
            textcoords="offset points",
            color=FG,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="#111827", ec="#374151", alpha=0.95),
        )
        self._assoc_pca_hover.set_visible(False)
        try:
            fig.tight_layout(pad=1.2)
        except Exception:
            pass
        canvas.draw()

    def _on_assoc_pca_row_clicked(self, index):
        if str((self._assoc_pca_result or {}).get("point_kind") or "variable") != "variable":
            return
        proxy = self._assoc_pca_proxy
        if proxy is None:
            return
        src = proxy.mapToSource(index)
        model = proxy.sourceModel()
        headers = [str(model.headerData(c, Qt.Horizontal) or "") for c in range(model.columnCount())]
        if "VAR" not in headers:
            return
        long_var = model.item(src.row(), headers.index("VAR")).text()
        self._set_assoc_seed_variable(long_var)

    def _on_assoc_pca_hover(self, event):
        if self._assoc_pca_artist is None or self._assoc_pca_hover is None:
            return
        if event.inaxes != getattr(self._assoc_pca_artist, "axes", None):
            if self._assoc_pca_hover.get_visible():
                self._assoc_pca_hover.set_visible(False)
                self._assoc_pca_canvas.draw_idle()
            return
        contains, info = self._assoc_pca_artist.contains(event)
        if not contains:
            if self._assoc_pca_hover.get_visible():
                self._assoc_pca_hover.set_visible(False)
                self._assoc_pca_canvas.draw_idle()
            return
        idx = int(info["ind"][0])
        if self._assoc_pca_xy is None or idx >= len(self._assoc_pca_labels):
            return
        self._assoc_pca_hover.xy = tuple(self._assoc_pca_xy[idx])
        self._assoc_pca_hover.set_text(self._assoc_pca_labels[idx])
        self._assoc_pca_hover.set_visible(True)
        self._assoc_pca_canvas.draw_idle()

    def _on_assoc_pca_click(self, event):
        if str((self._assoc_pca_result or {}).get("point_kind") or "variable") != "variable":
            return
        if event.button != 1 or self._assoc_pca_artist is None:
            return
        contains, info = self._assoc_pca_artist.contains(event)
        if not contains:
            return
        idx = int(info["ind"][0])
        if idx >= len(self._assoc_pca_labels):
            return
        self._set_assoc_seed_variable(self._assoc_pca_labels[idx], auto_run=True)

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
        if self._joint_mode:
            self._update_joint_action_buttons()
            return
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
        self._update_joint_controls_visibility()
        self._scatter_partial_btn.setEnabled(has_z)
        self._summary_btn.setChecked(False)
        if not has_z:
            self._scatter_raw_btn.setChecked(True)
            self._scatter_partial_btn.setChecked(False)
        # Honour whichever toggle is currently active
        partial = has_z and self._scatter_partial_btn.isChecked()
        self._request_scatter(x_var, y_var, partial=partial)

    def _on_scatter_toggle(self, partial: bool):
        """Raw / Partial button clicked — redraw with same X, Y."""
        if self._joint_mode:
            self._leave_joint_mode()
        self._summary_btn.setChecked(False)
        self._scatter_raw_btn.setChecked(not partial)
        self._scatter_partial_btn.setChecked(partial)
        if self._scatter_xvar and self._scatter_yvar:
            self._request_scatter(self._scatter_xvar, self._scatter_yvar, partial=partial)

    def _on_summary_clicked(self):
        """Return to volcano plot from scatter or joint mode."""
        if self._joint_mode:
            self._joint_mode = False
            self._joint_mode_btn.setChecked(False)
            if self._pre_joint_status_text:
                self._analyze_status.setText(self._pre_joint_status_text)
            self._pre_joint_status_text = ""
            self._update_joint_action_buttons()
        self._scatter_mode = False
        self._scatter_gen += 1
        self._summary_btn.setChecked(True)
        self._scatter_raw_btn.setChecked(False)
        self._scatter_partial_btn.setChecked(False)
        self._viz_stack.setCurrentIndex(0)
        self._update_joint_controls_visibility()
        self._render_timer.start()

    @staticmethod
    def _joint_model_worker(dat_path, x_var, y_vars, z_vars, dump_filters):
        from lunapi import gpa_dump

        cols = _unique_preserve([x_var] + list(y_vars) + list(z_vars))
        dump_opts = dict(dump_filters or {})
        dump_opts["lvars"] = ",".join(cols)
        raw_df = gpa_dump(dat_path, **dump_opts)
        return _fit_joint_model_frame(raw_df, x_var, y_vars, z_vars)

    def _request_scatter(self, x_var, y_var, partial=False):
        self._scatter_mode = True
        self._summary_btn.setChecked(False)
        self._render_timer.stop()
        self._scatter_gen += 1
        gen = self._scatter_gen
        zvars = self._last_gpa_z if partial else []
        dat_path = self._dat_edit.text().strip()
        dump_filters = self._joint_dump_filters()
        self._analyze_status.setText(
            f"Loading {'partial ' if partial else ''}scatter: {x_var} × {y_var}…")
        fut = self.ctrl._exec.submit(
            self._scatter_worker, dat_path, x_var, y_var, zvars, dump_filters)
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
    def _scatter_worker(dat_path, x_var, y_var, zvars, dump_filters=None):
        """Return (ids, xs_plot, ys_plot, xs_group).

        xs_plot / ys_plot  — values used for axes (residualized when partial)
        xs_group           — raw X values used for group detection / box labelling

        Uses gpa_dump to read from the .dat file directly so the call is safe
        regardless of what the C-library's cached GPA state happens to be.
        """
        from lunapi import gpa_dump

        cols = _unique_preserve([x_var, y_var] + list(zvars))
        dump_opts = dict(dump_filters or {})
        dump_opts["lvars"] = ",".join(cols)
        raw_df = gpa_dump(dat_path, **dump_opts)

        for col in cols:
            if col in raw_df.columns:
                raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

        missing = [c for c in [x_var, y_var] if c not in raw_df.columns]
        if missing:
            raise ValueError(f"Columns not found in .dat: {', '.join(missing)}")

        raw_df = raw_df.dropna(subset=[x_var, y_var])
        ids_raw = raw_df["ID"].tolist() if "ID" in raw_df.columns else list(range(len(raw_df)))
        xs_raw  = raw_df[x_var].tolist()
        ys_raw  = raw_df[y_var].tolist()

        if not zvars:
            return ids_raw, xs_raw, ys_raw, xs_raw

        # Partial: residualize X and Y on Z via OLS
        z_present = [z for z in zvars if z in raw_df.columns]
        complete  = raw_df[[x_var, y_var] + z_present].notna().all(axis=1)
        work      = raw_df[complete].copy()
        ids_p     = work["ID"].tolist() if "ID" in work.columns else list(range(len(work)))
        xs_raw_aligned = work[x_var].to_numpy(dtype=float)

        def _resid(col):
            y_arr = work[col].to_numpy(dtype=float)
            if not z_present:
                return y_arr.tolist()
            Z = np.column_stack(
                [np.ones(len(y_arr))] + [work[z].to_numpy(dtype=float) for z in z_present])
            beta, *_ = np.linalg.lstsq(Z, y_arr, rcond=None)
            return (y_arr - Z @ beta).tolist()

        return ids_p, _resid(x_var), _resid(y_var), xs_raw_aligned.tolist()

    def _on_scatter_ok(self, payload):
        ids, xs_plot, ys_plot, xs_group, x_var, y_var, partial = payload
        label = "partial " if partial else ""
        # detect dichotomous from raw group values
        xs_g = np.array(xs_group, dtype=float)
        unique_vals = np.unique(xs_g[~np.isnan(xs_g)])
        if len(unique_vals) == 2:
            self._render_boxplot(xs_g, np.array(ys_plot, dtype=float),
                                 np.array(xs_plot, dtype=float),
                                 unique_vals, x_var, y_var, partial=partial)
        else:
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
        r = _safe_corrcoef(xs_z, ys_z)
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

        r = _safe_corrcoef(xs_z, ys_z)
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
        t = payload.get("type")
        if t not in {"joint", "assoc_scatter"}:
            self._end_work()
        result = payload.get("result")

        if t == "prep":
            self._build_status.setText("Dataset built.")
            self._clear_assoc_matrix_cache()
            dat = self._build_dat_edit.text().strip()
            if dat:
                self._set_dat_path(dat)
                manifest_df = _parse_gpa_manifest_text(result)
                if not manifest_df.empty:
                    try:
                        _write_gpa_manifest_sidecar(dat, result)
                    except Exception:
                        pass
                    self._manifest_df = manifest_df
                    self._analyze_status.setText(
                        f"{len(manifest_df)} manifest row(s) loaded.")
                    self._assoc_status.setText(
                        f"{len(manifest_df)} manifest row(s) loaded.")
                    self._populate_manifest_table(manifest_df)
                    for p in (self._picker_x, self._picker_y, self._picker_z,
                              self._assoc_seed_picker, self._assoc_pool_picker):
                        p.populate(manifest_df)
                    self._populate_assoc_pca_color_combo()
                    self._update_selection_desc()
                    self._refresh_assoc_seed_long_list()
                    self._tabs.setCurrentIndex(1)  # switch to Manifest
                else:
                    # Fall back to the engine-derived manifest if prep did not
                    # return one for some reason.
                    self._run_load_manifest()

        elif t == "manifest":
            self._clear_assoc_matrix_cache()
            self._manifest_df = result
            self._analyze_status.setText(
                f"{len(result)} variables loaded." if result is not None else "Manifest empty.")
            self._assoc_status.setText(
                f"{len(result)} variables loaded." if result is not None else "Manifest empty.")
            if result is not None and not result.empty:
                self._tabs.setCurrentIndex(1)  # switch to Manifest
                self._populate_manifest_table(result)
                for p in (self._picker_x, self._picker_y, self._picker_z,
                          self._assoc_seed_picker, self._assoc_pool_picker):
                    p.populate(result)
                self._populate_assoc_pca_color_combo()
                self._update_selection_desc()
                self._refresh_assoc_seed_long_list(preferred=self._assoc_seed_long)

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
            self._tabs.setCurrentIndex(3)  # switch to Results

        elif t == "joint":
            self._joint_result = result
            self._joint_status_lbl.setText(self._joint_status_text(result))
            table = (result or {}).get("table")
            if table is not None and not table.empty:
                self._render_joint_table(table)
            else:
                self._joint_table.setModel(None)
            self._update_joint_action_buttons()

        elif t == "dump_tsv":
            path = payload.get("path", "")
            stdout = result or ""
            if not stdout.strip():
                QtWidgets.QMessageBox.warning(
                    self._root, "GPA dump", "Dump returned no data.")
                return
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(stdout)
                    if not stdout.endswith("\n"):
                        fh.write("\n")
                df = _parse_gpa_manifest_text(stdout)
                n_rows = len(df) if df is not None and not df.empty else "?"
                n_cols = len(df.columns) if df is not None and not df.empty else "?"
                self._analyze_status.setText(
                    f"Saved dump: {n_rows} rows × {n_cols} cols → {os.path.basename(path)}")
            except Exception as exc:
                QtWidgets.QMessageBox.critical(
                    self._root, "GPA dump", f"Failed to save file:\n{exc}")

        elif t == "assoc_corr":
            table = (result or {}).get("table")
            seed = (result or {}).get("seed", "")
            row_count = int((result or {}).get("row_count", 0))
            missing = (result or {}).get("missing") or []
            raw_df = (result or {}).get("raw_df")
            cache_key = (result or {}).get("cache_key")
            if raw_df is not None and cache_key is not None:
                self._store_assoc_matrix_cache(cache_key, raw_df)
            self._assoc_corr_df = table if isinstance(table, pd.DataFrame) else pd.DataFrame()
            self._populate_assoc_corr_table(self._assoc_corr_df)
            shown = len(self._filtered_assoc_corr_df())
            status = f"{seed}: {len(self._assoc_corr_df)} correlations computed from {row_count} rows; showing {shown}."
            if missing:
                status += f"  Dropped {len(missing)} unavailable variables."
            self._assoc_status.setText(status)
            self._assoc_plot_mode = "ranked"
            self._assoc_ranked_btn.setChecked(True)
            self._assoc_scatter_btn.setChecked(False)
            self._render_assoc_corr_ranked()

        elif t == "assoc_scatter":
            ids, xs, ys, xs_group, x_var, y_var, _partial = result
            self._render_assoc_scatter(xs, ys, x_var, y_var)

        elif t == "assoc_pca":
            self._assoc_pca_result = result or {}
            raw_df = self._assoc_pca_result.get("raw_df")
            cache_key = self._assoc_pca_result.get("cache_key")
            if raw_df is not None and cache_key is not None:
                self._store_assoc_matrix_cache(cache_key, raw_df)
            if self._assoc_pca_result.get("error"):
                self._assoc_status.setText(self._assoc_pca_result["error"])
            loadings = self._assoc_pca_result.get("loadings")
            self._assoc_pca_df = loadings if isinstance(loadings, pd.DataFrame) else pd.DataFrame()
            self._populate_assoc_pca_table(self._assoc_pca_df)
            self._populate_assoc_pca_color_combo()
            ratios = self._assoc_pca_result.get("explained_ratio") or []
            self._assoc_pca_x_combo.blockSignals(True)
            self._assoc_pca_y_combo.blockSignals(True)
            self._assoc_pca_x_combo.clear()
            self._assoc_pca_y_combo.clear()
            for idx, ratio in enumerate(ratios, start=1):
                label = f"PC{idx} ({ratio * 100:.1f}%)"
                key = f"PC{idx}"
                self._assoc_pca_x_combo.addItem(label, key)
                self._assoc_pca_y_combo.addItem(label, key)
            if self._assoc_pca_x_combo.count():
                self._assoc_pca_x_combo.setCurrentIndex(0)
            if self._assoc_pca_y_combo.count() > 1:
                self._assoc_pca_y_combo.setCurrentIndex(1)
            elif self._assoc_pca_y_combo.count():
                self._assoc_pca_y_combo.setCurrentIndex(0)
            self._assoc_pca_x_combo.blockSignals(False)
            self._assoc_pca_y_combo.blockSignals(False)
            used_rows = int(self._assoc_pca_result.get("n_rows_used", 0))
            used_cols = int(self._assoc_pca_result.get("n_cols_used", 0))
            missing_dump = self._assoc_pca_result.get("missing_dump_cols") or []
            low_obs = self._assoc_pca_result.get("low_obs_cols") or []
            dropped_const = self._assoc_pca_result.get("dropped_constant") or []
            self._assoc_pca_summary.setText(
                f"Rows={used_rows}/{self._assoc_pca_result.get('n_rows_input', 0)}  ·  Vars={used_cols}  ·  Missing={self._assoc_pca_result.get('row_mode', '')}"
            )
            if not self._assoc_pca_result.get("error"):
                bits = [f"PCA fit on {used_cols} variables using {used_rows} rows."]
                if missing_dump:
                    bits.append(f"{len(missing_dump)} requested vars unavailable")
                if low_obs:
                    bits.append(f"{len(low_obs)} below missingness threshold")
                if dropped_const:
                    bits.append(f"{len(dropped_const)} constant")
                self._assoc_status.setText("  ".join(bits))
            self._render_assoc_pca_plot()

    def _on_err(self, payload):
        self._end_work()
        err_type = ""
        tb = payload
        if isinstance(payload, dict):
            err_type = str(payload.get("type") or "")
            tb = payload.get("traceback") or ""
        if err_type != "assoc":
            self._clear_results_display()
            self._build_status.setText("Error.")
            self._analyze_status.setText("Error.")
        if hasattr(self, "_assoc_status"):
            self._assoc_status.setText("Error.")

        # Extract the final exception line for a readable summary, and keep
        # the full traceback in the detail section.
        tb_str = tb if isinstance(tb, str) else ""
        last_line = ""
        for line in reversed(tb_str.splitlines()):
            line = line.strip()
            if line and not line.startswith("File ") and not line.startswith("Traceback"):
                last_line = line
                break
        summary = last_line or tb_str[-300:].strip()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Critical, "GPA Error",
                                    summary, parent=self._root)
        msg.setDetailedText(tb_str)
        msg.exec()

    def _on_progress(self, msg):
        self._analyze_status.setText(msg)
