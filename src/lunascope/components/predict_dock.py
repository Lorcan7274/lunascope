
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
#  along with Luna. If not, see <http://www.gnu.org/licenses/>.
#
#  Please see LICENSE.txt for more details.
#
#  --------------------------------------------------------------------

"""Predict dock – Luna model-based prediction (e.g. brain age)."""

import threading
import urllib.request
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QMetaObject, Qt, QTimer, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..helpers import AuxiliaryWindow, is_dark_palette, screen_clamp
from ..runtime_paths import app_cache_root
from ..file_dialogs import open_file_name
from .soappops import MultiSelectComboBox


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

PREDICT_GITHUB_RAW = "https://raw.githubusercontent.com/remnrem/luna-api-notebooks/main/models"
PREDICT_FILE_SUFFIXES = ["-data.txt", "-features.txt", "-luna.txt"]

# Each entry: id, label, user_vars list of (name, tooltip), default_th, channel_var
PREDICT_MODELS = [
    {
        "id": "m1-adult-age",
        "label": "m1-adult-age  —  Brain Age (Sun 2019)",
        "user_vars": [("age", "Chronological age (years)")],
        "default_th": "3",
        "channel_var": "cen",
    },
]

_MODEL_BY_ID = {m["id"]: m for m in PREDICT_MODELS}

# Primary output columns to highlight
_KEY_COLS = ["YOBS", "Y1", "DIFF"]
_KEY_LABELS = {
    "YOBS": "YOBS\n(observed age)",
    "Y1":   "Y1\n(predicted age)",
    "DIFF": "DIFF\n(brain age gap)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _predict_cache_root() -> Path:
    return app_cache_root() / "predict"


def _model_dir(model_id: str) -> Path:
    return _predict_cache_root() / model_id


def _model_files_present(model_id: str) -> bool:
    d = _model_dir(model_id)
    return all((d / f"{model_id}{sfx}").exists() for sfx in PREDICT_FILE_SUFFIXES)


def _val_label(text: str, color: str = "#8ab4f8") -> QLabel:
    lbl = QLabel(text)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setMinimumWidth(130)
    lbl.setMinimumHeight(52)
    lbl.setFrameShape(QFrame.StyledPanel)
    lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    font = QFont()
    font.setPointSize(15)
    font.setBold(True)
    lbl.setFont(font)
    lbl.setStyleSheet(
        f"QLabel {{ color: {color}; border: 1px solid #444; border-radius: 4px; padding: 4px 8px; }}"
    )
    return lbl


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class PredictMixin:
    """Adds the Predict floating dock to the Lunascope controller."""

    _PREDICT_FLOAT_SIZE = (680, 560)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_predict(self):
        self._predict_last_df: pd.DataFrame | None = None
        self._predict_cancel_event = threading.Event()
        self._predict_proj_running = False

        # ---- Dock --------------------------------------------------------
        dock = AuxiliaryWindow("Predict", self.ui)
        dock.setObjectName("dock_predict")
        dock.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        dock.setWindowFlag(Qt.WindowMaximizeButtonHint, True)

        root = QWidget(dock)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # ---- Model + cache row -------------------------------------------
        model_frame = QFrame(root)
        ml = QHBoxLayout(model_frame)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(6)

        model_lbl = QLabel("Model:")
        model_combo = QComboBox()
        model_combo.setObjectName("predict_model_combo")
        for m in PREDICT_MODELS:
            model_combo.addItem(m["label"], m["id"])
        model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        path_lbl = QLabel("Cache:")
        path_edit = QLineEdit()
        path_edit.setObjectName("predict_path_edit")
        path_edit.setReadOnly(True)
        path_edit.setText(str(_predict_cache_root()))
        path_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        path_edit.setToolTip("Model files are downloaded into this folder")

        get_btn = QPushButton("Get…")
        get_btn.setObjectName("predict_get_btn")
        get_btn.setFixedWidth(56)
        get_btn.setToolTip("Download model files from GitHub")

        ml.addWidget(model_lbl)
        ml.addWidget(model_combo)
        ml.addWidget(path_lbl)
        ml.addWidget(path_edit)
        ml.addWidget(get_btn)
        outer.addWidget(model_frame)

        # ---- Status label ------------------------------------------------
        status_lbl = QLabel("")
        status_lbl.setObjectName("predict_status_lbl")
        status_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        outer.addWidget(status_lbl)

        # ---- Channel row ------------------------------------------------
        ch_frame = QFrame(root)
        cl = QHBoxLayout(ch_frame)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(6)
        ch_lbl = QLabel("EEG channel(s):")
        ch_combo = MultiSelectComboBox()
        ch_combo.setObjectName("predict_ch_combo")
        ch_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        cl.addWidget(ch_lbl)
        cl.addWidget(ch_combo)
        outer.addWidget(ch_frame)

        # ---- Variables section -------------------------------------------
        var_sep = QFrame(root)
        var_sep.setFrameShape(QFrame.HLine)
        var_sep.setFrameShadow(QFrame.Sunken)
        outer.addWidget(var_sep)

        var_frame = QFrame(root)
        vl = QHBoxLayout(var_frame)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(8)

        # Placeholder – populated dynamically when model changes
        self._predict_var_widgets: dict[str, QLineEdit] = {}
        self._predict_var_frame = var_frame
        self._predict_var_layout = vl

        outer.addWidget(var_frame)

        # Threshold (always shown)
        th_frame = QFrame(root)
        tl = QHBoxLayout(th_frame)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(6)
        th_lbl = QLabel("Outlier threshold (SD):")
        th_edit = QLineEdit("3")
        th_edit.setObjectName("predict_th_edit")
        th_edit.setMaximumWidth(60)
        tl.addWidget(th_lbl)
        tl.addWidget(th_edit)
        tl.addStretch(1)
        outer.addWidget(th_frame)

        # Vars file row
        varsfile_frame = QFrame(root)
        fl = QHBoxLayout(varsfile_frame)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(6)
        varsfile_lbl = QLabel("Vars file (optional):")
        varsfile_edit = QLineEdit()
        varsfile_edit.setObjectName("predict_varsfile_edit")
        varsfile_edit.setPlaceholderText("Path to per-individual variables file…")
        varsfile_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        varsfile_edit.setToolTip(
            "Tab-delimited file with IID column + one column per variable (e.g. age).\n"
            "In project mode each individual's row is looked up by ID.\n"
            "Values entered in the dialog above override the file."
        )
        varsfile_browse = QPushButton("Browse…")
        varsfile_browse.setObjectName("predict_varsfile_browse")
        varsfile_browse.setFixedWidth(72)
        fl.addWidget(varsfile_lbl)
        fl.addWidget(varsfile_edit)
        fl.addWidget(varsfile_browse)
        outer.addWidget(varsfile_frame)

        # ---- Run buttons -------------------------------------------------
        run_frame = QFrame(root)
        rl = QHBoxLayout(run_frame)
        rl.setContentsMargins(0, 2, 0, 2)
        rl.setSpacing(6)

        run_btn = QPushButton("Run Predict")
        run_btn.setObjectName("predict_run_btn")
        run_btn.setFixedHeight(28)
        run_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        run_btn.setToolTip("Run predict for the currently loaded individual")

        run_all_btn = QPushButton("Run All")
        run_all_btn.setObjectName("predict_run_all_btn")
        run_all_btn.setFixedHeight(28)
        run_all_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        run_all_btn.setToolTip(
            "Run predict for every visible sample in the S-list.\n"
            "Click again (shows as Stop) to interrupt after the current individual."
        )

        rl.addWidget(run_btn)
        rl.addWidget(run_all_btn)
        outer.addWidget(run_frame)

        # ---- Results divider --------------------------------------------
        res_sep = QFrame(root)
        res_sep.setFrameShape(QFrame.HLine)
        res_sep.setFrameShadow(QFrame.Sunken)
        outer.addWidget(res_sep)

        # Key-column badge row
        badge_frame = QFrame(root)
        bl = QHBoxLayout(badge_frame)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)

        badge_yobs = _val_label("—", "#8ab4f8")
        badge_y1   = _val_label("—", "#81c995")
        badge_diff = _val_label("—", "#f28b82")

        badge_yobs_title = QLabel(_KEY_LABELS["YOBS"])
        badge_yobs_title.setAlignment(Qt.AlignCenter)
        badge_y1_title   = QLabel(_KEY_LABELS["Y1"])
        badge_y1_title.setAlignment(Qt.AlignCenter)
        badge_diff_title = QLabel(_KEY_LABELS["DIFF"])
        badge_diff_title.setAlignment(Qt.AlignCenter)

        for title, badge in [
            (badge_yobs_title, badge_yobs),
            (badge_y1_title,   badge_y1),
            (badge_diff_title, badge_diff),
        ]:
            col_w = QWidget()
            col_l = QVBoxLayout(col_w)
            col_l.setContentsMargins(0, 0, 0, 0)
            col_l.setSpacing(2)
            col_l.addWidget(title)
            col_l.addWidget(badge)
            bl.addWidget(col_w, 1)

        outer.addWidget(badge_frame)

        # Full results table
        result_table = QTableWidget(0, 0)
        result_table.setObjectName("predict_result_table")
        result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        result_table.setSelectionBehavior(QTableWidget.SelectRows)
        result_table.verticalHeader().setVisible(False)
        result_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        fnt = QFont("Courier New")
        fnt.setPointSize(9)
        result_table.setFont(fnt)
        outer.addWidget(result_table, 1)

        # ---- Attach dock -------------------------------------------------
        dock.setWidget(root)
        _w, _h = screen_clamp(*self._PREDICT_FLOAT_SIZE)
        dock.resize(_w, _h)
        parent_geo = self.ui.frameGeometry()
        dock.move(
            parent_geo.x() + (parent_geo.width()  - _w) // 2,
            parent_geo.y() + 80,
        )
        dock.hide()

        # ---- Store refs --------------------------------------------------
        self.ui.dock_predict           = dock
        self.ui.predict_model_combo    = model_combo
        self.ui.predict_path_edit      = path_edit
        self.ui.predict_get_btn        = get_btn
        self.ui.predict_status_lbl     = status_lbl
        self.ui.predict_ch_combo       = ch_combo
        self.ui.predict_th_edit        = th_edit
        self.ui.predict_varsfile_edit  = varsfile_edit
        self.ui.predict_varsfile_browse = varsfile_browse
        self.ui.predict_run_btn        = run_btn
        self.ui.predict_run_all_btn    = run_all_btn
        self.ui.predict_badge_yobs     = badge_yobs
        self.ui.predict_badge_y1       = badge_y1
        self.ui.predict_badge_diff     = badge_diff
        self.ui.predict_result_table   = result_table

        # ---- Dark-mode styling -------------------------------------------
        if is_dark_palette():
            root.setStyleSheet("QLabel { color: #d7e3f4; }")

        # ---- Wire signals ------------------------------------------------
        model_combo.currentIndexChanged.connect(self._on_predict_model_changed)
        get_btn.clicked.connect(self._download_predict_model)
        varsfile_browse.clicked.connect(self._browse_predict_varsfile)
        run_btn.clicked.connect(self._run_predict)
        run_all_btn.clicked.connect(self._on_predict_run_all_clicked)

        self.sig_predict_proj_progress.connect(self._predict_proj_progress_update, Qt.QueuedConnection)
        self.sig_predict_proj_row.connect(self._predict_proj_row_done, Qt.QueuedConnection)
        self.sig_predict_proj_done.connect(self._predict_proj_done, Qt.QueuedConnection)
        self.sig_predict_proj_failed.connect(self._predict_proj_failed, Qt.QueuedConnection)

        # Populate var fields for default model
        self._rebuild_predict_var_fields()
        self._update_predict_status()

    # ------------------------------------------------------------------
    # Model changed
    # ------------------------------------------------------------------

    def _on_predict_model_changed(self, _idx: int):
        self._rebuild_predict_var_fields()
        self._update_predict_status()

    def _current_predict_model(self) -> dict | None:
        model_id = self.ui.predict_model_combo.currentData()
        return _MODEL_BY_ID.get(model_id)

    def _rebuild_predict_var_fields(self):
        """Recreate the dynamic variable-entry QLineEdits for the current model."""
        layout = self._predict_var_layout
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._predict_var_widgets.clear()

        model = self._current_predict_model()
        if model is None:
            return

        for var_name, var_tooltip in model.get("user_vars", []):
            lbl = QLabel(f"{var_name}:")
            lbl.setToolTip(var_tooltip)
            edit = QLineEdit()
            edit.setObjectName(f"predict_var_{var_name}")
            edit.setPlaceholderText(var_tooltip)
            edit.setMaximumWidth(100)
            edit.setToolTip(var_tooltip)
            layout.addWidget(lbl)
            layout.addWidget(edit)
            self._predict_var_widgets[var_name] = edit

        layout.addStretch(1)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def _update_predict_status(self):
        model = self._current_predict_model()
        if model is None:
            self.ui.predict_status_lbl.setText("")
            return
        mid = model["id"]
        if _model_files_present(mid):
            self.ui.predict_status_lbl.setText(
                f"<span style='color:#81c995'>&#9679;</span>&nbsp;{mid} — ready"
            )
        else:
            self.ui.predict_status_lbl.setText(
                f"<span style='color:#f28b82'>&#9679;</span>&nbsp;{mid} — not downloaded&nbsp;"
                "(click <b>Get…</b> to download)"
            )
        self.ui.predict_status_lbl.setTextFormat(Qt.RichText)

    # ------------------------------------------------------------------
    # Channel population (called when an individual is loaded)
    # ------------------------------------------------------------------

    def _update_predict_channels(self):
        if not hasattr(self, "p"):
            return
        prev = (
            self.ui.predict_ch_combo.checked_items()
            if hasattr(self.ui.predict_ch_combo, "checked_items")
            else []
        )
        try:
            df = self.p.headers()
        except Exception:
            df = None

        if df is not None:
            chs = df.loc[df["SR"] >= 32, "CH"].tolist()
        else:
            chs = []

        if hasattr(self.ui.predict_ch_combo, "set_items"):
            self.ui.predict_ch_combo.set_items(chs, checked_labels=prev)
        else:
            self.ui.predict_ch_combo.clear()
            self.ui.predict_ch_combo.addItems(chs)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def _download_predict_model(self):
        model = self._current_predict_model()
        if model is None:
            return
        mid = model["id"]
        dest_dir = _model_dir(mid)
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self.ui, "Predict Download Error", str(e))
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        errors: list[str] = []
        try:
            for sfx in PREDICT_FILE_SUFFIXES:
                fname = f"{mid}{sfx}"
                url = f"{PREDICT_GITHUB_RAW}/{fname}"
                dest = dest_dir / fname
                try:
                    with urllib.request.urlopen(url, timeout=60) as resp:
                        dest.write_bytes(resp.read())
                except Exception as e:
                    errors.append(f"{fname}: {e}")
        finally:
            QApplication.restoreOverrideCursor()

        if errors:
            QMessageBox.critical(
                self.ui,
                "Predict Download Error",
                "Some files failed to download:\n\n" + "\n".join(errors),
            )
        else:
            self._update_predict_status()
            QMessageBox.information(
                self.ui,
                "Predict",
                f"Downloaded {mid} model files to:\n{dest_dir}",
            )

    # ------------------------------------------------------------------
    # Vars file browser
    # ------------------------------------------------------------------

    def _browse_predict_varsfile(self):
        path, _ = open_file_name(
            self.ui,
            "Select vars file",
            file_filter="Text files (*.txt *.tsv *.csv);;All files (*)",
        )
        if path:
            self.ui.predict_varsfile_edit.setText(path)

    # ------------------------------------------------------------------
    # Vars file helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_predict_varsfile(path: str) -> "pd.DataFrame | None":
        """Load a vars file → DataFrame indexed by IID column. Returns None on failure."""
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        try:
            sep = "\t" if path.lower().endswith((".tsv", ".txt")) else ","
            df = pd.read_csv(path, sep=sep, dtype=str, encoding="utf-8-sig")
            if df.empty:
                return None
            for col_name in ("IID", "ID", "iid", "id"):
                if col_name in df.columns:
                    return df.set_index(col_name)
            return df.set_index(df.columns[0])
        except Exception:
            return None

    @staticmethod
    def _resolve_predict_var(
        id_str: str,
        var_name: str,
        direct_val: str,
        vars_df: "pd.DataFrame | None",
    ) -> "str | None":
        """Return resolved value for var_name: dialog first, then vars file, else None."""
        if direct_val:
            return direct_val
        if vars_df is not None:
            try:
                if id_str in vars_df.index and var_name in vars_df.columns:
                    val = str(vars_df.loc[id_str, var_name]).strip()
                    if val and val.lower() not in ("nan", "na", "none", ""):
                        return val
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Run (single individual)
    # ------------------------------------------------------------------

    def _run_predict(self):
        if not hasattr(self, "p"):
            QMessageBox.critical(self.ui, "Predict", "No individual loaded.")
            return

        model = self._current_predict_model()
        if model is None:
            QMessageBox.critical(self.ui, "Predict", "No model selected.")
            return

        mid = model["id"]
        if not _model_files_present(mid):
            QMessageBox.critical(
                self.ui,
                "Predict",
                f"Model files for '{mid}' are not downloaded.\n"
                "Click Get… to download them first.",
            )
            return

        channels = (
            self.ui.predict_ch_combo.checked_items()
            if hasattr(self.ui.predict_ch_combo, "checked_items")
            else [self.ui.predict_ch_combo.currentText().strip()]
        )
        channels = [c for c in channels if c]
        if not channels:
            QMessageBox.critical(self.ui, "Predict", "No EEG channel(s) selected.")
            return

        # Determine current individual ID for vars file lookup
        id_str = self._predict_current_id()

        # Direct vars from dialog widgets
        direct_vars = {
            var_name: edit.text().strip()
            for var_name, edit in self._predict_var_widgets.items()
        }

        th = self.ui.predict_th_edit.text().strip() or "3"
        vars_file = self.ui.predict_varsfile_edit.text().strip()
        vars_df = self._load_predict_varsfile(vars_file)

        # Resolve required vars; error if any are missing
        resolved_vars: dict[str, str] = {}
        missing_vars: list[str] = []
        for var_name, _ in model.get("user_vars", []):
            val = self._resolve_predict_var(id_str, var_name, direct_vars.get(var_name, ""), vars_df)
            if val is not None:
                resolved_vars[var_name] = val
            else:
                missing_vars.append(var_name)

        if missing_vars:
            msg = (
                f"Required variable(s) not set: {', '.join(missing_vars)}\n\n"
                "Enter a value in the dialog above, or provide a vars file whose "
                "IID column matches this individual's ID."
            )
            QMessageBox.critical(self.ui, "Predict", msg)
            return

        if getattr(self, "_busy", False):
            return

        self._busy = True
        self._buttons(False)
        self.sb_progress.setVisible(True)
        self.sb_progress.setRange(0, 0)
        self.sb_progress.setFormat("Running predict…")
        self.lock_ui()
        QTimer.singleShot(
            0,
            lambda: self._start_predict_worker(mid, channels, resolved_vars, th),
        )

    def _predict_current_id(self) -> str:
        """Return the ID of the currently selected slist row (column 0)."""
        view = self.ui.tbl_slist
        slist_model = view.model()
        curr_idx = view.currentIndex()
        if curr_idx.isValid() and slist_model is not None:
            col0 = slist_model.index(curr_idx.row(), 0)
            return str(slist_model.data(col0, Qt.DisplayRole) or "").strip()
        return ""

    def _start_predict_worker(
        self,
        model_id: str,
        channels: list[str],
        resolved_vars: dict[str, str],
        th: str,
    ):
        if not getattr(self, "_busy", False):
            return

        fut = self._exec.submit(
            self._derive_predict,
            self.p,
            model_id,
            channels,
            resolved_vars,
            th,
        )

        def _done(_f=fut):
            try:
                self._last_result = _f.result()
                QMetaObject.invokeMethod(self, "_predict_done_ok", Qt.QueuedConnection)
            except Exception as e:
                self._last_exc = e
                self._last_tb = f"{type(e).__name__}: {e}"
                QMetaObject.invokeMethod(self, "_predict_done_err", Qt.QueuedConnection)

        fut.add_done_callback(_done)

    @staticmethod
    def _derive_predict(
        p,
        model_id: str,
        channels: list[str],
        resolved_vars: dict[str, str],
        th: str,
    ) -> pd.DataFrame:
        from lunapi.results import cmdfile

        model = _MODEL_BY_ID[model_id]
        mpath = str(_model_dir(model_id))
        luna_txt = _model_dir(model_id) / f"{model_id}-luna.txt"

        p.var("mpath", mpath)
        p.var("th", th)

        ch_var = model.get("channel_var", "cen")
        p.var(ch_var, ",".join(channels))

        for var_name, val in resolved_vars.items():
            p.var(var_name, val)

        p.eval_lunascope(cmdfile(str(luna_txt)))
        df = p.table("PREDICT")
        if df is None:
            df = pd.DataFrame()
        return df

    @Slot()
    def _predict_done_ok(self):
        try:
            df: pd.DataFrame = self._last_result
            self._predict_last_df = df
            self._render_predict_results(df)
        finally:
            self.unlock_ui()
            self._busy = False
            self._buttons(True)
            self.sb_progress.setRange(0, 100)
            self.sb_progress.setValue(0)
            self.sb_progress.setVisible(False)

    @Slot()
    def _predict_done_err(self):
        try:
            QMessageBox.critical(self.ui, "Predict error", self._last_tb)
        finally:
            self.unlock_ui()
            self._busy = False
            self._buttons(True)
            self.sb_progress.setRange(0, 100)
            self.sb_progress.setValue(0)
            self.sb_progress.setVisible(False)

    # ------------------------------------------------------------------
    # Run All (project mode)
    # ------------------------------------------------------------------

    def _on_predict_run_all_clicked(self):
        if self._predict_proj_running:
            self._request_predict_cancel()
        else:
            self._run_predict_all()

    def _run_predict_all(self):
        if getattr(self, "_busy", False):
            return

        model = self._current_predict_model()
        if model is None:
            QMessageBox.critical(self.ui, "Predict", "No model selected.")
            return

        mid = model["id"]
        if not _model_files_present(mid):
            QMessageBox.critical(
                self.ui,
                "Predict",
                f"Model files for '{mid}' are not downloaded.\n"
                "Click Get… to download them first.",
            )
            return

        channels = (
            self.ui.predict_ch_combo.checked_items()
            if hasattr(self.ui.predict_ch_combo, "checked_items")
            else [self.ui.predict_ch_combo.currentText().strip()]
        )
        channels = [c for c in channels if c]
        if not channels:
            QMessageBox.critical(self.ui, "Predict", "No EEG channel(s) selected.")
            return

        # Collect visible slist rows
        view = self.ui.tbl_slist
        slist_model = view.model()
        if not slist_model or slist_model.rowCount() == 0:
            QMessageBox.critical(self.ui, "Predict", "No samples in the S-list.")
            return

        records: list[str] = []
        for row in range(slist_model.rowCount()):
            idx = slist_model.index(row, 0)
            label = str(slist_model.data(idx, Qt.DisplayRole) or "").strip()
            if label:
                records.append(label)

        if not records:
            QMessageBox.critical(self.ui, "Predict", "No samples found in the S-list.")
            return

        # Direct vars from dialog (override for all individuals)
        direct_vars = {
            var_name: edit.text().strip()
            for var_name, edit in self._predict_var_widgets.items()
        }

        th = self.ui.predict_th_edit.text().strip() or "3"
        vars_file = self.ui.predict_varsfile_edit.text().strip()

        # Guard: need at least one source for required vars
        required_vars = [v for v, _ in model.get("user_vars", [])]
        if required_vars:
            has_all_direct = all(direct_vars.get(v) for v in required_vars)
            if not has_all_direct and not vars_file:
                QMessageBox.critical(
                    self.ui,
                    "Predict",
                    f"Required variable(s): {', '.join(required_vars)}\n\n"
                    "Either enter a value in the dialog (applied to all individuals) "
                    "or provide a vars file with an IID column and these variable columns.",
                )
                return

        # Reset state
        self._predict_cancel_event.clear()
        self._predict_proj_running = True

        # Clear table & badges
        self.ui.predict_result_table.setRowCount(0)
        self.ui.predict_result_table.setColumnCount(0)
        self.ui.predict_badge_yobs.setText("—")
        self.ui.predict_badge_y1.setText("—")
        self.ui.predict_badge_diff.setText("—")

        # Lock UI
        self._busy = True
        self._buttons(False)
        self.ui.predict_run_btn.setEnabled(False)
        self.ui.predict_run_all_btn.setText("Stop")
        self.sb_progress.setVisible(True)
        self.sb_progress.setRange(0, len(records))
        self.sb_progress.setValue(0)
        self.sb_progress.setFormat(f"0 / {len(records)}")
        self.lock_ui(
            f"Running predict for {len(records)} samples…\n\n"
            "Click Stop to interrupt after the current individual."
        )

        fut = self._exec.submit(
            self._predict_project_worker,
            records, mid, channels, direct_vars, th, vars_file,
        )

        def _done(_f=fut):
            try:
                self.sig_predict_proj_done.emit(_f.result())
            except Exception as e:
                self.sig_predict_proj_failed.emit(f"{type(e).__name__}: {e}")

        fut.add_done_callback(_done)

    def _predict_project_worker(
        self,
        records: list[str],
        model_id: str,
        channels: list[str],
        direct_vars: dict[str, str],
        th: str,
        vars_file: str,
    ) -> dict:
        vars_df = self._load_predict_varsfile(vars_file)
        model = _MODEL_BY_ID[model_id]
        required_vars = [v for v, _ in model.get("user_vars", [])]

        all_dfs: list[pd.DataFrame] = []
        cancelled = False

        for i, id_str in enumerate(records, start=1):
            if self._predict_cancel_event.is_set():
                cancelled = True
                break

            # Resolve vars for this individual
            resolved: dict[str, str] = {}
            missing: list[str] = []
            for var_name in required_vars:
                val = self._resolve_predict_var(
                    id_str, var_name, direct_vars.get(var_name, ""), vars_df
                )
                if val is not None:
                    resolved[var_name] = val
                else:
                    missing.append(var_name)

            if missing:
                # Skip this individual but record an NA row
                na_row = pd.DataFrame({"ID": [id_str], "SKIPPED": [f"missing: {', '.join(missing)}"]})
                all_dfs.append(na_row)
                self.sig_predict_proj_row.emit(na_row)
                self.sig_predict_proj_progress.emit(i, len(records))
                continue

            try:
                self.proj.clear_vars()
                self.proj.reinit()
                p = self.proj.inst(id_str)
                row_df = self._derive_predict(p, model_id, channels, resolved, th)
                if not row_df.empty and "ID" not in row_df.columns:
                    row_df.insert(0, "ID", id_str)
                all_dfs.append(row_df)
                self.sig_predict_proj_row.emit(row_df)
            except Exception as e:
                err_row = pd.DataFrame({"ID": [id_str], "ERROR": [str(e)]})
                all_dfs.append(err_row)
                self.sig_predict_proj_row.emit(err_row)

            self.sig_predict_proj_progress.emit(i, len(records))

        combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        return {"df": combined, "cancelled": cancelled, "n": len(records)}

    def _request_predict_cancel(self):
        if not self._predict_proj_running:
            return
        self._predict_cancel_event.set()
        self.ui.predict_run_all_btn.setEnabled(False)
        self.ui.predict_run_all_btn.setText("Stopping…")

    # ------------------------------------------------------------------
    # Project mode Slots
    # ------------------------------------------------------------------

    @Slot(int, int)
    def _predict_proj_progress_update(self, i: int, n: int):
        self.sb_progress.setValue(i)
        self.sb_progress.setFormat(f"{i} / {n}")

    @Slot(object)
    def _predict_proj_row_done(self, row_df: pd.DataFrame):
        """Append one individual's result to the table as it arrives."""
        if row_df is None or row_df.empty:
            return
        self._append_predict_table_rows(row_df)

    @Slot(object)
    def _predict_proj_done(self, result: dict):
        try:
            df: pd.DataFrame = result.get("df", pd.DataFrame())
            self._predict_last_df = df
            # Update badges from first valid (non-skipped/error) row
            if not df.empty and "Y1" in df.columns:
                valid = df[pd.to_numeric(df["Y1"], errors="coerce").notna()]
                if not valid.empty:
                    self._render_predict_badges(valid.iloc[0:1])
        finally:
            self._predict_proj_cleanup()

    @Slot(str)
    def _predict_proj_failed(self, msg: str):
        try:
            QMessageBox.critical(self.ui, "Predict error", msg)
        finally:
            self._predict_proj_cleanup()

    def _predict_proj_cleanup(self):
        self.unlock_ui()
        self._busy = False
        self._predict_proj_running = False
        self._predict_cancel_event.clear()
        self._buttons(True)
        self.ui.predict_run_btn.setEnabled(True)
        self.ui.predict_run_all_btn.setText("Run All")
        self.ui.predict_run_all_btn.setEnabled(True)
        self.sb_progress.setRange(0, 100)
        self.sb_progress.setValue(0)
        self.sb_progress.setVisible(False)

    # ------------------------------------------------------------------
    # Render results
    # ------------------------------------------------------------------

    def _append_predict_table_rows(self, df: pd.DataFrame):
        """Append rows from df to the results table, creating columns on first call."""
        tbl = self.ui.predict_result_table

        # First batch: set up columns
        if tbl.columnCount() == 0 and not df.empty:
            cols = list(df.columns)
            tbl.setColumnCount(len(cols))
            tbl.setHorizontalHeaderLabels(cols)

        if tbl.columnCount() == 0:
            return

        cols = [
            tbl.horizontalHeaderItem(c).text()
            for c in range(tbl.columnCount())
        ]

        for _, row in df.iterrows():
            r = tbl.rowCount()
            tbl.insertRow(r)
            for c, col in enumerate(cols):
                val = row.get(col, None)
                if val is None:
                    txt = "NA"
                else:
                    try:
                        num = float(val)
                        txt = f"{num:.4f}" if not float(num).is_integer() else f"{int(num)}"
                    except (TypeError, ValueError, AttributeError):
                        txt = str(val)
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                if col in _KEY_COLS:
                    item.setForeground(
                        Qt.cyan   if col == "Y1"   else
                        Qt.green  if col == "YOBS" else
                        Qt.yellow
                    )
                tbl.setItem(r, c, item)

        tbl.resizeColumnsToContents()

    def _render_predict_badges(self, df: pd.DataFrame):
        def _fmt(col: str) -> str:
            if df.empty or col not in df.columns:
                return "—"
            try:
                val = pd.to_numeric(df[col].iloc[0], errors="coerce")
                return f"{val:.2f}" if pd.notna(val) else "—"
            except Exception:
                return "—"

        self.ui.predict_badge_yobs.setText(_fmt("YOBS"))
        self.ui.predict_badge_y1.setText(_fmt("Y1"))

        diff_text = _fmt("DIFF")
        self.ui.predict_badge_diff.setText(diff_text)
        try:
            diff_val = float(diff_text)
            if diff_val > 0:
                self.ui.predict_badge_diff.setStyleSheet(
                    "QLabel { color: #f28b82; border: 1px solid #444; border-radius: 4px;"
                    "padding: 4px 8px; font-size: 15pt; font-weight: bold; }"
                )
            elif diff_val < 0:
                self.ui.predict_badge_diff.setStyleSheet(
                    "QLabel { color: #81c995; border: 1px solid #444; border-radius: 4px;"
                    "padding: 4px 8px; font-size: 15pt; font-weight: bold; }"
                )
        except ValueError:
            pass

    def _render_predict_results(self, df: pd.DataFrame):
        """Full render for single-individual result."""
        self._render_predict_badges(df)

        tbl = self.ui.predict_result_table
        tbl.setRowCount(0)
        tbl.setColumnCount(0)
        if df.empty:
            return

        self._append_predict_table_rows(df)
