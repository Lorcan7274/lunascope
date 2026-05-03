
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
            "Optional tab-delimited vars file. The IID column must match the current individual.\n"
            "Direct variable entries above take precedence over file values."
        )
        varsfile_browse = QPushButton("Browse…")
        varsfile_browse.setObjectName("predict_varsfile_browse")
        varsfile_browse.setFixedWidth(72)
        fl.addWidget(varsfile_lbl)
        fl.addWidget(varsfile_edit)
        fl.addWidget(varsfile_browse)
        outer.addWidget(varsfile_frame)

        # ---- Run button --------------------------------------------------
        run_frame = QFrame(root)
        rl = QHBoxLayout(run_frame)
        rl.setContentsMargins(0, 2, 0, 2)
        rl.setSpacing(6)
        run_btn = QPushButton("Run Predict")
        run_btn.setObjectName("predict_run_btn")
        run_btn.setFixedHeight(28)
        run_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        rl.addWidget(run_btn)
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
        self.ui.dock_predict         = dock
        self.ui.predict_model_combo  = model_combo
        self.ui.predict_path_edit    = path_edit
        self.ui.predict_get_btn      = get_btn
        self.ui.predict_status_lbl   = status_lbl
        self.ui.predict_ch_combo     = ch_combo
        self.ui.predict_th_edit      = th_edit
        self.ui.predict_varsfile_edit  = varsfile_edit
        self.ui.predict_varsfile_browse = varsfile_browse
        self.ui.predict_run_btn      = run_btn
        self.ui.predict_badge_yobs   = badge_yobs
        self.ui.predict_badge_y1     = badge_y1
        self.ui.predict_badge_diff   = badge_diff
        self.ui.predict_result_table = result_table

        # ---- Dark-mode styling -------------------------------------------
        if is_dark_palette():
            root.setStyleSheet("QLabel { color: #d7e3f4; }")

        # ---- Wire signals ------------------------------------------------
        model_combo.currentIndexChanged.connect(self._on_predict_model_changed)
        get_btn.clicked.connect(self._download_predict_model)
        varsfile_browse.clicked.connect(self._browse_predict_varsfile)
        run_btn.clicked.connect(self._run_predict)

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
        # Remove old widgets
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
    # Run
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

        # Channels
        channels = (
            self.ui.predict_ch_combo.checked_items()
            if hasattr(self.ui.predict_ch_combo, "checked_items")
            else [self.ui.predict_ch_combo.currentText().strip()]
        )
        channels = [c for c in channels if c]
        if not channels:
            QMessageBox.critical(self.ui, "Predict", "No EEG channel(s) selected.")
            return

        # User vars
        direct_vars: dict[str, str] = {}
        for var_name, edit in self._predict_var_widgets.items():
            val = edit.text().strip()
            direct_vars[var_name] = val  # empty = not set directly

        th = self.ui.predict_th_edit.text().strip() or "3"
        vars_file = self.ui.predict_varsfile_edit.text().strip()

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
            lambda: self._start_predict_worker(mid, channels, direct_vars, th, vars_file),
        )

    def _start_predict_worker(
        self,
        model_id: str,
        channels: list[str],
        direct_vars: dict[str, str],
        th: str,
        vars_file: str,
    ):
        if not getattr(self, "_busy", False):
            return

        fut = self._exec.submit(
            self._derive_predict,
            self.p,
            model_id,
            channels,
            direct_vars,
            th,
            vars_file,
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
        direct_vars: dict[str, str],
        th: str,
        vars_file: str,
    ) -> pd.DataFrame:
        from lunapi.results import cmdfile

        model = _MODEL_BY_ID[model_id]
        mpath = str(_model_dir(model_id))
        luna_txt = _model_dir(model_id) / f"{model_id}-luna.txt"

        # Load vars file first (direct entries override below)
        if vars_file:
            p.var("vars", vars_file)

        # Set automatic vars
        p.var("mpath", mpath)
        p.var("th", th)

        # Channel var
        ch_var = model.get("channel_var", "cen")
        p.var(ch_var, ",".join(channels))

        # Direct user vars (skip if empty — let vars file supply them)
        for var_name, val in direct_vars.items():
            if val:
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
    # Render results
    # ------------------------------------------------------------------

    def _render_predict_results(self, df: pd.DataFrame):
        # Update badge labels
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
        # Tint DIFF badge: positive = older-than-expected (warm), negative = cooler
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

        # Populate full table
        tbl = self.ui.predict_result_table
        if df.empty:
            tbl.setRowCount(0)
            tbl.setColumnCount(0)
            return

        cols = list(df.columns)
        tbl.setColumnCount(len(cols))
        tbl.setRowCount(len(df))
        tbl.setHorizontalHeaderLabels(cols)

        for r, row in df.iterrows():
            for c, col in enumerate(cols):
                val = row[col]
                try:
                    num = float(val)
                    txt = f"{num:.4f}" if not num.is_integer() else f"{int(num)}"
                except (TypeError, ValueError, AttributeError):
                    txt = str(val)
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                if col in _KEY_COLS:
                    item.setForeground(
                        Qt.cyan if col == "Y1" else
                        Qt.green if col == "YOBS" else
                        Qt.yellow
                    )
                tbl.setItem(r, c, item)

        tbl.resizeColumnsToContents()
