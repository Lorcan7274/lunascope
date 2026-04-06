
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

import io
import math
import traceback

import lunapi as lp
import numpy as np

from ..helpers import screen_clamp, is_dark_palette

from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import QMetaObject, Qt, Slot
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class ActigraphyMixin:

    _ACTIGRAPHY_FLOAT_SIZE = (1200, 820)

    _ACT_EPOCH_STEPS = [
        (60, "1 min"),
        (120, "2 min"),
        (300, "5 min"),
        (600, "10 min"),
        (900, "15 min"),
        (1800, "30 min"),
        (3600, "60 min"),
    ]

    def _init_actigraphy(self):
        self.actigraphycanvas = None
        self._act_last_data = None
        self._act_raw_annot_cache = {}

        dock = QDockWidget("Actigraphy", self.ui)
        dock.setObjectName("dock_actigraphy")
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        dock.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        dock.setWindowFlag(Qt.WindowMaximizeButtonHint, True)

        root = QWidget(dock)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        controls = QWidget(root)
        cgrid = QGridLayout(controls)
        cgrid.setContentsMargins(0, 0, 0, 0)
        cgrid.setHorizontalSpacing(8)
        cgrid.setVerticalSpacing(6)

        lab_sig = QLabel("Signal", controls)
        combo_sig = QComboBox(controls)
        combo_sig.setObjectName("combo_actigraphy")
        combo_sig.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        lab_epoch = QLabel("Epoch", controls)
        combo_epoch = QComboBox(controls)
        combo_epoch.setObjectName("combo_actigraphy_epoch")
        for secs, label in self._ACT_EPOCH_STEPS:
            combo_epoch.addItem(label, secs)

        lab_anchor = QLabel("Day", controls)
        combo_anchor = QComboBox(controls)
        combo_anchor.setObjectName("combo_actigraphy_anchor")
        combo_anchor.addItem("Midnight", 0)
        combo_anchor.addItem("Noon", 12 * 3600)

        lab_raster = QLabel("Raster", controls)
        combo_mode = QComboBox(controls)
        combo_mode.setObjectName("combo_actigraphy_mode")
        combo_mode.addItem("Single", "single")
        combo_mode.addItem("Double", "double")

        lab_render = QLabel("Render", controls)
        combo_render = QComboBox(controls)
        combo_render.setObjectName("combo_actigraphy_render")
        combo_render.addItem("Heatmap", "heatmap")
        combo_render.addItem("Trace: row min/max", "trace_minmax")
        combo_render.addItem("Trace: row 5/95", "trace_robust")
        combo_render.addItem("Trace: study min/max", "trace_study_minmax")
        combo_render.addItem("Trace: study 5/95", "trace_study_robust")

        lab_source = QLabel("Source", controls)
        combo_source = QComboBox(controls)
        combo_source.setObjectName("combo_actigraphy_source")
        combo_source.addItem("Signal", "signal")
        combo_source.addItem("Raw", "raw")
        combo_source.addItem("Annotation", "annot")

        check_overlay = QCheckBox("Annot overlay", controls)
        check_overlay.setObjectName("check_actigraphy_overlay")

        check_detail = QCheckBox("Show lower plots", controls)
        check_detail.setObjectName("check_actigraphy_detail")
        check_detail.setChecked(True)

        lab_palette = QLabel("Palette", controls)
        combo_palette = QComboBox(controls)
        combo_palette.setObjectName("combo_actigraphy_palette")
        combo_palette.addItem("Turbo", "turbo")
        combo_palette.addItem("Viridis", "viridis")
        combo_palette.addItem("Gray", "gray")
        combo_palette.addItem("Icefire", "icefire")
        combo_palette.setCurrentIndex(3)

        lab_annot = QLabel("Annot", controls)
        combo_annot = QComboBox(controls)
        combo_annot.setObjectName("combo_actigraphy_annot")
        combo_annot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        combo_annot.setEnabled(False)

        cgrid.addWidget(lab_sig, 0, 0)
        cgrid.addWidget(combo_sig, 0, 1)
        cgrid.addWidget(lab_render, 0, 2)
        cgrid.addWidget(combo_render, 0, 3)
        cgrid.addWidget(lab_raster, 0, 4)
        cgrid.addWidget(combo_mode, 0, 5)
        cgrid.addWidget(lab_epoch, 0, 6)
        cgrid.addWidget(combo_epoch, 0, 7)
        cgrid.addWidget(check_overlay, 0, 8, 1, 2)
        cgrid.addWidget(check_detail, 1, 8, 1, 2)

        cgrid.addWidget(lab_annot, 1, 0)
        cgrid.addWidget(combo_annot, 1, 1)
        cgrid.addWidget(lab_source, 1, 2)
        cgrid.addWidget(combo_source, 1, 3)
        cgrid.addWidget(lab_palette, 1, 4)
        cgrid.addWidget(combo_palette, 1, 5)
        cgrid.addWidget(lab_anchor, 1, 6)
        cgrid.addWidget(combo_anchor, 1, 7)

        cgrid.setColumnStretch(1, 1)
        cgrid.setColumnStretch(3, 1)
        cgrid.setColumnStretch(5, 1)
        cgrid.setColumnStretch(7, 1)
        cgrid.setColumnStretch(9, 1)

        summary = QWidget(root)
        srow = QHBoxLayout(summary)
        srow.setContentsMargins(0, 0, 0, 0)
        srow.setSpacing(10)

        lab_days = QLabel("Days: -", summary)
        lab_epoch_info = QLabel("Epoch: -", summary)
        lab_ra = QLabel("RA: -", summary)
        lab_daily = QLabel("Daily mean: -", summary)
        for lab in (lab_days, lab_epoch_info, lab_ra, lab_daily):
            lab.setFrameStyle(QFrame.Panel | QFrame.Sunken)
            lab.setMinimumWidth(120)
            srow.addWidget(lab)
        srow.addStretch(1)

        host = QFrame(root)
        host.setObjectName("host_actigraphy")
        host.setFrameShape(QFrame.StyledPanel)
        host.setLayout(QVBoxLayout())
        host.layout().setContentsMargins(0, 0, 0, 0)

        outer.addWidget(controls)
        outer.addWidget(summary)
        outer.addWidget(host, 1)

        dock.setWidget(root)
        self.ui.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.ui.dock_actigraphy = dock
        self.ui.combo_actigraphy = combo_sig
        self.ui.combo_actigraphy_epoch = combo_epoch
        self.ui.combo_actigraphy_anchor = combo_anchor
        self.ui.combo_actigraphy_mode = combo_mode
        self.ui.combo_actigraphy_render = combo_render
        self.ui.combo_actigraphy_source = combo_source
        self.ui.check_actigraphy_overlay = check_overlay
        self.ui.check_actigraphy_detail = check_detail
        self.ui.combo_actigraphy_palette = combo_palette
        self.ui.combo_actigraphy_annot = combo_annot
        self.ui.host_actigraphy = host
        self.ui.act_days = lab_days
        self.ui.act_epoch = lab_epoch_info
        self.ui.act_ra = lab_ra
        self.ui.act_daily = lab_daily

        self._set_actigraphy_epoch_default(multiday=False)
        self._update_actigraphy_summary()
        # The canvas frame always gets a dark gradient background.
        # The text-colour override is only applied on dark-themed systems so that
        # the control-row labels remain readable on Windows light themes.
        _color_rule = """
            QComboBox, QLabel {
                color: #d7e3f4;
            }
        """ if is_dark_palette() else ""
        root.setStyleSheet(
            """
            QFrame#host_actigraphy {
                border: 1px solid rgba(120,140,170,0.45);
                border-radius: 8px;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(14,20,32,255),
                    stop:1 rgba(22,32,48,255)
                );
            }
            QLabel {
                font-weight: 500;
            }
            """
            + _color_rule
        )
        summary.setStyleSheet(
            """
            QLabel {
                color: #e9eef7;
                background: rgba(27,38,59,0.92);
                border: 1px solid rgba(130,160,210,0.35);
                border-radius: 6px;
                padding: 4px 8px;
            }
            """
        )

        self.ui.dock_actigraphy.visibilityChanged.connect(self._on_actigraphy_visibility_changed)
        self.ui.combo_actigraphy.currentIndexChanged.connect(self._schedule_actigraphy_update)
        self.ui.combo_actigraphy_epoch.currentIndexChanged.connect(self._schedule_actigraphy_update)
        self.ui.combo_actigraphy_anchor.currentIndexChanged.connect(self._schedule_actigraphy_update)
        self.ui.combo_actigraphy_mode.currentIndexChanged.connect(self._schedule_actigraphy_update)
        self.ui.combo_actigraphy_render.currentIndexChanged.connect(self._schedule_actigraphy_update)
        self.ui.combo_actigraphy_source.currentIndexChanged.connect(self._on_actigraphy_source_changed)
        self.ui.check_actigraphy_overlay.toggled.connect(self._schedule_actigraphy_update)
        self.ui.check_actigraphy_detail.toggled.connect(self._refresh_actigraphy_view)
        self.ui.combo_actigraphy_palette.currentIndexChanged.connect(self._schedule_actigraphy_update)
        self.ui.combo_actigraphy_annot.currentIndexChanged.connect(self._schedule_actigraphy_update)

    def _ensure_actigraphy_canvas(self, *_args):
        if getattr(self, "actigraphycanvas", None) is not None:
            return self.actigraphycanvas

        layout = self.ui.host_actigraphy.layout()
        if layout is None:
            layout = QVBoxLayout()
            self.ui.host_actigraphy.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)

        from .mplcanvas import MplCanvas

        self.actigraphycanvas = MplCanvas(self.ui.host_actigraphy)
        layout.addWidget(self.actigraphycanvas)
        self.actigraphycanvas.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.actigraphycanvas.customContextMenuRequested.connect(self._actigraphy_context_menu)
        return self.actigraphycanvas

    def _set_actigraphy_epoch_default(self, multiday: bool):
        target = 900 if multiday else 300
        for i in range(self.ui.combo_actigraphy_epoch.count()):
            if self.ui.combo_actigraphy_epoch.itemData(i) == target:
                self.ui.combo_actigraphy_epoch.setCurrentIndex(i)
                return

    def _get_actigraphy_epoch_dur(self) -> int:
        val = self.ui.combo_actigraphy_epoch.currentData()
        return int(val) if val else 300

    def _get_actigraphy_anchor_secs(self) -> int:
        val = self.ui.combo_actigraphy_anchor.currentData()
        return int(val) if val is not None else 0

    def _get_actigraphy_anchor_label(self) -> str:
        return self.ui.combo_actigraphy_anchor.currentText() or "Midnight"

    def _get_actigraphy_render_mode(self) -> str:
        return self.ui.combo_actigraphy_render.currentData() or "heatmap"

    def _get_actigraphy_palette(self):
        from matplotlib import colormaps
        from matplotlib.colors import LinearSegmentedColormap

        name = self.ui.combo_actigraphy_palette.currentData() or "turbo"
        if name == "icefire":
            cmap = LinearSegmentedColormap.from_list(
                "icefire",
                [
                    "#0b132b",
                    "#143d59",
                    "#1f7a8c",
                    "#7bdff2",
                    "#f7b267",
                    "#f25f5c",
                    "#fff3b0",
                ],
                N=256,
            )
        else:
            cmap = colormaps[name].copy()
        if name == "gray":
            cmap.set_bad("#b45309")
        else:
            cmap.set_bad("#64748b")
        return cmap

    def _on_actigraphy_source_changed(self, *_args):
        is_annot = self.ui.combo_actigraphy_source.currentData() == "annot"
        self.ui.combo_actigraphy_annot.setEnabled(is_annot)
        self.ui.combo_actigraphy.setEnabled(not is_annot)
        self._schedule_actigraphy_update()

    def _on_actigraphy_visibility_changed(self, visible):
        if visible:
            self._present_actigraphy_dock()
            self._schedule_actigraphy_update()

    def _schedule_actigraphy_update(self, *_args):
        if not hasattr(self, "ui") or not hasattr(self.ui, "dock_actigraphy"):
            return
        if not self.ui.dock_actigraphy.isVisible():
            return
        QtCore.QTimer.singleShot(0, self._calc_actigraphy_if_ready)

    def _calc_actigraphy_if_ready(self):
        if getattr(self, "_busy", False):
            return
        if not hasattr(self, "p"):
            return
        self._calc_actigraphy()

    def _refresh_actigraphy_view(self, *_args):
        if getattr(self, "_busy", False):
            return
        if not getattr(self.ui, "dock_actigraphy", None) or not self.ui.dock_actigraphy.isVisible():
            return
        if self._act_last_data is not None:
            self._complete_actigraphy(self._act_last_data)
            return
        self._schedule_actigraphy_update()

    def _present_actigraphy_dock(self):
        dock = getattr(self.ui, "dock_actigraphy", None)
        if dock is None or not dock.isVisible():
            return

        was_floating = dock.isFloating()
        if not was_floating:
            dock.setFloating(True)

        target_w, target_h = screen_clamp(*self._ACTIGRAPHY_FLOAT_SIZE)
        if dock.width() < target_w or dock.height() < target_h:
            dock.resize(target_w, target_h)

        if was_floating:
            return

        try:
            parent_geom = self.ui.frameGeometry()
            center = parent_geom.center()
            rect = dock.frameGeometry()
            rect.moveCenter(center)
            dock.move(rect.topLeft())
        except Exception:
            pass

    def _update_actigraphy_list(self):
        combo = self.ui.combo_actigraphy
        annot_combo = self.ui.combo_actigraphy_annot
        current_sig = combo.currentText()
        current_annot = annot_combo.currentText()
        sig_blocker = QtCore.QSignalBlocker(combo)
        annot_blocker = QtCore.QSignalBlocker(annot_combo)
        try:
            combo.clear()
            annot_combo.clear()
            self._act_raw_annot_cache = {}

            if not hasattr(self, "p"):
                self._update_actigraphy_summary()
                return

            df = self.p.headers()
            if df is None or df.empty:
                self._update_actigraphy_summary()
                return

            combo.addItem("<none>")
            combo.addItems(df["CH"].tolist())
            annots = [x for x in self.p.edf.annots() if x != "SleepStage"]
            annot_combo.addItems(annots)

            sig_idx = combo.findText(current_sig) if current_sig else -1
            if sig_idx >= 0:
                combo.setCurrentIndex(sig_idx)
            elif combo.count() > 0:
                combo.setCurrentIndex(0)

            annot_idx = annot_combo.findText(current_annot) if current_annot else -1
            if annot_idx >= 0:
                annot_combo.setCurrentIndex(annot_idx)
            elif annot_combo.count() > 0:
                annot_combo.setCurrentIndex(0)
        finally:
            del sig_blocker
            del annot_blocker

        self._set_actigraphy_epoch_default(multiday=getattr(self, "multiday_mode", False))
        self._on_actigraphy_source_changed()
        self._update_actigraphy_summary()
        self._schedule_actigraphy_update()

    def _sync_multiday_actigraphy_dock(self):
        hyp_action = self.ui.dock_hypno.toggleViewAction()
        act_action = self.ui.dock_actigraphy.toggleViewAction()

        if getattr(self, "multiday_mode", False):
            hyp_action.setShortcut(QKeySequence())
            act_action.setShortcut(QKeySequence("Ctrl+7"))
            self.ui.dock_hypno.hide()
        else:
            act_action.setShortcut(QKeySequence())
            hyp_action.setShortcut(QKeySequence("Ctrl+7"))

    def _actigraphy_context_menu(self, pos):
        self._ensure_actigraphy_canvas()
        menu = QtWidgets.QMenu(self.actigraphycanvas)
        act_copy = menu.addAction("Copy to Clipboard")
        act_save = menu.addAction("Save Figure…")
        action = menu.exec(self.actigraphycanvas.mapToGlobal(pos))
        if action == act_copy:
            self._actigraphy_copy_to_clipboard()
        elif action == act_save:
            self._actigraphy_save_figure()

    def _actigraphy_copy_to_clipboard(self):
        self._ensure_actigraphy_canvas()
        buf = io.BytesIO()
        self.actigraphycanvas.figure.savefig(buf, format="png", bbox_inches="tight")
        img = QtGui.QImage.fromData(buf.getvalue(), "PNG")
        QtWidgets.QApplication.clipboard().setImage(img)

    def _actigraphy_save_figure(self):
        self._ensure_actigraphy_canvas()
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.actigraphycanvas,
            "Save Figure",
            "actigraphy",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if not fn:
            return
        self.actigraphycanvas.figure.savefig(fn, bbox_inches="tight")

    def _calc_actigraphy(self):
        self._ensure_actigraphy_canvas()

        if not hasattr(self, "p"):
            QMessageBox.critical(self.ui, "Error", "No instance attached")
            return

        source = self.ui.combo_actigraphy_source.currentData()
        epoch_dur = self._get_actigraphy_epoch_dur()
        if 86400 % epoch_dur != 0:
            QMessageBox.critical(self.ui, "Error", "Epoch duration must evenly divide 24 hours")
            return
        ch = self.ui.combo_actigraphy.currentText()
        if source != "annot":
            if not ch:
                QMessageBox.critical(self.ui, "Error", "No suitable signal for actigraphy summaries")
                return
            if ch != "<none>" and ch not in self.p.edf.channels():
                return
        annot = self.ui.combo_actigraphy_annot.currentText()
        if source == "annot" and not annot:
            QMessageBox.critical(self.ui, "Error", "Select an annotation for the raster")
            return
        raw_annots = list(self.ui.tbl_desc_annots.checked()) if getattr(self.ui, "check_actigraphy_overlay", None) and self.ui.check_actigraphy_overlay.isChecked() and hasattr(self.ui, "tbl_desc_annots") else []

        self._busy = True
        self._buttons(False)
        self.sb_progress.setVisible(True)
        self.sb_progress.setRange(0, 0)
        self.sb_progress.setFormat("Running…")
        self.lock_ui()

        fut = self._exec.submit(
            self._derive_actigraphy,
            self.p,
            ch,
            epoch_dur,
            float(self.ui.spin_win.value()),
            float(getattr(self, "ns", 0.0)),
            int(getattr(self, "_record_start_tod_secs", 0)),
            self._get_actigraphy_anchor_secs(),
            source,
            annot,
            raw_annots,
            self._get_actigraphy_render_mode(),
        )

        def _done(_f=fut):
            try:
                self._last_result = _f.result()
                QMetaObject.invokeMethod(self, "_actigraphy_done_ok", Qt.QueuedConnection)
            except Exception as e:
                self._last_exc = e
                self._last_tb = f"{type(e).__name__}: {e}"
                QMetaObject.invokeMethod(self, "_actigraphy_done_err", Qt.QueuedConnection)

        fut.add_done_callback(_done)

    @Slot()
    def _actigraphy_done_ok(self):
        try:
            self._act_last_data = self._last_result
            self._actigraphy_debug(
                "done_ok",
                source=self._last_result.get("source"),
                days=self._last_result.get("days"),
                matrix_shape=np.shape(self._last_result.get("matrix")),
                empty_state=self._last_result.get("empty_state"),
            )
            try:
                self._complete_actigraphy(self._last_result)
                self._actigraphy_debug("done_ok_complete")
            except Exception as e:
                tb = traceback.format_exc()
                self._last_exc = e
                self._last_tb = tb
                self._actigraphy_debug("done_ok_exception", error=f"{type(e).__name__}: {e}")
                print(tb, flush=True)
                QMessageBox.critical(self.ui, "Error rendering actigraphy", tb)
        finally:
            self.unlock_ui()
            self._busy = False
            self._buttons(True)
            self.sb_progress.setRange(0, 100)
            self.sb_progress.setValue(0)
            self.sb_progress.setVisible(False)

    @Slot()
    def _actigraphy_done_err(self):
        try:
            QMessageBox.critical(self.ui, "Error deriving actigraphy summary", self._last_tb)
        finally:
            self.unlock_ui()
            self._busy = False
            self._buttons(True)
            self.sb_progress.setRange(0, 100)
            self.sb_progress.setValue(0)
            self.sb_progress.setVisible(False)

    def _derive_actigraphy(self, p, ch, epoch_dur, winsor_limit, total_seconds, start_tod_secs, anchor_secs, source, annot, raw_annots=None, render_mode="heatmap"):
        raw_annots = list(raw_annots or [])
        signal_ch = None if ch in ("", "<none>") else ch
        self._actigraphy_debug(
            "derive_start",
            source=source,
            signal=signal_ch,
            annot=annot,
            raw_annots="|".join(map(str, raw_annots)),
            epoch_dur=epoch_dur,
            total_seconds=total_seconds,
            anchor_secs=anchor_secs,
            render_mode=render_mode,
        )
        sr = self._get_signal_sr(p, signal_ch)
        warnings = []
        raw_annot_data = []
        if len(raw_annots):
            try:
                raw_annot_data = self._derive_actigraphy_raw_annotations(
                    p,
                    raw_annots,
                    total_seconds,
                    start_tod_secs,
                    anchor_secs,
                )
            except Exception as e:
                warnings.append(f"Raw annotation overlay unavailable: {type(e).__name__}: {e}")
        missing_mask = self._derive_actigraphy_missing_mask(
            p,
            signal_ch,
            epoch_dur,
            total_seconds,
            start_tod_secs,
            anchor_secs,
        )
        if signal_ch is None and source != "annot":
            bins_per_day = int(86400 // epoch_dur)
            ndays = self._calc_day_count(total_seconds, start_tod_secs, anchor_secs)
            matrix = np.full((ndays, bins_per_day), np.nan, dtype=float)
            return {
                "channel": ch,
                "source": source,
                "annotation": annot,
                "raw_annotations": raw_annot_data,
                "sr": sr,
                "epoch_dur": int(epoch_dur),
                "days": int(ndays),
                "anchor_secs": int(anchor_secs),
                "matrix": matrix,
                "missing_mask": missing_mask,
                "daily_mean": np.array([]),
                "profile": np.array([]),
                "ra": np.nan,
                "value_label": "",
                "skip_summaries": True,
                "empty_state": True,
                "warnings": warnings,
            }

        if source == "raw":
            matrix = self._derive_actigraphy_raw(p, signal_ch, epoch_dur, total_seconds, start_tod_secs, anchor_secs)
            value_label = "Raw mean"
        elif source == "annot":
            matrix = self._derive_actigraphy_annot(p, annot, epoch_dur, total_seconds, start_tod_secs, anchor_secs)
            value_label = "Occupancy"
        else:
            matrix = self._derive_actigraphy_signal(p, signal_ch, epoch_dur, winsor_limit, total_seconds, start_tod_secs, anchor_secs)
            value_label = "Epoch activity"

        daily_mean = np.array([self._actigraphy_safe_nanmean(row) for row in matrix], dtype=float)
        profile = np.array([self._actigraphy_safe_nanmean(matrix[:, i]) for i in range(matrix.shape[1])], dtype=float)
        ra = self._actigraphy_relative_amplitude(profile, epoch_dur)
        finite = np.isfinite(matrix)
        self._actigraphy_debug(
            "derive_done",
            matrix_shape=matrix.shape,
            finite=int(np.sum(finite)),
            nan=int(np.size(matrix) - np.sum(finite)),
            matrix_min=float(np.nanmin(matrix)) if np.any(finite) else np.nan,
            matrix_max=float(np.nanmax(matrix)) if np.any(finite) else np.nan,
            raw_annots=len(raw_annot_data),
            warnings=len(warnings),
            ra=ra,
        )

        return {
            "channel": ch,
            "source": source,
            "annotation": annot,
            "raw_annotations": raw_annot_data,
            "sr": sr,
            "epoch_dur": int(epoch_dur),
            "days": int(matrix.shape[0]),
            "anchor_secs": int(anchor_secs),
            "matrix": matrix,
            "missing_mask": missing_mask,
            "daily_mean": daily_mean,
            "profile": profile,
            "ra": ra,
            "value_label": value_label,
            "skip_summaries": False,
            "empty_state": False,
            "warnings": warnings,
        }

    def _get_signal_sr(self, p, ch):
        if not ch:
            return np.nan
        df = p.headers()
        if df is None or df.empty:
            return np.nan
        row = df.loc[df["CH"] == ch]
        if row.empty or "SR" not in row.columns:
            return np.nan
        return float(row["SR"].iloc[0])

    def _derive_actigraphy_missing_mask(self, p, ch, epoch_dur, total_seconds, start_tod_secs, anchor_secs):
        if not ch or ch not in p.edf.channels():
            bins_per_day = int(86400 // epoch_dur)
            ndays = self._calc_day_count(total_seconds, start_tod_secs, anchor_secs)
            return np.zeros((ndays, bins_per_day), dtype=bool)

        idx = p.s2i([(0, float(total_seconds))])
        arr = p.slice(idx, chs=ch, time=True)[1]
        bins_per_day = int(86400 // epoch_dur)
        ndays = self._calc_day_count(total_seconds, start_tod_secs, anchor_secs)
        if arr is None or len(arr) == 0:
            return np.ones((ndays, bins_per_day), dtype=bool)

        times = np.asarray(arr[:, 0], dtype=float)
        absolute = times + float(start_tod_secs) - float(anchor_secs)
        day_idx = np.floor_divide(absolute.astype(np.int64), 86400).astype(int)
        bin_idx = np.floor_divide((np.mod(absolute, 86400)).astype(np.int64), epoch_dur).astype(int)
        counts = np.zeros((ndays, bins_per_day), dtype=float)
        for i in range(len(day_idx)):
            d = day_idx[i]
            b = bin_idx[i]
            if 0 <= d < ndays and 0 <= b < bins_per_day:
                counts[d, b] += 1.0
        return counts == 0

    def _derive_actigraphy_signal(self, p, ch, epoch_dur, winsor_limit, total_seconds, start_tod_secs, anchor_secs):
        cmd = f"EPOCH dur={epoch_dur} verbose & SIGSTATS epoch sig={ch}"
        res = p.silent_proc_lunascope(cmd)

        df = res.get("SIGSTATS: CH_E")
        dt = res.get("EPOCH: E")
        if df is None or dt is None or df.empty or dt.empty:
            raise RuntimeError("No epochwise actigraphy summary returned")
        if "H1" not in df.columns or "E" not in df.columns:
            raise RuntimeError("SIGSTATS epoch output did not include Hjorth activity")
        if "START" not in dt.columns or "E" not in dt.columns:
            raise RuntimeError("EPOCH output did not include START times")

        merged = df[["E", "H1"]].merge(dt[["E", "START"]], on="E", how="inner")
        if merged.empty:
            raise RuntimeError("Could not align epoch summaries to epoch starts")

        starts = merged["START"].to_numpy(dtype=float)
        values = np.sqrt(np.maximum(merged["H1"].to_numpy(dtype=float), 0.0))
        if winsor_limit > 0:
            values = self._actigraphy_winsorize(values, winsor_limit)
        return self._build_day_matrix_from_points(starts, values, epoch_dur, total_seconds, start_tod_secs, anchor_secs)

    def _derive_actigraphy_raw(self, p, ch, epoch_dur, total_seconds, start_tod_secs, anchor_secs):
        idx = p.s2i([(0, float(total_seconds))])
        arr = p.slice(idx, chs=ch, time=True)[1]
        if arr is None or len(arr) == 0:
            raise RuntimeError("No raw data returned for selected signal")
        times = np.asarray(arr[:, 0], dtype=float)
        values = np.asarray(arr[:, 1], dtype=float)
        return self._build_day_matrix_from_samples(times, values, epoch_dur, total_seconds, start_tod_secs, anchor_secs)

    def _derive_actigraphy_annot(self, p, annot, epoch_dur, total_seconds, start_tod_secs, anchor_secs):
        events = p.fetch_annots([annot])
        bins_per_day = int(86400 // epoch_dur)
        ndays = self._calc_day_count(total_seconds, start_tod_secs, anchor_secs)
        matrix = np.zeros((ndays, bins_per_day), dtype=float)

        if events is None or len(events) == 0:
            return matrix

        for row in events.itertuples(index=False):
            start = float(getattr(row, "Start"))
            stop = float(getattr(row, "Stop"))
            a0 = start + float(start_tod_secs) - float(anchor_secs)
            a1 = stop + float(start_tod_secs) - float(anchor_secs)
            while a0 < a1:
                bin_start = math.floor(a0 / epoch_dur) * epoch_dur
                bin_stop = min(a1, bin_start + epoch_dur)
                overlap = max(0.0, bin_stop - a0)
                day = int(bin_start // 86400)
                slot = int((bin_start % 86400) // epoch_dur)
                if 0 <= day < ndays and 0 <= slot < bins_per_day:
                    matrix[day, slot] += overlap / epoch_dur
                a0 = bin_stop

        matrix = np.clip(matrix, 0.0, 1.0)
        return matrix

    def _derive_actigraphy_raw_annotations(self, p, annots, total_seconds, start_tod_secs, anchor_secs):
        colors = [
            "#ff6b6b",
            "#ffd166",
            "#06d6a0",
            "#4cc9f0",
            "#a78bfa",
            "#f72585",
            "#90be6d",
            "#f9844a",
        ]
        ndays = self._calc_day_count(total_seconds, start_tod_secs, anchor_secs)
        cache_key = (id(p), tuple(map(str, annots)), float(total_seconds))
        grouped = self._act_raw_annot_cache.get(cache_key)
        if grouped is None:
            grouped = {}
            if len(annots):
                try:
                    srv = lp.segsrv(p)
                    srv.populate(chs=[], anns=list(annots))
                    srv.set_annot_format6(False)
                    srv.set_clip_xaxes(False)
                    srv.window(0.0, float(total_seconds))
                    srv.compile_windowed_annots(list(annots))
                    for annot in annots:
                        a0 = np.asarray(srv.get_annots_xaxes(annot), dtype=float)
                        a1 = np.asarray(srv.get_annots_xaxes_ends(annot), dtype=float)
                        n = min(a0.size, a1.size)
                        grouped[str(annot)] = [] if n == 0 else list(zip(a0[:n].tolist(), a1[:n].tolist()))
                except Exception:
                    fetched = p.fetch_annots(annots, 1)
                    if fetched is not None and len(fetched) > 0 and "Class" in fetched.columns:
                        for name, df in fetched.groupby("Class", sort=False):
                            grouped[str(name)] = [
                                (float(row.Start), float(row.Stop))
                                for row in df.itertuples(index=False)
                            ]
                    for annot in annots:
                        if str(annot) in grouped:
                            continue
                        df = p.fetch_annots([annot], 1)
                        grouped[str(annot)] = [] if df is None or len(df) == 0 else [
                            (float(row.Start), float(row.Stop))
                            for row in df.itertuples(index=False)
                        ]
            self._act_raw_annot_cache[cache_key] = grouped
        out = []
        for ai, annot in enumerate(annots):
            rows = [[] for _ in range(ndays)]
            events = grouped.get(str(annot), [])
            if len(events) > 0:
                for start, stop in events:
                    a0 = start + float(start_tod_secs) - float(anchor_secs)
                    a1 = stop + float(start_tod_secs) - float(anchor_secs)
                    d0 = int(math.floor(a0 / 86400.0))
                    d1 = int(math.floor(max(a0, a1 - 1e-9) / 86400.0))
                    for day in range(max(0, d0), min(ndays - 1, d1) + 1):
                        day0 = day * 86400.0
                        x0 = max(a0, day0) - day0
                        x1 = min(a1, day0 + 86400.0) - day0
                        x0 /= 3600.0
                        x1 /= 3600.0
                        rows[day].append((x0, x1))
            out.append({
                "name": annot,
                "color": colors[ai % len(colors)],
                "rows": rows,
            })
        return out

    def _actigraphy_warn(self, msg, timeout_ms=10000):
        try:
            sb = getattr(self.ui, "statusbar", None)
            if sb is not None:
                sb.showMessage(msg, int(timeout_ms))
        except Exception:
            pass

    def _actigraphy_debug(self, stage, **kwargs):
        return

    def _calc_day_count(self, total_seconds, start_tod_secs, anchor_secs):
        shifted_start = float(start_tod_secs) - float(anchor_secs)
        return max(1, int(math.ceil((shifted_start + float(total_seconds)) / 86400.0)))

    def _build_day_matrix_from_points(self, starts, values, epoch_dur, total_seconds, start_tod_secs, anchor_secs):
        bins_per_day = int(86400 // epoch_dur)
        ndays = self._calc_day_count(total_seconds, start_tod_secs, anchor_secs)
        matrix = np.full((ndays, bins_per_day), np.nan, dtype=float)
        counts = np.zeros((ndays, bins_per_day), dtype=float)

        absolute = np.asarray(starts, dtype=float) + float(start_tod_secs) - float(anchor_secs)
        vals = np.asarray(values, dtype=float)
        day_idx = np.floor_divide(absolute.astype(np.int64), 86400).astype(int)
        bin_idx = np.floor_divide((np.mod(absolute, 86400)).astype(np.int64), epoch_dur).astype(int)

        for i in range(len(vals)):
            d = day_idx[i]
            b = bin_idx[i]
            if d < 0 or d >= ndays or b < 0 or b >= bins_per_day or not np.isfinite(vals[i]):
                continue
            if counts[d, b] == 0:
                matrix[d, b] = vals[i]
            else:
                matrix[d, b] += vals[i]
            counts[d, b] += 1.0

        with np.errstate(invalid="ignore", divide="ignore"):
            matrix = matrix / counts
        matrix[counts == 0] = np.nan
        return matrix

    def _build_day_matrix_from_samples(self, times, values, epoch_dur, total_seconds, start_tod_secs, anchor_secs):
        bins_per_day = int(86400 // epoch_dur)
        ndays = self._calc_day_count(total_seconds, start_tod_secs, anchor_secs)
        matrix = np.full((ndays, bins_per_day), np.nan, dtype=float)
        sums = np.zeros((ndays, bins_per_day), dtype=float)
        counts = np.zeros((ndays, bins_per_day), dtype=float)

        absolute = np.asarray(times, dtype=float) + float(start_tod_secs) - float(anchor_secs)
        vals = np.asarray(values, dtype=float)
        day_idx = np.floor_divide(absolute.astype(np.int64), 86400).astype(int)
        bin_idx = np.floor_divide((np.mod(absolute, 86400)).astype(np.int64), epoch_dur).astype(int)

        for i in range(len(vals)):
            d = day_idx[i]
            b = bin_idx[i]
            if d < 0 or d >= ndays or b < 0 or b >= bins_per_day or not np.isfinite(vals[i]):
                continue
            sums[d, b] += vals[i]
            counts[d, b] += 1.0

        with np.errstate(invalid="ignore", divide="ignore"):
            matrix = sums / counts
        matrix[counts == 0] = np.nan
        return matrix

    def _derive_actigraphy_trace_rows(self, p, ch, total_seconds, start_tod_secs, anchor_secs, double=False):
        idx = p.s2i([(0, float(total_seconds))])
        arr = p.slice(idx, chs=ch, time=True)[1]
        if arr is None or len(arr) == 0:
            return []

        times = np.asarray(arr[:, 0], dtype=float)
        values = np.asarray(arr[:, 1], dtype=float)
        absolute = times + float(start_tod_secs) - float(anchor_secs)
        day_idx = np.floor_divide(absolute.astype(np.int64), 86400).astype(int)
        tod = np.mod(absolute, 86400) / 3600.0
        ndays = self._calc_day_count(total_seconds, start_tod_secs, anchor_secs)
        rows = []

        for day in range(ndays):
            mask = day_idx == day
            if not np.any(mask):
                rows.append((np.array([]), np.array([])))
                continue
            x = tod[mask]
            y = values[mask]
            order = np.argsort(x, kind="stable")
            x = x[order]
            y = y[order]
            x, y = self._insert_trace_gaps(x, y)
            rows.append((x, y))

        if double:
            doubled = []
            for day in range(ndays):
                x1, y1 = rows[day]
                if day + 1 < ndays:
                    x2, y2 = rows[day + 1]
                    xd = np.concatenate([x1, x2 + 24.0]) if x1.size or x2.size else np.array([])
                    yd = np.concatenate([y1, y2]) if y1.size or y2.size else np.array([])
                else:
                    xd = x1.copy()
                    yd = y1.copy()
                xd, yd = self._insert_trace_gaps(xd, yd)
                doubled.append((xd, yd))
            rows = doubled

        return rows

    def _insert_trace_gaps(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < 2:
            return x, y
        dx = np.diff(x)
        pos = dx[dx > 0]
        if pos.size == 0:
            return x, y
        step = float(np.median(pos))
        if not np.isfinite(step) or step <= 0:
            return x, y
        gap_idx = np.where(dx > step * 3.0)[0]
        if gap_idx.size == 0:
            return x, y
        xo = []
        yo = []
        last = 0
        for gi in gap_idx:
            xo.append(x[last : gi + 1])
            yo.append(y[last : gi + 1])
            xo.append(np.array([np.nan]))
            yo.append(np.array([np.nan]))
            last = gi + 1
        xo.append(x[last:])
        yo.append(y[last:])
        return np.concatenate(xo), np.concatenate(yo)

    def _actigraphy_winsorize(self, values, limit):
        arr = np.asarray(values, dtype=float).copy()
        good = np.isfinite(arr)
        if not np.any(good):
            return arr
        lo = np.nanquantile(arr[good], float(limit))
        hi = np.nanquantile(arr[good], 1.0 - float(limit))
        arr[good] = np.clip(arr[good], lo, hi)
        return arr

    def _actigraphy_safe_nanmean(self, values):
        arr = np.asarray(values, dtype=float)
        good = np.isfinite(arr)
        if not np.any(good):
            return np.nan
        return float(np.mean(arr[good]))

    def _actigraphy_relative_amplitude(self, profile, epoch_dur):
        prof = np.asarray(profile, dtype=float)
        if prof.size == 0 or np.all(~np.isfinite(prof)):
            return np.nan

        def _window_mean(hours, pick_max):
            n = max(1, int(round(hours * 3600 / epoch_dur)))
            ext = np.concatenate([prof, prof[: max(0, n - 1)]])
            vals = []
            for i in range(prof.size):
                vals.append(self._actigraphy_safe_nanmean(ext[i : i + n]))
            vals = np.asarray(vals, dtype=float)
            if np.all(~np.isfinite(vals)):
                return np.nan
            idx = int(np.nanargmax(vals)) if pick_max else int(np.nanargmin(vals))
            return float(vals[idx])

        m10 = _window_mean(10, True)
        l5 = _window_mean(5, False)
        if not np.isfinite(m10) or not np.isfinite(l5) or (m10 + l5) == 0:
            return np.nan
        return (m10 - l5) / (m10 + l5)

    def _complete_actigraphy(self, data):
        self._actigraphy_debug(
            "complete_start",
            source=data.get("source"),
            days=data.get("days"),
            matrix_shape=np.shape(data.get("matrix")),
            raw_annotations=len(data.get("raw_annotations", [])),
        )
        self._ensure_actigraphy_canvas()
        fig = self.actigraphycanvas.figure
        fig.clf()
        fig.patch.set_facecolor("#0f1724")
        show_lower = bool(getattr(self.ui, "check_actigraphy_detail", None) and self.ui.check_actigraphy_detail.isChecked())
        if show_lower:
            gs = fig.add_gridspec(3, 2, width_ratios=[40, 2], height_ratios=[2.8, 1.25, 1.0])
            ax_raster = fig.add_subplot(gs[0, 0])
            cax = fig.add_subplot(gs[0, 1])
            ax_profile = fig.add_subplot(gs[1, 0], sharex=ax_raster)
            ax_daily = fig.add_subplot(gs[2, 0])
            fig.add_subplot(gs[1, 1]).set_axis_off()
            fig.add_subplot(gs[2, 1]).set_axis_off()
            axs = (ax_raster, ax_profile, ax_daily)
            fig.subplots_adjust(left=0.08, right=0.94, top=0.94, bottom=0.08, hspace=0.48, wspace=0.06)
        else:
            gs = fig.add_gridspec(1, 2, width_ratios=[40, 2])
            ax_raster = fig.add_subplot(gs[0, 0])
            cax = fig.add_subplot(gs[0, 1])
            axs = (ax_raster, None, None)
            fig.subplots_adjust(left=0.06, right=0.95, top=0.96, bottom=0.06, hspace=0.0, wspace=0.06)

        matrix = np.asarray(data["matrix"], dtype=float)
        epoch_dur = int(data["epoch_dur"])
        double = self.ui.combo_actigraphy_mode.currentData() == "double"
        render_mode = self._get_actigraphy_render_mode()
        trace_ok = (
            render_mode != "heatmap"
            and data["source"] != "annot"
            and np.isfinite(data.get("sr", np.nan))
            and float(data["sr"]) <= 1.0
        )
        self._actigraphy_debug(
            "complete_mode",
            double=double,
            render_mode=render_mode,
            trace_ok=trace_ok,
            empty_state=data.get("empty_state"),
        )
        if data.get("empty_state"):
            self._plot_empty_actigraphy_raster(
                axs[0],
                cax,
                int(data["days"]),
                int(data["anchor_secs"]),
                double=double,
                label="No signal selected",
            )
            self.actigraphycanvas.ax = axs[0]
            self._actigraphy_debug("complete_empty_raster")
        elif trace_ok:
            rows = self._derive_actigraphy_trace_rows(
                self.p,
                data["channel"],
                float(getattr(self, "ns", 0.0)),
                int(getattr(self, "_record_start_tod_secs", 0)),
                int(data["anchor_secs"]),
                double=double,
            )
            scale_mode = {
                "trace_robust": "robust_row",
                "trace_minmax": "minmax_row",
                "trace_study_minmax": "minmax_study",
                "trace_study_robust": "robust_study",
            }.get(render_mode, "robust_row")
            self._plot_actigraphy_traces(
                axs[0],
                cax,
                rows,
                data,
                double=double,
                scale_mode=scale_mode,
                mode_label=self.ui.combo_actigraphy_render.currentText(),
            )
            self.actigraphycanvas.ax = axs[0]
            self._actigraphy_debug("complete_trace_plot", rows=len(rows))
        else:
            self._plot_actigraphy_raster(axs[0], cax, matrix, epoch_dur, data, double=double)
            self.actigraphycanvas.ax = axs[0]
            self._actigraphy_debug("complete_heatmap_plot")

        warnings = list(data.get("warnings", []))
        try:
            if data.get("raw_annotations"):
                try:
                    self._plot_raw_annotation_overlay(axs[0], data.get("raw_annotations", []), double=double)
                    self._actigraphy_debug("complete_overlay_ok")
                except Exception as e:
                    warnings.append(f"Raw annotation overlay failed: {type(e).__name__}: {e}")
                    self._actigraphy_debug("complete_overlay_err", error=f"{type(e).__name__}: {e}")
            if show_lower:
                try:
                    self._plot_actigraphy_profile(axs[1], data["profile"], epoch_dur, data["value_label"], double=double)
                    self._actigraphy_debug("complete_profile_ok")
                except Exception as e:
                    warnings.append(f"Actigraphy profile plot failed: {type(e).__name__}: {e}")
                    axs[1].set_axis_off()
                    self._actigraphy_debug("complete_profile_err", error=f"{type(e).__name__}: {e}")
                try:
                    self._plot_actigraphy_daily(axs[2], data["daily_mean"], data["value_label"])
                    self._actigraphy_debug("complete_daily_ok")
                except Exception as e:
                    warnings.append(f"Actigraphy daily plot failed: {type(e).__name__}: {e}")
                    axs[2].set_axis_off()
                    self._actigraphy_debug("complete_daily_err", error=f"{type(e).__name__}: {e}")
            try:
                self._update_actigraphy_summary(data)
                self._actigraphy_debug("complete_summary_ok")
            except Exception as e:
                warnings.append(f"Actigraphy summary update failed: {type(e).__name__}: {e}")
                self._actigraphy_debug("complete_summary_err", error=f"{type(e).__name__}: {e}")
        finally:
            self.actigraphycanvas.draw_idle()
            self.actigraphycanvas.draw()

        if warnings:
            self._actigraphy_warn(warnings[0])

    def _style_actigraphy_axis(self, ax):
        ax.set_facecolor("#182334")
        for spine in ax.spines.values():
            spine.set_color("#46566f")
            spine.set_linewidth(0.8)
        ax.tick_params(colors="#cdd7e3", labelsize=9)
        ax.xaxis.label.set_color("#e2e8f0")
        ax.yaxis.label.set_color("#e2e8f0")
        ax.title.set_color("#f8fafc")

    def _format_clock_ticks(self, ticks, anchor_secs):
        anchor_h = int(anchor_secs // 3600)
        labels = []
        for t in ticks:
            hour = int((anchor_h + t) % 24)
            labels.append(f"{hour:02d}")
        return labels

    def _plot_actigraphy_raster(self, ax, cax, matrix, epoch_dur, data, double=False):
        if matrix.size == 0:
            ax.set_axis_off()
            cax.set_axis_off()
            return

        self._style_actigraphy_axis(ax)
        cax.set_facecolor("#182334")
        bins_per_day = matrix.shape[1]
        missing = np.asarray(data.get("missing_mask"), dtype=bool)
        if missing.shape != matrix.shape:
            missing = ~np.isfinite(matrix)
        cmap = self._get_actigraphy_palette()
        if double:
            ext = np.full((matrix.shape[0], bins_per_day * 2), np.nan, dtype=float)
            ext[:, :bins_per_day] = matrix
            if matrix.shape[0] > 1:
                ext[:-1, bins_per_day:] = matrix[1:, :]
            matrix = ext
            miss2 = np.ones((missing.shape[0], bins_per_day * 2), dtype=bool)
            miss2[:, :bins_per_day] = missing
            if missing.shape[0] > 1:
                miss2[:-1, bins_per_day:] = missing[1:, :]
            missing = miss2
            xmax = 48.0
            xticks = np.arange(0, 49, 6)
            title = "Double Raster"
        else:
            xmax = 24.0
            xticks = np.arange(0, 25, 6)
            title = "Raster"

        finite = np.isfinite(matrix)
        self._actigraphy_debug(
            "raster_stats",
            shape=matrix.shape,
            finite=int(np.sum(finite)),
            nan=int(np.size(matrix) - np.sum(finite)),
            missing=int(np.sum(missing)),
            matrix_min=float(np.nanmin(matrix)) if np.any(finite) else np.nan,
            matrix_max=float(np.nanmax(matrix)) if np.any(finite) else np.nan,
            double=double,
        )
        display = np.ma.masked_invalid(matrix)
        img = ax.imshow(
            display,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            cmap=cmap,
            extent=[0, xmax, matrix.shape[0] + 0.5, 0.5],
        )
        if np.any(missing):
            overlay = np.where(missing, 1.0, np.nan)
            ax.imshow(
                overlay,
                aspect="auto",
                interpolation="nearest",
                origin="upper",
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                alpha=0.55,
                extent=[0, xmax, matrix.shape[0] + 0.5, 0.5],
            )
        ticks = xticks
        ax.set_xlabel("Clock time (h)")
        ax.set_ylabel("Day")
        ax.set_xticks(ticks)
        ax.set_xticklabels(self._format_clock_ticks(ticks, data["anchor_secs"]))
        if matrix.shape[0] <= 12:
            yticks = np.arange(1, matrix.shape[0] + 1, 1)
        else:
            yticks = np.unique(np.rint(np.linspace(1, matrix.shape[0], 8)).astype(int))
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(int(y)) for y in yticks])

        name = data["channel"] if data["source"] != "annot" else data["annotation"]
        overlay_names = [str(rec.get("name", "")) for rec in data.get("raw_annotations", []) if rec.get("name")]
        overlay_suffix = ""
        if overlay_names:
            overlay_label = ", ".join(overlay_names[:4])
            if len(overlay_names) > 4:
                overlay_label += ", ..."
            overlay_suffix = f" | annots: {overlay_label}"
        ax.text(
            0.01,
            1.02,
            f"{name} | {data['source']} | {title}{overlay_suffix}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#dce6f2",
        )
        if np.any(missing):
            ax.text(
                0.995,
                1.02,
                "overlay = masked/no data",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#dce6f2",
            )
        cb = ax.figure.colorbar(img, cax=cax, label=data["value_label"])
        cb.outline.set_edgecolor("#516176")
        cb.ax.yaxis.label.set_color("#dce6f2")
        cb.ax.tick_params(colors="#cdd7e3", labelsize=8)
        cb.ax.set_facecolor("#182334")

    def _plot_empty_actigraphy_raster(self, ax, cax, ndays, anchor_secs, double=False, label=None):
        self._style_actigraphy_axis(ax)
        cax.set_axis_off()
        xmax = 48.0 if double else 24.0
        xticks = np.arange(0, 49, 6) if double else np.arange(0, 25, 6)
        ax.set_xlim(0, xmax)
        ax.set_ylim(max(1, ndays) + 0.5, 0.5)
        ax.set_xlabel("Clock time (h)")
        ax.set_ylabel("Day")
        ax.set_xticks(xticks)
        ax.set_xticklabels(self._format_clock_ticks(xticks, anchor_secs))
        yticks = np.arange(1, max(1, ndays) + 1, 1) if ndays <= 12 else np.unique(np.rint(np.linspace(1, ndays, 8)).astype(int))
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(int(y)) for y in yticks])
        for y in np.arange(1, ndays + 1):
            ax.hlines(y, 0, xmax, color="#334155", linewidth=0.5, alpha=0.45)
        ax.text(
            0.01,
            1.02,
            f"{label or 'raw annots'} | {'Double' if double else 'Single'} Raster",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#dce6f2",
        )

    def _trace_limits(self, y, mode, global_limits=None):
        yy = np.asarray(y, dtype=float)
        yy = yy[np.isfinite(yy)]
        if yy.size == 0:
            return 0.0, 1.0
        if mode in ("minmax_study", "robust_study") and global_limits is not None:
            lo, hi = global_limits
        elif mode == "minmax_row":
            lo = float(np.nanmin(yy))
            hi = float(np.nanmax(yy))
        else:
            lo = float(np.nanpercentile(yy, 5))
            hi = float(np.nanpercentile(yy, 95))
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi):
            hi = 1.0
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    def _plot_actigraphy_traces(self, ax, cax, rows, data, double=False, scale_mode="robust_row", mode_label="Trace"):
        self._style_actigraphy_axis(ax)
        cax.set_axis_off()
        xmax = 48.0 if double else 24.0
        ticks = np.arange(0, 49, 6) if double else np.arange(0, 25, 6)
        ndays = len(rows)
        bins_per_day = data["missing_mask"].shape[1] if np.asarray(data.get("missing_mask")).ndim == 2 else 0
        missing = np.asarray(data.get("missing_mask"), dtype=bool)
        global_limits = None
        if scale_mode in ("minmax_study", "robust_study"):
            all_y = [y[np.isfinite(y)] for _x, y in rows if y.size]
            if len(all_y):
                flat = np.concatenate(all_y)
                if flat.size:
                    if scale_mode == "minmax_study":
                        lo = float(np.nanmin(flat))
                        hi = float(np.nanmax(flat))
                    else:
                        lo = float(np.nanpercentile(flat, 5))
                        hi = float(np.nanpercentile(flat, 95))
                    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                        global_limits = (lo, hi)

        if ndays > 0:
            for y in np.arange(0.5, ndays + 1.0, 1.0):
                ax.hlines(y, 0, xmax, color="#334155", linewidth=0.5, alpha=0.45)

        for i, (x, y) in enumerate(rows, start=1):
            y0 = float(i)
            if x.size == 0 or y.size == 0:
                continue
            finite = np.isfinite(y)
            if not np.any(finite):
                continue
            yy = y[finite]
            xx = x[finite]
            lo, hi = self._trace_limits(yy, scale_mode, global_limits)
            rng = hi - lo
            mid = 0.5 * (lo + hi)
            half = 0.5 * rng
            if not np.isfinite(half) or half <= 1e-12:
                half = 1.0
            yn = (yy - mid) / half
            yt = y0 - 0.25 * yn
            if missing.ndim == 2 and missing.shape[0] == ndays and bins_per_day > 0:
                row_missing = missing[i - 1]
                if double:
                    miss2 = np.ones(bins_per_day * 2, dtype=bool)
                    miss2[:bins_per_day] = row_missing
                    if i < ndays:
                        miss2[bins_per_day:] = missing[i]
                    xx_idx = np.floor((xx / xmax) * (bins_per_day * 2)).astype(int)
                    xx_idx = np.clip(xx_idx, 0, bins_per_day * 2 - 1)
                    masked_pts = miss2[xx_idx]
                else:
                    xx_idx = np.floor((xx / xmax) * bins_per_day).astype(int)
                    xx_idx = np.clip(xx_idx, 0, bins_per_day - 1)
                    masked_pts = row_missing[xx_idx]
                yt = yt.copy()
                yt[masked_pts] = np.nan
            ax.plot(xx, yt, color="#7dd3fc", linewidth=0.8, alpha=0.95)

        if missing.ndim == 2 and missing.shape[0] == ndays:
            if double:
                bins_per_day = missing.shape[1]
                miss2 = np.ones((missing.shape[0], bins_per_day * 2), dtype=bool)
                miss2[:, :bins_per_day] = missing
                if missing.shape[0] > 1:
                    miss2[:-1, bins_per_day:] = missing[1:, :]
                missing = miss2
            extent = [0, xmax, ndays + 0.5, 0.5]
            overlay = np.where(missing, 1.0, np.nan)
            ax.imshow(
                overlay,
                aspect="auto",
                interpolation="nearest",
                origin="upper",
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                alpha=0.55,
                extent=extent,
                zorder=3,
            )

        ax.set_xlim(0, xmax)
        ax.set_ylim(ndays + 0.5, 0.5)
        if ndays <= 12:
            yticks = np.arange(1, ndays + 1, 1)
        else:
            yticks = np.unique(np.rint(np.linspace(1, ndays, 8)).astype(int))
        ax.set_yticks(yticks)
        ax.set_yticklabels([str(int(y)) for y in yticks])
        ax.set_xlabel("Clock time (h)")
        ax.set_ylabel("Day")
        ax.set_xticks(ticks)
        ax.set_xticklabels(self._format_clock_ticks(ticks, data["anchor_secs"]))
        overlay_names = [str(rec.get("name", "")) for rec in data.get("raw_annotations", []) if rec.get("name")]
        overlay_suffix = ""
        if overlay_names:
            overlay_label = ", ".join(overlay_names[:4])
            if len(overlay_names) > 4:
                overlay_label += ", ..."
            overlay_suffix = f" | annots: {overlay_label}"
        ax.text(
            0.01,
            1.02,
            f"{data['channel']} | raw trace | {mode_label} | {'Double' if double else 'Single'} Raster{overlay_suffix}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#dce6f2",
        )
        ax.text(
            0.995,
            1.02,
            "overlay = masked/no data",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#dce6f2",
        )

    def _plot_actigraphy_profile(self, ax, profile, epoch_dur, ylabel, double=False):
        prof = np.asarray(profile, dtype=float)
        if prof.size == 0:
            ax.set_axis_off()
            return
        self._style_actigraphy_axis(ax)
        x = np.arange(prof.size, dtype=float) * (epoch_dur / 3600.0)
        y = prof
        xmax = 24.0
        ticks = np.arange(0, 25, 4)
        if double:
            x = np.arange(prof.size * 2, dtype=float) * (epoch_dur / 3600.0)
            y = np.concatenate([prof, prof])
            xmax = 48.0
            ticks = np.arange(0, 49, 6)
        ax.plot(x, y, color="#6ee7f9", linewidth=1.8)
        ax.fill_between(x, 0, y, color="#4cc9f0", alpha=0.18)
        ax.set_xlim(0, xmax)
        ax.set_xticks(ticks)
        ax.set_xticklabels(self._format_clock_ticks(ticks, self._get_actigraphy_anchor_secs()))
        ax.set_xlabel("Clock time (h)")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="x", color="#7dd3fc", alpha=0.16, linewidth=0.8)

    def _plot_actigraphy_daily(self, ax, daily_mean, ylabel):
        vals = np.asarray(daily_mean, dtype=float)
        if vals.size == 0:
            ax.set_axis_off()
            return
        self._style_actigraphy_axis(ax)
        x = np.arange(1, vals.size + 1, dtype=float)
        ax.bar(x, vals, color="#f4a261", width=0.82, edgecolor="none", linewidth=0.0)
        ax.set_xlim(0.5, vals.size + 0.5)
        ax.set_xlabel("Day")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="#f4a261", alpha=0.16, linewidth=0.8)

    def _plot_raw_annotation_overlay(self, ax, annots, double=False):
        if len(annots) == 0:
            return
        lane_h = 0.12
        lane_gap = 0.03
        for ai, rec in enumerate(annots):
            color = rec["color"]
            for day_idx, spans in enumerate(rec["rows"], start=1):
                top = day_idx - 0.47 + ai * (lane_h + lane_gap)
                bars = [(x0, max(0.0, x1 - x0)) for x0, x1 in spans if x1 > x0]
                if len(bars):
                    ax.broken_barh(
                        bars,
                        (top, lane_h),
                        facecolors=color,
                        edgecolors="none",
                        alpha=0.92,
                        zorder=4,
                    )
                if double and day_idx < len(rec["rows"]):
                    bars2 = [(x0 + 24.0, max(0.0, x1 - x0)) for x0, x1 in rec["rows"][day_idx] if x1 > x0]
                    if len(bars2):
                        ax.broken_barh(
                            bars2,
                            (top, lane_h),
                            facecolors=color,
                            edgecolors="none",
                            alpha=0.92,
                            zorder=4,
                        )
        label = " | ".join([rec["name"] for rec in annots[:5]])
        if len(annots) > 5:
            label += " | ..."
        return

    def _update_actigraphy_summary(self, data=None):
        if data is None:
            self.ui.act_days.setText("Days: -")
            self.ui.act_epoch.setText("Epoch: -")
            self.ui.act_ra.setText("RA: -")
            self.ui.act_daily.setText("Daily mean: -")
            return

        self.ui.act_days.setText(f"Days: {data['days']}")
        if data.get("skip_summaries"):
            self.ui.act_epoch.setText("Epoch: n/a")
            self.ui.act_ra.setText("RA: n/a")
            self.ui.act_daily.setText("Summary: n/a")
            return
        self.ui.act_epoch.setText(f"Epoch: {data['epoch_dur'] // 60} min")
        ra = data.get("ra", np.nan)
        self.ui.act_ra.setText(f"RA: {ra:.3f}" if np.isfinite(ra) else "RA: -")
        dmean = self._actigraphy_safe_nanmean(data.get("daily_mean", []))
        anchor = self._get_actigraphy_anchor_label()
        self.ui.act_daily.setText(f"{anchor} mean: {dmean:.3g}" if np.isfinite(dmean) else f"{anchor} mean: -")
