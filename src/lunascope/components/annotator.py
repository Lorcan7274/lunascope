from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


ANNOTATOR_KEYS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")


@dataclass
class PendingAnnotation:
    key: str
    label: str
    mode: str
    start: float
    stop: float


class AnnotatorDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Annotator", parent)
        self.setObjectName("dock_annotator")
        self.setFloating(True)
        self.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )

        root = QWidget(self)
        self.setWidget(root)

        layout = QVBoxLayout(root)

        self.check_enabled = QCheckBox("Enable annotator")
        self.check_enabled.setObjectName("check_annotator_enabled")
        layout.addWidget(self.check_enabled)

        form = QFormLayout()
        self.combo_mode = QComboBox()
        self.combo_mode.setObjectName("combo_annotator_mode")
        self.combo_mode.addItem("Epoch", "epoch")
        self.combo_mode.addItem("Point", "point")
        self.combo_mode.addItem("Interval", "interval")
        form.addRow("Capture", self.combo_mode)

        self.check_use_selection = QCheckBox("Use current selection")
        self.check_use_selection.setObjectName("check_annotator_use_selection")
        self.check_use_selection.setChecked(True)
        form.addRow("", self.check_use_selection)

        self.spin_fixed_secs = QSpinBox()
        self.spin_fixed_secs.setObjectName("spin_annotator_fixed_secs")
        self.spin_fixed_secs.setRange(1, 3600)
        self.spin_fixed_secs.setValue(10)
        self.spin_fixed_secs.setSuffix(" s")
        form.addRow("Fixed span", self.spin_fixed_secs)
        layout.addLayout(form)

        group = QGroupBox("Key Bindings")
        grid = QGridLayout(group)
        grid.addWidget(QLabel("Key"), 0, 0)
        grid.addWidget(QLabel("Annotation"), 0, 1)
        self.key_combos: dict[str, QComboBox] = {}
        for row, key in enumerate(ANNOTATOR_KEYS, start=1):
            lab = QLabel(key)
            combo = QComboBox()
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.NoInsert)
            combo.setObjectName(f"combo_annotator_key_{key}")
            combo.addItem("")
            self.key_combos[key] = combo
            grid.addWidget(lab, row, 0)
            grid.addWidget(combo, row, 1)
        layout.addWidget(group)

        btn_row = QHBoxLayout()
        self.butt_clear = QPushButton("Clear bindings")
        self.butt_delete = QPushButton("Delete event")
        self.butt_delete.setEnabled(False)
        self.butt_delete.setToolTip("Per-event deletion is not supported yet.")
        btn_row.addWidget(self.butt_clear)
        btn_row.addWidget(self.butt_delete)
        layout.addLayout(btn_row)

        self.lbl_hint = QLabel(
            "Point mode commits immediately.\n"
            "Epoch/Interval mode: hold a bound key, adjust the current selection, release to commit."
        )
        self.lbl_hint.setWordWrap(True)
        layout.addWidget(self.lbl_hint)

        self.lbl_state = QLabel("Idle")
        self.lbl_state.setObjectName("lbl_annotator_state")
        self.lbl_state.setWordWrap(True)
        layout.addWidget(self.lbl_state)
        layout.addStretch(1)

        self.combo_mode.currentIndexChanged.connect(self._sync_mode_ui)
        self.butt_clear.clicked.connect(self.clear_bindings)
        self._sync_mode_ui()

    def _sync_mode_ui(self):
        mode = self.current_mode()
        is_interval = mode == "interval"
        self.check_use_selection.setEnabled(is_interval)
        self.spin_fixed_secs.setEnabled(is_interval and not self.check_use_selection.isChecked())

    def current_mode(self) -> str:
        return str(self.combo_mode.currentData() or "epoch")

    def bound_label_for_key(self, key: str) -> str:
        combo = self.key_combos.get(str(key))
        if combo is None:
            return ""
        return combo.currentText().strip()

    def binding_map(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, combo in self.key_combos.items():
            label = combo.currentText().strip()
            if label:
                out[key] = label
        return out

    def clear_bindings(self):
        for combo in self.key_combos.values():
            combo.setCurrentText("")

    def set_annotation_classes(self, annots: list[str]):
        vals = [str(x).strip() for x in annots if str(x).strip()]
        vals = sorted(set(vals), key=str.casefold)
        for combo in self.key_combos.values():
            current = combo.currentText().strip()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("")
            combo.addItems(vals)
            combo.setCurrentText(current)
            combo.blockSignals(False)

    def set_pending_text(self, text: str):
        self.lbl_state.setText(text)


class AnnotatorMixin:
    def _init_annotator(self):
        self.annotator = AnnotatorDock(self.ui)
        self.ui.addDockWidget(Qt.RightDockWidgetArea, self.annotator)
        self.annotator.hide()
        self.annotator.check_use_selection.toggled.connect(self.annotator._sync_mode_ui)
        self._annotator_pending: Optional[PendingAnnotation] = None
        self._annotator_epoch_secs = 30.0
        self._annotator_status("Idle")

    def _annotator_status(self, text: str, timeout_ms: int = 0):
        if hasattr(self, "annotator") and self.annotator is not None:
            self.annotator.set_pending_text(text)
        sb = getattr(self.ui, "statusbar", None)
        if sb is not None and timeout_ms > 0:
            sb.showMessage(text, timeout_ms)

    def _annotator_refresh_classes(self):
        if not hasattr(self, "annotator") or self.annotator is None:
            return
        annots: list[str] = []
        if hasattr(self, "p"):
            try:
                df = self.p.annots()
                if isinstance(df, pd.DataFrame) and "Annotations" in df.columns:
                    annots = [str(x) for x in df["Annotations"].tolist() if str(x) != "SleepStage"]
            except Exception:
                annots = []
        self.annotator.set_annotation_classes(annots)
        self._annotator_epoch_secs = self._annotator_resolve_epoch_secs()

    def _annotator_toggle_dock(self):
        self.annotator.setVisible(not self.annotator.isVisible())
        if self.annotator.isVisible():
            self.annotator.raise_()

    def _annotator_enabled(self) -> bool:
        return bool(
            hasattr(self, "annotator")
            and self.annotator is not None
            and self.annotator.isVisible()
            and self.annotator.check_enabled.isChecked()
            and hasattr(self, "p")
        )

    def _annotator_resolve_epoch_secs(self) -> float:
        if not hasattr(self, "p"):
            return 30.0
        try:
            self.p.silent_proc("EPOCH")
            df = self.p.table("EPOCH")
            if isinstance(df, pd.DataFrame) and not df.empty and {"START", "STOP"}.issubset(df.columns):
                dur = float(df.iloc[0]["STOP"]) - float(df.iloc[0]["START"])
                if dur > 0:
                    return dur
        except Exception:
            pass
        return 30.0

    def _annotator_current_range(self) -> tuple[float, float]:
        lo = float(getattr(self, "last_x1", 0.0))
        hi = float(getattr(self, "last_x2", lo))
        sel = getattr(self, "sel", None)
        if sel is not None:
            try:
                lo, hi = map(float, sel.region.getRegion())
            except Exception:
                pass
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi

    def _annotator_selection_center(self) -> float:
        lo, hi = self._annotator_current_range()
        return 0.5 * (lo + hi)

    def _annotator_snap_epoch_range(self, lo: float, hi: float) -> tuple[float, float]:
        step = max(1e-6, float(getattr(self, "_annotator_epoch_secs", 30.0) or 30.0))
        lo = max(0.0, float(lo))
        hi = max(lo, float(hi))
        start_idx = int(lo // step)
        if hi <= lo:
            stop_idx = start_idx + 1
        else:
            eps = 1e-9
            stop_idx = int((max(hi - eps, lo) // step) + 1)
        start = start_idx * step
        stop = stop_idx * step
        ns = float(getattr(self, "ns", stop))
        return max(0.0, start), min(ns, max(start, stop))

    def _annotator_interval_from_range(self) -> tuple[float, float]:
        lo, hi = self._annotator_current_range()
        mode = self.annotator.current_mode()
        if mode == "epoch":
            return self._annotator_snap_epoch_range(lo, hi)
        if mode == "interval" and not self.annotator.check_use_selection.isChecked():
            center = self._annotator_selection_center()
            half = 0.5 * float(self.annotator.spin_fixed_secs.value())
            return max(0.0, center - half), min(float(getattr(self, "ns", center + half)), center + half)
        return max(0.0, lo), max(lo, hi)

    def _annotator_describe_interval(self, start: float, stop: float) -> str:
        if abs(stop - start) < 1e-9:
            return f"{start:.2f}s"
        return f"{start:.2f}s -> {stop:.2f}s"

    def _annotator_begin_pending(self, key: str, label: str):
        start, stop = self._annotator_interval_from_range()
        self._annotator_pending = PendingAnnotation(
            key=key,
            label=label,
            mode=self.annotator.current_mode(),
            start=start,
            stop=stop,
        )
        self._annotator_status(
            f"Holding {key}: {label} [{self._annotator_describe_interval(start, stop)}]"
        )

    def _annotator_update_pending(self):
        pending = self._annotator_pending
        if pending is None:
            return
        start, stop = self._annotator_interval_from_range()
        pending.start = start
        pending.stop = stop
        self._annotator_status(
            f"Holding {pending.key}: {pending.label} [{self._annotator_describe_interval(start, stop)}]"
        )

    def _annotator_cancel_pending(self):
        self._annotator_pending = None
        self._annotator_status("Idle")

    def _annotator_insert_annotation(self, label: str, start: float, stop: float):
        if not hasattr(self, "p"):
            raise RuntimeError("No instance attached")
        self.p.insert_annot(str(label), [(float(start), float(stop))], durcol2=False)

    def _annotator_after_insert(self, label: str):
        curr_chs = list(self.ui.tbl_desc_signals.checked()) if hasattr(self.ui, "tbl_desc_signals") else []
        curr_anns = list(self.ui.tbl_desc_annots.checked()) if hasattr(self.ui, "tbl_desc_annots") else []
        try:
            self._update_metrics()
            if hasattr(self.ui, "tbl_desc_signals"):
                self.ui.tbl_desc_signals.set_checked_by_labels(curr_chs)
            if hasattr(self.ui, "tbl_desc_annots"):
                if hasattr(self.ui.tbl_desc_annots, "set_checked_by_labels_silent"):
                    self.ui.tbl_desc_annots.set_checked_by_labels_silent(curr_anns)
                else:
                    self.ui.tbl_desc_annots.set_checked_by_labels(curr_anns)
                if hasattr(self, "_mark_instances_dirty"):
                    self._mark_instances_dirty(curr_anns)
            self._render_hypnogram()
            self._render_signals_simple()
        finally:
            self._annotator_refresh_classes()
        hidden = label not in curr_anns
        suffix = " (hidden by current annotation selection)" if hidden else ""
        self._annotator_status(f"Added annotation: {label}{suffix}", 3000)

    def _annotator_commit(self, label: str, start: float, stop: float):
        self._annotator_insert_annotation(label, start, stop)
        self._annotator_after_insert(label)

    def _annotator_handle_maintrace_key_press(self, ev) -> bool:
        if not self._annotator_enabled():
            return False
        if ev.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
            return False
        key_text = ev.text()
        if key_text not in ANNOTATOR_KEYS:
            return False
        label = self.annotator.bound_label_for_key(key_text)
        if not label:
            self._annotator_status(f"No annotation bound to key {key_text}", 2000)
            return True
        mode = self.annotator.current_mode()
        if mode == "point":
            if ev.isAutoRepeat():
                return True
            point = self._annotator_selection_center()
            self._annotator_commit(label, point, point)
            return True
        if ev.isAutoRepeat():
            return True
        if self._annotator_pending is not None and self._annotator_pending.key != key_text:
            return True
        self._annotator_begin_pending(key_text, label)
        return True

    def _annotator_handle_maintrace_key_release(self, ev) -> bool:
        pending = self._annotator_pending
        if pending is None:
            return False
        key_text = ev.text()
        if key_text != pending.key:
            return False
        if ev.isAutoRepeat():
            return True
        start, stop = pending.start, pending.stop
        label = pending.label
        self._annotator_pending = None
        self._annotator_commit(label, start, stop)
        return True

    def _annotator_on_window_range_changed(self, lo: float, hi: float):
        _ = (lo, hi)
        if self._annotator_pending is not None:
            self._annotator_update_pending()
