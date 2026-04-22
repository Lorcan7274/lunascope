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

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


ANNOTATOR_KEYS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")
_ANNOTATOR_KEY_BY_QT = {
    Qt.Key_0: "0",
    Qt.Key_1: "1",
    Qt.Key_2: "2",
    Qt.Key_3: "3",
    Qt.Key_4: "4",
    Qt.Key_5: "5",
    Qt.Key_6: "6",
    Qt.Key_7: "7",
    Qt.Key_8: "8",
    Qt.Key_9: "9",
}


@dataclass
class PendingAnnotation:
    key: str
    label: str
    mode: str
    start: float
    stop: float


@dataclass
class StagedAnnotation:
    key: str
    label: str
    mode: str
    start: float
    stop: float


# ---------------------------------------------------------------------------
# Annotation editor tab widget  (form panel — driven by Dock5 selection)
# ---------------------------------------------------------------------------

class AnnotEditorWidget(QWidget):
    """The 'Edit' tab inside AnnotatorDock.

    Populated via set_instance() when a row is selected in the Instances dock.
    Queues edits/deletes against self.ssa; user then applies or cancels all pending changes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._identity: dict | None = None   # current annotation identity
        self._pending: int = 0               # count of queued ops this session

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        # ---- header ------------------------------------------------------
        self.lbl_header = QLabel("No annotation selected")
        self.lbl_header.setWordWrap(True)
        self.lbl_header.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.lbl_header)

        # ---- form --------------------------------------------------------
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.edit_inst = QLineEdit()
        self.edit_inst.setObjectName("edit_anneditor_inst")
        form.addRow("Instance ID:", self.edit_inst)

        self.edit_start = QLineEdit()
        self.edit_start.setObjectName("edit_anneditor_start")
        self.edit_start.setPlaceholderText("seconds")
        form.addRow("Start (s):", self.edit_start)

        self.edit_stop = QLineEdit()
        self.edit_stop.setObjectName("edit_anneditor_stop")
        self.edit_stop.setPlaceholderText("seconds")
        form.addRow("Stop (s):", self.edit_stop)

        self.edit_ch = QLineEdit()
        self.edit_ch.setObjectName("edit_anneditor_ch")
        form.addRow("Channel:", self.edit_ch)

        self.edit_meta = QLineEdit()
        self.edit_meta.setObjectName("edit_anneditor_meta")
        self.edit_meta.setPlaceholderText("key=value pairs e.g.  note=adjusted  conf=0.9")
        form.addRow("Metadata:", self.edit_meta)

        layout.addLayout(form)

        # ---- queue buttons -----------------------------------------------
        q_row = QHBoxLayout()
        q_row.setSpacing(10)
        self.butt_queue_edit   = QPushButton("Queue Edit")
        self.butt_queue_edit.setObjectName("butt_anneditor_queue_edit")
        self.butt_queue_delete = QPushButton("Queue Delete")
        self.butt_queue_delete.setObjectName("butt_anneditor_queue_delete")
        q_row.addWidget(self.butt_queue_edit)
        q_row.addWidget(self.butt_queue_delete)
        layout.addLayout(q_row)

        # ---- pending status ----------------------------------------------
        self.lbl_pending = QLabel("No pending edits")
        self.lbl_pending.setObjectName("lbl_anneditor_pending")
        layout.addWidget(self.lbl_pending)

        layout.addStretch(1)

        # ---- apply / discard --------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 8, 0, 0)
        btn_row.setSpacing(12)
        self.butt_apply   = QPushButton("Apply Pending Changes")
        self.butt_apply.setObjectName("butt_anneditor_apply")
        self.butt_apply.setMinimumWidth(160)
        self.butt_discard = QPushButton("Cancel Pending Changes")
        self.butt_discard.setObjectName("butt_anneditor_discard")
        self.butt_discard.setMinimumWidth(160)
        btn_row.addStretch()
        btn_row.addWidget(self.butt_discard)
        btn_row.addWidget(self.butt_apply)
        layout.addLayout(btn_row)

        self._set_fields_enabled(False)

    # ------------------------------------------------------------------
    def _set_fields_enabled(self, on: bool):
        for w in (self.edit_inst, self.edit_start, self.edit_stop, self.edit_ch,
                  self.edit_meta, self.butt_queue_edit, self.butt_queue_delete):
            w.setEnabled(on)

    def set_classes(self, names: list[str]):
        pass  # retained for API compatibility; no combo needed any more

    def set_instance(self, identity: dict):
        """Populate the form from an identity dict (from _events_identity)."""
        self._identity = identity
        self.lbl_header.setText(f"{identity['aclass']}")
        self.edit_inst.setText(identity.get("inst_id", ""))
        self.edit_start.setText(identity.get("start_sec", ""))
        self.edit_stop.setText(identity.get("stop_sec", ""))
        self.edit_ch.setText(identity.get("ch_str", ""))
        self.edit_meta.setText(identity.get("meta", ""))
        self._set_fields_enabled(True)

    def clear_instance(self):
        self._identity = None
        self.lbl_header.setText("No annotation selected")
        for w in (self.edit_inst, self.edit_start, self.edit_stop, self.edit_ch, self.edit_meta):
            w.clear()
        self._set_fields_enabled(False)

    def bump_pending(self, delta: int = 1):
        self._pending += delta
        n = self._pending
        if n == 0:
            self.lbl_pending.setText("No pending edits")
        else:
            self.lbl_pending.setText(f"{n} pending edit{'s' if n != 1 else ''} — not yet applied")

    def reset_pending(self):
        self._pending = 0
        self.lbl_pending.setText("No pending edits")

    def current_values(self) -> dict:
        return {
            "inst_id":   self.edit_inst.text().strip(),
            "start_sec": self.edit_start.text().strip(),
            "stop_sec":  self.edit_stop.text().strip(),
            "ch_str":    self.edit_ch.text().strip(),
            "meta":      self.edit_meta.text().strip(),
        }


# ---------------------------------------------------------------------------
# Dock shell — tabbed: "Add" (original) + "Edit" (new)
# ---------------------------------------------------------------------------

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
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)

        tabs = QTabWidget()
        outer.addWidget(tabs)

        # ---- Tab 0: Add (original annotator UI) -------------------------
        add_widget = QWidget()
        layout = QVBoxLayout(add_widget)

        self.check_enabled = QCheckBox("Enable annotator")
        self.check_enabled.setObjectName("check_annotator_enabled")
        self.check_enabled.setChecked(True)
        self.check_enabled.hide()

        form = QFormLayout()
        self.combo_mode = QComboBox()
        self.combo_mode.setObjectName("combo_annotator_mode")
        self.combo_mode.addItem("Epoch", "epoch")
        self.combo_mode.addItem("Interval", "interval")
        self.combo_mode.setCurrentIndex(1)
        form.addRow("Capture", self.combo_mode)
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
            if combo.lineEdit() is not None:
                combo.lineEdit().setPlaceholderText("Existing or new class")
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
            "Epoch mode: fixed 30s at the current selection/window center.\n"
            "Interval mode: drag/select a visible region, then press a bound key to stage.\n"
            "Use Commit Staged to write staged additions into Luna."
        )
        self.lbl_hint.setWordWrap(True)
        layout.addWidget(self.lbl_hint)

        self.lbl_state = QLabel("Idle")
        self.lbl_state.setObjectName("lbl_annotator_state")
        self.lbl_state.setWordWrap(True)
        layout.addWidget(self.lbl_state)

        self.lbl_stage_pending = QLabel("No staged additions")
        self.lbl_stage_pending.setObjectName("lbl_annotator_stage_pending")
        self.lbl_stage_pending.setWordWrap(True)
        layout.addWidget(self.lbl_stage_pending)

        stage_btn_row = QHBoxLayout()
        self.butt_stage_clear = QPushButton("Clear Staged")
        self.butt_stage_clear.setObjectName("butt_annotator_stage_clear")
        self.butt_stage_commit = QPushButton("Commit Staged")
        self.butt_stage_commit.setObjectName("butt_annotator_stage_commit")
        self.butt_stage_clear.setEnabled(False)
        self.butt_stage_commit.setEnabled(False)
        stage_btn_row.addWidget(self.butt_stage_clear)
        stage_btn_row.addWidget(self.butt_stage_commit)
        layout.addLayout(stage_btn_row)
        layout.addStretch(1)

        # ---- Tab 0: Edit ------------------------------------------------
        self.editor = AnnotEditorWidget()
        tabs.addTab(self.editor, "Edit")

        tabs.addTab(add_widget, "Add")

        self._tabs = tabs

        # signals
        self.combo_mode.currentIndexChanged.connect(self._sync_mode_ui)
        self.butt_clear.clicked.connect(self.clear_bindings)
        self._sync_mode_ui()

    def _sync_mode_ui(self):
        pass

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
        self.editor.set_classes(vals)

    def set_pending_text(self, text: str):
        self.lbl_state.setText(text)


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class AnnotatorMixin:
    def _annotator_key_text(self, ev) -> str:
        txt = (ev.text() or "").strip()
        if txt in ANNOTATOR_KEYS:
            return txt
        return _ANNOTATOR_KEY_BY_QT.get(ev.key(), "")

    def _init_annotator(self):
        self.annotator = AnnotatorDock(self.ui)
        self.ui.addDockWidget(Qt.RightDockWidgetArea, self.annotator)
        self.annotator.hide()
        self._annotator_pending: Optional[PendingAnnotation] = None
        self._annotator_staged: list[StagedAnnotation] = []
        self._annotator_epoch_secs = 30.0
        self._annotator_status("Idle")
        self._queued_deletes:           set  = set()   # (aclass, inst_id) queued for deletion
        self._queued_edits:             set  = set()   # (aclass, inst_id) queued for editing
        self._queued_prior_edits:       set  = set()   # edits superseded by a delete (for restore)
        self._queued_positions:         dict = {}      # (aclass, inst_id) -> (start_sec, stop_sec)
        self._queued_delete_kwargs:     dict = {}      # full kwargs for srv.delete_annot replay
        self._queued_edit_kwargs:       dict = {}      # full kwargs for srv.edit_annot replay
        self._queued_original_identity: dict = {}      # identity snapshot when edit first queued
        self._annot_queue_overlay_items: list = []
        self._annot_cursor_item = None                 # current-selection highlight on pg1

        self.annotator.visibilityChanged.connect(self._annot_cursor_on_dock_visibility)
        self.annotator.visibilityChanged.connect(self._annotator_sync_pg1_selector_visibility)
        self.annotator.combo_mode.currentIndexChanged.connect(self._annotator_sync_pg1_selector_visibility)
        self.annotator._tabs.currentChanged.connect(self._annotator_sync_pg1_selector_visibility)
        self.annotator.installEventFilter(self)
        self.annotator._tabs.installEventFilter(self)
        self.annotator.combo_mode.installEventFilter(self)
        for combo in self.annotator.key_combos.values():
            combo.installEventFilter(self)
            if combo.lineEdit() is not None:
                combo.lineEdit().installEventFilter(self)

        # wire editor buttons
        ed = self.annotator.editor
        ed.butt_queue_edit.clicked.connect(self._annot_editor_queue_edit)
        ed.butt_queue_delete.clicked.connect(self._annot_editor_queue_delete)
        ed.butt_apply.clicked.connect(self._annot_editor_apply)
        ed.butt_discard.clicked.connect(self._annot_editor_discard)
        self.annotator.butt_stage_clear.clicked.connect(self._annotator_clear_staged)
        self.annotator.butt_stage_commit.clicked.connect(self._annotator_commit_staged)
        self._annotator_sync_stage_widgets()

    def _annotator_sync_pg1_selector_visibility(self, *_):
        sel = getattr(self, "annot_sel", None)
        if sel is None:
            return
        if not self._annotator_enabled() or self.annotator.current_mode() != "interval":
            sel.clear_visuals()
        else:
            self._annotator_status("Add annotation mode activated")

    def _annotator_add_mode_active(self) -> bool:
        return bool(
            hasattr(self, "annotator")
            and self.annotator is not None
            and self.annotator.isVisible()
            and getattr(self.annotator, "_tabs", None) is not None
            and self.annotator._tabs.currentIndex() == 1
            and hasattr(self, "p")
        )

    def _annot_identity_key(self, identity: dict) -> tuple[str, str, str, str, str]:
        return (
            str(identity.get("aclass", "")),
            str(identity.get("start_tp", "")),
            str(identity.get("stop_tp", "")),
            str(identity.get("inst_id", "")),
            str(identity.get("ch_str", "")),
        )

    # ------------------------------------------------------------------
    # Add-annotator helpers (unchanged)
    # ------------------------------------------------------------------

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
        annots.extend(x.label for x in getattr(self, "_annotator_staged", []))
        self.annotator.set_annotation_classes(annots)
        self._annotator_epoch_secs = self._annotator_resolve_epoch_secs()

    def _annotator_toggle_dock(self):
        self.annotator.setVisible(not self.annotator.isVisible())
        if self.annotator.isVisible():
            self.annotator.raise_()

    def _annotator_enabled(self) -> bool:
        return self._annotator_add_mode_active()

    def _annotator_handles_widget(self, obj) -> bool:
        annotator = getattr(self, "annotator", None)
        if annotator is None or obj is None:
            return False
        if obj is annotator or obj is annotator._tabs or obj is annotator.combo_mode:
            return True
        if obj in annotator.key_combos.values():
            return True
        for combo in annotator.key_combos.values():
            if combo.lineEdit() is obj:
                return True
        return False

    def _annotator_handle_widget_key_event(self, obj, event) -> bool:
        if not self._annotator_add_mode_active():
            return False
        if not self._annotator_handles_widget(obj):
            return False
        if event.type() == QEvent.KeyPress:
            return self._annotator_handle_maintrace_key_press(event)
        if event.type() == QEvent.KeyRelease:
            return self._annotator_handle_maintrace_key_release(event)
        return False

    def _annotator_resolve_epoch_secs(self) -> float:
        # Add-annotation epoch capture is intentionally fixed to 30s.
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

    def _annotator_snap_epoch_range(self, lo: float, hi: float) -> tuple[float, float]:
        epoch = float(getattr(self, "_annotator_epoch_secs", 30.0) or 30.0)
        lo = max(0.0, float(lo))
        hi = max(lo, float(hi))
        center = 0.5 * (lo + hi)
        half = 0.5 * epoch
        ns = float(getattr(self, "ns", center + half))
        start = max(0.0, center - half)
        stop = start + epoch
        if stop > ns:
            stop = ns
            start = max(0.0, stop - epoch)
        return start, max(start, stop)

    def _annotator_visible_selection_range(self) -> Optional[tuple[float, float]]:
        sel = getattr(self, "annot_sel", None)
        if sel is None or not hasattr(sel, "region"):
            return None
        point_x = getattr(sel, "_point_x", None)
        if point_x is not None:
            x = max(0.0, min(float(getattr(self, "ns", point_x)), float(point_x)))
            return x, x
        region = getattr(sel, "region", None)
        if region is None:
            return None
        try:
            if not region.isVisible():
                return None
            lo, hi = map(float, region.getRegion())
        except Exception:
            return None
        if hi < lo:
            lo, hi = hi, lo
        if (hi - lo) <= 1e-9:
            return lo, lo
        lo = max(0.0, lo)
        ns = float(getattr(self, "ns", hi))
        hi = min(ns, hi)
        if hi < lo:
            return None
        return lo, hi

    def _annotator_interval_from_range(self) -> Optional[tuple[float, float]]:
        lo, hi = self._annotator_current_range()
        mode = self.annotator.current_mode()
        if mode == "epoch":
            return self._annotator_snap_epoch_range(lo, hi)
        if mode == "interval":
            return self._annotator_visible_selection_range()
        return max(0.0, lo), max(lo, hi)

    def _annotator_describe_interval(self, start: float, stop: float) -> str:
        if abs(stop - start) < 1e-9:
            return f"{start:.2f}s"
        return f"{start:.2f}s -> {stop:.2f}s"

    def _annotator_begin_pending(self, key: str, label: str):
        span = self._annotator_interval_from_range()
        if span is None:
            self._annotator_status(
                "Interval mode requires a visible dragged selection; drag on the trace first.",
                2500,
            )
            self._annotator_pending = None
            return
        start, stop = span
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
        span = self._annotator_interval_from_range()
        if span is None:
            self._annotator_status(
                f"Holding {pending.key}: drag/select an interval region",
            )
            return
        start, stop = span
        pending.start = start
        pending.stop = stop
        self._annotator_status(
            f"Holding {pending.key}: {pending.label} [{self._annotator_describe_interval(start, stop)}]"
        )

    def _annotator_cancel_pending(self):
        self._annotator_pending = None
        self._annotator_status("Idle")

    def _annotator_insert_annotation(self, label: str, spans: list[tuple[float, float]]):
        if not hasattr(self, "p"):
            raise RuntimeError("No instance attached")
        self.p.insert_annot(
            str(label),
            [(float(start), float(stop)) for start, stop in spans],
            durcol2=False,
        )

    def _annotator_after_insert(self, labels: list[str]):
        curr_chs = list(self.ui.tbl_desc_signals.checked()) if hasattr(self.ui, "tbl_desc_signals") else []
        curr_anns = list(self.ui.tbl_desc_annots.checked()) if hasattr(self.ui, "tbl_desc_annots") else []
        keep_anns = list(dict.fromkeys(curr_anns + [str(x) for x in labels if str(x).strip()]))
        try:
            self._update_metrics()
            if hasattr(self.ui, "tbl_desc_signals"):
                self.ui.tbl_desc_signals.set_checked_by_labels(curr_chs)
            if hasattr(self.ui, "tbl_desc_annots"):
                if hasattr(self.ui.tbl_desc_annots, "set_checked_by_labels_silent"):
                    self.ui.tbl_desc_annots.set_checked_by_labels_silent(keep_anns)
                else:
                    self.ui.tbl_desc_annots.set_checked_by_labels(keep_anns)
                if hasattr(self, "_mark_instances_dirty"):
                    self._mark_instances_dirty(keep_anns)
            self._render_hypnogram()
            if getattr(self, "rendered", False):
                self._render_signals()
            else:
                self._render_signals_simple()
        finally:
            self._annotator_refresh_classes()
        n_labels = len({str(x) for x in labels if str(x).strip()})
        self._annotator_status(
            f"Committed {len(labels)} staged annotation{'s' if len(labels) != 1 else ''}"
            + (f" across {n_labels} classes" if n_labels > 1 else ""),
            3000,
        )

    def _annotator_staged_signature(self, label: str, start: float, stop: float, mode: str) -> tuple:
        return (
            str(label).strip(),
            round(float(start), 6),
            round(float(stop), 6),
            str(mode).strip(),
        )

    def _annotator_sync_stage_widgets(self):
        if not hasattr(self, "annotator") or self.annotator is None:
            return
        n = len(self._annotator_staged)
        if n == 0:
            text = "No staged additions"
        else:
            labels = sorted({x.label for x in self._annotator_staged}, key=str.casefold)
            noun = "addition" if n == 1 else "additions"
            text = f"{n} staged {noun}: {', '.join(labels)}"
        self.annotator.lbl_stage_pending.setText(text)
        self.annotator.butt_stage_clear.setEnabled(n > 0)
        self.annotator.butt_stage_commit.setEnabled(n > 0)

    def _annotator_stage_annotation(self, key: str, label: str, mode: str, start: float, stop: float):
        label = str(label).strip()
        staged = StagedAnnotation(
            key=str(key),
            label=label,
            mode=str(mode),
            start=float(start),
            stop=float(stop),
        )
        sig = self._annotator_staged_signature(label, start, stop, mode)
        existing = {
            self._annotator_staged_signature(x.label, x.start, x.stop, x.mode)
            for x in self._annotator_staged
        }
        if sig in existing:
            self._annotator_status(
                f"Already staged: {label} [{self._annotator_describe_interval(start, stop)}]",
                2000,
            )
            return
        self._annotator_staged.append(staged)
        self._annotator_sync_stage_widgets()
        self._annotator_refresh_classes()
        self._annot_queue_draw_overlay()
        self._annotator_status(
            f"Staged: {label} [{self._annotator_describe_interval(start, stop)}]",
            3000,
        )

    def _annotator_clear_staged(self):
        if not self._annotator_staged:
            self._annotator_status("No staged additions to clear", 2000)
            return
        self._annotator_staged.clear()
        self._annotator_sync_stage_widgets()
        self._annotator_refresh_classes()
        self._annot_queue_draw_overlay()
        self._annotator_status("Cleared staged additions", 3000)

    def _annotator_commit_staged(self):
        if not self._annotator_staged:
            self._annotator_status("No staged additions to commit", 2000)
            return

        grouped: dict[str, list[tuple[float, float]]] = {}
        staged_labels: list[str] = []
        for item in self._annotator_staged:
            grouped.setdefault(item.label, []).append((item.start, item.stop))
            staged_labels.append(item.label)

        try:
            for label, spans in grouped.items():
                self._annotator_insert_annotation(label, spans)
        except Exception as exc:
            self._annotator_status(f"Commit failed: {exc}", 3000)
            return

        self._annotator_staged.clear()
        self._annotator_sync_stage_widgets()
        self._annot_queue_draw_overlay()
        self._annotator_after_insert(staged_labels)

    def _annotator_handle_maintrace_key_press(self, ev) -> bool:
        if ev.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
            return False
        raw_text = (ev.text() or "").strip()
        key_text = self._annotator_key_text(ev)
        # d/D: toggle queue-delete on the currently selected annotation (annotator dock must be up)
        if raw_text in ("d", "D"):
            if (hasattr(self, "annotator") and self.annotator is not None
                    and self.annotator.isVisible()
                    and self.annotator.editor._identity is not None):
                if not ev.isAutoRepeat():
                    self._annot_editor_queue_delete()
                return True
        if not self._annotator_enabled():
            return False
        if key_text not in ANNOTATOR_KEYS:
            return False
        if ev.isAutoRepeat():
            return True
        label = self.annotator.bound_label_for_key(key_text)
        if not label:
            self._annotator_status(f"No annotation bound to key {key_text}", 2000)
            return True
        span = self._annotator_interval_from_range()
        if span is None:
            self._annotator_status(
                "Interval mode requires a visible dragged selection; drag on the trace first.",
                2500,
            )
            return True
        start, stop = span
        self._annotator_stage_annotation(
            key_text,
            label,
            self.annotator.current_mode(),
            start,
            stop,
        )
        return True

    def _annotator_handle_maintrace_key_release(self, ev) -> bool:
        pending = self._annotator_pending
        if pending is None:
            return False
        key_text = self._annotator_key_text(ev)
        if key_text != pending.key:
            return False
        if ev.isAutoRepeat():
            return True
        start, stop = pending.start, pending.stop
        label = pending.label
        mode = pending.mode
        key = pending.key
        self._annotator_pending = None
        self._annotator_stage_annotation(key, label, mode, start, stop)
        return True

    def _annotator_mode_active(self) -> bool:
        """True when the Annotator dock is open (drives click behaviour on pg1)."""
        return (hasattr(self, "annotator") and
                self.annotator is not None and
                self.annotator.isVisible())

    def _annotator_on_window_range_changed(self, lo: float, hi: float):
        _ = (lo, hi)
        if self._annotator_pending is not None:
            self._annotator_update_pending()
        if self._queued_deletes or self._queued_edits or self._annotator_staged:
            self._annot_queue_draw_overlay()
        self._annot_draw_cursor()

    # ------------------------------------------------------------------
    # Annotation editor
    # ------------------------------------------------------------------
    # Annotation editor — form panel driven by Dock5 selection
    # ------------------------------------------------------------------

    def _annot_editor_srv(self):
        return getattr(self, "ssa", None)

    def _annot_editor_from_instance(self, identity: dict):
        """Called by _on_row_changed when a Dock5 row is selected."""
        if not hasattr(self, "annotator") or self.annotator is None:
            return
        key = self._annot_identity_key(identity)
        ed = self.annotator.editor
        ed.set_instance(identity)
        if key in getattr(self, "_queued_deletes", set()):
            ed.lbl_header.setText(f"{identity['aclass']}  [QUEUED DELETE]")
            ed.lbl_header.setStyleSheet("font-weight: bold; color: #e05050;")
        elif key in getattr(self, "_queued_edits", set()):
            ed.lbl_header.setText(f"{identity['aclass']}  [QUEUED EDIT]")
            ed.lbl_header.setStyleSheet("font-weight: bold; color: #d0a020;")
        else:
            ed.lbl_header.setStyleSheet("font-weight: bold;")
        self._annot_draw_cursor()

    def _annot_editor_queue_edit(self):
        ed = self.annotator.editor
        identity = ed._identity
        if identity is None:
            return
        if self._annot_editor_srv() is None:
            return

        key = self._annot_identity_key(identity)

        # Always queue / update — repeated clicks update the queued values.
        # Original identity is preserved on first queue via setdefault below.
        vals = ed.current_values()
        kwargs: dict = {
            "aclass":   identity["aclass"],
            "inst_id":  identity["inst_id"],
            "start_tp": int(float(identity["start_tp"])),
            "stop_tp":  int(float(identity["stop_tp"])),
            "ch_str":   identity["ch_str"],
        }
        try:
            new_start_s = float(vals["start_sec"])
            new_stop_s  = float(vals["stop_sec"])
            kwargs["new_start"] = int(new_start_s * 1_000_000_000)
            kwargs["new_stop"]  = int(new_stop_s  * 1_000_000_000)
        except ValueError:
            pass
        new_inst = vals["inst_id"]
        if new_inst and new_inst != identity["inst_id"]:
            kwargs["new_inst_id"] = new_inst
        new_ch = vals["ch_str"]
        if new_ch != identity["ch_str"]:
            kwargs["new_ch"] = new_ch
        meta_str = vals["meta"].strip()
        if meta_str:
            meta_dict: dict[str, str] = {}
            for part in meta_str.split():
                if "=" in part:
                    k, _, v = part.partition("=")
                    meta_dict[k.strip()] = v.strip()
                else:
                    meta_dict[part] = ""
            if meta_dict:
                kwargs["meta"] = meta_dict
        try:
            self._queued_deletes.discard(key)
            self._queued_edits.add(key)
            self._queued_edit_kwargs[key] = kwargs
            self._queued_original_identity.setdefault(key, dict(identity))
            try:
                self._queued_positions[key] = (float(identity["start_sec"]), float(identity["stop_sec"]))
            except (KeyError, ValueError):
                pass
            self._annot_queue_replay_srv()
        except Exception as exc:
            ed.lbl_pending.setText(f"Error queuing edit: {exc}")
            return

        self._annot_queue_sync_pending()
        # Update header only — do NOT call set_instance, which would overwrite
        # the user's edited field values with the original identity data.
        ed.lbl_header.setText(f"{identity['aclass']}  [QUEUED EDIT]")
        ed.lbl_header.setStyleSheet("font-weight: bold; color: #d0a020;")
        self._annot_draw_cursor()
        self._annot_queue_refresh_dock5()
        self._annot_queue_draw_overlay()

    def _annot_editor_queue_delete(self):
        ed = self.annotator.editor
        identity = ed._identity
        if identity is None:
            return
        if self._annot_editor_srv() is None:
            return

        key = self._annot_identity_key(identity)

        if key in self._queued_deletes:
            # --- toggle OFF: unqueue this delete ---
            self._queued_deletes.discard(key)
            self._queued_delete_kwargs.pop(key, None)
            self._queued_positions.pop(key, None)
            # restore prior edit state if applicable
            if key in self._queued_prior_edits:
                self._queued_edits.add(key)
                self._queued_prior_edits.discard(key)
            self._annot_queue_replay_srv()
        else:
            # --- toggle ON: queue this delete ---
            del_kwargs = {
                "aclass":   identity["aclass"],
                "inst_id":  identity["inst_id"],
                "start_tp": int(float(identity["start_tp"])),
                "stop_tp":  int(float(identity["stop_tp"])),
                "ch_str":   identity["ch_str"],
            }
            if key in self._queued_edits:
                self._queued_prior_edits.add(key)
                self._queued_edits.discard(key)
                self._queued_edit_kwargs.pop(key, None)
            self._queued_deletes.add(key)
            self._queued_delete_kwargs[key] = del_kwargs
            try:
                self._queued_positions[key] = (float(identity["start_sec"]), float(identity["stop_sec"]))
            except (KeyError, ValueError):
                pass
            self._annot_queue_replay_srv()

        self._annot_queue_sync_pending()
        self._annot_editor_from_instance(identity)
        self._annot_queue_refresh_dock5()
        self._annot_queue_draw_overlay()

    def _annot_queue_refresh_dock5(self):
        curr_anns = list(self.ui.tbl_desc_annots.checked()) if hasattr(self.ui, "tbl_desc_annots") else []
        if hasattr(self, "_update_instances"):
            self._update_instances(curr_anns)

    def _annot_queue_sync_pending(self):
        """Sync the editor pending label to the actual number of unique queued ops."""
        if not hasattr(self, "annotator") or self.annotator is None:
            return
        n = len(self._queued_deletes) + len(self._queued_edits)
        ed = self.annotator.editor
        ed._pending = n
        if n == 0:
            ed.lbl_pending.setText("No pending edits")
        else:
            ed.lbl_pending.setText(f"{n} pending edit{'s' if n != 1 else ''} — not yet applied")

    def _annot_queue_replay_srv(self):
        """Rebuild the srv queue from Python state (needed after any toggle-off)."""
        srv = self._annot_editor_srv()
        if srv is None:
            return
        try:
            srv.clear_annot_edits()
        except Exception:
            pass
        for kw in self._queued_delete_kwargs.values():
            try:
                srv.delete_annot(**kw)
            except Exception:
                pass
        for kw in self._queued_edit_kwargs.values():
            try:
                srv.edit_annot(**kw)
            except Exception:
                pass

    def _annot_queue_clear_all_state(self):
        """Clear all Python-side queue tracking (used by apply and discard)."""
        self._queued_deletes.clear()
        self._queued_edits.clear()
        self._queued_prior_edits.clear()
        self._queued_positions.clear()
        self._queued_delete_kwargs.clear()
        self._queued_edit_kwargs.clear()
        self._queued_original_identity.clear()

    # ------------------------------------------------------------------
    # Queue overlay — lightweight marker layer on pg1
    # ------------------------------------------------------------------

    def _annot_queue_on_curves_reinit(self):
        """Called by _initiate_curves after pi.clear() — items are gone, just reset lists."""
        self._annot_queue_overlay_items.clear()
        self._annot_cursor_item = None

    def _annot_queue_overlay_on_trace_redraw(self):
        """Hook called at end of _update_pg1 and _update_pg1_simple."""
        self._annot_queue_draw_overlay()
        self._annot_draw_cursor()

    def _annot_queue_draw_overlay(self):
        """Draw colored outline markers over queued-delete/edit annotation bars."""
        import numpy as np
        import pyqtgraph as pg

        pw = getattr(self.ui, "pg1", None)
        if pw is None:
            return

        # remove stale overlay items
        for item in self._annot_queue_overlay_items:
            try:
                pw.removeItem(item)
            except Exception:
                pass
        self._annot_queue_overlay_items.clear()

        if not self._queued_deletes and not self._queued_edits:
            if not self._annotator_staged:
                return
        if not getattr(self, "annot_mgr", None):
            return

        del_rect_xs: list = []
        del_rect_ys: list = []
        del_slash_xs: list = []
        del_slash_ys: list = []
        edit_xs: list = []
        edit_ys: list = []
        add_xs: list = []
        add_ys: list = []

        all_queued = {k: "del" for k in self._queued_deletes}
        all_queued.update({k: "edit" for k in self._queued_edits})

        nan = float("nan")
        for key, status in all_queued.items():
            aclass = key[0]
            if aclass not in self.annot_mgr.tracks:
                continue
            pos = self._queued_positions.get(key)
            if pos is None:
                continue
            x0, x1 = float(pos[0]), float(pos[1])
            track = self.annot_mgr.tracks[aclass]
            if len(track["y0"]) == 0:
                continue
            y0 = float(track["y0"][0])
            y1 = float(track["y1"][0])

            rect   = [x0, x1, x1, x0, x0, nan]
            rect_y = [y0, y0, y1, y1, y0, nan]

            if status == "del":
                del_rect_xs.extend(rect)
                del_rect_ys.extend(rect_y)
                del_slash_xs.extend([x0, x1, nan])
                del_slash_ys.extend([y1, y0, nan])
            else:
                edit_xs.extend(rect)
                edit_ys.extend(rect_y)

        fallback_labels: list[str] = []
        for item in self._annotator_staged:
            if item.label not in self.annot_mgr.tracks and item.label not in fallback_labels:
                fallback_labels.append(item.label)
        fallback_idx = {label: idx for idx, label in enumerate(fallback_labels)}

        def _fallback_y_bounds(idx: int) -> tuple[float, float]:
            lane_h = 0.035
            gap = 0.01
            top = 0.96 - idx * (lane_h + gap)
            bottom = max(0.02, top - lane_h)
            return bottom, max(bottom + 0.01, top)

        for item in self._annotator_staged:
            x0 = float(item.start)
            x1 = float(item.stop)
            if item.label in self.annot_mgr.tracks and len(self.annot_mgr.tracks[item.label]["y0"]) != 0:
                track = self.annot_mgr.tracks[item.label]
                y0 = float(track["y0"][0])
                y1 = float(track["y1"][0])
            else:
                y0, y1 = _fallback_y_bounds(fallback_idx.get(item.label, 0))
            rect = [x0, x1, x1, x0, x0, nan]
            rect_y = [y0, y0, y1, y1, y0, nan]
            add_xs.extend(rect)
            add_ys.extend(rect_y)

        def _add(xs, ys, color, width, style=Qt.SolidLine):
            if not xs:
                return
            pen = pg.mkPen(color, width=width, cosmetic=True, style=style)
            item = pg.PlotCurveItem(
                x=np.asarray(xs, dtype=float),
                y=np.asarray(ys, dtype=float),
                pen=pen, connect="finite",
            )
            pw.addItem(item)
            self._annot_queue_overlay_items.append(item)

        _add(del_rect_xs, del_rect_ys, (220, 60, 60, 220), 3)
        _add(del_slash_xs, del_slash_ys, (220, 60, 60, 200), 2)
        _add(edit_xs, edit_ys, (210, 160, 20, 220), 3, Qt.DashLine)
        _add(add_xs, add_ys, (70, 210, 235, 220), 3, Qt.DashLine)

    # ------------------------------------------------------------------
    # Cursor highlight — white/gray outline on the currently selected bar
    # ------------------------------------------------------------------

    def _annot_cursor_on_dock_visibility(self, visible: bool):
        if visible:
            self._annot_draw_cursor()
            if hasattr(self, "_hide_pg1_probe"):
                self._hide_pg1_probe()
        else:
            self._annot_clear_cursor()

    def _annot_clear_cursor(self):
        pw = getattr(self.ui, "pg1", None)
        if pw is not None and self._annot_cursor_item is not None:
            try:
                pw.removeItem(self._annot_cursor_item)
            except Exception:
                pass
        self._annot_cursor_item = None

    def _annot_draw_cursor(self):
        import numpy as np
        import pyqtgraph as pg

        pw = getattr(self.ui, "pg1", None)
        if pw is None:
            return

        self._annot_clear_cursor()

        if not getattr(self, "annotator", None) or not self.annotator.isVisible():
            return

        identity = self.annotator.editor._identity
        if identity is None:
            return

        key = self._annot_identity_key(identity)
        if key in self._queued_deletes or key in self._queued_edits:
            return

        aclass = identity["aclass"]
        if not getattr(self, "annot_mgr", None) or aclass not in self.annot_mgr.tracks:
            return

        track = self.annot_mgr.tracks[aclass]
        if len(track["y0"]) == 0:
            return

        try:
            x0, x1 = float(identity["start_sec"]), float(identity["stop_sec"])
        except (KeyError, ValueError):
            return

        y0, y1 = float(track["y0"][0]), float(track["y1"][0])
        nan = float("nan")
        pen = pg.mkPen((210, 210, 210, 200), width=2, cosmetic=True, style=Qt.DashLine)
        item = pg.PlotCurveItem(
            x=np.asarray([x0, x1, x1, x0, x0, nan], dtype=float),
            y=np.asarray([y0, y0, y1, y1, y0, nan], dtype=float),
            pen=pen, connect="finite",
        )
        pw.addItem(item)
        self._annot_cursor_item = item

    def _annot_editor_apply(self):
        srv = self._annot_editor_srv()
        ed = self.annotator.editor
        if srv is None:
            return

        # Save kwargs before clearing state — needed to re-apply to the new
        # self.ss segsrv created by _render_signals() in rendered mode.
        saved_del_kwargs  = dict(self._queued_delete_kwargs)
        saved_edit_kwargs = dict(self._queued_edit_kwargs)

        try:
            n = srv.apply_annot_edits()
        except Exception as exc:
            ed.lbl_pending.setText(f"Apply failed: {exc}")
            return

        self._annot_queue_clear_all_state()
        self._annot_queue_draw_overlay()
        ed.reset_pending()
        ed.lbl_pending.setText(f"Applied: {n} instance{'s' if n != 1 else ''} changed")
        if ed._identity:
            ed.lbl_header.setStyleSheet("font-weight: bold;")

        # full refresh — same as _annotator_after_insert
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
            if getattr(self, "rendered", False):
                self._render_signals()
                # _render_signals() creates a fresh self.ss = lp.segsrv(self.p)
                # synchronously before starting the background thread.
                # apply_annot_edits() on self.ssa may not propagate to a new
                # segsrv, so re-queue and apply to self.ss here while it is
                # still on the main thread and before the thread queries annots.
                ss = getattr(self, "ss", None)
                if ss is not None and (saved_del_kwargs or saved_edit_kwargs):
                    try:
                        for kw in saved_del_kwargs.values():
                            ss.delete_annot(**kw)
                        for kw in saved_edit_kwargs.values():
                            ss.edit_annot(**kw)
                        ss.apply_annot_edits()
                    except Exception:
                        pass
            else:
                # Bust the ssa windowed-annotation cache: segsrv skips
                # recompiling if window bounds are unchanged, so a deleted
                # annotation stays visible until the window moves.
                # Setting a dummy window first forces a full recompile.
                try:
                    if getattr(self, "ssa", None) is not None:
                        self.ssa.window(0.0, 0.0)
                except Exception:
                    pass
                self._render_signals_simple()
        finally:
            self._annotator_refresh_classes()

    def _annot_editor_discard(self):
        srv = self._annot_editor_srv()
        ed = self.annotator.editor
        if srv is not None:
            try:
                srv.clear_annot_edits()
            except Exception:
                pass
        self._annot_queue_clear_all_state()
        self._annot_queue_draw_overlay()
        ed.reset_pending()
        ed.lbl_pending.setText("Edits discarded")
        if ed._identity:
            ed.lbl_header.setStyleSheet("font-weight: bold;")
        self._annot_queue_refresh_dock5()

    # ------------------------------------------------------------------
    # Canvas hit-test: double-click on annotation bar → select in Dock5
    # ------------------------------------------------------------------

    def _annot_hit_test(self, scene_pos, vb) -> tuple | None:
        """Return (ann_class, x_start, x_stop) for the bar under scene_pos, or None."""
        if not hasattr(self, "annot_mgr") or self.annot_mgr is None:
            return None
        vp = vb.mapSceneToView(scene_pos)
        x, y = vp.x(), vp.y()
        for name, track in self.annot_mgr.tracks.items():
            if name == "__#gaps__":
                continue
            x0s, x1s = track["x0"], track["x1"]
            y0s, y1s = track["y0"], track["y1"]
            hits: list[tuple[float, float]] = []
            for i in range(len(x0s)):
                if x0s[i] <= x <= x1s[i] and min(y0s[i], y1s[i]) <= y <= max(y0s[i], y1s[i]):
                    hits.append((float(x0s[i]), float(x1s[i])))
            if hits:
                # If multiple bars of the same class overlap at the cursor,
                # prefer the narrowest matching interval. This makes stacked/
                # nested annotations with the same start much easier to target.
                hits.sort(key=lambda span: ((span[1] - span[0]), abs((span[0] + span[1]) * 0.5 - x)))
                return (name, hits[0][0], hits[0][1])
        return None

    def _annot_select_in_dock5(self, ann_class: str, x_start: float, x_stop: float, zoom: bool = True):
        """Find the best-matching row in Dock5, select it, and raise the Annotator."""
        from PySide6.QtCore import QSortFilterProxyModel
        from PySide6.QtWidgets import QAbstractItemView

        # Ensure dock5 is visible and its table is populated before searching.
        # _update_instances defers the rebuild when dock5 is hidden, so we must
        # show the dock and flush the pending update first.
        dock5 = getattr(self.ui, "dock_annots", None)
        if dock5 is not None and not dock5.isVisible():
            dock5.show()
            dock5.raise_()
            if hasattr(self, "_flush_instances_update"):
                self._flush_instances_update()

        model = getattr(self, "events_model", None)
        if model is None:
            return

        headers = [
            str(model.headerData(c, Qt.Horizontal) or "")
            for c in range(model.columnCount())
        ]
        try:
            class_col = headers.index("class")
            start_col = headers.index("start")
            dur_col   = headers.index("dur")
        except ValueError:
            return

        candidates: list[tuple[tuple[float, float, float], int]] = []
        for r in range(model.rowCount()):
            cls = str(model.data(model.index(r, class_col)) or "")
            # strip any queued-status prefix before comparing
            for _pfx in ("(X) ", "(E) "):
                if cls.startswith(_pfx):
                    cls = cls[len(_pfx):]
                    break
            if cls != ann_class:
                continue
            try:
                start = float(model.data(model.index(r, start_col)))
                dur   = float(model.data(model.index(r, dur_col)))
            except (TypeError, ValueError):
                continue
            stop = start + dur
            if start <= x_stop and stop >= x_start:
                score = (
                    abs(start - x_start) + abs(stop - x_stop),
                    abs(dur - (x_stop - x_start)),
                    abs(start - x_start),
                )
                candidates.append((score, r))

        if not candidates:
            return

        candidates.sort(key=lambda item: item[0])
        candidate_rows = [row for _, row in candidates]
        best_row = candidate_rows[0]

        view = getattr(self.ui, "tbl_desc_events", None)
        if view is None:
            return

        proxy = view.model()
        curr_proxy_idx = view.currentIndex()
        curr_src_row = None
        if curr_proxy_idx.isValid():
            curr_src_idx = curr_proxy_idx
            curr_model = proxy
            while curr_model is not None and hasattr(curr_model, "mapToSource"):
                curr_src_idx = curr_model.mapToSource(curr_src_idx)
                try:
                    curr_model = curr_model.sourceModel()
                except Exception:
                    curr_model = None
            if curr_src_idx.isValid():
                curr_src_row = curr_src_idx.row()

        if curr_src_row in candidate_rows and len(candidate_rows) > 1:
            curr_pos = candidate_rows.index(curr_src_row)
            best_row = candidate_rows[(curr_pos + 1) % len(candidate_rows)]

        src_idx = model.index(best_row, 0)
        if proxy is not None and hasattr(proxy, "mapFromSource"):
            proxy_idx = proxy.mapFromSource(src_idx)
        else:
            proxy_idx = src_idx
        if not proxy_idx.isValid():
            return

        if not zoom:
            self._annot_select_no_zoom = True
        view.setCurrentIndex(proxy_idx)
        self._annot_select_no_zoom = False
        view.scrollTo(proxy_idx, QAbstractItemView.PositionAtCenter)

        # show and raise Annotator, switch to Edit tab
        if hasattr(self, "annotator") and self.annotator is not None:
            self.annotator.show()
            self.annotator.raise_()
            self.annotator._tabs.setCurrentIndex(0)
