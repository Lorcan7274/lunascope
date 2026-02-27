"""Session state save/restore helpers for Lunascope (Phase 1).

Phase 1 scope:
- Main window geometry/state (including dock placements)
- Explicit dock visibility/floating fallback map
- Text buffers and common UI control values

Design goals:
- Self-contained module
- Robust to future UI additions by introspecting widget types dynamically
- Graceful restore when widgets are missing/changed
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PySide6.QtCore import QByteArray
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QWidget,
)

SCHEMA_VERSION = 1

# Phase 1 intentionally excludes runtime/data-dependent combo boxes.
# They depend on attached EDF/sample content and transforms, which are not
# frozen/restored in Phase 1.
EXCLUDED_COMBO_BOXES = {
    "combo_pops",
    "combo_soap",
    "combo_ifnot_mask",
    "combo_if_mask",
    "combo_spectrogram",
}


@dataclass
class RestoreReport:
    restored: int = 0
    deferred: int = 0
    skipped: int = 0
    missing: int = 0
    deferred_items: list[str] | None = None
    skipped_items: list[str] | None = None
    missing_items: list[str] | None = None

    def __post_init__(self) -> None:
        if self.deferred_items is None:
            self.deferred_items = []
        if self.skipped_items is None:
            self.skipped_items = []
        if self.missing_items is None:
            self.missing_items = []


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _b64_qba(data: QByteArray) -> str:
    return base64.b64encode(bytes(data)).decode("ascii")


def _qba_from_b64(text: str) -> QByteArray:
    raw = base64.b64decode(text.encode("ascii"))
    return QByteArray(raw)


def _iter_named_widgets(root: QWidget, klass: type[QWidget]):
    for w in root.findChildren(klass):
        name = w.objectName()
        if name:
            yield name, w


def _collect_window(ui: QMainWindow) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if hasattr(ui, "saveGeometry"):
        out["geometry_b64"] = _b64_qba(ui.saveGeometry())
    if hasattr(ui, "saveState"):
        out["state_b64"] = _b64_qba(ui.saveState())
    return out


def _collect_docks(ui: QMainWindow) -> dict[str, dict[str, Any]]:
    docks: dict[str, dict[str, Any]] = {}
    for name, dock in _iter_named_widgets(ui, QDockWidget):
        geom_b64 = ""
        try:
            geom_b64 = _b64_qba(dock.saveGeometry())
        except Exception:
            geom_b64 = ""
        docks[name] = {
            "visible": bool(dock.isVisible()),
            "floating": bool(dock.isFloating()),
            "window_title": dock.windowTitle(),
            "geometry_b64": geom_b64,
        }
    return docks


def _collect_line_edits(ui: QWidget) -> dict[str, str]:
    return {name: w.text() for name, w in _iter_named_widgets(ui, QLineEdit)}


def _collect_plain_text_edits(ui: QWidget) -> dict[str, str]:
    return {name: w.toPlainText() for name, w in _iter_named_widgets(ui, QPlainTextEdit)}


def _collect_spin_boxes(ui: QWidget) -> dict[str, int]:
    return {name: int(w.value()) for name, w in _iter_named_widgets(ui, QSpinBox)}


def _collect_double_spin_boxes(ui: QWidget) -> dict[str, float]:
    return {name: float(w.value()) for name, w in _iter_named_widgets(ui, QDoubleSpinBox)}


def _collect_check_boxes(ui: QWidget) -> dict[str, bool]:
    return {name: bool(w.isChecked()) for name, w in _iter_named_widgets(ui, QCheckBox)}


def _collect_radio_buttons(ui: QWidget) -> dict[str, bool]:
    return {name: bool(w.isChecked()) for name, w in _iter_named_widgets(ui, QRadioButton)}


def _collect_combo_boxes(ui: QWidget) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, w in _iter_named_widgets(ui, QComboBox):
        if name in EXCLUDED_COMBO_BOXES:
            continue
        out[name] = {
            "index": int(w.currentIndex()),
            "text": w.currentText(),
        }
    return out


def _collect_tab_widgets(ui: QWidget) -> dict[str, int]:
    return {name: int(w.currentIndex()) for name, w in _iter_named_widgets(ui, QTabWidget)}


def collect_session_state(
    ui: QMainWindow,
    *,
    app_meta: dict[str, Any] | None = None,
    session_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = {
        "schema_version": SCHEMA_VERSION,
        "saved_utc": _now_utc(),
        "app": app_meta or {},
        "session": session_meta or {},
        "window": _collect_window(ui),
        "docks": _collect_docks(ui),
        "widgets": {
            "line_edits": _collect_line_edits(ui),
            "plain_text_edits": _collect_plain_text_edits(ui),
            "spin_boxes": _collect_spin_boxes(ui),
            "double_spin_boxes": _collect_double_spin_boxes(ui),
            "check_boxes": _collect_check_boxes(ui),
            "radio_buttons": _collect_radio_buttons(ui),
            "combo_boxes": _collect_combo_boxes(ui),
            "tab_widgets": _collect_tab_widgets(ui),
        },
    }
    return state


def _apply_map_bool(ui: QWidget, mapping: dict[str, bool], klass: type[QWidget], report: RestoreReport, setter: str) -> None:
    for name, val in mapping.items():
        w = ui.findChild(klass, name)
        if w is None:
            report.missing += 1
            report.missing_items.append(f"{klass.__name__}:{name}")
            continue
        fn = getattr(w, setter, None)
        if callable(fn):
            fn(bool(val))
            report.restored += 1
        else:
            report.skipped += 1
            report.skipped_items.append(f"{klass.__name__}:{name}:no_{setter}")


def _apply_map_text(ui: QWidget, mapping: dict[str, str], klass: type[QWidget], report: RestoreReport, setter: str) -> None:
    for name, val in mapping.items():
        w = ui.findChild(klass, name)
        if w is None:
            report.missing += 1
            report.missing_items.append(f"{klass.__name__}:{name}")
            continue
        fn = getattr(w, setter, None)
        if callable(fn):
            fn(str(val))
            report.restored += 1
        else:
            report.skipped += 1
            report.skipped_items.append(f"{klass.__name__}:{name}:no_{setter}")


def _apply_map_numeric(ui: QWidget, mapping: dict[str, Any], klass: type[QWidget], report: RestoreReport) -> None:
    for name, val in mapping.items():
        w = ui.findChild(klass, name)
        if w is None:
            report.missing += 1
            report.missing_items.append(f"{klass.__name__}:{name}")
            continue
        try:
            w.setValue(val)
            report.restored += 1
        except Exception:
            report.skipped += 1
            report.skipped_items.append(f"{klass.__name__}:{name}:invalid_value")


def _apply_combo_boxes(ui: QWidget, mapping: dict[str, dict[str, Any]], report: RestoreReport) -> None:
    for name, rec in mapping.items():
        if name in EXCLUDED_COMBO_BOXES:
            continue
        w = ui.findChild(QComboBox, name)
        if w is None:
            report.missing += 1
            report.missing_items.append(f"QComboBox:{name}")
            continue

        text = str(rec.get("text", ""))
        idx = int(rec.get("index", -1))

        used = False
        if text:
            i = w.findText(text)
            if i >= 0:
                w.setCurrentIndex(i)
                used = True
            elif w.count() == 0:
                # Data-dependent combo not populated yet: defer instead of skip.
                w.setProperty("_session_pending_text", text)
                report.deferred += 1
                report.deferred_items.append(f"QComboBox:{name}:pending_text={text}")
                continue
        if (not used) and 0 <= idx < w.count():
            w.setCurrentIndex(idx)
            used = True

        if used:
            report.restored += 1
        else:
            report.skipped += 1
            report.skipped_items.append(
                f"QComboBox:{name}:unmatched(text={text!r},index={idx},count={w.count()})"
            )


def _apply_tab_widgets(ui: QWidget, mapping: dict[str, int], report: RestoreReport) -> None:
    for name, idx in mapping.items():
        w = ui.findChild(QTabWidget, name)
        if w is None:
            report.missing += 1
            report.missing_items.append(f"QTabWidget:{name}")
            continue
        if 0 <= int(idx) < w.count():
            w.setCurrentIndex(int(idx))
            report.restored += 1
        else:
            report.skipped += 1
            report.skipped_items.append(
                f"QTabWidget:{name}:index_out_of_range(saved={idx},count={w.count()})"
            )


def apply_session_state(ui: QMainWindow, state: dict[str, Any]) -> RestoreReport:
    report = RestoreReport()

    # 1) Restore geometry/state first for dock/layout placement.
    window = state.get("window", {})
    geom_b64 = window.get("geometry_b64")
    if geom_b64:
        try:
            ok = ui.restoreGeometry(_qba_from_b64(geom_b64))
            report.restored += 1 if ok else 0
            report.skipped += 0 if ok else 1
            if not ok:
                report.skipped_items.append("window:restoreGeometry:false")
        except Exception:
            report.skipped += 1
            report.skipped_items.append("window:restoreGeometry:exception")

    state_b64 = window.get("state_b64")
    if state_b64:
        try:
            ok = ui.restoreState(_qba_from_b64(state_b64))
            report.restored += 1 if ok else 0
            report.skipped += 0 if ok else 1
            if not ok:
                report.skipped_items.append("window:restoreState:false")
        except Exception:
            report.skipped += 1
            report.skipped_items.append("window:restoreState:exception")

    # 2) Fallback dock visibility/floating (robust when new/changed docks exist).
    docks = state.get("docks", {})
    for name, rec in docks.items():
        d = ui.findChild(QDockWidget, name)
        if d is None:
            report.missing += 1
            report.missing_items.append(f"QDockWidget:{name}")
            continue
        try:
            if "floating" in rec:
                d.setFloating(bool(rec["floating"]))
                report.restored += 1
            geom_b64 = rec.get("geometry_b64")
            if geom_b64:
                ok = d.restoreGeometry(_qba_from_b64(str(geom_b64)))
                report.restored += 1 if ok else 0
                report.skipped += 0 if ok else 1
                if not ok:
                    report.skipped_items.append(f"QDockWidget:{name}:restoreGeometry:false")
            if "visible" in rec:
                vis = bool(rec["visible"])
                d.setVisible(vis)
                if vis:
                    d.show()
                    d.raise_()
                report.restored += 1
        except Exception:
            report.skipped += 1
            report.skipped_items.append(f"QDockWidget:{name}:setFloating/setVisible_exception")

    # Final visibility enforcement for floating docks after all state operations.
    for name, rec in docks.items():
        if not bool(rec.get("visible", False)):
            continue
        d = ui.findChild(QDockWidget, name)
        if d is None:
            continue
        try:
            d.show()
            if bool(rec.get("floating", False)):
                d.raise_()
        except Exception:
            pass

    # 3) Restore generic widget values.
    widgets = state.get("widgets", {})
    _apply_map_text(ui, widgets.get("line_edits", {}), QLineEdit, report, "setText")
    _apply_map_text(ui, widgets.get("plain_text_edits", {}), QPlainTextEdit, report, "setPlainText")
    _apply_map_numeric(ui, widgets.get("spin_boxes", {}), QSpinBox, report)
    _apply_map_numeric(ui, widgets.get("double_spin_boxes", {}), QDoubleSpinBox, report)
    _apply_map_bool(ui, widgets.get("check_boxes", {}), QCheckBox, report, "setChecked")
    _apply_map_bool(ui, widgets.get("radio_buttons", {}), QRadioButton, report, "setChecked")
    _apply_combo_boxes(ui, widgets.get("combo_boxes", {}), report)
    _apply_tab_widgets(ui, widgets.get("tab_widgets", {}), report)

    return report


def save_session_file(
    path: str | Path,
    ui: QMainWindow,
    *,
    app_meta: dict[str, Any] | None = None,
    session_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    p = Path(path).expanduser()
    if p.suffix.lower() != ".lss":
        p = p.with_suffix(".lss")

    state = collect_session_state(ui, app_meta=app_meta, session_meta=session_meta)
    p.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
    return {"path": str(p), "state": state}


def load_session_file(path: str | Path, ui: QMainWindow) -> dict[str, Any]:
    p = Path(path).expanduser()
    raw = p.read_text(encoding="utf-8")
    state = json.loads(raw)

    report = apply_session_state(ui, state)
    return {
        "path": str(p),
        "state": state,
        "report": {
            "restored": report.restored,
            "deferred": report.deferred,
            "skipped": report.skipped,
            "missing": report.missing,
            "deferred_items": report.deferred_items,
            "skipped_items": report.skipped_items,
            "missing_items": report.missing_items,
        },
    }
