"""Tests for :mod:`lunascope.session_state`.

These tests build a minimal ``QMainWindow`` with a known set of named
widgets, save the state to disk, and verify a round-trip restores all the
expected values.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.qt


def _make_window(qapp):
    """Construct a window populated with one of every supported widget type."""
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

    win = QMainWindow()
    win.setObjectName("test_window")

    central = QWidget()
    central.setObjectName("central")
    win.setCentralWidget(central)

    # one of each
    le = QLineEdit(parent=central)
    le.setObjectName("le_demo")
    le.setText("hello")

    pte = QPlainTextEdit(parent=central)
    pte.setObjectName("pte_demo")
    pte.setPlainText("multi\nline")

    sp = QSpinBox(parent=central)
    sp.setObjectName("sp_demo")
    sp.setRange(0, 100)
    sp.setValue(42)

    dsp = QDoubleSpinBox(parent=central)
    dsp.setObjectName("dsp_demo")
    dsp.setRange(-10.0, 10.0)
    dsp.setValue(1.25)

    cb = QCheckBox(parent=central)
    cb.setObjectName("cb_demo")
    cb.setChecked(True)

    rb = QRadioButton(parent=central)
    rb.setObjectName("rb_demo")
    rb.setChecked(True)

    combo = QComboBox(parent=central)
    combo.setObjectName("combo_demo")
    combo.addItems(["alpha", "beta", "gamma"])
    combo.setCurrentIndex(2)

    tabs = QTabWidget(parent=central)
    tabs.setObjectName("tabs_demo")
    tabs.addTab(QWidget(), "T0")
    tabs.addTab(QWidget(), "T1")
    tabs.setCurrentIndex(1)

    from PySide6.QtCore import Qt

    dock = QDockWidget("dock_title", win)
    dock.setObjectName("dock_demo")
    dock.setWidget(QWidget())
    win.addDockWidget(Qt.LeftDockWidgetArea, dock)

    # store widgets so the caller can clear & re-check them
    win._test_widgets = dict(
        le=le,
        pte=pte,
        sp=sp,
        dsp=dsp,
        cb=cb,
        rb=rb,
        combo=combo,
        tabs=tabs,
        dock=dock,
    )
    return win


def test_collect_session_state_schema(qapp):
    from lunascope import session_state

    win = _make_window(qapp)
    state = session_state.collect_session_state(
        win, app_meta={"version": "1.0"}, session_meta={"k": "v"}
    )

    assert state["schema_version"] == session_state.SCHEMA_VERSION
    assert state["app"]["version"] == "1.0"
    assert state["session"]["k"] == "v"
    assert "geometry_b64" in state["window"]
    assert "le_demo" in state["widgets"]["line_edits"]
    assert "pte_demo" in state["widgets"]["plain_text_edits"]
    assert "sp_demo" in state["widgets"]["spin_boxes"]
    assert "dsp_demo" in state["widgets"]["double_spin_boxes"]
    assert "cb_demo" in state["widgets"]["check_boxes"]
    assert "rb_demo" in state["widgets"]["radio_buttons"]
    assert "combo_demo" in state["widgets"]["combo_boxes"]
    assert "tabs_demo" in state["widgets"]["tab_widgets"]
    assert "dock_demo" in state["docks"]


def test_excluded_combo_boxes_are_skipped(qapp):
    from PySide6.QtWidgets import QComboBox

    from lunascope import session_state

    win = _make_window(qapp)
    pops = QComboBox(parent=win.centralWidget())
    pops.setObjectName("combo_pops")
    pops.addItems(["x", "y"])
    pops.setCurrentIndex(1)

    state = session_state.collect_session_state(win)
    assert "combo_pops" not in state["widgets"]["combo_boxes"]


def test_round_trip_restores_widget_values(qapp, tmp_path: Path):
    from lunascope import session_state

    win = _make_window(qapp)
    out_path = tmp_path / "session"  # no .lss suffix → save_session_file adds it
    info = session_state.save_session_file(out_path, win)
    saved = Path(info["path"])
    assert saved.suffix == ".lss"
    assert saved.exists()

    # Mutate widget values, then load — they should snap back.
    w = win._test_widgets
    w["le"].setText("changed")
    w["pte"].setPlainText("")
    w["sp"].setValue(1)
    w["dsp"].setValue(0.0)
    w["cb"].setChecked(False)
    w["rb"].setChecked(False)
    w["combo"].setCurrentIndex(0)
    w["tabs"].setCurrentIndex(0)

    result = session_state.load_session_file(saved, win)
    report = result["report"]
    assert report["restored"] > 0
    assert report["missing"] == 0

    assert w["le"].text() == "hello"
    assert w["pte"].toPlainText() == "multi\nline"
    assert w["sp"].value() == 42
    assert w["dsp"].value() == pytest.approx(1.25)
    assert w["cb"].isChecked() is True
    assert w["rb"].isChecked() is True
    assert w["combo"].currentText() == "gamma"
    assert w["tabs"].currentIndex() == 1


def test_geometry_round_trip(qapp, tmp_path: Path):
    from lunascope import session_state

    win = _make_window(qapp)
    win.resize(640, 480)
    geom_path = tmp_path / "geom.json"
    session_state.save_geometry_file(geom_path, win, app_meta={"v": "1"})

    payload = geom_path.read_text(encoding="utf-8")
    assert '"geometry_b64"' in payload

    # New window — load geometry
    win2 = _make_window(qapp)
    info = session_state.load_geometry_file(geom_path, win2)
    assert info["report"]["restored"] >= 1


def test_apply_session_state_reports_missing_widgets(qapp):
    from lunascope import session_state

    win = _make_window(qapp)
    fake_state = {
        "window": {},
        "docks": {"dock_does_not_exist": {"visible": True, "floating": False}},
        "widgets": {
            "line_edits": {"le_does_not_exist": "x"},
            "plain_text_edits": {},
            "spin_boxes": {},
            "double_spin_boxes": {},
            "check_boxes": {},
            "radio_buttons": {},
            "combo_boxes": {},
            "tab_widgets": {},
        },
    }
    report = session_state.apply_session_state(win, fake_state)
    assert report.missing >= 2
    assert any("le_does_not_exist" in s for s in report.missing_items)
    assert any("dock_does_not_exist" in s for s in report.missing_items)


def test_combo_box_pending_text_is_deferred(qapp):
    """Saved combo with no matching option in the empty target should defer."""
    from PySide6.QtWidgets import QComboBox

    from lunascope import session_state

    win = _make_window(qapp)
    # Replace the demo combo with an empty one of the same name
    central = win.centralWidget()
    old = win._test_widgets["combo"]
    old.setParent(None)
    new = QComboBox(parent=central)
    new.setObjectName("combo_demo")  # no items added
    win._test_widgets["combo"] = new

    fake_state = {
        "window": {},
        "docks": {},
        "widgets": {
            "line_edits": {},
            "plain_text_edits": {},
            "spin_boxes": {},
            "double_spin_boxes": {},
            "check_boxes": {},
            "radio_buttons": {},
            "combo_boxes": {"combo_demo": {"index": 0, "text": "gamma"}},
            "tab_widgets": {},
        },
    }
    report = session_state.apply_session_state(win, fake_state)
    assert report.deferred == 1
    assert new.property("_session_pending_text") == "gamma"
