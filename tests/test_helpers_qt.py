"""Qt-flavoured tests for :mod:`lunascope.helpers` widgets.

Run under the ``offscreen`` Qt platform; require the ``qapp`` fixture.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.qt


def test_screen_clamp_under_available_size(qapp):
    from lunascope.helpers import screen_clamp

    w, h = screen_clamp(10_000, 8_000, frac=0.85)
    # Clamp should never inflate values
    assert w <= 10_000
    assert h <= 8_000
    assert w > 0 and h > 0


def test_is_dark_palette_returns_bool(qapp):
    from lunascope.helpers import is_dark_palette

    out = is_dark_palette()
    assert isinstance(out, bool)


def test_clear_rows_removes_all_rows(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QStandardItem, QStandardItemModel

    from lunascope.helpers import clear_rows

    model = QStandardItemModel(3, 2)
    model.setHorizontalHeaderLabels(["a", "b"])
    for r in range(3):
        for c in range(2):
            model.setItem(r, c, QStandardItem(f"r{r}c{c}"))

    assert model.rowCount() == 3
    clear_rows(model, keep_headers=True)
    assert model.rowCount() == 0
    # Headers preserved
    assert model.headerData(0, Qt.Horizontal) == "a"
    assert model.headerData(1, Qt.Horizontal) == "b"


def test_clear_rows_blanks_headers_when_requested(qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QStandardItem, QStandardItemModel

    from lunascope.helpers import clear_rows

    model = QStandardItemModel(2, 2)
    model.setHorizontalHeaderLabels(["foo", "bar"])
    model.setItem(0, 0, QStandardItem("x"))
    clear_rows(model, keep_headers=False)

    assert model.rowCount() == 0
    h0 = model.headerData(0, Qt.Horizontal)
    h1 = model.headerData(1, Qt.Horizontal)
    assert (h0 or "") == ""
    assert (h1 or "") == ""


def test_clear_rows_handles_none_model(qapp):
    """Passing a target whose model is None should not raise."""
    from PySide6.QtWidgets import QTableView

    from lunascope.helpers import clear_rows

    view = QTableView()
    # No model attached → must be a graceful no-op
    clear_rows(view)


def test_add_dock_shortcuts_registers_windows_friendly_reset_shortcut(qapp):
    from PySide6.QtGui import QAction
    from PySide6.QtWidgets import QMainWindow, QMenu

    from lunascope.helpers import add_dock_shortcuts

    win = QMainWindow()
    view_menu = QMenu("View", win)
    win.menuView = view_menu
    mask_action = QAction("(-) Masks / Subset", win)
    view_menu.addAction(mask_action)

    add_dock_shortcuts(win, view_menu, reset_layout=lambda: None)

    reset_action = next(
        act for act in view_menu.actions() if act.text() == "Reset to Default Layout"
    )
    shortcuts = {seq.toString() for seq in reset_action.shortcuts()}

    assert "Ctrl+)" in shortcuts
    assert "Ctrl+Shift+0" in shortcuts
    assert mask_action.shortcut().toString() == "Ctrl+-"


def test_font_controller_clamps_persists_and_applies_app_font(qapp, tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QFont

    from lunascope.helpers import (
        AppFontController,
        FONT_SCALE_MAX,
        FONT_SCALE_MIN,
        saved_font_scale,
    )

    settings = QSettings(str(tmp_path / "font.ini"), QSettings.IniFormat)
    original = QFont(qapp.font())
    original_prop = qapp.property("lunascope_font_scale")
    base_size = original.pointSizeF()
    if base_size <= 0:
        base_size = float(original.pointSize() if original.pointSize() > 0 else 10.0)

    try:
        ctl = AppFontController(settings=settings, apply_delay_ms=1)
        ctl.set_scale(99, immediate=True)
        assert ctl.scale == FONT_SCALE_MAX
        assert saved_font_scale(settings) == FONT_SCALE_MAX
        assert qapp.property("lunascope_font_scale") == FONT_SCALE_MAX

        ctl.set_scale(-99, immediate=True)
        assert ctl.scale == FONT_SCALE_MIN
        assert saved_font_scale(settings) == FONT_SCALE_MIN
        assert qapp.font().pointSizeF() == pytest.approx(base_size * FONT_SCALE_MIN)
    finally:
        qapp.setFont(original)
        qapp.setProperty("lunascope_font_scale", original_prop)


def test_font_controller_actions_do_not_use_mask_subset_shortcut(qapp, tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QMainWindow

    from lunascope.helpers import AppFontController

    settings = QSettings(str(tmp_path / "font.ini"), QSettings.IniFormat)
    win = QMainWindow()
    ctl = AppFontController(win, settings=settings)
    actions = ctl.create_actions(win)
    shortcuts = {
        seq.toString()
        for action in actions
        for seq in action.shortcuts()
    }

    assert [action.text() for action in actions] == [
        "Larger Text",
        "Smaller Text",
        "Reset Text Size",
    ]
    assert "Ctrl+-" not in shortcuts


def test_wide_popup_combo_box_expands_popup_for_long_items(qapp):
    from lunascope.components.explorer_waveform import _WidePopupComboBox

    combo = _WidePopupComboBox()
    combo.resize(120, combo.sizeHint().height())
    combo.addItems(["Short", "SP_15_PZ_negative_peak_annotation_name"])

    combo.show()
    qapp.processEvents()
    combo.showPopup()
    qapp.processEvents()

    expected = combo.popup_width_hint()
    assert combo.view().minimumWidth() == expected
    assert combo.view().minimumWidth() > combo.width()

    combo.hidePopup()
