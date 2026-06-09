from PySide6.QtCore import Qt, QSortFilterProxyModel
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QTableView

from lunascope.components.metrics import MetricsMixin


class _UI:
    def __init__(self):
        self.tbl_desc_signals = QTableView()
        self.tbl_desc_annots = QTableView()


class _Dummy(MetricsMixin):
    def __init__(self):
        self.ui = _UI()
        self.rendered = True
        self._rendered_chs = ["C3"]
        self._rendered_anns = ["spindle"]


def _model(headers, rows):
    model = QStandardItemModel()
    model.setHorizontalHeaderLabels(headers)
    for row in rows:
        model.appendRow([QStandardItem(str(value)) for value in row])
    return model


def _proxy_for(model):
    proxy = QSortFilterProxyModel()
    proxy.setSourceModel(model)
    return proxy


def test_rendered_membership_colors_gray_unrendered_signal_rows(qapp):
    obj = _Dummy()
    src = _model(["Sel", "CH", "SR"], [["", "C3", "256"], ["", "O1", "256"]])
    obj.ui.tbl_desc_signals.setModel(_proxy_for(src))

    obj._apply_rendered_membership_colors()

    included = src.item(0, 1).foreground().color().name().lower()
    excluded = src.item(1, 1).foreground().color().name().lower()
    assert included == "#d7dce2"
    assert excluded == "#777777"


def test_rendered_membership_colors_gray_unrendered_annotation_rows(qapp):
    obj = _Dummy()
    src = _model(["Sel", "Annotations"], [["", "spindle"], ["", "arousal"]])
    obj.ui.tbl_desc_annots.setModel(_proxy_for(src))

    obj._apply_rendered_membership_colors()

    assert src.item(0, 1).foreground().color().name().lower() == "#d7dce2"
    assert src.item(1, 1).foreground().color().name().lower() == "#777777"


def test_rendered_membership_colors_clear_when_not_rendered(qapp):
    obj = _Dummy()
    obj.rendered = False
    src = _model(["Sel", "CH"], [["", "C3"], ["", "O1"]])
    obj.ui.tbl_desc_signals.setModel(_proxy_for(src))

    obj._apply_rendered_membership_colors()

    for row in range(src.rowCount()):
        assert src.item(row, 1).foreground().color().name().lower() == "#d7dce2"
