from __future__ import annotations

from lunascope.components.metrics import MetricsMixin


class _Dock:
    def __init__(self, visible: bool = False):
        self._visible = visible

    def isVisible(self) -> bool:
        return self._visible


class _AnnotationTable:
    def __init__(self):
        self.labels = ["spindle", "arousal"]

    def checked(self):
        return list(self.labels)


class _Dummy(MetricsMixin):
    def __init__(self, dock_visible: bool = True):
        self.ui = type("UI", (), {})()
        self.ui.dock_annots = _Dock(visible=dock_visible)
        self.ui.tbl_desc_annots = _AnnotationTable()
        self.ssa = object()
        self.rebuilt = None

    def _rebuild_instances_table(self, anns):
        self.rebuilt = list(anns)


def test_update_instances_skips_rebuild_when_dock_hidden():
    obj = _Dummy(dock_visible=False)
    obj._update_instances(["spindle", "arousal"])
    assert obj.rebuilt is None


def test_update_instances_rebuilds_immediately_when_dock_visible():
    obj = _Dummy(dock_visible=True)
    obj._update_instances(["spindle", "arousal"])
    assert obj.rebuilt == ["spindle", "arousal"]


def test_mark_instances_dirty_rebuilds_immediately_when_dock_visible():
    obj = _Dummy(dock_visible=True)
    obj._mark_instances_dirty(["spindle"])
    assert obj.rebuilt == ["spindle"]


def test_mark_instances_dirty_skips_rebuild_when_dock_hidden():
    obj = _Dummy(dock_visible=False)
    obj._mark_instances_dirty(["spindle"])
    assert obj.rebuilt is None


def test_visibility_change_rebuilds_from_checked_annotations():
    obj = _Dummy(dock_visible=False)
    obj.ui.tbl_desc_annots.labels = ["slow_wave"]
    obj._on_instances_dock_visibility_changed(True)
    assert obj.rebuilt == ["slow_wave"]
