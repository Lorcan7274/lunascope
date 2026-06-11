from __future__ import annotations

from lunascope.components.metrics import MetricsMixin


class _Dock:
    def __init__(self, visible: bool = False):
        self._visible = visible

    def isVisible(self) -> bool:
        return self._visible


class _Timer:
    def __init__(self):
        self.started = []

    def start(self, delay_ms: int):
        self.started.append(delay_ms)


class _Dummy(MetricsMixin):
    def __init__(self, dock_visible: bool = True):
        self.ui = type("UI", (), {})()
        self.ui.dock_annots = _Dock(visible=dock_visible)
        self._instances_update_timer = _Timer()
        self._pending_instance_annots = []
        self._instances_update_dirty = False
        self.ssa = object()
        self.rebuilt = None

    def _rebuild_instances_table(self, anns):
        self.rebuilt = list(anns)


def test_update_instances_defers_when_dock_hidden():
    obj = _Dummy(dock_visible=False)
    obj._update_instances(["spindle", "arousal"])
    assert obj._instances_update_dirty is True
    assert obj.rebuilt is None
    assert obj._instances_update_timer.started == []


def test_update_instances_fires_immediately_when_dock_visible():
    obj = _Dummy(dock_visible=True)
    obj._update_instances(["spindle", "arousal"])
    assert obj._instances_update_dirty is False
    assert obj._instances_update_timer.started == [0]


def test_flush_rebuilds_regardless_of_dock_visibility():
    # _flush_instances_update must rebuild unconditionally (no isVisible gate)
    # so the Qt-race case where visibilityChanged(True) fires but isVisible()
    # momentarily returns False does not deadlock dock5.
    for visible in (True, False):
        obj = _Dummy(dock_visible=visible)
        obj._pending_instance_annots = ["spindle"]
        obj._instances_update_dirty = True
        obj._flush_instances_update()
        assert obj.rebuilt == ["spindle"], f"flush failed with dock_visible={visible}"
        assert obj._instances_update_dirty is False


def test_mark_instances_dirty_starts_timer_when_dock_visible():
    # _mark_instances_dirty must fire the timer immediately when dock5 is open
    # so that mask / anal / annotator operations are reflected without needing
    # a hide/show cycle.
    obj = _Dummy(dock_visible=True)
    obj._mark_instances_dirty(["spindle"])
    assert obj._instances_update_dirty is True
    assert obj._instances_update_timer.started == [0]


def test_mark_instances_dirty_defers_when_dock_hidden():
    obj = _Dummy(dock_visible=False)
    obj._mark_instances_dirty(["spindle"])
    assert obj._instances_update_dirty is True
    assert obj._instances_update_timer.started == []


def test_visibility_change_flushes_deferred_update():
    obj = _Dummy(dock_visible=False)
    obj._update_instances(["spindle"])
    assert obj._instances_update_dirty is True

    obj.ui.dock_annots._visible = True
    obj._on_instances_dock_visibility_changed(True)
    assert obj._instances_update_timer.started == [0]

    obj._flush_instances_update()
    assert obj.rebuilt == ["spindle"]
    assert obj._instances_update_dirty is False
