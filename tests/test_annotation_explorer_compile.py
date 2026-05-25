from __future__ import annotations

import pandas as pd

from lunascope.components.annot_explorer_funcs import compile_cohort


class _FakeEDF:
    def __init__(self, annots, *, nr=10, rs=30):
        self._annots = list(annots)
        self._stat = {"nr": nr, "rs": rs}

    def stat(self):
        return dict(self._stat)

    def annots(self):
        return list(self._annots)


class _FakeInstance:
    def __init__(self, annots, events, *, nr=10, rs=30):
        self.edf = _FakeEDF(annots, nr=nr, rs=rs)
        self._events = pd.DataFrame(events)

    def fetch_annots(self, classes):
        return self._events[self._events["Class"].isin(classes)].copy()


class _FakeProj:
    def __init__(self, mapping):
        self._mapping = dict(mapping)
        self.calls = []

    def inst(self, id_str):
        self.calls.append(str(id_str))
        return self._mapping[str(id_str)]


def test_compile_cohort_uses_live_current_instance_for_selected_subject():
    disk_instance = _FakeInstance(
        ["SPINDLE"],
        [{"Class": "SPINDLE", "Start": 10.0, "Stop": 11.0}],
    )
    live_instance = _FakeInstance(
        ["SPINDLE", "SO"],
        [
            {"Class": "SPINDLE", "Start": 10.0, "Stop": 11.0},
            {"Class": "SO", "Start": 20.0, "Stop": 21.5},
        ],
    )
    other_instance = _FakeInstance(
        ["KCOMPLEX"],
        [{"Class": "KCOMPLEX", "Start": 5.0, "Stop": 6.0}],
    )
    proj = _FakeProj({"S1": disk_instance, "S2": other_instance})

    cohort = compile_cohort(
        proj,
        ["S1", "S2"],
        current_instance=live_instance,
        current_id="S1",
    )

    assert proj.calls == ["S2"]
    assert cohort["annot_classes"] == ["KCOMPLEX", "SO", "SPINDLE"]
    subj1 = next(subj for subj in cohort["subjects"] if subj["id"] == "S1")
    assert subj1["events"]["Class"].tolist() == ["SPINDLE", "SO"]
    assert subj1["duration"] == 300.0


def test_compile_cohort_can_target_only_the_attached_record():
    disk_instance = _FakeInstance(
        ["SPINDLE"],
        [{"Class": "SPINDLE", "Start": 10.0, "Stop": 11.0}],
    )
    live_instance = _FakeInstance(
        ["SPINDLE", "SO"],
        [
            {"Class": "SPINDLE", "Start": 10.0, "Stop": 11.0},
            {"Class": "SO", "Start": 20.0, "Stop": 21.5},
        ],
    )
    proj = _FakeProj({"S1": disk_instance})

    cohort = compile_cohort(
        proj,
        ["S1"],
        current_instance=live_instance,
        current_id="S1",
    )

    assert proj.calls == []
    assert cohort["n_subjects"] == 1
    assert cohort["annot_classes"] == ["SO", "SPINDLE"]
    assert cohort["subjects"][0]["id"] == "S1"
    assert cohort["subjects"][0]["events"]["Class"].tolist() == ["SPINDLE", "SO"]
