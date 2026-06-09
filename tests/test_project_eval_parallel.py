import pandas as pd

from lunascope.components.anal import (
    _clamp_project_eval_workers,
    _default_project_eval_workers,
    _normalize_project_result_table,
    _project_eval_slices,
)


def test_default_project_eval_workers_is_safe_for_unknown_cpu_count(monkeypatch):
    monkeypatch.setattr("lunascope.components.anal.os.cpu_count", lambda: None)

    assert _default_project_eval_workers() == 1


def test_default_project_eval_workers_caps_at_ten():
    assert _default_project_eval_workers(64) == 10


def test_default_project_eval_workers_uses_half_available_cpus():
    assert _default_project_eval_workers(8) == 4


def test_clamp_project_eval_workers_bounds_and_record_count():
    assert _clamp_project_eval_workers(0) == 1
    assert _clamp_project_eval_workers(99) == 10
    assert _clamp_project_eval_workers(8, total_records=3) == 3


def test_project_eval_slices_are_contiguous_and_cover_all_rows():
    tasks = [
        {"ordinal": i, "sample_row": [f"S{i}", f"{i}.edf", "."], "label": f"S{i}"}
        for i in range(1, 8)
    ]

    slices = _project_eval_slices(tasks, workers=3)

    assert [(s["start_ordinal"], s["end_ordinal"]) for s in slices] == [
        (1, 3),
        (4, 6),
        (7, 7),
    ]
    assert [row[0] for s in slices for row in s["rows"]] == [
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
    ]


def test_normalize_project_result_table_adds_id_column():
    df = pd.DataFrame({"X": [1, 2]})

    out = _normalize_project_result_table(df, "S1")

    assert list(out.columns) == ["ID", "X"]
    assert out["ID"].tolist() == ["S1", "S1"]


def test_normalize_project_result_table_fills_blank_ids():
    df = pd.DataFrame({"ID": ["", None, "S2"], "X": [1, 2, 3]})

    out = _normalize_project_result_table(df, "S1")

    assert list(out.columns) == ["ID", "X"]
    assert out["ID"].tolist() == ["S1", "S1", "S2"]
