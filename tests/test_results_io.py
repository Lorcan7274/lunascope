"""Tests for the file format helpers in :mod:`lunascope.components.results_io`.

The full ``ResultsIOMixin`` requires a live UI; here we exercise the static
load/save logic by binding the mixin methods to a lightweight fake
controller object.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from lunascope.components import results_io


class _FakeCtrl:
    """Minimal stand-in providing the attributes used by the load helpers."""

    _save_results_pkl = results_io.ResultsIOMixin._save_results_pkl
    _save_results_zip = results_io.ResultsIOMixin._save_results_zip
    _load_results_pkl = results_io.ResultsIOMixin._load_results_pkl
    _load_results_zip = results_io.ResultsIOMixin._load_results_zip

    def __init__(self):
        self.results = {
            "HYPNO_B": pd.DataFrame({"ID": ["S1"], "TST": [400.0]}),
            "HYPNO_SS": pd.DataFrame(
                {"ID": ["S1", "S1"], "SS": ["N1", "N2"], "MINS": [10.0, 20.0]}
            ),
        }


def _pairs():
    # Use non-empty strata names because the zip backend serialises through
    # TSV+pandas, where an empty string round-trips to ``"nan"``.
    return [("HYPNO", "B"), ("HYPNO", "SS")]


def test_save_and_load_pkl_round_trip(tmp_path):
    ctrl = _FakeCtrl()
    path = tmp_path / "results.pkl"
    ctrl._save_results_pkl(str(path), _pairs())

    results, tree = ctrl._load_results_pkl(str(path))
    assert set(results.keys()) == set(ctrl.results.keys())
    pd.testing.assert_frame_equal(results["HYPNO_B"], ctrl.results["HYPNO_B"])
    assert tree == _pairs()


def test_save_and_load_zip_round_trip(tmp_path):
    ctrl = _FakeCtrl()
    path = tmp_path / "results.zip"
    ctrl._save_results_zip(str(path), _pairs())

    # Verify zip layout has manifest and per-table tsv files
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        assert "_manifest.tsv" in names
        assert any(n.endswith("HYPNO_SS.tsv") for n in names)

    results, tree = ctrl._load_results_zip(str(path))
    pd.testing.assert_frame_equal(
        results["HYPNO_SS"].reset_index(drop=True),
        ctrl.results["HYPNO_SS"].reset_index(drop=True),
    )
    assert tree == _pairs()


def test_save_and_load_zip_round_trip_utf8_text(tmp_path):
    ctrl = _FakeCtrl()
    ctrl.results = {
        "标注_中文": pd.DataFrame(
            {"ID": ["样本一"], "ANNOT": ["睡眠阶段"], "PATH": ["数据/受试者一.edf"]}
        )
    }
    path = tmp_path / "结果.zip"
    ctrl._save_results_zip(str(path), [("标注", "中文")])

    with zipfile.ZipFile(path) as zf:
        assert any("标注_中文.tsv" in name for name in zf.namelist())

    results, tree = ctrl._load_results_zip(str(path))

    assert tree == [("标注", "中文")]
    pd.testing.assert_frame_equal(results["标注_中文"], ctrl.results["标注_中文"])


def test_load_pkl_rejects_non_dict(tmp_path):
    import pickle

    bad = tmp_path / "bad.pkl"
    with open(bad, "wb") as f:
        pickle.dump([1, 2, 3], f)

    with pytest.raises(ValueError, match="dict"):
        _FakeCtrl()._load_results_pkl(str(bad))


def test_load_pkl_requires_results_and_tree_keys(tmp_path):
    import pickle

    bad = tmp_path / "missing.pkl"
    with open(bad, "wb") as f:
        pickle.dump({"results": {}}, f)  # missing 'tree'

    with pytest.raises(ValueError, match="tree"):
        _FakeCtrl()._load_results_pkl(str(bad))


def test_load_pkl_rejects_non_dataframe_values(tmp_path):
    import pickle

    bad = tmp_path / "wrongtype.pkl"
    with open(bad, "wb") as f:
        pickle.dump({"results": {"foo": [1, 2, 3]}, "tree": []}, f)

    with pytest.raises(ValueError, match="DataFrame"):
        _FakeCtrl()._load_results_pkl(str(bad))


def test_load_zip_rejects_missing_manifest(tmp_path):
    bad = tmp_path / "no_manifest.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("foo.tsv", "a\tb\n1\t2\n")
    with pytest.raises(ValueError, match="_manifest"):
        _FakeCtrl()._load_results_zip(str(bad))


def test_load_zip_rejects_manifest_with_missing_columns(tmp_path):
    bad = tmp_path / "bad_manifest.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("_manifest.tsv", "key\tcommand\n")  # missing 'strata'
    with pytest.raises(ValueError, match="missing columns"):
        _FakeCtrl()._load_results_zip(str(bad))
