"""Tests for the pure-logic helpers in :mod:`lunascope.helpers`.

These tests are independent of any Qt event loop; importing the module pulls
PySide6 in but no widgets are constructed.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt

from lunascope.helpers import (
    _canon,
    override_colors,
    random_darkbg_colors,
    sort_df_by_list,
    winsorize_array,
)
from lunascope.components.slist import (
    SListMixin,
    _is_absolute_sample_path,
    _read_sample_list_rows,
)


# ---------------------------------------------------------------------------
# sort_df_by_list
# ---------------------------------------------------------------------------


def test_sort_df_by_list_orders_known_values_and_keeps_unknown_at_end():
    df = pd.DataFrame({"stage": ["R", "N1", "W", "Other", "N2"], "v": [1, 2, 3, 4, 5]})
    out = sort_df_by_list(df, col_idx=0, order_list=["W", "N1", "N2", "N3", "R"])
    assert list(out["stage"]) == ["W", "N1", "N2", "R", "Other"]
    assert list(out["v"]) == [3, 2, 5, 1, 4]


def test_sort_df_by_list_is_case_insensitive():
    df = pd.DataFrame({"k": ["w", "N1", "r"]})
    out = sort_df_by_list(df, col_idx=0, order_list=["W", "N1", "R"])
    assert list(out["k"]) == ["w", "N1", "r"]


def test_sort_df_by_list_does_not_mutate_input():
    df = pd.DataFrame({"k": ["b", "a"]})
    snapshot = df.copy()
    sort_df_by_list(df, col_idx=0, order_list=["a", "b"])
    pd.testing.assert_frame_equal(df, snapshot)


# ---------------------------------------------------------------------------
# winsorize_array
# ---------------------------------------------------------------------------


def test_winsorize_clips_to_quantile_bounds():
    arr = np.arange(101, dtype=float)
    out = winsorize_array(arr, 0.05)
    # 5th percentile = 5.0, 95th = 95.0
    assert out.min() == pytest.approx(5.0)
    assert out.max() == pytest.approx(95.0)


def test_winsorize_with_zero_limit_is_identity():
    arr = np.array([1.0, 2.0, 3.0])
    out = winsorize_array(arr, 0.0)
    np.testing.assert_array_equal(out, arr)


def test_winsorize_clamps_limit_at_half():
    arr = np.arange(100, dtype=float)
    # limit > 0.5 should be clamped to 0.5 (median collapse)
    out = winsorize_array(arr, 0.9)
    median = np.median(arr)
    np.testing.assert_allclose(out, np.full_like(arr, median))


def test_winsorize_handles_nans():
    arr = np.array([np.nan, 0.0, 5.0, 10.0, np.nan])
    out = winsorize_array(arr, 0.1)
    assert np.isnan(out[0]) and np.isnan(out[-1])
    assert np.all(np.isfinite(out[1:4]))


def test_winsorize_does_not_mutate_input():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    snapshot = arr.copy()
    winsorize_array(arr, 0.1)
    np.testing.assert_array_equal(arr, snapshot)


def test_winsorize_all_nan_returns_all_nan():
    arr = np.array([np.nan, np.nan, np.nan])
    out = winsorize_array(arr, 0.1)
    assert np.all(np.isnan(out))


# ---------------------------------------------------------------------------
# _canon and override_colors
# ---------------------------------------------------------------------------


def test_canon_uppercases_and_strips():
    assert _canon("  fp1 ") == "FP1"
    assert _canon("Cz") == "CZ"


def test_override_colors_replaces_only_named():
    colors = ["#000000", "#111111", "#222222"]
    names = ["Fp1", "Cz", "O1"]
    out = override_colors(colors, names, {"cz": "#ff0000"})
    assert out[0] == "#000000"
    assert out[1].lower() == "#ff0000"
    assert out[2] == "#222222"


def test_override_colors_preserves_tuple_form():
    colors = [(10, 20, 30), (40, 50, 60)]
    names = ["A", "B"]
    out = override_colors(colors, names, {"A": "#ffffff"})
    assert isinstance(out[0], tuple)
    assert out[0] == (255, 255, 255)
    assert out[1] == (40, 50, 60)


def test_override_colors_no_overrides_passthrough():
    colors = ["#aaa", "#bbb"]
    names = ["x", "y"]
    out = override_colors(colors, names, {})
    assert out == colors


# ---------------------------------------------------------------------------
# random_darkbg_colors
# ---------------------------------------------------------------------------


def test_random_darkbg_colors_count_and_determinism():
    a = random_darkbg_colors(5, seed=42)
    b = random_darkbg_colors(5, seed=42)
    assert len(a) == 5 and len(b) == 5
    # The same seed should produce the same colors
    assert [c.name() for c in a] == [c.name() for c in b]


def test_random_darkbg_colors_zero_is_empty():
    assert random_darkbg_colors(0, seed=1) == []


def test_is_absolute_sample_path_handles_windows_and_posix_forms():
    assert _is_absolute_sample_path("/tmp/file.edf")
    assert _is_absolute_sample_path(r"C:\Users\john\data\file.edf")
    assert _is_absolute_sample_path("C:/Users/john/data/file.edf")
    assert _is_absolute_sample_path(r"\\server\share\file.edf")
    assert not _is_absolute_sample_path("data/file.edf")
    assert not _is_absolute_sample_path(".")


def test_read_sample_list_rows_resolves_relatives_but_keeps_windows_absolutes(tmp_path):
    slist = tmp_path / "study.lst"
    slist.write_text(
        "win_abs\tC:/Users/john/data/file.edf\t.\n"
        "relative\tedf/file2.edf\tannots/file2.annot\n"
    )

    rows = _read_sample_list_rows(str(slist))
    base = tmp_path.resolve()

    assert rows[0] == ["win_abs", "C:/Users/john/data/file.edf", "."]
    assert rows[1] == [
        "relative",
        os.path.normpath(str(base / "edf" / "file2.edf")),
        os.path.normpath(str(base / "annots" / "file2.annot")),
    ]


def test_sample_list_model_preserves_dot_placeholder():
    df = pd.DataFrame(
        {
            "ID": ["10822_10051"],
            "EDF": ["."],
            "Annotations": [{"pops-orig/10822_10051.annot"}],
        }
    )

    model = SListMixin.sample_list_df_to_model(df)

    assert model.data(model.index(0, 0), Qt.DisplayRole) == "10822_10051"
    assert model.data(model.index(0, 1), Qt.DisplayRole) == "."
