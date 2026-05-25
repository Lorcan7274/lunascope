from __future__ import annotations

import numpy as np
import pandas as pd

from lunascope.components.explorer_gpa import (
    _fit_joint_model_frame,
    _normalize_gpa_role_overlap,
)


def test_fit_joint_model_frame_linear_includes_y_and_z_terms():
    df = pd.DataFrame(
        {
            "X": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "Y1": [0.2, 1.1, 1.8, 2.7, 3.3, 4.1],
            "Z1": [10, 11, 9, 13, 12, 14],
        }
    )

    result = _fit_joint_model_frame(df, "X", ["Y1"], ["Z1"])

    assert "error" not in result
    assert result["model_type"] == "linear"
    assert result["n_complete"] == 6
    assert list(result["table"]["TERM"]) == ["Intercept", "Y1", "Z1"]
    assert np.isfinite(result["table"]["P"]).all()


def test_fit_joint_model_frame_binary_uses_logistic():
    df = pd.DataFrame(
        {
            "X": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "Y1": [0.1, 0.4, 0.7, 0.3, 0.2, 0.8, 0.5, 0.6, 0.9, 0.45],
            "Z1": [0, 0, 1, 1, 0, 1, 1, 0, 1, 0],
        }
    )

    result = _fit_joint_model_frame(df, "X", ["Y1"], ["Z1"])

    assert "error" not in result
    assert result["model_type"] == "logistic"
    assert result["binary_labels"] == (0.0, 1.0)
    assert set(result["table"]["TERM"]) == {"Intercept", "Y1", "Z1"}
    assert np.isfinite(result["table"]["P"]).all()


def test_fit_joint_model_frame_binary_tolerates_normalized_levels():
    df = pd.DataFrame(
        {
            "X": [-0.999999999, 1.000000001, -1.0, 1.0, -1.0, 1.0, -0.999999999, 1.000000001],
            "Y1": [0.1, 0.4, 0.7, 0.3, 0.2, 0.8, 0.5, 0.6],
            "Z1": [0, 0, 1, 1, 0, 1, 1, 0],
        }
    )

    result = _fit_joint_model_frame(df, "X", ["Y1"], ["Z1"])

    assert "error" not in result
    assert result["model_type"] == "logistic"
    assert result["binary_labels"] == (-1.0, 1.000000001)
    assert set(result["table"]["TERM"]) == {"Intercept", "Y1", "Z1"}
    assert np.isfinite(result["table"]["P"]).all()


def test_fit_joint_model_frame_drops_collinear_terms():
    df = pd.DataFrame(
        {
            "X": [2, 4, 6, 8, 10, 12],
            "Y1": [1, 2, 3, 4, 5, 6],
            "Y2": [1, 2, 3, 4, 5, 6],
            "Z1": [0, 1, 0, 1, 0, 1],
        }
    )

    result = _fit_joint_model_frame(df, "X", ["Y1", "Y2"], ["Z1"])

    assert "error" not in result
    assert "Y2" not in set(result["table"]["TERM"])
    assert any("Dropped Y2 (collinear)" in note for note in result["warnings"])


def test_normalize_gpa_role_overlap_drops_y_overlap_with_x_and_z():
    result = _normalize_gpa_role_overlap(
        ["X1", "X2"],
        ["Y1", "X1", "Z1", "Y2"],
        ["Z1", "Z2"],
    )

    assert result["ok"] is True
    assert result["x_vars"] == ["X1", "X2"]
    assert result["y_vars"] == ["Y1", "Y2"]
    assert result["z_vars"] == ["Z1", "Z2"]
    assert result["dropped_from_y"] == ["X1", "Z1"]
    assert any("also selected in X" in line for line in result["warning_lines"])
    assert any("also selected in Z" in line for line in result["warning_lines"])


def test_normalize_gpa_role_overlap_rejects_xz_overlap():
    result = _normalize_gpa_role_overlap(
        ["X1", "Z1"],
        ["Y1"],
        ["Z1", "Z2"],
    )

    assert result["ok"] is False
    assert result["error_lines"] == ["X and Z: Z1"]
