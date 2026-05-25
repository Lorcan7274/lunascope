from __future__ import annotations

import numpy as np
import pandas as pd


def test_rank_seed_correlations_uses_pairwise_complete_obs():
    from lunascope.components.explorer_gpa import _rank_seed_correlations

    df = pd.DataFrame(
        {
            "seed": [1.0, 2.0, 3.0, 4.0, None],
            "a": [2.0, 4.0, 6.0, 8.0, 10.0],
            "b": [1.0, None, 3.0, 2.0, 5.0],
            "c": [9.0, 9.0, 9.0, 9.0, 9.0],
        }
    )
    meta = pd.DataFrame(
        {
            "VAR": ["seed", "a", "b", "c"],
            "BASE": ["seed", "A", "B", "C"],
            "GRP": [".", "g1", "g2", "g3"],
            "NI": ["4", "5", "4", "5"],
        }
    )

    out = _rank_seed_correlations(df, "seed", ["a", "b", "c"], meta_df=meta)

    assert out["TARGET"].tolist() == ["a", "b"]
    assert out.iloc[0]["N"] == 4
    assert out.iloc[0]["R"] == 1.0
    assert out.iloc[0]["TARGET_BASE"] == "A"
    assert out.iloc[1]["N"] == 3
    assert out.iloc[1]["TARGET_GRP"] == "g2"


def test_safe_corrcoef_returns_nan_for_constant_inputs():
    from lunascope.components.explorer_gpa import _safe_corrcoef

    assert np.isnan(_safe_corrcoef([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]))
    assert np.isnan(_safe_corrcoef([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]))
    assert _safe_corrcoef([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == 1.0


def test_with_dump_qc_disabled_forces_qc_false():
    from lunascope.components.explorer_gpa import _with_dump_qc_disabled

    assert _with_dump_qc_disabled({}) == {"qc": "F"}
    assert _with_dump_qc_disabled({"subset": "SEX=1"}) == {"subset": "SEX=1", "qc": "F"}
    assert _with_dump_qc_disabled({"qc": "T", "knn": "5"}) == {"qc": "F", "knn": "5"}


def test_fit_variable_pca_default_complete_columns_only():
    from lunascope.components.explorer_gpa import _fit_variable_pca

    df = pd.DataFrame(
        {
            "v1": [1.0, 2.0, 3.0, 4.0],
            "v2": [2.0, 3.0, 4.0, 5.0],
            "v3": [1.0, None, 3.0, 4.0],
        }
    )
    meta = pd.DataFrame({"VAR": ["v1", "v2", "v3"], "BASE": ["V1", "V2", "V3"], "GRP": [".", ".", "."]})

    result = _fit_variable_pca(df, meta_df=meta, min_col_prop=1.0, row_mode="complete", standardize=True)

    assert result["error"] == ""
    assert result["n_rows_used"] == 4
    assert result["n_cols_used"] == 2
    assert result["low_obs_cols"] == ["v3"]
    assert set(result["loadings"]["VAR"]) == {"v1", "v2"}


def test_fit_variable_pca_can_keep_block_missing_columns_with_imputation():
    from lunascope.components.explorer_gpa import _fit_variable_pca

    df = pd.DataFrame(
        {
            "v1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "v2": [5.0, 4.0, 3.0, 2.0, 1.0],
            "block": [None, None, 10.0, 11.0, 12.0],
        }
    )

    result = _fit_variable_pca(df, min_col_prop=0.5, row_mode="median", standardize=True)

    assert result["error"] == ""
    assert result["n_rows_used"] == 5
    assert result["n_cols_used"] == 3
    assert "block" in set(result["loadings"]["VAR"])
    assert len(result["explained_ratio"]) >= 2


def test_build_repeated_matrix_creates_unique_row_ids_and_applies_filters():
    from lunascope.components.explorer_gpa import _build_repeated_matrix

    df = pd.DataFrame(
        {
            "ID": ["S1", "S1", "S1", "S2"],
            "N": [1, 1, 2, 1],
            "CH": ["Cz", "Pz", "Cz", "Cz"],
            "STG": ["N2", "N2", "N3", "N2"],
            "AMP": [10.0, 11.0, 12.0, 13.0],
            "DUR": [0.5, 0.6, 0.7, 0.8],
        }
    )

    out = _build_repeated_matrix(
        df,
        subject_id_col="ID",
        obs_key_cols=["N", "CH", "STG"],
        metric_cols=["AMP", "DUR"],
        meta_cols=["CH", "STG"],
        filters={"STG": ["N2"]},
    )

    matrix = out["matrix"]
    assert out["row_count"] == 3
    assert out["subject_count"] == 2
    assert matrix["RID"].tolist() == ["S1|1|Cz|N2", "S1|1|Pz|N2", "S2|1|Cz|N2"]
    assert set(matrix["CH"]) == {"Cz", "Pz"}
    assert set(out["manifest"]["VAR"]) == {"AMP", "DUR"}


def test_build_repeated_matrix_rejects_duplicate_keys():
    from lunascope.components.explorer_gpa import _build_repeated_matrix

    df = pd.DataFrame(
        {
            "ID": ["S1", "S1"],
            "N": [1, 1],
            "CH": ["Cz", "Cz"],
            "AMP": [10.0, 11.0],
        }
    )

    try:
        _build_repeated_matrix(
            df,
            subject_id_col="ID",
            obs_key_cols=["N", "CH"],
            metric_cols=["AMP"],
            meta_cols=["CH"],
        )
    except ValueError as exc:
        assert "does not uniquely identify rows" in str(exc)
    else:
        raise AssertionError("Expected duplicate repeated-observation keys to fail")


def test_fit_observation_pca_returns_row_scores_and_metadata():
    from lunascope.components.explorer_gpa import _fit_observation_pca

    df = pd.DataFrame(
        {
            "RID": ["r1", "r2", "r3"],
            "ID": ["S1", "S1", "S2"],
            "CH": ["Cz", "Pz", "Cz"],
            "v1": [1.0, 2.0, 3.0],
            "v2": [2.0, 1.5, 0.5],
            "v3": [0.2, 0.3, 0.9],
        }
    )

    result = _fit_observation_pca(
        df,
        ["v1", "v2", "v3"],
        min_col_prop=1.0,
        row_mode="complete",
        standardize=True,
    )

    assert result["error"] == ""
    assert result["n_rows_used"] == 3
    assert "RID" in result["scores"].columns
    assert "CH" in result["scores"].columns
    assert "PC1" in result["scores"].columns
