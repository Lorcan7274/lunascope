"""Tests for :mod:`lunascope.components.harmonizer_funcs`.

The harmonizer module is intentionally Qt-free; we exercise its analytic
functions directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lunascope.components import harmonizer_funcs as hf


# ---------------------------------------------------------------------------
# build_presence
# ---------------------------------------------------------------------------


def test_build_presence_basic_matrix(tiny_channels_df):
    names, ids, mat = hf.build_presence(tiny_channels_df, name_col="CH")
    assert "EEG_F3" in names
    assert set(ids) == {"S1", "S2", "S3"}
    # EEG_F3 is in all three subjects
    row = mat[names.index("EEG_F3")]
    assert row.sum() == 3
    # ECG only in S1
    ecg_row = mat[names.index("ECG")]
    assert ecg_row.sum() == 1


def test_build_presence_remap_collapses_aliases(tiny_annots_df):
    names, _ids, mat = hf.build_presence(
        tiny_annots_df,
        name_col="ANNOT",
        remap={"Arousal": "arousal"},
    )
    assert "arousal" in names
    assert "Arousal" not in names
    # arousal is now in S1 and S2
    row = mat[names.index("arousal")]
    assert int(row.sum()) == 2


def test_build_presence_ignore_drops_entries(tiny_channels_df):
    names, _ids, _mat = hf.build_presence(
        tiny_channels_df, name_col="CH", ignore={"ECG"}
    )
    assert "ECG" not in names


def test_build_presence_empty_returns_zero_matrix():
    names, ids, mat = hf.build_presence(
        pd.DataFrame(columns=["ID", "CH"]), name_col="CH"
    )
    assert names == [] and ids == []
    assert mat.shape == (0, 0)


def test_build_presence_preserves_id_order():
    df = pd.DataFrame(
        [
            {"ID": "z", "CH": "EEG"},
            {"ID": "a", "CH": "EEG"},
            {"ID": "m", "CH": "EEG"},
        ]
    )
    _names, ids, _mat = hf.build_presence(
        df, name_col="CH", ordered_ids=["z", "a", "m"]
    )
    assert ids == ["z", "a", "m"]


# ---------------------------------------------------------------------------
# channel_summary / annot_summary
# ---------------------------------------------------------------------------


def test_channel_summary_counts_subjects(tiny_channels_df):
    out = hf.channel_summary(tiny_channels_df)
    row = out[out["CH"] == "EEG_F3"].iloc[0]
    assert row["N"] == 3
    # Multiple SR values for EEG_F3 → should be marked with a star
    assert "*" in str(row["SR"])


def test_channel_summary_split_by_sr_separates_rows(tiny_channels_df):
    out = hf.channel_summary(tiny_channels_df, split_by_sr=True)
    f3_rows = out[out["CH"] == "EEG_F3"]
    assert len(f3_rows) == 2  # 200 + 256 Hz variants


def test_annot_summary_counts(tiny_annots_df):
    out = hf.annot_summary(tiny_annots_df)
    # Without remap: lowercase "arousal" appears for S1 and S2; "Arousal"
    # only for S2; "spindle" for S1 and S3.
    arousal = out[out["ANNOT"] == "arousal"]
    Arousal = out[out["ANNOT"] == "Arousal"]
    spindle = out[out["ANNOT"] == "spindle"]
    assert int(arousal["N"].iloc[0]) == 2
    assert int(Arousal["N"].iloc[0]) == 1
    assert int(spindle["N"].iloc[0]) == 2


def test_annot_summary_remap_combines_classes(tiny_annots_df):
    out = hf.annot_summary(tiny_annots_df, remap={"Arousal": "arousal"})
    arousal = out[out["ANNOT"] == "arousal"]
    assert int(arousal["N"].iloc[0]) == 2


# ---------------------------------------------------------------------------
# normalize_domain / infer_channel_domain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("EEG", "EEG"),
        ("ekg", "ECG"),
        ("EOG", "EOG"),
        ("RESP", "RESP"),
        ("oximetry", "SpO2"),
        ("", ""),
        ("aux", "OTHER"),
    ],
)
def test_normalize_domain_canonical_forms(value, expected):
    assert hf.normalize_domain(value) == expected


def test_normalize_domain_keyword_inference():
    assert hf.normalize_domain("Heart Rate") == "ECG"
    assert hf.normalize_domain("Chin EMG") == "EMG"
    assert hf.normalize_domain("Pleth") == "SpO2"
    assert hf.normalize_domain("Nasal cannula") == "RESP"


def test_normalize_domain_eeg_keyword_inference():
    assert hf.normalize_domain("Fp1-A2") == "EEG"
    assert hf.normalize_domain("Cz") == "EEG"


def test_infer_channel_domain_prefers_typed_value():
    """A non-empty TYPE entry should override name-based inference."""
    out = hf.infer_channel_domain("Fp1", "ecg")
    assert out == "ECG"


def test_infer_channel_domain_falls_back_to_name():
    out = hf.infer_channel_domain("Fp1", "")
    assert out == "EEG"


# ---------------------------------------------------------------------------
# domain_assignments / coverage_stats
# ---------------------------------------------------------------------------


def test_domain_assignments_uses_user_overrides(tiny_channels_df):
    out = hf.domain_assignments(
        tiny_channels_df,
        user_domains={"EEG_F3": "EMG"},
    )
    row = out[out["CH"] == "EEG_F3"].iloc[0]
    assert row["Domain"] == "EMG"
    assert row["Source"] == "user"


def test_domain_assignments_uses_types(tiny_channels_df):
    types_df = pd.DataFrame(
        [
            {"CH": "ECG", "TYPE": "ECG"},
            {"CH": "EOG_L", "TYPE": "EOG"},
        ]
    )
    out = hf.domain_assignments(tiny_channels_df, types_df=types_df)
    ecg_row = out[out["CH"] == "ECG"].iloc[0]
    assert ecg_row["Domain"] == "ECG"
    assert ecg_row["Source"] == "types"


def test_coverage_stats_percent_of_canonical(tiny_channels_df):
    canonical = ["EEG_F3", "EEG_C3", "EEG_O1", "ECG"]
    out = hf.coverage_stats(tiny_channels_df, canonical=canonical)
    assert set(out.columns) == {"ID", "N_present", "N_canonical", "Pct"}
    s1 = out[out["ID"] == "S1"].iloc[0]
    # S1 has F3, C3, ECG  → 3/4 = 75%
    assert s1["N_present"] == 3
    assert s1["Pct"] == pytest.approx(75.0, abs=0.1)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_round_trip(tmp_path):
    scan = hf.ScanResult(
        channels_df=pd.DataFrame([{"ID": "S1", "CH": "EEG", "SR": "256", "TRANS": "", "PDIM": ""}]),
        annots_df=pd.DataFrame(columns=["ID", "ANNOT"]),
        types_df=pd.DataFrame(columns=["CH", "TYPE"]),
        ids=["S1"],
        n_total=1,
        scan_ts="2026-01-01T00:00:00",
    )
    cache_path = tmp_path / "scan.cache"
    hf.save_cache(str(cache_path), scan)
    loaded = hf.load_cache(str(cache_path))
    pd.testing.assert_frame_equal(loaded.channels_df, scan.channels_df)
    assert loaded.ids == ["S1"]
    assert loaded.scan_ts == scan.scan_ts


def test_load_cache_rejects_unknown_magic(tmp_path):
    import pickle

    bad = tmp_path / "bad.cache"
    with open(bad, "wb") as f:
        pickle.dump({"magic": "not-our-magic", "scan": None}, f)
    with pytest.raises(ValueError, match="Harmonizer cache"):
        hf.load_cache(str(bad))


# ---------------------------------------------------------------------------
# write_param_file
# ---------------------------------------------------------------------------


def test_write_param_file_emits_aliases_and_drops(tmp_path):
    path = tmp_path / "params.txt"
    hf.write_param_file(
        str(path),
        remap_ch={"EEG 1": "EEG", "EEG_1": "EEG"},
        ignore_ch={"BAD CH"},
        remap_ann={"Arousal_RESP": "arousal"},
        ignore_ann=set(),
        sig_names=["EEG", "ECG"],
        annot_names=["arousal"],
    )
    text = path.read_text(encoding="utf-8")
    assert "alias\t" in text
    assert "EEG" in text
    assert "drop\t" in text
    assert "\"BAD CH\"" in text  # quoted because of the space
    assert "remap\tarousal|Arousal_RESP" in text
    assert "annot\tarousal" in text


# ---------------------------------------------------------------------------
# Rare-cooccurrence pairs
# ---------------------------------------------------------------------------


def test_rare_cooccurrence_pairs_finds_disjoint_channels():
    df = pd.DataFrame(
        [
            {"ID": "S1", "CH": "EEG_A", "SR": "256", "TRANS": "AC", "PDIM": "uV"},
            {"ID": "S2", "CH": "EEG_A", "SR": "256", "TRANS": "AC", "PDIM": "uV"},
            {"ID": "S3", "CH": "EEG_B", "SR": "256", "TRANS": "AC", "PDIM": "uV"},
            {"ID": "S4", "CH": "EEG_B", "SR": "256", "TRANS": "AC", "PDIM": "uV"},
        ]
    )
    out = hf.rare_cooccurrence_pairs(df, min_subjects=2)
    pair = out[(out["CH_A"] == "EEG_A") & (out["CH_B"] == "EEG_B")]
    assert len(pair) == 1
    row = pair.iloc[0]
    assert row["Both"] == 0
    assert row["Union"] == 4


def test_rare_cooccurrence_excludes_cross_domain_pairs():
    df = pd.DataFrame(
        [
            {"ID": "S1", "CH": "EEG_A", "SR": "256", "TRANS": "AC", "PDIM": "uV"},
            {"ID": "S2", "CH": "EEG_A", "SR": "256", "TRANS": "AC", "PDIM": "uV"},
            {"ID": "S3", "CH": "ECG",   "SR": "128", "TRANS": "AC", "PDIM": "mV"},
            {"ID": "S4", "CH": "ECG",   "SR": "128", "TRANS": "AC", "PDIM": "mV"},
        ]
    )
    types = pd.DataFrame(
        [
            {"CH": "EEG_A", "TYPE": "EEG"},
            {"CH": "ECG", "TYPE": "ECG"},
        ]
    )
    out = hf.rare_cooccurrence_pairs(df, types_df=types, min_subjects=2)
    # Different domains — should be filtered out entirely.
    assert out.empty or not (
        ((out["CH_A"] == "EEG_A") & (out["CH_B"] == "ECG"))
        | ((out["CH_A"] == "ECG") & (out["CH_B"] == "EEG_A"))
    ).any()
