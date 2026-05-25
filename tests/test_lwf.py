"""Tests for :mod:`lunascope.lwf`.

The reader is exercised against synthetic LWF v3 binary files constructed
in-test. These tests verify the on-disk format Lunascope expects from
Luna's WAVEFORMS writer.
"""

from __future__ import annotations

import struct
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lunascope import lwf as lwf_mod


# ---------------------------------------------------------------------------
# Helpers to build a synthetic .lwf file
# ---------------------------------------------------------------------------


def _pack_string(s: str) -> bytes:
    data = s.encode("utf-8")
    return struct.pack("<I", len(data)) + data


def _build_lwf_v3(
    *,
    id_str: str = "subj01",
    edf: str = "subj01.edf",
    outfile: str = "subj01.out",
    start_date: str = "01.01.20",
    start_time: str = "22.00.00",
    tag: str = "tag",
    align: str = "anchor",
    def_annots=("spindle",),
    channels=(("C3", "uV", 100, 10_000_000), ("C4", "uV", 100, 10_000_000)),
    feature_names=("amp", "dur"),
    waves=None,
) -> bytes:
    """Return bytes for a single-shard LWF v3 file with N waves and channels."""
    if waves is None:
        waves = [
            {
                "annot": "spindle",
                "instance": "i0",
                "annot_ch": "C3",
                "meta": "meta0",
                "annot_start_sec": 1.0,
                "annot_stop_sec": 2.0,
                "anchor_sec": 1.5,
                "wave_start_sec": 0.5,
                "wave_stop_sec": 2.5,
                "blocks": {  # per-channel sample arrays
                    "C3": np.array([0.0, 1.0, 2.0, 3.0], dtype="<f4"),
                    "C4": np.array([4.0, 5.0, 6.0, 7.0], dtype="<f4"),
                },
                "features": {"amp": 0.7, "dur": 1.0},
                "feature_qc": 0,
            }
        ]

    n_waves = len(waves)
    n_channels = len(channels)
    n_features = len(feature_names)

    # ---- header ---------------------------------------------------------
    header = BytesIO()
    header.write(b"LWF1")
    header.write(struct.pack("<i", 3))  # version
    for s in (id_str, edf, outfile, start_date, start_time, tag, align):
        header.write(_pack_string(s))
    header.write(struct.pack("<i", len(def_annots)))
    for a in def_annots:
        header.write(_pack_string(a))
    header.write(struct.pack("<i", n_channels))
    for label, unit, sr, sample_step_tp in channels:
        header.write(_pack_string(label))
        header.write(_pack_string(unit))
        header.write(struct.pack("<Q", int(sample_step_tp)))
        header.write(struct.pack("<d", float(sr)))
    header.write(struct.pack("<i", n_features))
    for fn in feature_names:
        header.write(_pack_string(fn))
    header.write(struct.pack("<i", n_waves))

    # ---- compute index entry size, then payload offsets ----------------
    # Each index entry size depends only on its strings + fixed fields.
    header_data = header.getvalue()
    header_size = len(header_data)
    # Build each index entry, but with placeholder payload_offset (filled
    # after we know payload_offsets).
    index_buffers = []
    for w in waves:
        ib = BytesIO()
        ib.write(_pack_string(w["annot"]))
        ib.write(_pack_string(w["instance"]))
        ib.write(_pack_string(w["annot_ch"]))
        ib.write(struct.pack("<d", w["annot_start_sec"]))
        ib.write(struct.pack("<d", w["annot_stop_sec"]))
        ib.write(struct.pack("<d", w["anchor_sec"]))
        ib.write(struct.pack("<d", w["wave_start_sec"]))
        ib.write(struct.pack("<d", w["wave_stop_sec"]))
        ib.write(struct.pack("<Q", 0))  # placeholder payload_offset
        # n_blocks + per-block (n, data_start, data_stop)
        ib.write(struct.pack("<i", n_channels))
        for label, _unit, _sr, _sample_step_tp in channels:
            arr = w["blocks"][label]
            ib.write(struct.pack("<i", int(arr.size)))
            ib.write(struct.pack("<d", float(w["wave_start_sec"])))
            ib.write(struct.pack("<d", float(w["wave_stop_sec"])))
        index_buffers.append(ib.getvalue())

    index_size = sum(len(b) for b in index_buffers)
    payload_offsets = []
    payload_buffers = []
    next_offset = header_size + index_size
    for w in waves:
        payload_offsets.append(next_offset)
        pb = BytesIO()
        pb.write(_pack_string(w["annot"]))
        pb.write(_pack_string(w["instance"]))
        pb.write(_pack_string(w["annot_ch"]))
        pb.write(_pack_string(w["meta"]))
        pb.write(struct.pack("<d", w["annot_start_sec"]))
        pb.write(struct.pack("<d", w["annot_stop_sec"]))
        pb.write(struct.pack("<d", w["anchor_sec"]))
        pb.write(struct.pack("<d", w["wave_start_sec"]))
        pb.write(struct.pack("<d", w["wave_stop_sec"]))
        pb.write(struct.pack("<i", n_channels))
        for label, _unit, _sr, _sample_step_tp in channels:
            arr = w["blocks"][label]
            pb.write(struct.pack("<i", int(arr.size)))
            pb.write(struct.pack("<d", float(w["wave_start_sec"])))
            pb.write(struct.pack("<d", float(w["wave_stop_sec"])))
            pb.write(struct.pack("<i", int(w["feature_qc"])))
            for fn in feature_names:
                pb.write(struct.pack("<d", float(w["features"][fn])))
            pb.write(arr.astype("<f4").tobytes())
        payload_bytes = pb.getvalue()
        payload_buffers.append(payload_bytes)
        next_offset += len(payload_bytes)

    # ---- patch payload offsets into index entries ----------------------
    patched_index = []
    for i, ib in enumerate(index_buffers):
        # Locate the placeholder we wrote: it's the 8-byte u64 immediately
        # after the three packed strings + 5*8 doubles.  Easier: we know
        # the structure — recompute the offset of the placeholder.
        b = bytearray(ib)
        # Find the structure header lengths exactly: skip 3 strings + 5 f64
        cursor = 0
        for _ in range(3):
            n = struct.unpack_from("<I", b, cursor)[0]
            cursor += 4 + n
        cursor += 8 * 5  # five f64s
        struct.pack_into("<Q", b, cursor, payload_offsets[i])
        patched_index.append(bytes(b))

    return header_data + b"".join(patched_index) + b"".join(payload_buffers)


@pytest.fixture
def synthetic_lwf_dir(tmp_path: Path) -> Path:
    """Write two synthetic .lwf files into a directory and return the dir."""
    for i in range(2):
        path = tmp_path / f"subj{i}.lwf"
        data = _build_lwf_v3(id_str=f"subj{i}", edf=f"subj{i}.edf")
        path.write_bytes(data)
    return tmp_path


# ---------------------------------------------------------------------------
# Reader tests
# ---------------------------------------------------------------------------


def test_load_lwf_directory_reads_all_shards(synthetic_lwf_dir):
    ds = lwf_mod.load_lwf_directory(synthetic_lwf_dir)
    assert len(ds.shards) == 2
    assert set(ds.files["LWF_ID"]) == {"subj0", "subj1"}
    assert "EDF" in ds.files.columns
    assert ds.waves["ANNOT"].iloc[0] == "spindle"
    assert ds.channels["CH"].iloc[0] in ("C3", "C4")
    assert ds.channels["SR"].iloc[0] == pytest.approx(100.0)


def test_load_lwf_block_payload_matches_input(synthetic_lwf_dir):
    ds = lwf_mod.load_lwf_directory(synthetic_lwf_dir)
    shard = ds.shards[0]
    wave = shard.waves[0]
    np.testing.assert_array_equal(
        wave.blocks["C3"].values, np.array([0.0, 1.0, 2.0, 3.0], dtype="<f4")
    )
    np.testing.assert_array_equal(
        wave.blocks["C4"].values, np.array([4.0, 5.0, 6.0, 7.0], dtype="<f4")
    )
    # Feature values are preserved
    assert wave.blocks["C3"].features["amp"] == pytest.approx(0.7)
    assert wave.blocks["C3"].features["dur"] == pytest.approx(1.0)


def test_load_lwf_preserves_fractional_sr_and_step(tmp_path):
    data = _build_lwf_v3(
        id_str="subj_frac",
        channels=(("PP_N1", "prob", 0.2, 5_000_000_000),),
        waves=[
            {
                "annot": "hd_3_neither",
                "instance": "i0",
                "annot_ch": "",
                "meta": "meta0",
                "annot_start_sec": 10.0,
                "annot_stop_sec": 20.0,
                "anchor_sec": 15.0,
                "wave_start_sec": 0.0,
                "wave_stop_sec": 30.0,
                "blocks": {"PP_N1": np.array([0.0, 0.2, 0.4], dtype="<f4")},
                "features": {"amp": 0.7, "dur": 1.0},
                "feature_qc": 0,
            }
        ],
    )
    path = tmp_path / "frac.lwf"
    path.write_bytes(data)
    ds = lwf_mod.load_lwf_directory(tmp_path)
    shard = ds.shards[0]
    assert shard.channels[0].sr == pytest.approx(0.2)
    assert shard.channels[0].sample_step_tp == 5_000_000_000
    assert ds.channels["SR"].iloc[0] == pytest.approx(0.2)


def test_load_lwf_rejects_invalid_magic(tmp_path):
    bad = tmp_path / "bad.lwf"
    bad.write_bytes(b"NOPE" + b"\x00" * 32)
    (tmp_path / "good.lwf").write_bytes(b"LWF1" + b"\x00")  # truncated good file
    with pytest.raises(ValueError, match="Invalid .lwf magic"):
        lwf_mod.load_lwf_directory(tmp_path)


def test_load_lwf_rejects_unsupported_version(tmp_path):
    p = tmp_path / "old.lwf"
    p.write_bytes(b"LWF1" + struct.pack("<i", 99))
    with pytest.raises(ValueError, match="version"):
        lwf_mod.load_lwf_directory(tmp_path)


def test_load_lwf_directory_errors_when_empty(tmp_path):
    with pytest.raises(ValueError, match="No .lwf files"):
        lwf_mod.load_lwf_directory(tmp_path)


def test_load_lwf_directory_errors_when_not_a_dir(tmp_path):
    with pytest.raises(ValueError, match="Not a directory"):
        lwf_mod.load_lwf_directory(tmp_path / "missing")


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def test_summarize_lwf_dataset_aggregates(synthetic_lwf_dir):
    ds = lwf_mod.load_lwf_directory(synthetic_lwf_dir)
    s = lwf_mod.summarize_lwf_dataset(ds)
    assert s["n_files"] == 2
    assert s["n_waves"] == 2
    assert s["n_ids"] == 2
    assert "C3" in s["channels"] and "C4" in s["channels"]
    assert s["annots"] == ["spindle"]
    assert "amp" in s["features"]


def test_format_lwf_summary_returns_multiline(synthetic_lwf_dir):
    ds = lwf_mod.load_lwf_directory(synthetic_lwf_dir)
    text = lwf_mod.format_lwf_summary(ds)
    assert "Files: 2" in text
    assert "Waveforms: 2" in text
    # Must include channels
    assert "C3" in text


def test_format_lwf_summary_compact(synthetic_lwf_dir):
    ds = lwf_mod.load_lwf_directory(synthetic_lwf_dir)
    text = lwf_mod.format_lwf_summary_compact(ds)
    assert "2 files" in text
    assert "CH:" in text
