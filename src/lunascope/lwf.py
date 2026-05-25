from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
import pandas as pd


_I32 = struct.Struct("<i")
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_F32 = struct.Struct("<f")
_F64 = struct.Struct("<d")

_MAGIC = b"LWF1"
_VERSION = 3


@dataclass(slots=True)
class LWFChannel:
    label: str
    unit: str
    sr: float
    sample_step_tp: int


@dataclass(slots=True)
class LWFBlock:
    n: int
    data_start_sec: float
    data_stop_sec: float
    feature_qc: int
    features: dict[str, float]
    values: np.ndarray


@dataclass(slots=True)
class LWFWave:
    annot: str
    instance: str
    annot_ch: str
    meta: str
    annot_start_sec: float
    annot_stop_sec: float
    anchor_sec: float
    wave_start_sec: float
    wave_stop_sec: float
    blocks: dict[str, LWFBlock]


@dataclass(slots=True)
class LWFShard:
    path: str
    id: str
    edf: str
    outfile: str
    start_date: str
    start_time: str
    tag: str
    align: str
    def_annots: list[str]
    feature_names: list[str]
    channels: list[LWFChannel]
    waves: list[LWFWave]


@dataclass(slots=True)
class LWFDataset:
    root: str
    shards: list[LWFShard]
    files: pd.DataFrame
    channels: pd.DataFrame
    waves: pd.DataFrame


def _read_exact(fh, n: int) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise ValueError("Unexpected end of .lwf file")
    return data


def _read_i32(fh) -> int:
    return _I32.unpack(_read_exact(fh, _I32.size))[0]


def _read_u32(fh) -> int:
    return _U32.unpack(_read_exact(fh, _U32.size))[0]


def _read_u64(fh) -> int:
    return _U64.unpack(_read_exact(fh, _U64.size))[0]


def _read_f64(fh) -> float:
    return _F64.unpack(_read_exact(fh, _F64.size))[0]


def _read_string(fh) -> str:
    n = _read_u32(fh)
    if n == 0:
        return ""
    return _read_exact(fh, n).decode("utf-8", errors="replace")


def _load_lwf_file(path: str | Path) -> LWFShard:
    path = str(path)
    with open(path, "rb") as fh:
        if _read_exact(fh, 4) != _MAGIC:
            raise ValueError(f"Invalid .lwf magic: {path}")
        version = _read_i32(fh)
        if version != _VERSION:
            raise ValueError(
                f"{path} uses .lwf version {version}, but Lunascope expects version {_VERSION}. "
                "Please regenerate the waveform shards with the current Luna WAVEFORMS writer."
            )

        id_str = _read_string(fh)
        edf = _read_string(fh)
        outfile = _read_string(fh)
        start_date = _read_string(fh)
        start_time = _read_string(fh)
        tag = _read_string(fh)
        align = _read_string(fh)

        n_annots = _read_i32(fh)
        def_annots = [_read_string(fh) for _ in range(n_annots)]

        n_channels = _read_i32(fh)
        channels = [
            LWFChannel(
                label=_read_string(fh),
                unit=_read_string(fh),
                sample_step_tp=_read_u64(fh),
                sr=_read_f64(fh),
            )
            for _ in range(n_channels)
        ]
        n_features = _read_i32(fh)
        feature_names = [_read_string(fh) for _ in range(n_features)]

        n_waves = _read_i32(fh)
        index_rows = []
        for _ in range(n_waves):
            annot = _read_string(fh)
            instance = _read_string(fh)
            annot_ch = _read_string(fh)
            annot_start_sec = _read_f64(fh)
            annot_stop_sec = _read_f64(fh)
            anchor_sec = _read_f64(fh)
            wave_start_sec = _read_f64(fh)
            wave_stop_sec = _read_f64(fh)
            payload_offset = _read_u64(fh)
            n_blocks = _read_i32(fh)
            blocks = []
            for _ in range(n_blocks):
                blocks.append(
                    (
                        _read_i32(fh),
                        _read_f64(fh),
                        _read_f64(fh),
                    )
                )
            index_rows.append(
                (
                    annot,
                    instance,
                    annot_ch,
                    annot_start_sec,
                    annot_stop_sec,
                    anchor_sec,
                    wave_start_sec,
                    wave_stop_sec,
                    payload_offset,
                    blocks,
                )
            )

        waves: list[LWFWave] = []
        for annot, instance, annot_ch, annot_start_sec, annot_stop_sec, anchor_sec, wave_start_sec, wave_stop_sec, payload_offset, indexed_blocks in index_rows:
            fh.seek(payload_offset)
            payload_annot = _read_string(fh)
            payload_instance = _read_string(fh)
            payload_annot_ch = _read_string(fh)
            meta = _read_string(fh)
            payload_annot_start_sec = _read_f64(fh)
            payload_annot_stop_sec = _read_f64(fh)
            payload_anchor_sec = _read_f64(fh)
            payload_wave_start_sec = _read_f64(fh)
            payload_wave_stop_sec = _read_f64(fh)
            n_blocks = _read_i32(fh)

            if (payload_annot, payload_instance, payload_annot_ch) != (annot, instance, annot_ch):
                raise ValueError(f"Index/payload mismatch in {path}")
            if n_blocks != len(indexed_blocks) or n_blocks != len(channels):
                raise ValueError(f"Channel/block count mismatch in {path}")

            block_map: dict[str, LWFBlock] = {}
            for ch, (idx_n, idx_start, idx_stop) in zip(channels, indexed_blocks):
                n = _read_i32(fh)
                data_start_sec = _read_f64(fh)
                data_stop_sec = _read_f64(fh)
                if version >= 2 and feature_names:
                    feature_qc = _read_i32(fh)
                    feature_values = [_read_f64(fh) for _ in feature_names]
                    features = dict(zip(feature_names, feature_values))
                else:
                    feature_qc = -1
                    features = {}
                values = np.frombuffer(_read_exact(fh, n * _F32.size), dtype="<f4").copy()
                if n != idx_n:
                    raise ValueError(f"Indexed sample count mismatch in {path}")
                block_map[ch.label] = LWFBlock(
                    n=n,
                    data_start_sec=data_start_sec if np.isfinite(data_start_sec) else idx_start,
                    data_stop_sec=data_stop_sec if np.isfinite(data_stop_sec) else idx_stop,
                    feature_qc=feature_qc,
                    features=features,
                    values=values,
                )

            waves.append(
                LWFWave(
                    annot=payload_annot,
                    instance=payload_instance,
                    annot_ch=payload_annot_ch,
                    meta=meta,
                    annot_start_sec=payload_annot_start_sec,
                    annot_stop_sec=payload_annot_stop_sec,
                    anchor_sec=payload_anchor_sec,
                    wave_start_sec=payload_wave_start_sec,
                    wave_stop_sec=payload_wave_stop_sec,
                    blocks=block_map,
                )
            )

    return LWFShard(
        path=path,
        id=id_str,
        edf=edf,
        outfile=outfile,
        start_date=start_date,
        start_time=start_time,
        tag=tag,
        align=align,
        def_annots=def_annots,
        feature_names=feature_names,
        channels=channels,
        waves=waves,
    )


def load_lwf_directory(directory: str | Path, recursive: bool = False) -> LWFDataset:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    files = sorted(root.rglob("*.lwf") if recursive else root.glob("*.lwf"))
    if not files:
        raise ValueError(f"No .lwf files found in {root}")

    shards = [_load_lwf_file(path) for path in files]

    file_rows = []
    channel_rows = []
    wave_rows = []
    for shard in shards:
        file_rows.append(
            {
                "FILE": shard.path,
                "LWF_ID": shard.id,
                "EDF": shard.edf,
                "TAG": shard.tag or ".",
                "ALIGN": shard.align,
                "N_FEATURES": len(shard.feature_names),
                "FEATURES": ",".join(shard.feature_names) if shard.feature_names else ".",
                "START_DATE": shard.start_date,
                "START_TIME": shard.start_time,
                "N_ANNOTS": len(shard.def_annots),
                "ANNOTS": ",".join(shard.def_annots),
                "N_WAVES": len(shard.waves),
                "N_CH": len(shard.channels),
            }
        )
        for ch in shard.channels:
            counts = [wave.blocks[ch.label].n for wave in shard.waves if ch.label in wave.blocks]
            channel_rows.append(
                {
                    "FILE": shard.path,
                    "CH": ch.label,
                    "SR": ch.sr,
                    "SAMPLE_STEP_TP": ch.sample_step_tp,
                    "UNIT": ch.unit,
                    "MIN_SAMPLES": min(counts) if counts else 0,
                    "MAX_SAMPLES": max(counts) if counts else 0,
                }
            )
        for wave_idx, wave in enumerate(shard.waves):
            row = {
                "FILE": shard.path,
                "LWF_ID": shard.id,
                "TAG": shard.tag or ".",
                "ALIGN": shard.align,
                "WAVE_IDX": wave_idx,
                "ANNOT": wave.annot,
                "INSTANCE": wave.instance,
                "ANNOT_CH": wave.annot_ch,
                "META": wave.meta,
                "ANNOT_START_SEC": wave.annot_start_sec,
                "ANNOT_STOP_SEC": wave.annot_stop_sec,
                "ANCHOR_SEC": wave.anchor_sec,
                "WAVE_START_SEC": wave.wave_start_sec,
                "WAVE_STOP_SEC": wave.wave_stop_sec,
                "CHANNELS": tuple(wave.blocks.keys()),
                "BLOCKS": wave.blocks,
                "FEATURE_NAMES": tuple(shard.feature_names),
            }
            wave_rows.append(row)

    return LWFDataset(
        root=str(root),
        shards=shards,
        files=pd.DataFrame(file_rows),
        channels=pd.DataFrame(channel_rows),
        waves=pd.DataFrame(wave_rows),
    )


def summarize_lwf_dataset(dataset: LWFDataset) -> dict[str, object]:
    files_df = dataset.files
    waves_df = dataset.waves
    channels_df = dataset.channels
    feature_names = []
    if not files_df.empty and "FEATURES" in files_df.columns:
        seen = set()
        for raw in files_df["FEATURES"].dropna().astype(str).tolist():
            if raw == ".":
                continue
            for name in raw.split(","):
                name = name.strip()
                if name and name not in seen:
                    seen.add(name)
                    feature_names.append(name)
    return {
        "root": dataset.root,
        "n_files": int(len(dataset.shards)),
        "n_waves": int(len(waves_df)),
        "n_ids": int(files_df["LWF_ID"].nunique()) if not files_df.empty else 0,
        "ids": sorted(files_df["LWF_ID"].dropna().astype(str).unique().tolist()) if not files_df.empty else [],
        "tags": sorted({str(x) for x in files_df["TAG"].dropna().tolist() if str(x) != "."}),
        "annots": sorted(waves_df["ANNOT"].dropna().astype(str).unique().tolist()) if not waves_df.empty else [],
        "channels": sorted(channels_df["CH"].dropna().astype(str).unique().tolist()) if not channels_df.empty else [],
        "features": feature_names,
    }


def _format_preview(values: list[str], limit: int = 8) -> str:
    if not values:
        return ""
    preview = values[:limit]
    suffix = "" if len(values) <= limit else f" +{len(values) - limit} more"
    return ", ".join(preview) + suffix


def format_lwf_summary(dataset: LWFDataset) -> str:
    s = summarize_lwf_dataset(dataset)
    lines = [
        f"Loaded .lwf dataset from {s['root']}",
        f"Files: {s['n_files']}",
        f"Waveforms: {s['n_waves']}",
        f"Individuals: {s['n_ids']}",
    ]
    if s["channels"]:
        lines.append("Channels: " + _format_preview(s["channels"]))
    if s["annots"]:
        lines.append("Annotations: " + _format_preview(s["annots"]))
    if s["tags"]:
        lines.append("Tags: " + _format_preview(s["tags"]))
    if s["features"]:
        lines.append("Features: " + _format_preview(s["features"]))
    if s["ids"]:
        lines.append("IDs: " + _format_preview(s["ids"]))
    return "\n".join(lines)


def format_lwf_summary_compact(dataset: LWFDataset) -> str:
    s = summarize_lwf_dataset(dataset)
    left = (
        f"{s['n_files']} files  |  {s['n_waves']} waves  |  {s['n_ids']} IDs"
    )
    parts = []
    if s["channels"]:
        parts.append("CH: " + _format_preview(s["channels"], limit=4))
    if s["annots"]:
        parts.append("ANNOT: " + _format_preview(s["annots"], limit=4))
    if s["tags"]:
        parts.append("TAG: " + _format_preview(s["tags"], limit=4))
    if s["features"]:
        parts.append("FTR: " + _format_preview(s["features"], limit=4))
    right = "  |  ".join(parts)
    return left if not right else f"{left}\n{right}"
