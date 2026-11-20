"""Results alignment and bounded recorded-signal playback (no Qt dependency)."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time
from pathlib import Path

import numpy as np
import pandas as pd


def signal_unit(header_unit, stored_unit=""):
    """Actual header units take priority over PROJECT stored-unit information."""
    unknown = {"", "unknown", "na", "n/a", "nan", "none", "."}
    for unit in (header_unit, stored_unit):
        text = str(unit).strip()
        if text.casefold() not in unknown:
            return text
    return "unknown"


def project_resources(results):
    """Read PROJECT's FILE/PATH and OUTPUT/PARCEL_ID/LABEL writer tables.

    Ambiguous cohort or merged outputs require explicit user selection.
    """
    directories, labels, units = set(), {}, {}
    for key, table in results.items():
        command = key[0] if isinstance(key, tuple) else str(key).split("_")[0]
        if command != "PROJECT" or table is None:
            continue
        if "ID" in table and table.ID.nunique() > 1:
            raise ValueError("Cohort PROJECT outputs require explicit atlas and signal mapping selection")
        if {"FILE", "PATH"} <= set(table):
            for path in table.loc[table.FILE == "metadata", "PATH"].dropna():
                directories.add(Path(str(path)).parent)
        if {"PARCEL_ID", "LABEL"} <= set(table):
            for row in table.to_dict("records"):
                pid, label = int(row["PARCEL_ID"]), str(row["LABEL"])
                if pid in labels and labels[pid] != label:
                    raise ValueError("Ambiguous PROJECT mappings; select an atlas and mapping explicitly")
                labels[pid] = label
                if "STORED_UNIT" in row:
                    unit_key = label.casefold()
                    unit = str(row["STORED_UNIT"])
                    if unit_key in units and units[unit_key] != unit:
                        raise ValueError(f"Conflicting PROJECT stored units for {label}")
                    units[unit_key] = unit
    if len(directories) > 1:
        raise ValueError("Multiple PROJECT atlas directories; select an atlas and mapping explicitly")
    return next(iter(directories)) if len(directories) == 1 else None, labels, units


def result_values(atlas, frame, variable, *, filters=None, aggregation=None,
                  prefix="", mapping=None):
    """Align scalars without implicit pooling or last-row-wins semantics."""
    if "CH" not in frame or any(c in frame for c in ("CH1", "CH2")):
        raise ValueError("Select a scalar-per-CH table; pairwise tables are unsupported")
    data = frame
    for column, value in (filters or {}).items():
        data = data[data[column].isna()] if pd.isna(value) else data[data[column] == value]
    # Even explicit aggregation cannot silently combine subjects/stages/epochs.
    for column in ("ID", "STAGE", "SS", "E", "EPOCH", "BAND"):
        if column in data and data[column].nunique(dropna=False) > 1:
            raise ValueError(f"Filter or step {column} before mapping regions")
    names = atlas.match(data["CH"].dropna().astype(str).unique(), prefix=prefix, mapping=mapping)
    data = data.assign(_signal=data["CH"].astype(str).str.casefold())
    values = np.full(len(atlas.parcels), np.nan)
    for i, name in enumerate(names):
        if name is None:
            continue
        rows = data.loc[data._signal == name.casefold(), variable]
        if len(rows) > 1 and aggregation is None:
            raise ValueError(f"Multiple rows for {name}; filter further or choose aggregation")
        if not pd.api.types.is_numeric_dtype(rows.dtype):
            raise ValueError("Select a numeric variable")
        values[i] = rows.iloc[0] if len(rows) == 1 else getattr(rows, aggregation)()
    values[~np.isfinite(values)] = np.nan
    units = frame.attrs.get("units", {})
    return values, units.get(variable, "") if isinstance(units, dict) else str(units)


class PlaybackClock:
    def __init__(self, now=time.monotonic):
        self.now = now
        self.cursor = 0.
        self.speed = 1.
        self.started = None

    def position(self):
        return self.cursor if self.started is None else self.cursor + (self.now() - self.started) * self.speed

    def pause(self):
        self.cursor = self.position()
        self.started = None

    def seek(self, seconds):
        self.cursor = max(0., seconds)
        if self.started is not None:
            self.started = self.now()

    def play(self, speed=1.):
        self.pause()
        self.speed = speed
        self.started = self.now()


class ChunkCache:
    """At most three short blocks, independent of recording length."""
    def __init__(self, capacity=3):
        self.capacity = capacity
        self.blocks = OrderedDict()

    def put(self, key, block):
        self.blocks[key] = block
        self.blocks.move_to_end(key)
        while len(self.blocks) > self.capacity:
            self.blocks.popitem(last=False)

    def get(self, key):
        block = self.blocks.get(key)
        if block is not None:
            self.blocks.move_to_end(key)
        return block


def recording_info(recording, atlas, prefix="", mapping=None):
    """Only call through the controller's serialized executor."""
    headers = recording.headers()
    names = atlas.match(headers.CH.astype(str), prefix=prefix, mapping=mapping)
    units = dict(zip(headers.CH.astype(str).str.casefold(), headers.PDIM.astype(str)))
    segment_table = recording.segments()
    if not {"START", "STOP"} <= set(segment_table):
        raise ValueError("Recording segments must provide START and STOP timestamps")
    segments = np.asarray(segment_table[["START", "STOP"]], dtype=float)
    if segments.ndim != 2 or segments.shape[1] != 2 or not len(segments):
        raise ValueError("Recording has no timestamped segments")
    if (not np.isfinite(segments).all() or np.any(segments[:, 1] <= segments[:, 0])
            or np.any(segments[1:, 0] < segments[:-1, 1])):
        raise ValueError("Recording segments must be finite, ordered and nonoverlapping")
    return names, [units.get(s.casefold(), "unknown") if s else "" for s in names], segments


def summarize(times, samples, cursors, mode, window, band):
    """Compute frames in a worker; incomplete/gapped windows remain neutral."""
    out = np.full(len(cursors), np.nan)
    if len(times) < 2:
        return out
    dt = float(np.median(np.diff(times)))
    if dt <= 0 or not np.isfinite(dt):
        return out
    right = np.searchsorted(times, cursors)
    if mode == "Signed amplitude":
        indices = np.minimum(right, len(times) - 1)
        previous = np.maximum(indices - 1, 0)
        indices = np.where(abs(times[previous] - cursors) < abs(times[indices] - cursors), previous, indices)
        valid = abs(times[indices] - cursors) <= dt * .55
        out[valid] = samples[indices[valid]]
        return out
    n = max(2, int(round(window / dt)))
    starts = np.searchsorted(times, cursors - window / 2)
    ends = starts + n
    valid = (ends <= len(times)) & (cursors - window / 2 >= times[0] - dt * .1)
    safe = np.minimum(ends - 1, len(times) - 1)
    valid &= abs(times[safe] - (cursors + window / 2 - dt)) <= dt * 1.1
    # A cumulative discontinuity/invalid-sample count avoids crossing EDF+D gaps.
    bad = np.r_[0, np.cumsum((np.diff(times) > dt * 1.5).astype(int))]
    valid &= bad[safe] == bad[np.minimum(starts, len(times) - 1)]
    locations = np.flatnonzero(valid)
    if not len(locations):
        return out
    if mode == "RMS amplitude":
        squares = np.r_[0., np.cumsum(np.where(np.isfinite(samples), samples, 0.).astype(float) ** 2)]
        invalid = np.r_[0, np.cumsum(~np.isfinite(samples))]
        good = invalid[ends[locations]] == invalid[starts[locations]]
        locations = locations[good]
        out[locations] = np.sqrt(np.maximum(0., (squares[ends[locations]] - squares[starts[locations]]) / n))
    else:
        from scipy.signal import periodogram
        lo, hi = band
        if not 0 <= lo < hi <= .5 / dt or n < 4:
            return out
        # Bound FFT working memory as well as raw/cache memory.
        for offset in range(0, len(locations), 32):
            loc = locations[offset:offset + 32]
            slices = samples[starts[loc, None] + np.arange(n)]
            freq, psd = periodogram(slices, fs=1 / dt, window="hann", axis=1)
            tolerance = np.finfo(float).eps * max(1., .5 / dt) * 32
            selected = (freq >= lo - tolerance) & (freq <= hi + tolerance)
            if np.count_nonzero(selected) >= 2:
                out[loc] = np.trapezoid(psd[:, selected], freq[selected], axis=1)
    return out


@dataclass
class SignalChunk:
    cursors: np.ndarray
    values: np.ndarray
    traces: dict


def fetch_chunk(recording, names, segments, index, mode, window, band, cancelled,
                chunk_seconds=10., fps=25):
    """Fetch ≤30 s/channel, preserve times, and cache 25 Hz scalar summaries."""
    start = index * chunk_seconds
    stop = min(start + chunk_seconds, float(segments[:, 1].max()))
    cursors = np.arange(start, stop, 1 / fps)
    values = np.full((len(cursors), len(names)), np.nan, dtype=np.float32)
    traces = {}
    a, b = max(0., start - window / 2), stop + window / 2
    intervals = [(max(a, s), min(b, e)) for s, e in segments if min(b, e) > max(a, s)]
    if not intervals:
        return SignalChunk(cursors, values, traces)
    for i, name in enumerate(names):
        if cancelled.is_set():
            return None
        if name is None:
            continue
        raw = recording.slice(recording.s2i(intervals), chs=name, time=True)
        if raw is None or len(raw) < 2 or raw[1] is None or not len(raw[1]):
            continue
        matrix = np.asarray(raw[1])
        times, samples = matrix[:, 0].astype(float), matrix[:, 1].astype(np.float32)
        if len(times) > 1 and (np.diff(times) <= 0).any():
            raise ValueError("Signal timestamps must increase")
        values[:, i] = summarize(times, samples, cursors, mode, window, band)
        for s, e in segments:
            # End is exclusive; require a full analysis window within one segment.
            half = 0 if mode == "Signed amplitude" else window / 2
            inside = (cursors >= s + half) & (cursors < e - half)
            if s == segments[0, 0]:
                valid_segment = inside
            else:
                valid_segment |= inside
        values[~valid_segment, i] = np.nan
        stride = max(1, int(np.ceil(len(times) / 2000)))
        tx, sy = times[::stride].copy(), samples[::stride].copy()
        # Keep explicit segment breaks even when a gap is shorter than the
        # downsampled trace spacing.
        for s, _ in segments:
            if len(tx) and tx[0] < s <= tx[-1]:
                sy[np.searchsorted(tx, s)] = np.nan
        traces[i] = (tx, sy)
    return SignalChunk(cursors, values, traces)
