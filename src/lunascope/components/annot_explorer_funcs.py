
#  --------------------------------------------------------------------
#
#  This file is part of Luna.
#
#  LUNA is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Luna is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Luna. If not, see <http:#www.gnu.org/licenses/>.
#
#  Please see LICENSE.txt for more details.
#
#  --------------------------------------------------------------------

"""
Annotation Explorer: cohort-level annotation analysis functions.

All functions are pure (no Qt dependency) and operate on the 'cohort' dict
returned by compile_cohort().

Cohort structure::

    {
      'subjects': [
        {
          'id': str,
          'duration': float,        # recording length in seconds
          'events': pd.DataFrame,   # columns: Class, Start, Stop, Dur
        }, ...
      ],
      'annot_classes': [str],        # sorted unique annotation class names
      'total_events': int,
      'n_subjects': int,
    }

All time values are in seconds.  Within-subject temporal coordinates are
preserved (i.e. events are not shifted unless event_raster_data() is called
with a non-zero gap).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

ANNOT_PALETTE = [
    "#ff6b6b",  # coral red
    "#ffd166",  # golden yellow
    "#06d6a0",  # emerald green
    "#4cc9f0",  # sky blue
    "#a78bfa",  # lavender purple
    "#f72585",  # hot pink
    "#90be6d",  # lime green
    "#f9844a",  # orange
    "#43aa8b",  # teal
    "#577590",  # steel blue
    "#f94144",  # bright red
    "#90e0ef",  # light cyan
    "#c77dff",  # violet
    "#48cae4",  # aqua
    "#fb8500",  # amber
    "#8ecae6",  # pale blue
]

ANNEX_CACHE_HEADER = "# lunascope-annot-explorer v1"
ANNEX_CACHE_COLUMNS = "class\tinstance\tchannel\tstart\tstop\tmeta"
ANNEX_SUBJECT_CLASS = "indiv_int_mrkr"
ANNEX_CACHE_GAP_SECS = 10.0


def get_annot_color(cls: str, classes: List[str]) -> str:
    """Return a consistent palette colour for annotation class *cls*."""
    try:
        idx = classes.index(cls)
        return ANNOT_PALETTE[idx % len(ANNOT_PALETTE)]
    except ValueError:
        return "#aaaaaa"


# ---------------------------------------------------------------------------
# Cohort compilation
# ---------------------------------------------------------------------------

def compile_cohort(proj, ids: List[str], exclude_classes=None, progress_cb=None) -> dict:
    """Iterate over *ids* and collect annotation events from each subject.

    Designed to run in a background thread.  Modifies the project's current
    individual as it iterates; the caller should restore it afterwards.

    Parameters
    ----------
    proj:
        A ``lunapi.proj`` instance with a loaded sample list.
    ids:
        Ordered list of individual IDs to process.
    exclude_classes:
        Annotation class names to skip (default: ``['SleepStage']``).

    Returns
    -------
    dict
        Cohort dict as described in the module docstring.
    """
    import lunapi as lp

    exclude = set(exclude_classes or ["SleepStage"])
    subjects = []
    all_classes: set = set()

    for id_str in ids:
        try:
            p = proj.inst(id_str)
        except Exception as e:
            print(f"[AnnotExplorer] Cannot attach {id_str!r}: {e}")
            continue

        # ---- recording duration ----------------------------------------
        dur = 0.0
        try:
            stat = p.edf.stat()
            nr = float(stat.get("nr", 0) or 0)
            rs = float(stat.get("rs", 1) or 1)
            dur = nr * rs
        except Exception:
            pass

        # Fall back to segsrv if stat didn't work
        if dur <= 0:
            try:
                ss = lp.segsrv(p)
                dur = float(ss.num_seconds_clocktime_original())
            except Exception:
                pass

        # ---- annotation class names ------------------------------------
        try:
            raw_classes = [c for c in (p.edf.annots() or []) if c not in exclude]
        except Exception:
            raw_classes = []

        # ---- fetch events ----------------------------------------------
        ev = pd.DataFrame(columns=["Class", "Start", "Stop", "Dur"])
        if raw_classes:
            try:
                fetched = p.fetch_annots(raw_classes)
                if fetched is not None and not fetched.empty:
                    # Normalise column names (lunapi versions differ slightly)
                    col_rename = {}
                    for col in fetched.columns:
                        lc = col.lower()
                        if lc in ("class", "annotation", "ann"):
                            col_rename[col] = "Class"
                        elif lc == "start":
                            col_rename[col] = "Start"
                        elif lc in ("stop", "end"):
                            col_rename[col] = "Stop"
                    if col_rename:
                        fetched = fetched.rename(columns=col_rename)

                    needed = [c for c in ("Class", "Start", "Stop") if c in fetched.columns]
                    if len(needed) == 3:
                        ev = fetched[needed].copy()
                        ev["Start"] = pd.to_numeric(ev["Start"], errors="coerce").fillna(0.0)
                        ev["Stop"] = pd.to_numeric(ev["Stop"], errors="coerce").fillna(0.0)
                        ev["Dur"] = (ev["Stop"] - ev["Start"]).clip(lower=0.0)
                        ev = ev[~ev["Class"].isin(exclude)].reset_index(drop=True)

                        if dur <= 0 and not ev.empty:
                            dur = float(ev["Stop"].max())

                        all_classes.update(ev["Class"].unique())
            except Exception as e:
                print(f"[AnnotExplorer] Cannot fetch annots for {id_str!r}: {e}")

        subjects.append({"id": id_str, "duration": dur, "events": ev})
        if progress_cb is not None:
            try:
                progress_cb(len(subjects), len(ids))
            except Exception:
                pass

    total_events = sum(len(s["events"]) for s in subjects)

    return {
        "subjects": subjects,
        "annot_classes": sorted(all_classes),
        "total_events": total_events,
        "n_subjects": len(subjects),
    }


def save_annex_cache(path: str, cohort: dict) -> None:
    """Write a compiled annotation cohort to a single pooled .annot file.

    Subjects are concatenated with a fixed 10-second spacer, mirroring
    Luna's multi-sample OVERLAP merge behavior.
    """
    subjects = cohort.get("subjects", []) if cohort else []
    offset = 0.0

    with open(path, "w") as fh:
        fh.write(f"{ANNEX_CACHE_HEADER}\n")
        fh.write("# Subject marker rows use a reserved annotation class.\n")
        fh.write(f"{ANNEX_CACHE_COLUMNS}\n")

        for subj in subjects:
            subj_id = str(subj.get("id", ""))
            dur = float(subj.get("duration", 0.0) or 0.0)
            start = offset
            stop = offset + dur
            fh.write(
                f"{ANNEX_SUBJECT_CLASS}\t{subj_id}\t.\t{start:.6f}\t{stop:.6f}\t.\n"
            )

            ev = subj.get("events")
            if isinstance(ev, pd.DataFrame) and not ev.empty:
                for _, row in ev.iterrows():
                    cls = str(row.get("Class", "."))
                    ev_start = float(row.get("Start", 0.0) or 0.0) + offset
                    ev_stop = float(row.get("Stop", ev_start) or ev_start) + offset
                    fh.write(f"{cls}\t.\t.\t{ev_start:.6f}\t{ev_stop:.6f}\t.\n")

            offset = stop + ANNEX_CACHE_GAP_SECS


def load_annex_cache(path: str) -> dict:
    """Read a pooled .annot file produced by save_annex_cache()."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("class\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            cls = parts[0]
            inst = parts[1] if len(parts) > 1 else "."
            start = float(parts[3])
            stop = float(parts[4])
            rows.append((cls, inst, start, stop))

    if not rows:
        return {"subjects": [], "annot_classes": [], "total_events": 0, "n_subjects": 0}

    subj_rows = [(inst, start, stop) for cls, inst, start, stop in rows if cls == ANNEX_SUBJECT_CLASS]
    if not subj_rows:
        raise ValueError("No subject marker rows found in .annot file.")

    subj_rows.sort(key=lambda x: x[1])
    event_rows = [(cls, start, stop) for cls, _, start, stop in rows if cls != ANNEX_SUBJECT_CLASS]

    subjects = []
    all_classes = set()
    eps = 1e-6

    for subj_id, subj_start, subj_stop in subj_rows:
        subj_events = []
        for cls, start, stop in event_rows:
            if start >= subj_start - eps and stop <= subj_stop + eps:
                subj_events.append({
                    "Class": cls,
                    "Start": max(0.0, start - subj_start),
                    "Stop": max(0.0, stop - subj_start),
                })
                all_classes.add(cls)

        if subj_events:
            ev = pd.DataFrame(subj_events)
            ev["Dur"] = (ev["Stop"] - ev["Start"]).clip(lower=0.0)
        else:
            ev = pd.DataFrame(columns=["Class", "Start", "Stop", "Dur"])

        subjects.append({
            "id": subj_id,
            "duration": max(0.0, subj_stop - subj_start),
            "events": ev,
        })

    total_events = sum(len(s["events"]) for s in subjects)
    return {
        "subjects": subjects,
        "annot_classes": sorted(all_classes),
        "total_events": total_events,
        "n_subjects": len(subjects),
    }


# ---------------------------------------------------------------------------
# Temporal occupancy
# ---------------------------------------------------------------------------

def temporal_occupancy(cohort: dict, classes: List[str], bin_secs: float = 30.0) -> dict:
    """Per-class temporal occupancy probability across the cohort.

    For each time bin and each annotation class, returns the fraction of
    subjects that have at least one event of that class active (overlapping)
    in that bin.  The denominator is the number of subjects whose recording
    covers the bin, so shorter recordings do not deflate later bins.

    Parameters
    ----------
    cohort   : cohort dict from compile_cohort / load_annex_cache
    classes  : annotation class names to include
    bin_secs : bin width in seconds (controls resolution vs. noise)

    Returns
    -------
    dict with keys:
        bins       – ndarray (n_bins,)  bin centre times in seconds
        occupancy  – {cls: ndarray (n_bins,)} values in [0, 1]
        n_active   – ndarray (n_bins,)  subjects with data per bin
        n_subjects – int
        max_dur    – float  longest recording (seconds)
        bin_secs   – float  actual bin width used
    """
    subjects = cohort.get("subjects", [])
    if not subjects:
        return {"bins": np.array([]), "occupancy": {}, "n_active": np.array([]),
                "n_subjects": 0, "max_dur": 0.0, "bin_secs": bin_secs}

    max_dur = max((float(s.get("duration", 0) or 0) for s in subjects), default=0.0)
    if max_dur <= 0 or bin_secs <= 0:
        return {"bins": np.array([]), "occupancy": {}, "n_active": np.array([]),
                "n_subjects": len(subjects), "max_dur": 0.0, "bin_secs": bin_secs}

    n_bins = max(1, min(int(np.ceil(max_dur / bin_secs)), 5000))
    actual_bin = max_dur / n_bins
    bin_edges  = np.arange(n_bins + 1, dtype=np.float64) * actual_bin
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # How many subjects have a recording covering each bin
    n_active = np.zeros(n_bins, dtype=np.float32)
    for s in subjects:
        dur = float(s.get("duration", 0) or 0)
        if dur > 0:
            last = min(int(np.ceil(dur / actual_bin)), n_bins)
            n_active[:last] += 1.0

    occupancy: Dict[str, np.ndarray] = {}
    for cls in classes:
        occ = np.zeros(n_bins, dtype=np.float32)
        for s in subjects:
            ev = s.get("events")
            if not isinstance(ev, pd.DataFrame) or ev.empty:
                continue
            cls_ev = ev[ev["Class"] == cls]
            if cls_ev.empty:
                continue
            starts_b = (cls_ev["Start"].values / actual_bin).astype(int)
            stops_b  = np.ceil(cls_ev["Stop"].values / actual_bin).astype(int)
            hit = np.zeros(n_bins, dtype=bool)
            for b0, b1 in zip(starts_b, stops_b):
                b0 = int(np.clip(b0, 0, n_bins - 1))
                b1 = int(np.clip(b1, 0, n_bins))
                if b0 < b1:
                    hit[b0:b1] = True
            occ += hit.astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            occupancy[cls] = np.where(n_active > 0, occ / n_active, np.nan)

    return {
        "bins":       bin_centers,
        "occupancy":  occupancy,
        "n_active":   n_active,
        "n_subjects": len(subjects),
        "max_dur":    max_dur,
        "bin_secs":   actual_bin,
    }


# ---------------------------------------------------------------------------
# Peri-event time histogram  (PETH)
# ---------------------------------------------------------------------------

def peri_event_histogram(
    cohort: dict,
    ref_class: str,
    target_classes: List[str],
    window_secs: float = 60.0,
    bin_secs: float = 2.0,
    ref_anchor: str = "mid",
    target_mode: str = "span",
) -> dict:
    """Compute peri-event time histograms centred on *ref_class* events.

    Parameters
    ----------
    ref_anchor : "start" | "mid" | "end"
        Which part of the reference event is used as the time-zero anchor.
    target_mode : "span" | "onset"
        "span"  — for each lag bin, counts how many reference events have the
                  target annotation *active* (spanning) at that lag.  Gives
                  P(active) on the y-axis; natural for long/epoch annotations.
        "onset" — counts target event *onsets* (start times) at each lag
                  relative to the reference anchor.  Gives rate (events / ref / s);
                  natural for brief point-process events.

    Returns
    -------
    dict with keys:
        bins, edges, counts, density, n_ref, ref_class, target_classes,
        window, bin_secs, ref_anchor, target_mode
    """
    bin_secs = max(bin_secs, 0.01)
    window_secs = max(window_secs, bin_secs)

    edges = np.arange(-window_secs, window_secs + bin_secs * 0.5, bin_secs)
    bins = (edges[:-1] + edges[1:]) / 2.0
    n_bins = len(bins)

    counts = {cls: np.zeros(n_bins, dtype=float) for cls in target_classes}
    n_ref = 0

    for subj in cohort["subjects"]:
        ev = subj["events"]
        if ev is None or ev.empty or "Class" not in ev.columns:
            continue

        ref_ev = ev[ev["Class"] == ref_class]
        if ref_ev.empty:
            continue

        if ref_anchor == "start":
            ref_times = ref_ev["Start"].values.astype(float)
        elif ref_anchor == "end":
            ref_times = ref_ev["Stop"].values.astype(float)
        else:  # "mid"
            ref_times = ((ref_ev["Start"].values + ref_ev["Stop"].values) / 2.0)
        n_ref += len(ref_times)

        for cls in target_classes:
            tgt_ev = ev[ev["Class"] == cls]
            if tgt_ev.empty:
                continue

            is_self = (cls == ref_class)
            if target_mode == "span":
                # Each target interval [ts, te] is active at lag t when
                # ts - ref <= t <= te - ref.  Accumulate via diff+cumsum.
                tgt_starts = tgt_ev["Start"].values.astype(float)
                tgt_stops  = tgt_ev["Stop"].values.astype(float)
                # All pairwise lag intervals — shape (N_tgt, N_ref)
                N_tgt, N_ref = len(tgt_starts), len(ref_times)
                lag_lo_mat = tgt_starts[:, np.newaxis] - ref_times[np.newaxis, :]
                lag_hi_mat = tgt_stops[:, np.newaxis]  - ref_times[np.newaxis, :]
                if is_self and N_tgt == N_ref:
                    # exclude self-pairs (diagonal)
                    mask = ~np.eye(N_tgt, dtype=bool)
                    lag_lo = lag_lo_mat[mask]
                    lag_hi = lag_hi_mat[mask]
                else:
                    lag_lo = lag_lo_mat.ravel()
                    lag_hi = lag_hi_mat.ravel()
                # Clip to window and convert to bin indices
                b0 = np.clip(
                    np.searchsorted(edges, lag_lo, side="right") - 1, 0, n_bins)
                b1 = np.clip(
                    np.searchsorted(edges, lag_hi, side="left"),  0, n_bins)
                valid = b1 > b0
                if valid.any():
                    delta = np.zeros(n_bins + 1, dtype=float)
                    np.add.at(delta, b0[valid].astype(int),  1.0)
                    np.add.at(delta, b1[valid].astype(int), -1.0)
                    counts[cls] += np.cumsum(delta)[:n_bins]
            else:
                # "onset": histogram of target start times relative to ref anchor
                tgt_times = tgt_ev["Start"].values.astype(float)
                N_tgt, N_ref = len(tgt_times), len(ref_times)
                lags_mat = tgt_times[:, np.newaxis] - ref_times[np.newaxis, :]
                if is_self and N_tgt == N_ref:
                    # exclude self-pairs (diagonal)
                    lags = lags_mat[~np.eye(N_tgt, dtype=bool)]
                else:
                    lags = lags_mat.ravel()
                in_win = lags[(lags >= edges[0]) & (lags < edges[-1])]
                if len(in_win):
                    c, _ = np.histogram(in_win, bins=edges)
                    counts[cls] += c

    density = {}
    for cls in target_classes:
        if n_ref > 0:
            if target_mode == "span":
                # P(target active at lag t) — dimensionless probability
                density[cls] = counts[cls] / n_ref
            else:
                # rate: target onsets per reference event per second
                density[cls] = counts[cls] / (n_ref * bin_secs)
        else:
            density[cls] = np.zeros(n_bins, dtype=float)

    return {
        "bins": bins,
        "edges": edges,
        "counts": counts,
        "density": density,
        "n_ref": n_ref,
        "window": window_secs,
        "bin_secs": bin_secs,
        "ref_class": ref_class,
        "target_classes": target_classes,
        "ref_anchor": ref_anchor,
        "target_mode": target_mode,
    }


# ---------------------------------------------------------------------------
# Overlap / co-occurrence matrix
# ---------------------------------------------------------------------------

def overlap_matrix(
    cohort: dict,
    annot_classes: List[str],
    bin_secs: float = 5.0,
) -> dict:
    """Compute pairwise overlap statistics using binary occupancy arrays.

    Each subject's recording is discretised into *bin_secs*-wide bins.  An
    annotation occupies a bin if any part of it falls within that bin.

    Returns
    -------
    dict with keys:
        jaccard   – (n×n) symmetric Jaccard similarity matrix
        directed  – (n×n) directed overlap: directed[i,j] = P(j present | i present)
        labels    – annotation class names (row/column order)
        event_rate – {class: events per hour, pooled across all subjects}
        occ_frac  – {class: fraction of total recording time occupied}
    """
    n = len(annot_classes)
    if n == 0:
        return {
            "jaccard": np.zeros((0, 0)),
            "directed": np.zeros((0, 0)),
            "labels": [],
            "event_rate": {},
            "occ_frac": {},
        }

    occ_totals = np.zeros(n, dtype=float)
    cooccupy = np.zeros((n, n), dtype=float)
    total_bins = 0.0
    event_counts = np.zeros(n, dtype=float)
    cls_idx = {c: i for i, c in enumerate(annot_classes)}

    for subj in cohort["subjects"]:
        ev = subj["events"]
        dur = subj["duration"]
        if ev is None or ev.empty or dur <= 0:
            continue

        n_bins = max(1, int(np.ceil(dur / bin_secs)))
        total_bins += n_bins

        # Build occupancy arrays with the cumsum trick (O(n_events + n_bins))
        occ: Dict[str, Optional[np.ndarray]] = {}
        for cls in annot_classes:
            cls_ev = ev[ev["Class"] == cls]
            if cls_ev.empty:
                occ[cls] = None
                continue

            starts = np.clip(
                (cls_ev["Start"].values / bin_secs).astype(int), 0, n_bins
            )
            stops = np.clip(
                np.ceil(cls_ev["Stop"].values / bin_secs).astype(int), 0, n_bins
            )

            diff = np.zeros(n_bins + 1, dtype=np.int32)
            np.add.at(diff, starts, 1)
            np.add.at(diff, stops, -1)
            arr = np.cumsum(diff[:n_bins]) > 0
            occ[cls] = arr

            i = cls_idx[cls]
            event_counts[i] += len(cls_ev)

        # Stack non-None occupancy arrays into a matrix for vectorised ops
        present_idx  = [i for i in range(n)
                        if occ.get(annot_classes[i]) is not None]
        if not present_idx:
            continue

        occ_mat = np.vstack(
            [occ[annot_classes[i]].astype(np.int32) for i in present_idx]
        )   # shape (n_present, n_bins)

        # Per-class occupied bin counts
        row_sums = occ_mat.sum(axis=1).astype(float)
        for pi, i in enumerate(present_idx):
            occ_totals[i] += row_sums[pi]

        # Pairwise co-occupancy via matrix multiply
        co_sub = occ_mat @ occ_mat.T   # (n_present, n_present), all int32
        ix = np.ix_(present_idx, present_idx)
        cooccupy[ix] += co_sub.astype(float)

    # Derive Jaccard and directed overlap
    jaccard = np.zeros((n, n), dtype=float)
    directed = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            co = cooccupy[i, j]
            union = occ_totals[i] + occ_totals[j] - co
            if union > 0:
                jaccard[i, j] = co / union
            if occ_totals[i] > 0:
                directed[i, j] = co / occ_totals[i]

    total_hours = (total_bins * bin_secs) / 3600.0
    event_rate = {
        cls: (event_counts[cls_idx[cls]] / total_hours if total_hours > 0 else 0.0)
        for cls in annot_classes
    }
    occ_frac = {
        cls: (occ_totals[cls_idx[cls]] / max(total_bins, 1))
        for cls in annot_classes
    }

    return {
        "jaccard": jaccard,
        "directed": directed,
        "labels": annot_classes,
        "event_rate": event_rate,
        "occ_frac": occ_frac,
    }


# ---------------------------------------------------------------------------
# Nearest-neighbour distances
# ---------------------------------------------------------------------------

def nearest_neighbor_distances(
    cohort: dict,
    ref_class: str,
    target_classes: List[str],
) -> dict:
    """For each event of *ref_class*, find the time to the nearest event of
    each *target_classes* annotation **within the same subject**.

    Returns
    -------
    {class: sorted np.ndarray of distances in seconds}
    """
    distances: Dict[str, list] = {cls: [] for cls in target_classes}

    for subj in cohort["subjects"]:
        ev = subj["events"]
        if ev is None or ev.empty or "Class" not in ev.columns:
            continue

        ref_ev = ev[ev["Class"] == ref_class]
        if ref_ev.empty:
            continue
        ref_times = (ref_ev["Start"].values + ref_ev["Stop"].values) / 2.0

        for cls in target_classes:
            tgt_ev = ev[ev["Class"] == cls]
            if tgt_ev.empty:
                continue
            tgt_times = (tgt_ev["Start"].values + tgt_ev["Stop"].values) / 2.0

            # For each ref event, minimum absolute distance to any target event
            diffs = np.abs(ref_times[:, np.newaxis] - tgt_times[np.newaxis, :])
            min_d = diffs.min(axis=1)
            distances[cls].extend(min_d.tolist())

    return {cls: np.sort(np.array(v)) for cls, v in distances.items()}


# ---------------------------------------------------------------------------
# Inter-event intervals
# ---------------------------------------------------------------------------

def inter_event_intervals(cohort: dict, annot_classes: List[str]) -> dict:
    """Compute gap between consecutive events (Stop[i] → Start[i+1]) within
    each subject for each annotation class.

    Only positive gaps (non-overlapping events) are returned.

    Returns
    -------
    {class: sorted np.ndarray of IEI values in seconds}
    """
    ieis: Dict[str, list] = {cls: [] for cls in annot_classes}

    for subj in cohort["subjects"]:
        ev = subj["events"]
        if ev is None or ev.empty or "Class" not in ev.columns:
            continue

        for cls in annot_classes:
            cls_ev = ev[ev["Class"] == cls].sort_values("Start")
            if len(cls_ev) < 2:
                continue
            stops = cls_ev["Stop"].values[:-1]
            starts = cls_ev["Start"].values[1:]
            gaps = starts - stops
            ieis[cls].extend(gaps[gaps > 0].tolist())

    return {cls: np.sort(np.array(v)) for cls, v in ieis.items() if v}


# ---------------------------------------------------------------------------
# Event raster (pooled timeline)
# ---------------------------------------------------------------------------

def event_raster_data(
    cohort: dict,
    annot_classes: List[str],
    gap_secs: float = 10.0,
) -> dict:
    """Build pooled timeline data for raster rendering.

    Subjects are concatenated end-to-end with *gap_secs* separating them.
    All event start/stop times are shifted by the subject's cumulative offset.

    Returns
    -------
    dict with keys:
        by_class        – {class: list of (x_start, x_stop)} in merged time
        subject_bounds  – [(offset, offset+dur), ...] for shading
        subject_ids     – [str, ...]
        total_duration  – float (seconds)
        gap_secs        – float
    """
    by_class: Dict[str, List[Tuple[float, float]]] = {cls: [] for cls in annot_classes}
    subject_bounds = []
    current_offset = 0.0

    for subj in cohort["subjects"]:
        ev = subj["events"]
        dur = max(subj["duration"], 0.0)
        subj_end = current_offset + dur
        subject_bounds.append((current_offset, subj_end))

        if ev is not None and not ev.empty:
            for cls in annot_classes:
                cls_ev = ev[ev["Class"] == cls]
                if cls_ev.empty:
                    continue
                s0 = cls_ev["Start"].values.astype(float) + current_offset
                s1 = cls_ev["Stop"].values.astype(float) + current_offset
                by_class[cls].extend(zip(s0.tolist(), s1.tolist()))

        current_offset = subj_end + gap_secs

    total_duration = max(current_offset - gap_secs, 1.0)

    return {
        "by_class": by_class,
        "subject_bounds": subject_bounds,
        "total_duration": total_duration,
        "subject_ids": [s["id"] for s in cohort["subjects"]],
        "gap_secs": gap_secs,
    }


# ---------------------------------------------------------------------------
# Normalized occupancy profile
# ---------------------------------------------------------------------------

def _merge_intervals(starts: np.ndarray, stops: np.ndarray) -> List[Tuple[float, float]]:
    """Merge overlapping intervals."""
    if len(starts) == 0 or len(stops) == 0:
        return []
    order = np.argsort(starts)
    starts = starts[order].astype(float)
    stops = stops[order].astype(float)

    merged: List[Tuple[float, float]] = []
    cur_s = starts[0]
    cur_e = stops[0]
    for s, e in zip(starts[1:], stops[1:]):
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def normalized_occupancy(
    cohort: dict,
    annot_classes: List[str],
    bin_pct: float = 2.0,
) -> dict:
    """Mean occupancy fraction across a normalized 0..1 recording axis.

    Each subject's recording is rescaled to [0, 1]. For each selected class,
    the function computes the fraction of each normalized bin occupied by that
    class, then averages those occupancy fractions across subjects.

    `bin_pct` is the normalized bin width expressed as a percentage of the
    recording duration, so 2.0 means 50 bins across the night.
    """
    bin_pct = float(np.clip(bin_pct, 0.25, 50.0))
    n_bins = max(4, int(np.ceil(100.0 / bin_pct)))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_width = edges[1] - edges[0]

    occ_sum = {cls: np.zeros(n_bins, dtype=float) for cls in annot_classes}
    class_subject_counts = {cls: 0 for cls in annot_classes}
    n_subjects_used = 0

    for subj in cohort["subjects"]:
        ev = subj["events"]
        dur = float(subj.get("duration", 0.0) or 0.0)
        if dur <= 0:
            continue

        n_subjects_used += 1
        if ev is None or ev.empty or "Class" not in ev.columns:
            continue

        for cls in annot_classes:
            cls_ev = ev[ev["Class"] == cls]
            if cls_ev.empty:
                continue

            starts = np.clip(cls_ev["Start"].values.astype(float) / dur, 0.0, 1.0)
            stops = np.clip(cls_ev["Stop"].values.astype(float) / dur, 0.0, 1.0)
            merged = _merge_intervals(starts, stops)
            if not merged:
                continue

            occ = np.zeros(n_bins, dtype=float)
            for s, e in merged:
                if e <= s:
                    continue
                left = max(0, int(np.floor(s / bin_width)))
                right = min(n_bins - 1, int(np.ceil(e / bin_width)) - 1)
                for bi in range(left, right + 1):
                    ov = max(0.0, min(e, edges[bi + 1]) - max(s, edges[bi]))
                    if ov > 0:
                        occ[bi] += ov / bin_width

            occ_sum[cls] += np.clip(occ, 0.0, 1.0)
            class_subject_counts[cls] += 1

    mean_occ = {
        cls: occ_sum[cls] / max(n_subjects_used, 1)
        for cls in annot_classes
    }

    return {
        "x": centers,
        "edges": edges,
        "mean_occ": mean_occ,
        "bin_pct": bin_pct,
        "n_bins": n_bins,
        "n_subjects_used": n_subjects_used,
        "class_subject_counts": class_subject_counts,
    }


# ---------------------------------------------------------------------------
# Duration distributions
# ---------------------------------------------------------------------------

def duration_stats(cohort: dict, annot_classes: List[str]) -> dict:
    """Collect event durations (in seconds) for each annotation class.

    Returns
    -------
    {class: np.ndarray of durations > 0}
    """
    durs: Dict[str, list] = {cls: [] for cls in annot_classes}

    for subj in cohort["subjects"]:
        ev = subj["events"]
        if ev is None or ev.empty or "Class" not in ev.columns:
            continue
        for cls in annot_classes:
            cls_ev = ev[ev["Class"] == cls]
            if cls_ev.empty:
                continue
            vals = cls_ev["Dur"].values
            durs[cls].extend(vals[vals > 0].tolist())

    return {cls: np.array(v) for cls, v in durs.items() if v}


# ---------------------------------------------------------------------------
# Cross-correlogram (bidirectional PETH between two specific classes)
# ---------------------------------------------------------------------------

def cross_correlogram(
    cohort: dict,
    class_a: str,
    class_b: str,
    max_lag_secs: float = 60.0,
    bin_secs: float = 1.0,
) -> dict:
    """Bidirectional cross-correlogram between *class_a* and *class_b*.

    Positive lags mean a class_b event *follows* the class_a reference event.

    Returns
    -------
    dict with keys: lags, edges, count, density, n_a, n_b, class_a, class_b
    """
    bin_secs = max(bin_secs, 0.01)
    edges = np.arange(-max_lag_secs, max_lag_secs + bin_secs * 0.5, bin_secs)
    bins = (edges[:-1] + edges[1:]) / 2.0
    n_bins = len(bins)
    count = np.zeros(n_bins, dtype=float)
    n_a = 0
    n_b = 0

    for subj in cohort["subjects"]:
        ev = subj["events"]
        if ev is None or ev.empty or "Class" not in ev.columns:
            continue
        ev_a = ev[ev["Class"] == class_a]
        ev_b = ev[ev["Class"] == class_b]
        if ev_a.empty or ev_b.empty:
            continue

        t_a = (ev_a["Start"].values + ev_a["Stop"].values) / 2.0
        t_b = (ev_b["Start"].values + ev_b["Stop"].values) / 2.0
        n_a += len(t_a)
        n_b += len(t_b)

        lags = (t_b[:, np.newaxis] - t_a[np.newaxis, :]).ravel()
        in_win = lags[(lags >= edges[0]) & (lags < edges[-1])]
        if len(in_win):
            c, _ = np.histogram(in_win, bins=edges)
            count += c

    return {
        "lags": bins,
        "edges": edges,
        "count": count,
        "density": count / (n_a * bin_secs) if n_a > 0 else count,
        "n_a": n_a,
        "n_b": n_b,
        "class_a": class_a,
        "class_b": class_b,
    }
