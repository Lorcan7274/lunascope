
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
Harmonizer: pure scan and analysis functions.  No Qt dependency.

The scan collects channel metadata (SR, TRANS, PDIM) and annotation label
names from every subject in the sample list, then provides helpers to build
presence matrices, similarity suggestions, coverage stats, and @param file
exports.
"""

import re
import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


CACHE_MAGIC = "lunascope-harmonizer-v1"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    channels_df: pd.DataFrame   # columns: ID, CH, SR, TRANS, PDIM
    annots_df:   pd.DataFrame   # columns: ID, ANNOT
    types_df:    pd.DataFrame   # columns: CH, TYPE  (from TYPES cmd; may be empty)
    ids:         List[str]      # scanned subject IDs in scan order
    n_total:     int            # subjects in sample list at scan time
    scan_ts:     str            # ISO-8601 timestamp


# ---------------------------------------------------------------------------
# Cohort scan
# ---------------------------------------------------------------------------

def scan_cohort(
    proj,
    ids: List[str],
    scan_channels: bool = True,
    scan_annots: bool = True,
    scan_edfplus: bool = False,
    stop_flag=None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> ScanResult:
    """Iterate over *ids* collecting channel/annotation metadata.

    Designed to run in a background thread.
    stop_flag  : threading.Event; scan aborts when .is_set().
    progress_cb: called as (n_done, n_total) after each subject.
    """
    ch_rows:   list = []
    ann_rows:  list = []
    type_rows: list = []
    scanned:   list = []
    n_total = len(ids)

    for i, id_str in enumerate(ids):
        if stop_flag is not None and stop_flag.is_set():
            break
        if progress_cb:
            progress_cb(i, n_total)

        try:
            p = proj.inst(id_str)
        except Exception as exc:
            print(f"[Harmonizer] Cannot attach {id_str!r}: {exc}")
            continue

        scanned.append(id_str)

        # ---- channels + types -------------------------------------------
        if scan_channels:
            hdr_ok = False
            try:
                p.silent_proc('HEADERS & TYPES')
                hdr_ok = True
            except Exception:
                try:
                    p.silent_proc('HEADERS')
                    hdr_ok = True
                except Exception as exc2:
                    print(f"[Harmonizer] HEADERS failed for {id_str!r}: {exc2}")

            if hdr_ok:
                try:
                    df = p.table('HEADERS', 'CH')
                    if df is not None and not df.empty:
                        for _, row in df.iterrows():
                            ch_rows.append({
                                'ID':    id_str,
                                'CH':    str(row.get('CH',    '')),
                                'SR':    _fmt_num(row.get('SR',    '')),
                                'TRANS': str(row.get('TRANS', '')),
                                'PDIM':  str(row.get('PDIM',  '')),
                            })
                except Exception:
                    pass

                try:
                    df_t = p.table('TYPES', 'CH')
                    if df_t is not None and not df_t.empty:
                        for _, row in df_t.iterrows():
                            type_rows.append({
                                'CH':   str(row.get('CH',   '')),
                                'TYPE': str(row.get('TYPE', '')),
                            })
                except Exception:
                    pass

        # ---- annotations ------------------------------------------------
        if scan_annots:
            try:
                if scan_edfplus:
                    try:
                        p.silent_proc('ANNOTS')
                    except Exception:
                        pass
                for cls in (p.edf.annots() or []):
                    ann_rows.append({'ID': id_str, 'ANNOT': str(cls)})
            except Exception as exc:
                print(f"[Harmonizer] annots failed for {id_str!r}: {exc}")

    if progress_cb:
        progress_cb(len(scanned), n_total)

    # Deduplicate types: most common TYPE per CH across all subjects.
    if type_rows:
        tdf = pd.DataFrame(type_rows)
        tdf = (
            tdf.groupby(['CH', 'TYPE'])
               .size().reset_index(name='n')
               .sort_values('n', ascending=False)
               .drop_duplicates('CH', keep='first')[['CH', 'TYPE']]
               .reset_index(drop=True)
        )
    else:
        tdf = pd.DataFrame(columns=['CH', 'TYPE'])

    return ScanResult(
        channels_df=(pd.DataFrame(ch_rows)
                     if ch_rows
                     else pd.DataFrame(columns=['ID', 'CH', 'SR', 'TRANS', 'PDIM'])),
        annots_df=(pd.DataFrame(ann_rows)
                   if ann_rows
                   else pd.DataFrame(columns=['ID', 'ANNOT'])),
        types_df=tdf,
        ids=scanned,
        n_total=n_total,
        scan_ts=datetime.now().isoformat(timespec='seconds'),
    )


def _fmt_num(v) -> str:
    try:
        return f"{float(v):g}"
    except (TypeError, ValueError):
        return str(v) if v is not None else ''


# ---------------------------------------------------------------------------
# Presence matrix
# ---------------------------------------------------------------------------

def build_presence(
    df: pd.DataFrame,
    name_col: str,
    id_col: str = 'ID',
    ordered_ids: Optional[List[str]] = None,
    remap:  Optional[Dict[str, str]] = None,
    ignore: Optional[Set[str]] = None,
) -> Tuple[List[str], List[str], np.ndarray]:
    """Build a boolean presence matrix.

    Returns (row_names, col_ids, matrix) where
    matrix[i, j] is True iff row_names[i] appears in col_ids[j].
    ordered_ids preserves the scan order for columns.
    """
    if df is None or df.empty:
        return [], [], np.zeros((0, 0), dtype=bool)

    remap  = remap  or {}
    ignore = ignore or set()

    df2 = df.copy()
    df2[name_col] = df2[name_col].map(lambda x: remap.get(x, x))
    df2 = df2[~df2[name_col].isin(ignore)]
    df2 = df2[df2[name_col].astype(str).str.strip() != '']

    if df2.empty:
        return [], [], np.zeros((0, 0), dtype=bool)

    names = sorted(df2[name_col].unique())

    present_ids = set(df2[id_col])
    if ordered_ids is not None:
        id_order = [x for x in ordered_ids if x in present_ids]
    else:
        id_order = list(dict.fromkeys(df2[id_col]))

    name_idx = {n: i for i, n in enumerate(names)}
    id_idx   = {iid: i for i, iid in enumerate(id_order)}

    mat = np.zeros((len(names), len(id_order)), dtype=bool)
    for _, row in df2.drop_duplicates([id_col, name_col]).iterrows():
        r = name_idx.get(row[name_col])
        c = id_idx.get(row[id_col])
        if r is not None and c is not None:
            mat[r, c] = True

    return names, id_order, mat


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def channel_summary(
    channels_df: pd.DataFrame,
    remap:  Optional[Dict[str, str]] = None,
    ignore: Optional[Set[str]] = None,
    split_by_sr: bool = False,
    split_by_trans: bool = False,
    split_by_pdim: bool = False,
) -> pd.DataFrame:
    """Per-channel summary: CH, N (subjects), SR, TRANS, PDIM."""
    if channels_df is None or channels_df.empty:
        return pd.DataFrame(columns=['CH', 'N', 'SR', 'TRANS', 'PDIM'])

    remap  = remap  or {}
    ignore = ignore or set()

    df2 = channels_df.copy()
    df2['CH'] = df2['CH'].map(lambda x: remap.get(x, x))
    df2 = df2[~df2['CH'].isin(ignore)]
    df2 = df2[df2['CH'].astype(str).str.strip() != '']

    def _mode(s):
        u = s.dropna().astype(str)
        u = u[u.str.strip() != '']
        if u.empty:
            return ''
        vc = u.value_counts()
        return vc.index[0] if len(vc) == 1 else f"{vc.index[0]} *"

    group_cols = ['CH']
    if split_by_sr:
        group_cols.append('SR')
    if split_by_trans:
        group_cols.append('TRANS')
    if split_by_pdim:
        group_cols.append('PDIM')

    rows = []
    for keys, grp in df2.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        rows.append({
            'CH':    key_map.get('CH', ''),
            'N':     grp['ID'].nunique(),
            'SR':    str(key_map['SR']) if split_by_sr else _mode(grp['SR']),
            'TRANS': str(key_map['TRANS']) if split_by_trans else _mode(grp['TRANS']),
            'PDIM':  str(key_map['PDIM']) if split_by_pdim else _mode(grp['PDIM']),
        })

    return pd.DataFrame(rows).sort_values(
        ['CH', 'SR', 'TRANS', 'PDIM'],
        key=lambda s: s.astype(str).str.lower(),
        ignore_index=True,
    )


def annot_summary(
    annots_df: pd.DataFrame,
    remap:  Optional[Dict[str, str]] = None,
    ignore: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Per-annotation-class summary: ANNOT, N (subjects)."""
    if annots_df is None or annots_df.empty:
        return pd.DataFrame(columns=['ANNOT', 'N'])

    remap  = remap  or {}
    ignore = ignore or set()

    df2 = annots_df.copy()
    df2['ANNOT'] = df2['ANNOT'].map(lambda x: remap.get(x, x))
    df2 = df2[~df2['ANNOT'].isin(ignore)]

    rows = []
    for ann, grp in df2.groupby('ANNOT'):
        rows.append({'ANNOT': ann, 'N': grp['ID'].nunique()})

    return pd.DataFrame(rows).sort_values(
        'ANNOT',
        key=lambda s: s.astype(str).str.lower(),
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Domain inference / normalization
# ---------------------------------------------------------------------------

def normalize_domain(value: str) -> str:
    """Map engine/user/free-text domain labels onto a small canonical set."""
    text = str(value or '').strip()
    if not text:
        return ''

    upper = text.upper()
    if upper in {'EEG'}:
        return 'EEG'
    if upper in {'ECG', 'EKG'}:
        return 'ECG'
    if upper in {'EMG'}:
        return 'EMG'
    if upper in {'EOG'}:
        return 'EOG'
    if upper in {'RESP', 'RESPIRATION', 'RESPIRATORY', 'AIRFLOW'}:
        return 'RESP'
    if upper in {'SPO2', 'SAO2', 'OXIMETRY', 'OXYGEN'}:
        return 'SpO2'
    if upper in {'OTHER', 'AUX', 'MISC'}:
        return 'OTHER'

    low = text.lower()
    if any(tok in low for tok in ('ecg', 'ekg', 'heart', 'hr')):
        return 'ECG'
    if any(tok in low for tok in ('eog', 'roc', 'loc')):
        return 'EOG'
    if any(tok in low for tok in ('emg', 'chin', 'leg', 'plm')):
        return 'EMG'
    if any(tok in low for tok in ('spo2', 'sao2', 'o2sat', 'ox', 'pleth')):
        return 'SpO2'
    if any(tok in low for tok in ('resp', 'airflow', 'thor', 'abd', 'nasal', 'can', 'effort')):
        return 'RESP'

    eeg_tokens = (
        'eeg', 'fp', 'fz', 'cz', 'pz', 'oz', 'f3', 'f4', 'f7', 'f8', 'c3',
        'c4', 'o1', 'o2', 't3', 't4', 't5', 't6', 't7', 't8', 'p3', 'p4',
        'a1', 'a2', 'm1', 'm2'
    )
    if any(tok in low for tok in eeg_tokens):
        return 'EEG'

    return text if len(text) <= 12 else 'OTHER'


def infer_channel_domain(ch_name: str, type_name: str = '') -> str:
    """Infer a canonical domain from TYPES output and/or the channel name."""
    typed = normalize_domain(type_name)
    if typed:
        return typed
    return normalize_domain(ch_name)


def domain_assignments(
    channels_df: pd.DataFrame,
    types_df: Optional[pd.DataFrame] = None,
    remap: Optional[Dict[str, str]] = None,
    ignore: Optional[Set[str]] = None,
    user_domains: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Per-channel domain assignments with source provenance."""
    if channels_df is None or channels_df.empty:
        return pd.DataFrame(columns=['CH', 'Domain', 'Source'])

    remap = remap or {}
    ignore = ignore or set()
    user_domains = user_domains or {}

    df2 = channels_df.copy()
    df2['CH'] = df2['CH'].map(lambda x: remap.get(x, x))
    df2 = df2[~df2['CH'].isin(ignore)]
    df2 = df2[df2['CH'].astype(str).str.strip() != '']
    names = sorted(df2['CH'].astype(str).unique())

    types_lookup: Dict[str, str] = {}
    if types_df is not None and not types_df.empty:
        ch_col = 'CH' if 'CH' in types_df.columns else None
        type_col = 'TYPE' if 'TYPE' in types_df.columns else None
        if ch_col and type_col:
            for _, row in types_df.iterrows():
                ch = remap.get(str(row.get(ch_col, '')), str(row.get(ch_col, '')))
                ch = ch.strip()
                if not ch:
                    continue
                domain = normalize_domain(str(row.get(type_col, '')))
                if domain:
                    types_lookup[ch] = domain

    rows = []
    for name in names:
        user_dom = normalize_domain(user_domains.get(name, ''))
        if user_dom:
            rows.append({'CH': name, 'Domain': user_dom, 'Source': 'user'})
            continue

        typed_dom = types_lookup.get(name, '')
        inferred = infer_channel_domain(name, typed_dom)
        if typed_dom:
            source = 'types'
        elif inferred:
            source = 'name'
        else:
            source = ''
        rows.append({'CH': name, 'Domain': inferred, 'Source': source})

    return pd.DataFrame(rows).sort_values('CH', ignore_index=True)


# ---------------------------------------------------------------------------
# Rare co-occurrence / alias candidates
# ---------------------------------------------------------------------------

def rare_cooccurrence_pairs(
    channels_df: pd.DataFrame,
    types_df: Optional[pd.DataFrame] = None,
    remap: Optional[Dict[str, str]] = None,
    ignore: Optional[Set[str]] = None,
    user_domains: Optional[Dict[str, str]] = None,
    min_subjects: int = 2,
    max_overlap_ratio: float = 0.1,
    max_jaccard: float = 0.1,
    top_n: int = 150,
) -> pd.DataFrame:
    """Find channel pairs that rarely co-occur despite each appearing repeatedly."""
    if channels_df is None or channels_df.empty:
        return pd.DataFrame(columns=[
            'CH_A', 'CH_B', 'Domain', 'N_A', 'N_B', 'Both',
            'OverlapPct', 'JaccardPct', 'Union', 'Score'
        ])

    remap = remap or {}
    ignore = ignore or set()

    df2 = channels_df.copy()
    df2['CH'] = df2['CH'].map(lambda x: remap.get(x, x))
    df2 = df2[~df2['CH'].isin(ignore)]
    df2 = df2[df2['CH'].astype(str).str.strip() != '']
    df2 = df2[['ID', 'CH']].drop_duplicates()
    if df2.empty:
        return pd.DataFrame(columns=[
            'CH_A', 'CH_B', 'Domain', 'N_A', 'N_B', 'Both',
            'OverlapPct', 'JaccardPct', 'Union', 'Score'
        ])

    domains_df = domain_assignments(
        channels_df, types_df=types_df, remap=remap, ignore=ignore,
        user_domains=user_domains,
    )
    domain_lookup = dict(zip(domains_df['CH'], domains_df['Domain']))

    ch_to_ids: Dict[str, Set[str]] = {
        ch: set(grp['ID'].astype(str))
        for ch, grp in df2.groupby('CH')
    }
    names = sorted(ch_to_ids)

    rows = []
    for i, ch_a in enumerate(names):
        ids_a = ch_to_ids[ch_a]
        n_a = len(ids_a)
        if n_a < min_subjects:
            continue
        dom_a = domain_lookup.get(ch_a, '')
        for ch_b in names[i + 1:]:
            ids_b = ch_to_ids[ch_b]
            n_b = len(ids_b)
            if n_b < min_subjects:
                continue
            dom_b = domain_lookup.get(ch_b, '')
            if dom_a and dom_b and dom_a != dom_b:
                continue

            both = len(ids_a & ids_b)
            union = len(ids_a | ids_b)
            if union == 0:
                continue
            overlap_ratio = both / min(n_a, n_b)
            jaccard = both / union
            if overlap_ratio > max_overlap_ratio or jaccard > max_jaccard:
                continue

            score = min(n_a, n_b) * (1.0 - overlap_ratio)
            rows.append({
                'CH_A': ch_a,
                'CH_B': ch_b,
                'Domain': dom_a or dom_b,
                'N_A': n_a,
                'N_B': n_b,
                'Both': both,
                'OverlapPct': round(100.0 * overlap_ratio, 1),
                'JaccardPct': round(100.0 * jaccard, 1),
                'Union': union,
                'Score': round(score, 2),
            })

    if not rows:
        return pd.DataFrame(columns=[
            'CH_A', 'CH_B', 'Domain', 'N_A', 'N_B', 'Both',
            'OverlapPct', 'JaccardPct', 'Union', 'Score'
        ])

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ['Score', 'Union', 'OverlapPct', 'JaccardPct', 'CH_A', 'CH_B'],
        ascending=[False, False, True, True, True, True],
        ignore_index=True,
    )
    return out.head(top_n)


def annot_rare_cooccurrence_pairs(
    annots_df: pd.DataFrame,
    remap: Optional[Dict[str, str]] = None,
    ignore: Optional[Set[str]] = None,
    min_subjects: int = 2,
    max_overlap_ratio: float = 0.1,
    max_jaccard: float = 0.1,
    top_n: int = 100,
) -> pd.DataFrame:
    """Find annotation pairs that rarely co-occur despite each appearing in many subjects."""
    COLS = ['ANN_A', 'ANN_B', 'N_A', 'N_B', 'Both', 'OverlapPct', 'JaccardPct', 'Union', 'Score']
    if annots_df is None or annots_df.empty:
        return pd.DataFrame(columns=COLS)

    remap = remap or {}
    ignore = ignore or set()

    df2 = annots_df.copy()
    df2['ANNOT'] = df2['ANNOT'].map(lambda x: remap.get(x, x))
    df2 = df2[~df2['ANNOT'].isin(ignore)]
    df2 = df2[df2['ANNOT'].astype(str).str.strip() != '']
    df2 = df2[['ID', 'ANNOT']].drop_duplicates()
    if df2.empty:
        return pd.DataFrame(columns=COLS)

    ann_to_ids: Dict[str, Set[str]] = {
        ann: set(grp['ID'].astype(str))
        for ann, grp in df2.groupby('ANNOT')
    }
    names = sorted(ann_to_ids)

    rows = []
    for i, ann_a in enumerate(names):
        ids_a = ann_to_ids[ann_a]
        n_a = len(ids_a)
        if n_a < min_subjects:
            continue
        for ann_b in names[i + 1:]:
            ids_b = ann_to_ids[ann_b]
            n_b = len(ids_b)
            if n_b < min_subjects:
                continue
            both = len(ids_a & ids_b)
            union = len(ids_a | ids_b)
            if union == 0:
                continue
            overlap_ratio = both / min(n_a, n_b)
            jaccard = both / union
            if overlap_ratio > max_overlap_ratio or jaccard > max_jaccard:
                continue
            score = min(n_a, n_b) * (1.0 - overlap_ratio)
            rows.append({
                'ANN_A': ann_a, 'ANN_B': ann_b,
                'N_A': n_a, 'N_B': n_b, 'Both': both,
                'OverlapPct': round(100.0 * overlap_ratio, 1),
                'JaccardPct': round(100.0 * jaccard, 1),
                'Union': union, 'Score': round(score, 2),
            })

    if not rows:
        return pd.DataFrame(columns=COLS)

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ['Score', 'Union', 'OverlapPct', 'JaccardPct', 'ANN_A', 'ANN_B'],
        ascending=[False, False, True, True, True, True],
        ignore_index=True,
    )
    return out.head(top_n)


# ---------------------------------------------------------------------------
# Coverage stats
# ---------------------------------------------------------------------------

def coverage_stats(
    channels_df: pd.DataFrame,
    remap:     Optional[Dict[str, str]] = None,
    ignore:    Optional[Set[str]] = None,
    canonical: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Per-subject coverage against the canonical channel set."""
    if channels_df is None or channels_df.empty:
        return pd.DataFrame(columns=['ID', 'N_present', 'N_canonical', 'Pct'])

    remap  = remap  or {}
    ignore = ignore or set()

    df2 = channels_df.copy()
    df2['CH'] = df2['CH'].map(lambda x: remap.get(x, x))
    df2 = df2[~df2['CH'].isin(ignore)]
    df2 = df2[df2['CH'].astype(str).str.strip() != '']

    if canonical is None:
        canonical = sorted(df2['CH'].unique())
    if not canonical:
        return pd.DataFrame(columns=['ID', 'N_present', 'N_canonical', 'Pct'])

    canon_set = set(canonical)
    n_total   = len(canonical)

    rows = []
    for id_str, grp in df2.groupby('ID'):
        n = len(set(grp['CH'].unique()) & canon_set)
        rows.append({
            'ID':          id_str,
            'N_present':   n,
            'N_canonical': n_total,
            'Pct':         round(100.0 * n / n_total, 1),
        })

    return pd.DataFrame(rows).sort_values('Pct', ignore_index=True)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def save_cache(path: str, scan: ScanResult):
    with open(path, 'wb') as f:
        pickle.dump({'magic': CACHE_MAGIC, 'scan': scan}, f, protocol=4)


def load_cache(path: str) -> ScanResult:
    with open(path, 'rb') as f:
        d = pickle.load(f)
    if d.get('magic') != CACHE_MAGIC:
        raise ValueError("Not a valid Harmonizer cache file")
    return d['scan']


# ---------------------------------------------------------------------------
# @param file export
# ---------------------------------------------------------------------------

def _quote(name: str) -> str:
    """Quote a name that contains spaces, pipes, or tabs."""
    if any(c in name for c in (' ', '|', '\t')):
        return f'"{name}"'
    return name


def write_param_file(
    path:       str,
    remap_ch:   Dict[str, str],
    ignore_ch:  Set[str],
    remap_ann:  Dict[str, str],
    ignore_ann: Set[str],
    sig_names:  Optional[List[str]] = None,
    annot_names: Optional[List[str]] = None,
):
    """Write a Luna @param file with signal aliases and annotation remaps.

    The Harmonizer stores remapping as original→canonical.  This function
    inverts signal remapping to the Luna alias format: canonical|orig1|orig2|…

    Output format:
        alias         EEG|EEG1|EEG 1
        remap         arousal|Arousal_RESP_ARO
        drop=         BAD_CH,ARTIFACT
        sig=          EEG,ECG,EOG
        annot=        arousal,leg_movement
    """
    lines = [
        "% Lunascope Harmonizer @param file",
        f"% Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    def _invert(remap: Dict[str, str]) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for orig, canon in remap.items():
            out.setdefault(canon, []).append(orig)
        return out

    ch_aliases  = _invert(remap_ch)
    sig_names = sorted(set(sig_names or []), key=str.lower)
    annot_names = sorted(set(annot_names or []), key=str.lower)

    if ch_aliases:
        lines.append("% Channel aliases  (format: alias<TAB>canonical|orig1|orig2|...)")
        for canon in sorted(ch_aliases):
            parts = [_quote(canon)] + [_quote(a) for a in sorted(ch_aliases[canon])]
            lines.append(f"alias\t{'|'.join(parts)}")
        lines.append("")

    if ignore_ch:
        lines.append("% Drop / ignore channels")
        lines.append(f"drop\t{','.join(_quote(c) for c in sorted(ignore_ch))}")
        lines.append("")

    if sig_names:
        lines.append("% Signal whitelist")
        lines.append(f"sig\t{','.join(_quote(c) for c in sig_names)}")
        lines.append("")

    if remap_ann:
        lines.append("% Annotation remaps  (format: remap<TAB>canonical|secondary)")
        for orig, canon in sorted(remap_ann.items(), key=lambda item: (str(item[1]).lower(), str(item[0]).lower())):
            lines.append(f"remap\t{_quote(canon)}|{_quote(orig)}")
        lines.append("")

    if ignore_ann:
        lines.append("% Drop / ignore annotations")
        lines.append(f"drop\t{','.join(_quote(c) for c in sorted(ignore_ann))}")
        lines.append("")

    if annot_names:
        lines.append("% Annotation whitelist")
        lines.append(f"annot\t{','.join(_quote(c) for c in annot_names)}")
        lines.append("")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
