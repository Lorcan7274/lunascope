
#  --------------------------------------------------------------------
#  Luna / Lunascope  —  Explorer: Hypnoscope tab
#  --------------------------------------------------------------------

"""Cohort hypnogram viewer.

Compiles staging annotations (N1/N2/N3/R/W/L/?) across all subjects,
renders an imshow-based grid (scales to thousands of records), and
provides save/load of the compiled cache so re-compilation is rare.
"""

import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from .explorer_base import BG, FG, GRID, _ExplorerTab


# ---------------------------------------------------------------------------
# Stage constants
# ---------------------------------------------------------------------------

STAGE_CLASSES = ["W", "N1", "N2", "N3", "R", "L", "?"]
STAGE_CODE    = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4, "L": 5, "?": 6}
STAGE_LABEL   = {"W": "Wake", "N1": "N1", "N2": "N2", "N3": "N3",
                 "R": "REM", "L": "Lights", "?": "Unknown"}
EPOCH_DUR     = 30.0        # seconds
MAX_SECS      = 30 * 3600   # hard trim at 30 h

# Match the stage colors used elsewhere in Lunascope.
# RGBA colours (float 0-1) keyed by integer code
STAGE_RGBA = {
    -1: (0.07, 0.07, 0.07, 0.6),   # no data
     0: (0.00, 0.50, 0.00, 1.0),   # W
     1: (32/255.0, 178/255.0, 218/255.0, 1.0),  # N1
     2: (0.00, 0.00, 1.00, 1.0),   # N2
     3: (0.00, 0.00, 128/255.0, 1.0),   # N3
     4: (1.00, 0.00, 0.00, 1.0),   # R
     5: (1.00, 1.00, 0.00, 1.0),   # L
     6: (0.50, 0.50, 0.50, 1.0),   # ?
}

# Pre-built lookup: index = code + 1  (handles code=-1 → index=0)
_STAGE_CMAP = np.array(
    [STAGE_RGBA[k] for k in range(-1, 7)], dtype=np.float32
)  # shape (8, 4)

# Compact epoch encoding for save/load
_CODE_TO_CHAR = {-1: ".", 0: "w", 1: "1", 2: "2", 3: "3", 4: "r", 5: "l", 6: "?"}
_CHAR_TO_CODE = {v: k for k, v in _CODE_TO_CHAR.items()}


# ---------------------------------------------------------------------------
# Save / load helpers
# ---------------------------------------------------------------------------

CACHE_HEADER  = "# lunascope-hypnoscope v1"
CACHE_COLUMNS = "ID\tSTART_TOD\tN_EPOCHS\tTST\tEFF\tSOL\tEPOCHS"


def save_hypnoscope_cache(path: str, subjects: list):
    """Write compiled staging data to a compact TSV cache file."""
    with open(path, "w") as fh:
        fh.write(f"{CACHE_HEADER}\n")
        fh.write(f"# Generated: {datetime.now():%Y-%m-%d %H:%M}\n")
        fh.write(f"{CACHE_COLUMNS}\n")
        for s in subjects:
            epoch_str = "".join(_CODE_TO_CHAR.get(int(c), ".") for c in s["epochs"])
            sol = f"{s['sol_secs']:.1f}" if s["sol_secs"] is not None else "NA"
            fh.write(
                f"{s['id']}\t{s['start_tod_secs']:.2f}\t{s['n_epochs']}\t"
                f"{s['tst_epochs']}\t{s['sleep_efficiency']:.6f}\t"
                f"{sol}\t{epoch_str}\n"
            )


def load_hypnoscope_cache(path: str) -> list:
    """Read a cache file produced by save_hypnoscope_cache. Returns subjects list."""
    subjects = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#") or line.startswith("ID\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 7:
                continue
            (id_str, start_tod, n_ep_str, tst_str,
             eff_str, sol_str, epoch_str) = parts[:7]
            sol = None if sol_str == "NA" else float(sol_str)
            epochs = np.array(
                [_CHAR_TO_CODE.get(c, -1) for c in epoch_str], dtype=np.int8
            )
            sol_ep = (int(round(float(sol_str) / EPOCH_DUR))
                      if sol_str != "NA" else None)
            subjects.append({
                "id":               id_str,
                "start_tod_secs":   float(start_tod),
                "epochs":           epochs,
                "n_epochs":         len(epochs),
                "duration_secs":    len(epochs) * EPOCH_DUR,
                "sleep_onset_ep":   sol_ep,
                "sol_secs":         sol,
                "tst_epochs":       int(tst_str),
                "sleep_efficiency": float(eff_str),
            })
    return subjects


# ---------------------------------------------------------------------------
# Compilation (background)
# ---------------------------------------------------------------------------

def _compile_hypnoscope(proj, ids: list) -> dict:
    """Load staging + EDF start-time for each subject. Thread-safe for one worker."""
    subjects = []
    for id_str in ids:
        try:
            p = proj.inst(id_str)
        except Exception as e:
            print(f"[Hypnoscope] skip {id_str!r}: {e}")
            continue

        # Start-of-recording clock time (seconds from midnight)
        start_tod = 0.0
        try:
            p.silent_proc("HEADERS")
            df_h = p.table("HEADERS")
            if df_h is not None and not df_h.empty:
                t_str = str(df_h["START_TIME"].iloc[0])
                parts = t_str.replace(":", ".").split(".")
                if len(parts) >= 3:
                    start_tod = (int(parts[0]) * 3600
                                 + int(parts[1]) * 60
                                 + int(parts[2]))
        except Exception:
            pass

        # Staging events
        try:
            ev = p.fetch_annots(STAGE_CLASSES, int(EPOCH_DUR))
        except Exception:
            ev = None

        if not isinstance(ev, pd.DataFrame) or ev.empty:
            continue

        # Normalise column names
        col_map = {}
        for col in ev.columns:
            lc = col.lower()
            if lc in ("class", "annotation"):
                col_map[col] = "Class"
            elif lc == "start":
                col_map[col] = "Start"
            elif lc in ("stop", "end"):
                col_map[col] = "Stop"
        if col_map:
            ev = ev.rename(columns=col_map)
        if not {"Class", "Start", "Stop"}.issubset(ev.columns):
            continue

        ev = ev[ev["Class"].isin(STAGE_CLASSES)].copy()
        ev["Start"] = pd.to_numeric(ev["Start"], errors="coerce")
        ev["Stop"]  = pd.to_numeric(ev["Stop"],  errors="coerce")
        ev = ev.dropna(subset=["Start", "Stop"])
        if ev.empty:
            continue

        ev = ev[ev["Start"] < MAX_SECS].copy()
        ev["Stop"] = ev["Stop"].clip(upper=MAX_SECS)

        duration  = float(ev["Stop"].max())
        n_epochs  = max(1, int(np.ceil(duration / EPOCH_DUR)))
        epochs    = np.full(n_epochs, -1, dtype=np.int8)

        # Vectorised fill
        for cls, code in STAGE_CODE.items():
            cls_ev = ev[ev["Class"] == cls]
            if cls_ev.empty:
                continue
            e0s = np.clip((cls_ev["Start"].values / EPOCH_DUR).astype(int), 0, n_epochs - 1)
            e1s = np.clip(np.ceil(cls_ev["Stop"].values / EPOCH_DUR).astype(int), 0, n_epochs)
            for b0, b1 in zip(e0s, e1s):
                if b0 < b1:
                    epochs[b0:b1] = code

        # Sleep onset (first NREM/REM epoch)
        sleep_ep = None
        for i, code in enumerate(epochs):
            if code in (1, 2, 3, 4):
                sleep_ep = i
                break

        sleep_eps = int(np.sum(np.isin(epochs, [1, 2, 3, 4])))
        valid_eps = int(np.sum(epochs >= 0))
        eff       = sleep_eps / valid_eps if valid_eps > 0 else 0.0
        sol_secs  = sleep_ep * EPOCH_DUR if sleep_ep is not None else None

        subjects.append({
            "id":               id_str,
            "start_tod_secs":   start_tod,
            "epochs":           epochs,
            "n_epochs":         n_epochs,
            "duration_secs":    duration,
            "sleep_onset_ep":   sleep_ep,
            "sol_secs":         sol_secs,
            "tst_epochs":       sleep_eps,
            "sleep_efficiency": eff,
        })

    return {"subjects": subjects}


# ---------------------------------------------------------------------------
# Tab widget
# ---------------------------------------------------------------------------

class HypnoscopeTab(_ExplorerTab):
    """Hypnoscope tab: cohort-level hypnogram visualisation."""

    _sig_ok  = QtCore.Signal(object)
    _sig_err = QtCore.Signal(str)

    _ALIGN_OPTS = [
        ("clock",   "Clock time"),
        ("elapsed", "Elapsed recording"),
        ("sleep",   "Elapsed sleep"),
    ]
    _SORT_OPTS = [
        ("alpha", "Alphabetical"),
        ("clock", "Clock start"),
        ("eff",   "Sleep efficiency ↓"),
        ("tst",   "TST ↓"),
        ("sol",   "Sleep-onset latency ↑"),
    ]

    def __init__(self, ctrl, parent=None):
        super().__init__(ctrl, parent)
        self._data: list | None = None   # list of subject dicts
        self._sig_ok.connect(self._on_ok,  Qt.QueuedConnection)
        self._sig_err.connect(self._on_err, Qt.QueuedConnection)
        self._build_widget()

    # ------------------------------------------------------------------
    # Widget
    # ------------------------------------------------------------------

    def _build_widget(self):
        root  = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 4, 6, 4); outer.setSpacing(4)

        row = QWidget(); rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(6)

        btn_compile = QPushButton("Compile All");  btn_compile.setFixedWidth(100)
        btn_load    = QPushButton("Load cache…");  btn_load.setFixedWidth(100)
        btn_save    = QPushButton("Save cache…");  btn_save.setFixedWidth(100)
        btn_export  = QPushButton("Export…");      btn_export.setFixedWidth(80)

        lbl_status = QLabel("No data")
        lbl_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lbl_status.setStyleSheet("color:#888;")

        combo_align = QComboBox(); combo_align.setMinimumWidth(140)
        for key, lbl in self._ALIGN_OPTS:
            combo_align.addItem(lbl, key)

        combo_sort = QComboBox(); combo_sort.setMinimumWidth(160)
        for key, lbl in self._SORT_OPTS:
            combo_sort.addItem(lbl, key)

        rl.addWidget(btn_compile); rl.addWidget(btn_load); rl.addWidget(btn_save)
        rl.addWidget(lbl_status, 1)
        rl.addWidget(QLabel("Align:")); rl.addWidget(combo_align)
        rl.addWidget(QLabel("Sort:"));  rl.addWidget(combo_sort)
        rl.addWidget(btn_export)

        canvas_host = QFrame()
        canvas_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas_host.setFrameShape(QFrame.NoFrame)
        canvas_host.setLayout(QVBoxLayout())
        canvas_host.layout().setContentsMargins(0, 0, 0, 0)
        self._canvas_host = canvas_host

        outer.addWidget(row); outer.addWidget(canvas_host, 1)

        self._root        = root
        self._lbl_status  = lbl_status
        self._combo_align = combo_align
        self._combo_sort  = combo_sort

        btn_compile.clicked.connect(self._compile)
        btn_load.clicked.connect(self._load_cache)
        btn_save.clicked.connect(self._save_cache)
        btn_export.clicked.connect(self._save_figure)
        combo_align.currentIndexChanged.connect(self._rerender)
        combo_sort.currentIndexChanged.connect(self._rerender)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_all_ids(self):
        try:
            df = self.ctrl.proj.sample_list()
            return [] if (df is None or df.empty) else df.iloc[:, 0].astype(str).tolist()
        except Exception:
            return []

    def _get_current_id(self):
        view = getattr(self.ctrl.ui, "tbl_slist", None)
        if view is None:
            return None
        idx = view.currentIndex()
        return idx.siblingAtColumn(0).data(Qt.DisplayRole) if idx.isValid() else None

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _compile(self):
        ids = self._get_all_ids()
        if not ids:
            QtWidgets.QMessageBox.warning(self._root, "Hypnoscope",
                                          "No subjects in the sample list.")
            return
        if not self._start_work(f"Compiling staging from {len(ids)} subjects…"):
            return
        self._saved_id = self._get_current_id()
        fut = self.ctrl._exec.submit(_compile_hypnoscope, self.ctrl.proj, ids)
        def _done(_f=fut):
            try:
                self._sig_ok.emit(_f.result())
            except Exception:
                self._sig_err.emit(traceback.format_exc())
        fut.add_done_callback(_done)

    # ------------------------------------------------------------------
    # Save / load cache
    # ------------------------------------------------------------------

    def _save_cache(self):
        if not self._data:
            QtWidgets.QMessageBox.warning(self._root, "Hypnoscope",
                                          "No data to save. Compile first.")
            return
        fn, _ = QFileDialog.getSaveFileName(
            self._root, "Save Hypnoscope Cache", "hypnoscope_cache.tsv",
            "Hypnoscope cache (*.tsv);;All files (*)"
        )
        if fn:
            try:
                save_hypnoscope_cache(fn, self._data)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self._root, "Save error", str(e))

    def _load_cache(self):
        fn, _ = QFileDialog.getOpenFileName(
            self._root, "Load Hypnoscope Cache", "",
            "Hypnoscope cache (*.tsv);;All files (*)"
        )
        if not fn:
            return
        try:
            subjects = load_hypnoscope_cache(fn)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self._root, "Load error", str(e))
            return
        self._data = subjects
        self._lbl_status.setStyleSheet("color:#06d6a0;")
        self._lbl_status.setText(f"{len(subjects)} subjects loaded from cache")
        self._rerender()

    # ------------------------------------------------------------------
    # Done callbacks
    # ------------------------------------------------------------------

    def _on_ok(self, result):
        try:
            subjects = result.get("subjects", [])
            self._data = subjects
            n = len(subjects)
            self._lbl_status.setStyleSheet("color:#06d6a0;")
            self._lbl_status.setText(f"{n} subjects compiled")
            saved = getattr(self, "_saved_id", None)
            if saved:
                try:
                    self.ctrl.p = self.ctrl.proj.inst(saved)
                except Exception:
                    pass
            self._rerender()
        finally:
            self._end_work()

    def _on_err(self, tb_str):
        try:
            QtWidgets.QMessageBox.critical(
                self._root, "Hypnoscope compile error", tb_str[:800])
        finally:
            self._end_work()

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _rerender(self, *_):
        if self._data is None:
            return
        self._render(self._data,
                     self._combo_align.currentData(),
                     self._combo_sort.currentData())

    def _render(self, all_subjects: list, align: str, sort_by: str):
        if not all_subjects:
            self._render_empty("No staging data found.\n\n"
                               "Staging annotations (N1/N2/N3/R/W) must exist for "
                               "subjects in the sample list.")
            return

        # Filter to subjects with sleep data (for sleep alignment)
        if align == "sleep":
            subjects = [s for s in all_subjects if s["sleep_onset_ep"] is not None]
            if not subjects:
                self._render_empty("No subjects with a detectable sleep onset.")
                return
        else:
            subjects = all_subjects

        # Sort
        key_fn = {
            "alpha": lambda s: s["id"],
            "clock": lambda s: s["start_tod_secs"],
            "eff":   lambda s: -s["sleep_efficiency"],
            "tst":   lambda s: -s["tst_epochs"],
            "sol":   lambda s: (s["sol_secs"] if s["sol_secs"] is not None else 1e9),
        }.get(sort_by, lambda s: s["id"])
        subjects = sorted(subjects, key=key_fn)
        if sort_by == "clock":
            # Rows render bottom→top because imshow uses origin="lower".
            # Reverse the clock-order list so the earliest start appears at the top.
            subjects.reverse()
        n_subj   = len(subjects)

        EPOCH_H  = EPOCH_DUR / 3600.0  # epoch width in hours

        # ---- build per-subject x_start (hours) -------------------------
        if align == "clock":
            x_starts = [s["start_tod_secs"] / 3600.0 for s in subjects]
        elif align == "elapsed":
            x_starts = [0.0] * n_subj
        else:  # sleep
            x_starts = [-s["sleep_onset_ep"] * EPOCH_H for s in subjects]

        x_min = min(x_starts)
        x_max = max(
            xs + s["duration_secs"] / 3600.0
            for xs, s in zip(x_starts, subjects)
        )

        # Cap columns to avoid excessive memory (1 col = 1 epoch = 30 s)
        n_cols = max(1, min(int(np.ceil((x_max - x_min) / EPOCH_H)), 14400))
        x_max  = x_min + n_cols * EPOCH_H

        # ---- fill image array ------------------------------------------
        # img[row, col] = stage code (-1..6)
        img = np.full((n_subj, n_cols), -1, dtype=np.int8)

        for row_idx, (subj, x_start) in enumerate(zip(subjects, x_starts)):
            epochs    = subj["epochs"]
            col_start = int(round((x_start - x_min) / EPOCH_H))
            n_ep      = len(epochs)
            c0        = max(col_start, 0)
            e0        = c0 - col_start          # epoch index offset (if col_start < 0)
            n_fill    = min(n_ep - max(e0, 0), n_cols - c0)
            if n_fill > 0:
                img[row_idx, c0 : c0 + n_fill] = epochs[max(e0, 0) : max(e0, 0) + n_fill]

        # ---- vectorised RGBA lookup ------------------------------------
        # (img + 1) maps -1..6 → 0..7 as index into _STAGE_CMAP
        rgba_img = _STAGE_CMAP[(img.astype(np.int16) + 1)]   # (n_subj, n_cols, 4)

        # ---- draw ------------------------------------------------------
        canvas = self._ensure_canvas()
        fig    = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)

        ax.imshow(
            rgba_img,
            aspect="auto",
            interpolation="nearest",
            extent=[x_min, x_max, -0.5, n_subj - 0.5],
            origin="lower",
        )

        # ---- Y axis (adaptive) ----------------------------------------
        ax.set_ylim(-0.5, n_subj - 0.5)
        ax.tick_params(axis="y", length=0, colors=FG)

        if n_subj <= 50:
            ax.set_yticks(range(n_subj))
            fs = max(4.5, min(8.5, 350.0 / n_subj))
            ax.set_yticklabels(
                [s["id"][:16] + "…" if len(s["id"]) > 17 else s["id"]
                 for s in subjects],
                fontsize=fs, color=FG,
            )
        elif n_subj <= 300:
            step = max(1, n_subj // 20)
            ticks = list(range(0, n_subj, step))
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(t) for t in ticks], fontsize=6.5, color=FG)
        else:
            # Too many rows for labels — just show count
            ax.set_yticks([])

        # ---- X axis ---------------------------------------------------
        ax.tick_params(axis="x", colors=FG, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)

        if align == "clock":
            import matplotlib.ticker as mticker
            ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda h, _: f"{int(h) % 24:02d}:00"))
            ax.set_xlabel("Clock time", color=FG, fontsize=9)
            # Midnight marker
            midnights = [v for v in np.arange(0, 72, 24) if x_min <= v <= x_max]
            for mn in midnights:
                ax.axvline(mn, color=GRID, lw=0.8, ls="--", alpha=0.7)
        elif align == "elapsed":
            ax.set_xlabel("Elapsed time (h)", color=FG, fontsize=9)
        else:
            ax.axvline(0.0, color="#ffffff", lw=0.8, ls="--", alpha=0.55)
            ax.set_xlabel("Time from sleep onset (h)", color=FG, fontsize=9)

        # Sleep-onset markers (vertical bars at each row's onset x)
        if align != "sleep":
            for row_idx, (subj, x_start) in enumerate(zip(subjects, x_starts)):
                sol_ep = subj["sleep_onset_ep"]
                if sol_ep is None:
                    continue
                x_sol = x_start + sol_ep * EPOCH_H
                ax.plot(x_sol, row_idx, marker="|", color="#ffffff",
                        markersize=6, markeredgewidth=0.8, alpha=0.6)

        # ---- Title ----------------------------------------------------
        align_label = {"clock": "clock time", "elapsed": "elapsed time",
                       "sleep": "sleep onset"}.get(align, align)
        ax.set_title(
            f"Hypnoscope — {n_subj} subjects  |  aligned by {align_label}",
            color=FG, fontsize=10, pad=6,
        )

        fig.subplots_adjust(left=0.10, right=0.98, top=0.95, bottom=0.07)
        canvas.draw()
