
#  --------------------------------------------------------------------
#  Luna / Lunascope  —  Explorer: Output Plotter tab
#  --------------------------------------------------------------------

"""Generic plotter for tables in the Outputs dock.

Reads from ctrl.results (dict of {key: DataFrame}) and provides
scatter / line / bar / histogram / box viewer with two independent
grouping variables (Group 1 and Group 2), each switchable between
Overlay (colour / marker coding) and Separate (sub-plot panels).

The four mode combinations for two groups:
  ov  × ov   → single panel, G1=colour, G2=marker/linestyle
  sep × ov   → G1-panels, G2 colour-coded inside each
  ov  × sep  → G2-panels, G1 colour-coded inside each
  sep × sep  → G1 × G2 grid of panels
"""

from math import ceil

import numpy as np
import pandas as pd

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
    QFileDialog,
)

from .explorer_base import BG, FG, GRID, _ExplorerTab


# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

_PALETTE = [
    "#4cc9f0", "#f9844a", "#06d6a0", "#a78bfa",
    "#ffd166", "#f72585", "#90be6d", "#ff6b6b",
    "#43aa8b", "#577590", "#c77dff", "#fb8500",
]
_MARKERS    = ["o", "s", "^", "D", "v", "P", "X", "*"]
_LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
_HATCHES    = ["", "///", "...", "xxx", "\\\\\\", "+++"]

_MAX_OVERLAY = 8
_MAX_PANELS  = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_plot_type(df, xcol, ycol):
    if xcol not in df.columns or ycol not in df.columns:
        return "scatter"
    x_num    = pd.api.types.is_numeric_dtype(df[xcol])
    n_uniq_x = df[xcol].nunique()
    if not x_num or n_uniq_x <= 20:
        return "bar"
    if x_num and pd.api.types.is_numeric_dtype(df[ycol]) and n_uniq_x < len(df) * 0.6:
        return "line"
    return "scatter"


def _table_display_name(key):
    if not key or "_" not in key:
        return key
    head, *tail = str(key).split("_")
    return f"{head} : {' x '.join(tail)}" if tail else str(key)


def _combo_label(*vals):
    """Join non-None group values into a legend label."""
    return " / ".join(str(v) for v in vals if v is not None)


# ---------------------------------------------------------------------------
# PlotterTab
# ---------------------------------------------------------------------------

class PlotterTab(_ExplorerTab):
    """Output plotter tab with two-variable grouping."""

    def __init__(self, ctrl, parent=None):
        super().__init__(ctrl, parent)
        self._df: pd.DataFrame | None = None
        self._aux_df: pd.DataFrame | None = None   # uploaded covariate file
        self._aux_path: str = ""
        self._plot_timer = QTimer(self)
        self._plot_timer.setSingleShot(True)
        self._plot_timer.setInterval(300)
        self._plot_timer.timeout.connect(self._plot)
        self._build_widget()

    # ------------------------------------------------------------------
    # Widget
    # ------------------------------------------------------------------

    def _build_widget(self):
        root  = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 4, 6, 4); outer.setSpacing(4)

        # ---- row 1: table selector ------------------------------------
        row1 = QWidget(); rl1 = QHBoxLayout(row1)
        rl1.setContentsMargins(0, 0, 0, 0); rl1.setSpacing(6)

        btn_refresh = QPushButton("↻"); btn_refresh.setFixedWidth(30)
        btn_refresh.setToolTip("Reload available tables from the Outputs dock")
        combo_table = QComboBox()
        combo_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lbl_shape = QLabel("")
        lbl_shape.setStyleSheet("color:#888; font-size:11px;")

        rl1.addWidget(QLabel("Table:")); rl1.addWidget(combo_table, 1)
        rl1.addWidget(lbl_shape); rl1.addWidget(btn_refresh)

        # ---- row 1b: covariate file -----------------------------------
        row1b = QWidget(); rl1b = QHBoxLayout(row1b)
        rl1b.setContentsMargins(0, 0, 0, 0); rl1b.setSpacing(6)

        btn_load_cov  = QPushButton("Load covariates…"); btn_load_cov.setFixedWidth(140)
        btn_load_cov.setToolTip("Upload a TSV/CSV file with an ID column to merge as covariates")
        btn_clear_cov = QPushButton("✕"); btn_clear_cov.setFixedWidth(26)
        btn_clear_cov.setToolTip("Remove loaded covariate file")
        lbl_cov_file  = QLabel("(none)")
        lbl_cov_file.setStyleSheet("color:#888; font-size:11px;")
        lbl_cov_file.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        rl1b.addWidget(QLabel("Covariates:")); rl1b.addWidget(btn_load_cov)
        rl1b.addWidget(lbl_cov_file, 1); rl1b.addWidget(btn_clear_cov)

        # ---- row 2: axes + plot type ----------------------------------
        row2 = QWidget(); rl2 = QHBoxLayout(row2)
        rl2.setContentsMargins(0, 0, 0, 0); rl2.setSpacing(6)

        combo_x = QComboBox(); combo_x.setMinimumWidth(100)
        combo_y = QComboBox(); combo_y.setMinimumWidth(100)
        combo_type = QComboBox(); combo_type.setFixedWidth(90)
        for key, lbl in [("auto", "Auto"), ("scatter", "Scatter"),
                          ("line", "Line"), ("bar", "Bar"),
                          ("hist", "Histogram"), ("box", "Box")]:
            combo_type.addItem(lbl, key)
        chk_log_x  = QCheckBox("log X")
        chk_log_y  = QCheckBox("log Y")
        btn_export = QPushButton("Export…"); btn_export.setFixedWidth(80)

        rl2.addWidget(QLabel("X:")); rl2.addWidget(combo_x)
        rl2.addWidget(QLabel("Y:")); rl2.addWidget(combo_y)
        rl2.addWidget(QLabel("Type:")); rl2.addWidget(combo_type)
        rl2.addWidget(chk_log_x); rl2.addWidget(chk_log_y)
        rl2.addStretch(1); rl2.addWidget(btn_export)

        # ---- row 3: group selectors -----------------------------------
        row3 = QWidget(); rl3 = QHBoxLayout(row3)
        rl3.setContentsMargins(0, 0, 0, 0); rl3.setSpacing(6)

        def _group_combo():
            c = QComboBox()
            c.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            c.setMinimumWidth(100)
            c.addItem("(none)", None)
            return c

        def _mode_combo():
            c = QComboBox(); c.setFixedWidth(90)
            c.addItem("Overlay",  "overlay")
            c.addItem("Separate", "separate")
            return c

        combo_group1 = _group_combo(); combo_mode1 = _mode_combo()
        combo_group2 = _group_combo(); combo_mode2 = _mode_combo()

        rl3.addWidget(QLabel("Group 1:")); rl3.addWidget(combo_group1, 1)
        rl3.addWidget(combo_mode1)
        rl3.addWidget(QLabel("Group 2:")); rl3.addWidget(combo_group2, 1)
        rl3.addWidget(combo_mode2)
        rl3.addStretch(1)

        # ---- canvas ---------------------------------------------------
        canvas_host = QFrame()
        canvas_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas_host.setFrameShape(QFrame.NoFrame)
        canvas_host.setLayout(QVBoxLayout())
        canvas_host.layout().setContentsMargins(0, 0, 0, 0)
        self._canvas_host = canvas_host

        outer.addWidget(row1); outer.addWidget(row1b)
        outer.addWidget(row2); outer.addWidget(row3)
        outer.addWidget(canvas_host, 1)

        # ---- store refs -----------------------------------------------
        self._root         = root
        self._combo_table  = combo_table
        self._combo_x      = combo_x
        self._combo_y      = combo_y
        self._combo_group1 = combo_group1
        self._combo_mode1  = combo_mode1
        self._combo_group2 = combo_group2
        self._combo_mode2  = combo_mode2
        self._combo_type   = combo_type
        self._chk_log_x    = chk_log_x
        self._chk_log_y    = chk_log_y
        self._lbl_shape    = lbl_shape
        self._lbl_cov_file = lbl_cov_file

        # ---- wire signals ---------------------------------------------
        btn_refresh.clicked.connect(self.refresh_tables)
        btn_load_cov.clicked.connect(self._load_aux_file)
        btn_clear_cov.clicked.connect(self._clear_aux_file)
        combo_table.currentIndexChanged.connect(self._on_table_changed)
        btn_export.clicked.connect(self._save_figure)
        for w in (combo_x, combo_y, combo_group1, combo_mode1,
                  combo_group2, combo_mode2, combo_type):
            w.currentIndexChanged.connect(self._schedule_plot)
        chk_log_x.stateChanged.connect(self._schedule_plot)
        chk_log_y.stateChanged.connect(self._schedule_plot)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def refresh_tables(self):
        """Populate the table combo from ctrl.results."""
        results = getattr(self.ctrl, "results", None) or {}
        cur = self._combo_table.currentData()
        self._combo_table.blockSignals(True)
        self._combo_table.clear()
        for key in sorted(results.keys()):
            self._combo_table.addItem(_table_display_name(key), key)
        idx = self._combo_table.findData(cur)
        if idx >= 0:
            self._combo_table.setCurrentIndex(idx)
        self._combo_table.blockSignals(False)
        self._on_table_changed()

    def _on_table_changed(self, *_):
        key     = self._combo_table.currentData()
        results = getattr(self.ctrl, "results", None) or {}
        df      = results.get(key) if key else None
        self._df = df if isinstance(df, pd.DataFrame) and not df.empty else None

        for c in (self._combo_x, self._combo_y,
                  self._combo_group1, self._combo_group2):
            c.blockSignals(True); c.clear()
        self._combo_group1.addItem("(none)", None)
        self._combo_group2.addItem("(none)", None)

        if self._df is not None:
            eff_df, aux_cols = self._get_effective_df()
            aux_set = set(aux_cols)
            cols = list(eff_df.columns)
            for col in cols:
                label = f"{col} [cov]" if col in aux_set else col
                self._combo_x.addItem(label, col)
                self._combo_y.addItem(label, col)
                self._combo_group1.addItem(label, col)
                self._combo_group2.addItem(label, col)
            n_aux = len(aux_cols)
            shape_txt = f"{len(self._df)} rows × {len(self._df.columns)} cols"
            if n_aux:
                shape_txt += f"  +{n_aux} cov"
            self._lbl_shape.setText(shape_txt)
            # Default: pick first two numeric cols from main table for X/Y
            num_cols = [c for c in self._df.columns if pd.api.types.is_numeric_dtype(self._df[c])]
            if num_cols:
                self._combo_x.setCurrentIndex(
                    self._combo_x.findData(num_cols[0]))
                self._combo_y.setCurrentIndex(
                    self._combo_y.findData(num_cols[min(1, len(num_cols) - 1)]))
        else:
            self._lbl_shape.setText("")

        for c in (self._combo_x, self._combo_y,
                  self._combo_group1, self._combo_group2):
            c.blockSignals(False)

        self._schedule_plot()

    # ------------------------------------------------------------------
    # Covariate file
    # ------------------------------------------------------------------

    def _load_aux_file(self):
        fn, _ = QFileDialog.getOpenFileName(
            self._root, "Load Covariate File", "",
            "Tabular files (*.tsv *.csv *.txt);;All files (*)"
        )
        if not fn:
            return
        try:
            sep = "\t" if fn.lower().endswith(".tsv") or fn.lower().endswith(".txt") else ","
            df = pd.read_csv(fn, sep=sep, dtype=str)
            # Try comma if TSV parse yielded one column (might actually be CSV)
            if len(df.columns) == 1:
                df = pd.read_csv(fn, sep=",", dtype=str)
            # Normalise NA representations
            df.replace(["NA", "na", "N/A", "n/a", ".", ""], np.nan, inplace=True)
            # Require ID column (case-insensitive)
            id_col = next((c for c in df.columns if c.strip().upper() == "ID"), None)
            if id_col is None:
                QtWidgets.QMessageBox.warning(
                    self._root, "Covariates",
                    "File must contain a column named 'ID'."
                )
                return
            # Normalise column name to 'ID'
            if id_col != "ID":
                df = df.rename(columns={id_col: "ID"})
            # Coerce numeric columns where possible
            for col in df.columns:
                if col == "ID":
                    continue
                coerced = pd.to_numeric(df[col], errors="coerce")
                if coerced.notna().any():
                    df[col] = coerced
            self._aux_df   = df
            self._aux_path = fn
            import os
            self._lbl_cov_file.setText(os.path.basename(fn) +
                                       f"  ({len(df)} rows, {len(df.columns)-1} covariate cols)")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self._root, "Covariates", f"Could not load file:\n{e}")
            return
        # Refresh combos with new columns available
        self._on_table_changed()

    def _clear_aux_file(self):
        self._aux_df   = None
        self._aux_path = ""
        self._lbl_cov_file.setText("(none)")
        self._on_table_changed()

    def _get_effective_df(self):
        """Return (merged_df, aux_col_names).

        If a covariate file is loaded and the main table has an ID column,
        returns a left-joined DataFrame and the list of covariate column names
        that were added.  Otherwise returns (self._df, []).
        """
        df = self._df
        if df is None:
            return None, []
        if self._aux_df is None:
            return df, []
        # Locate ID column in main table (case-insensitive)
        id_col = next((c for c in df.columns if c.strip().upper() == "ID"), None)
        if id_col is None:
            return df, []
        # Aux columns to add (all except ID)
        aux_cols = [c for c in self._aux_df.columns if c != "ID"]
        # For conflict resolution: suffix _cov for cols already in main df
        existing = set(df.columns)
        rename_map = {}
        for c in aux_cols:
            if c in existing:
                rename_map[c] = c + "_cov"
        aux = self._aux_df.rename(columns=rename_map)
        merged_aux_cols = [rename_map.get(c, c) for c in aux_cols]
        try:
            # Coerce both ID columns to stripped strings so int/str mismatches join correctly
            df_m = df.copy()
            df_m[id_col] = df_m[id_col].astype(str).str.strip()
            aux_m = aux.copy()
            aux_m["ID"] = aux_m["ID"].astype(str).str.strip()
            merged = pd.merge(df_m, aux_m, left_on=id_col, right_on="ID", how="left")
            # Drop duplicate ID col if main df id_col != 'ID'
            if id_col != "ID" and "ID" in merged.columns:
                merged = merged.drop(columns=["ID"])
        except Exception:
            return df, []
        return merged, merged_aux_cols

    # ------------------------------------------------------------------
    # Plot dispatch
    # ------------------------------------------------------------------

    def _schedule_plot(self, *_):
        if self._df is not None or self._aux_df is not None:
            self._plot_timer.start()

    def _plot(self):
        df, _ = self._get_effective_df()
        if df is None:
            return

        xcol  = self._combo_x.currentData() or self._combo_x.currentText()
        ycol  = self._combo_y.currentData() or self._combo_y.currentText()
        gcol1 = self._combo_group1.currentData() or None
        gcol2 = self._combo_group2.currentData() or None
        mode1 = self._combo_mode1.currentData() or "overlay"
        mode2 = self._combo_mode2.currentData() or "overlay"
        ptype = self._combo_type.currentData()
        log_x = self._chk_log_x.isChecked()
        log_y = self._chk_log_y.isChecked()

        if not xcol or not ycol:
            return
        if ptype == "auto":
            ptype = _auto_plot_type(df, xcol, ycol)

        groups1 = self._group_levels(df, gcol1)
        groups2 = self._group_levels(df, gcol2)

        # Validate limits
        n1_sep = len(groups1) if (groups1 and mode1 == "separate") else 0
        n2_sep = len(groups2) if (groups2 and mode2 == "separate") else 0
        n1_ov  = len(groups1) if (groups1 and mode1 == "overlay")  else 0
        n2_ov  = len(groups2) if (groups2 and mode2 == "overlay")  else 0

        if max(n1_sep, 1) * max(n2_sep, 1) > _MAX_PANELS:
            QtWidgets.QMessageBox.information(
                self._root, "Plotter",
                f"Too many panels ({max(n1_sep,1) * max(n2_sep,1)}). "
                f"Reduce groups or switch to Overlay.")
            return
        for n, col, tag in [(n1_ov, gcol1, "Group 1"), (n2_ov, gcol2, "Group 2")]:
            if n > _MAX_OVERLAY:
                QtWidgets.QMessageBox.information(
                    self._root, "Plotter",
                    f"{tag} '{col}' has {n} levels — overlay limit is {_MAX_OVERLAY}.")
                return

        canvas = self._ensure_canvas()
        fig    = canvas.figure
        fig.clear(); fig.patch.set_facecolor(BG)

        ctx = self._make_plot_context(fig, groups1, mode1, groups2, mode2, gcol1, gcol2)

        try:
            if ptype == "hist":
                self._plot_hist(ctx, df, xcol, log_x)
            elif ptype == "bar":
                self._plot_bar(ctx, df, xcol, ycol)
            elif ptype == "box":
                self._plot_box(ctx, df, xcol, ycol)
            elif ptype == "line":
                self._plot_line(ctx, df, xcol, ycol)
            else:
                self._plot_scatter(ctx, df, xcol, ycol)

            for ax in fig.axes:
                if log_x and ptype not in ("hist", "bar", "box"):
                    ax.set_xscale("log")
                if log_y and ptype not in ("hist", "bar", "box"):
                    ax.set_yscale("log")
                ax.set_xlabel(xcol, color=FG, fontsize=9)
                if ptype != "hist":
                    ax.set_ylabel(ycol, color=FG, fontsize=9)
                ax.tick_params(colors=FG, labelsize=8)
                for sp in ax.spines.values():
                    sp.set_edgecolor(GRID)

            tbl_key = self._combo_table.currentText()
            all_axes = fig.axes
            if len(all_axes) == 1:
                all_axes[0].set_title(tbl_key, color=FG, fontsize=9, pad=6)
            else:
                fig.suptitle(tbl_key, color=FG, fontsize=9, y=0.99)

            fig.subplots_adjust(left=0.12, right=0.97, top=0.92, bottom=0.12,
                                hspace=0.42, wspace=0.35)
            canvas.draw()
        except Exception:
            fig.clear()
            fig.patch.set_facecolor(BG)
            ax = fig.add_subplot(111); ax.set_facecolor(BG)
            ax.text(0.5, 0.5, "Selected inputs cannot be plotted", color=FG,
                    ha="center", va="center", fontsize=9, transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw()

    # ------------------------------------------------------------------
    # Context builder
    # ------------------------------------------------------------------

    def _make_plot_context(self, fig, groups1, mode1, groups2, mode2, gcol1, gcol2):
        """Build axes grid and encoding callables for the two-group layout.

        Returns a context dict consumed by the individual plot methods.
        """
        sep1 = bool(groups1) and mode1 == "separate"
        sep2 = bool(groups2) and mode2 == "separate"
        ov1  = bool(groups1) and mode1 == "overlay"
        ov2  = bool(groups2) and mode2 == "overlay"
        n1   = len(groups1) if groups1 else 0
        n2   = len(groups2) if groups2 else 0

        n_rows = n1 if sep1 else 1
        n_cols = n2 if sep2 else 1

        if n_rows == 1 and n_cols == 1:
            single = fig.add_subplot(111); single.set_facecolor(BG)
            _grid  = [[single]]
        else:
            _grid = fig.subplots(n_rows, n_cols, squeeze=False)
            for r in range(n_rows):
                for c in range(n_cols):
                    _grid[r][c].set_facecolor(BG)
                    parts = []
                    if sep1 and groups1: parts.append(str(groups1[r]))
                    if sep2 and groups2: parts.append(str(groups2[c]))
                    if parts:
                        _grid[r][c].set_title(
                            " / ".join(parts), color=FG, fontsize=8, pad=4)

        def ax_for(g1i=0, g2i=0):
            return _grid[g1i if sep1 else 0][g2i if sep2 else 0]

        def color_for(g1i=0, g2i=0):
            if ov1: return _PALETTE[g1i % len(_PALETTE)]
            if ov2: return _PALETTE[g2i % len(_PALETTE)]
            return _PALETTE[0]

        def marker_for(g1i=0, g2i=0):
            # G2 drives marker when both overlay (G1 already drives colour)
            if ov2: return _MARKERS[g2i % len(_MARKERS)]
            return "o"

        def ls_for(g1i=0, g2i=0):
            if ov2: return _LINESTYLES[g2i % len(_LINESTYLES)]
            return "-"

        def hatch_for(g1i=0, g2i=0):
            if ov2: return _HATCHES[g2i % len(_HATCHES)]
            return ""

        # Bar-width offset helpers ----------------------------------------
        # Within a single panel the number of bars per x-category depends on
        # which group dimensions are in overlay mode.
        if ov1 and ov2:
            _n_bars = n1 * n2
            def bar_offset(g1i, g2i):
                idx = g1i * n2 + g2i
                w   = 0.8 / _n_bars
                return idx, w, _n_bars
        elif ov1:
            _n_bars = n1
            def bar_offset(g1i, g2i):
                w = 0.8 / _n_bars
                return g1i, w, _n_bars
        elif ov2:
            _n_bars = n2
            def bar_offset(g1i, g2i):
                w = 0.8 / _n_bars
                return g2i, w, _n_bars
        else:
            def bar_offset(g1i, g2i):
                return 0, 0.8, 1

        return dict(
            ax_for=ax_for, color_for=color_for, marker_for=marker_for,
            ls_for=ls_for, hatch_for=hatch_for, bar_offset=bar_offset,
            groups1=groups1, groups2=groups2,
            gcol1=gcol1, gcol2=gcol2,
            sep1=sep1, sep2=sep2, ov1=ov1, ov2=ov2,
        )

    # ------------------------------------------------------------------
    # Legend helper
    # ------------------------------------------------------------------

    @staticmethod
    def _add_legend(ctx, ax):
        """Add colour/marker legend for any overlay groups on *ax*."""
        from matplotlib.lines import Line2D
        ov1, ov2 = ctx["ov1"], ctx["ov2"]
        groups1, groups2 = ctx["groups1"], ctx["groups2"]
        if not ov1 and not ov2:
            return
        handles = []
        if ov1:
            for i, g in enumerate(groups1):
                handles.append(Line2D([0], [0], color=_PALETTE[i % len(_PALETTE)],
                                      linewidth=2, label=str(g)))
        if ov2:
            for i, g in enumerate(groups2):
                mkr = _MARKERS[i % len(_MARKERS)]
                col = "#cccccc" if ov1 else _PALETTE[i % len(_PALETTE)]
                handles.append(Line2D([0], [0], marker=mkr, color="none",
                                      markerfacecolor=col, markersize=6,
                                      label=str(g)))
        if handles:
            leg = ax.legend(handles=handles, fontsize=8,
                            framealpha=0.3, facecolor="#1a1a1a", edgecolor=GRID)
            for t in leg.get_texts():
                t.set_color(FG)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _group_levels(self, df, gcol):
        if not gcol or gcol not in df.columns:
            return None
        return [g for g in pd.unique(df[gcol]) if not pd.isna(g)]

    def _iter_groups(self, ctx):
        """Yield (g1i, g1v, g2i, g2v) for every (group1, group2) pair."""
        g1_seq = list(enumerate(ctx["groups1"])) if ctx["groups1"] else [(0, None)]
        g2_seq = list(enumerate(ctx["groups2"])) if ctx["groups2"] else [(0, None)]
        for g1i, g1v in g1_seq:
            for g2i, g2v in g2_seq:
                yield g1i, g1v, g2i, g2v

    def _subset(self, df, ctx, g1v, g2v):
        sub = df
        if g1v is not None: sub = sub[sub[ctx["gcol1"]] == g1v]
        if g2v is not None: sub = sub[sub[ctx["gcol2"]] == g2v]
        return sub

    # ------------------------------------------------------------------
    # Scatter
    # ------------------------------------------------------------------

    def _plot_scatter(self, ctx, df, xcol, ycol):
        used = set()
        for g1i, g1v, g2i, g2v in self._iter_groups(ctx):
            sub = self._subset(df, ctx, g1v, g2v)
            if sub.empty: continue
            ax  = ctx["ax_for"](g1i, g2i)
            ax.scatter(sub[xcol], sub[ycol],
                       color=ctx["color_for"](g1i, g2i),
                       marker=ctx["marker_for"](g1i, g2i),
                       s=18, alpha=0.75, linewidths=0,
                       label=_combo_label(g1v, g2v))
            used.add(ax)
        for ax in used:
            self._add_legend(ctx, ax)

    # ------------------------------------------------------------------
    # Line
    # ------------------------------------------------------------------

    def _plot_line(self, ctx, df, xcol, ycol):
        used = set()
        for g1i, g1v, g2i, g2v in self._iter_groups(ctx):
            sub = self._subset(df, ctx, g1v, g2v)
            if sub.empty: continue
            sub = sub.sort_values(xcol)
            ax  = ctx["ax_for"](g1i, g2i)
            ax.plot(sub[xcol], sub[ycol],
                    color=ctx["color_for"](g1i, g2i),
                    linestyle=ctx["ls_for"](g1i, g2i),
                    marker=ctx["marker_for"](g1i, g2i),
                    lw=1.5, markersize=3,
                    label=_combo_label(g1v, g2v))
            used.add(ax)
        for ax in used:
            self._add_legend(ctx, ax)

    # ------------------------------------------------------------------
    # Bar  (mean ± SE per x-category)
    # ------------------------------------------------------------------

    def _plot_bar(self, ctx, df, xcol, ycol):
        used = set()
        for g1i, g1v, g2i, g2v in self._iter_groups(ctx):
            sub = self._subset(df, ctx, g1v, g2v)
            if sub.empty: continue
            ax  = ctx["ax_for"](g1i, g2i)
            col = ctx["color_for"](g1i, g2i)
            htc = ctx["hatch_for"](g1i, g2i)
            bar_idx, bar_w, _ = ctx["bar_offset"](g1i, g2i)

            agg = (sub.groupby(xcol)[ycol]
                      .agg(["mean", "sem"])
                      .reset_index())
            x_vals = sorted(agg[xcol].unique(), key=str)
            x_locs = np.arange(len(x_vals))
            n_bars = int(round(0.8 / bar_w))
            offset = (bar_idx - (n_bars - 1) / 2.0) * bar_w

            means = [agg[agg[xcol] == xv]["mean"].values[0]
                     if xv in agg[xcol].values else np.nan for xv in x_vals]
            sems  = [agg[agg[xcol] == xv]["sem"].values[0]
                     if xv in agg[xcol].values else 0   for xv in x_vals]

            ax.bar(x_locs + offset, means, width=bar_w,
                   color=col, alpha=0.85, hatch=htc,
                   label=_combo_label(g1v, g2v),
                   yerr=sems, capsize=3,
                   error_kw={"ecolor": FG, "elinewidth": 0.8})
            ax.set_xticks(x_locs)
            ax.set_xticklabels([str(v) for v in x_vals],
                               rotation=30, ha="right", fontsize=7, color=FG)
            used.add(ax)
        for ax in used:
            self._add_legend(ctx, ax)

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------

    def _plot_hist(self, ctx, df, xcol, log_x):
        is_num = pd.api.types.is_numeric_dtype(df[xcol])
        used   = set()
        for g1i, g1v, g2i, g2v in self._iter_groups(ctx):
            sub = self._subset(df, ctx, g1v, g2v)
            if sub.empty: continue
            ax  = ctx["ax_for"](g1i, g2i)
            col = ctx["color_for"](g1i, g2i)
            lbl = _combo_label(g1v, g2v)

            if not is_num:
                counts = sub[xcol].value_counts()
                ax.bar(range(len(counts)), counts.values,
                       color=col, alpha=0.65, label=lbl)
                ax.set_xticks(range(len(counts)))
                ax.set_xticklabels(counts.index.astype(str),
                                   rotation=30, ha="right", fontsize=7, color=FG)
            else:
                vals = sub[xcol].dropna()
                if log_x: vals = vals[vals > 0]
                if vals.empty: continue
                bins = (np.logspace(np.log10(vals.min()), np.log10(vals.max()), 40)
                        if log_x else 40)
                ax.hist(vals, bins=bins, color=col, alpha=0.5, edgecolor="none",
                        label=lbl)
            ax.set_ylabel("Count", color=FG, fontsize=9)
            used.add(ax)
        for ax in used:
            self._add_legend(ctx, ax)

    # ------------------------------------------------------------------
    # Box
    # ------------------------------------------------------------------

    def _plot_box(self, ctx, df, xcol, ycol):
        used = set()
        for g1i, g1v, g2i, g2v in self._iter_groups(ctx):
            sub = self._subset(df, ctx, g1v, g2v)
            if sub.empty: continue
            ax  = ctx["ax_for"](g1i, g2i)
            col = ctx["color_for"](g1i, g2i)

            plot_df = sub.copy()
            if pd.api.types.is_numeric_dtype(plot_df[xcol]):
                try:
                    plot_df["_xbin"] = pd.qcut(
                        plot_df[xcol],
                        q=min(8, plot_df[xcol].nunique()),
                        duplicates="drop")
                    _xc = "_xbin"
                except Exception:
                    _xc = xcol
            else:
                _xc = xcol

            cats       = plot_df[_xc].unique()
            data_grps  = [plot_df[plot_df[_xc] == g][ycol].dropna().values
                          for g in cats]
            ax.boxplot(data_grps, patch_artist=True,
                       medianprops={"color": "#ffffff", "linewidth": 1.5},
                       boxprops={"facecolor": col, "alpha": 0.6},
                       whiskerprops={"color": FG},
                       capprops={"color": FG},
                       flierprops={"marker": ".", "color": col,
                                   "alpha": 0.4, "markersize": 3})
            ax.set_xticks(range(1, len(cats) + 1))
            ax.set_xticklabels([str(g)[:12] for g in cats],
                               rotation=30, ha="right", fontsize=7, color=FG)
            used.add(ax)
        for ax in used:
            self._add_legend(ctx, ax)
