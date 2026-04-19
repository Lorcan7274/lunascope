
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

#  --------------------------------------------------------------------
#  Luna / Lunascope  —  Explorer: Annotation tab
#  --------------------------------------------------------------------

"""Cohort-level annotation explorer tab (peri-event, overlap, nearest, etc.)"""

import traceback

import numpy as np

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, QSignalBlocker, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton, QScrollArea, QSizePolicy, QSplitter,
    QVBoxLayout, QWidget,
)

from .explorer_base import BG, FG, GRID, SEP, _ExplorerTab
from ..file_dialogs import open_file_name, save_file_name
from .annot_explorer_funcs import (
    ANNOT_PALETTE,
    compile_cohort,
    duration_stats,
    event_raster_data,
    inter_event_intervals,
    load_annex_cache,
    nearest_neighbor_distances,
    overlap_matrix,
    peri_event_histogram,
    save_annex_cache,
    temporal_occupancy,
)


class AnnotTab(_ExplorerTab):
    """Annotation Explorer tab: cohort-level annotation visualisation."""

    _sig_ok       = QtCore.Signal(object)   # analysis result dict
    _sig_err      = QtCore.Signal(str)       # traceback
    _sig_progress = QtCore.Signal(int, int)  # (done, total) during compile

    # view-mode keys and labels
    _VIEWS = [
        ("peth",      "Peri-event (PETH)"),
        ("overlap",   "Overlap matrix"),
        ("nearest",   "Nearest-neighbour"),
        ("raster",    "Event raster"),
        ("occupancy", "Temporal occupancy"),
        ("duration",  "Duration distribution"),
        ("iei",       "Inter-event intervals"),
    ]

    def __init__(self, ctrl, parent=None):
        super().__init__(ctrl, parent)
        self._cohort        = None
        self._render_result = None
        self._render_timer  = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(250)
        self._render_timer.timeout.connect(self._render_view)

        self._sig_ok.connect(self._on_ok,           Qt.QueuedConnection)
        self._sig_err.connect(self._on_err,          Qt.QueuedConnection)
        self._sig_progress.connect(self._on_progress, Qt.QueuedConnection)

        self._build_widget()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_widget(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(4)

        # ---- row 1: compile / status / view / export ------------------
        row1 = QWidget()
        rl1  = QHBoxLayout(row1)
        rl1.setContentsMargins(0, 0, 0, 0); rl1.setSpacing(6)

        btn_compile = QPushButton("Compile All")
        btn_compile.setFixedWidth(100)
        btn_compile.setToolTip("Load annotations from every subject in the sample list")

        btn_load = QPushButton("Load cache…"); btn_load.setFixedWidth(100)
        btn_save = QPushButton("Save cache…"); btn_save.setFixedWidth(100)

        lbl_status = QLabel("No data compiled")
        lbl_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lbl_status.setStyleSheet("color:#888;")

        combo_view = QComboBox(); combo_view.setMinimumWidth(180)
        for key, label in self._VIEWS:
            combo_view.addItem(label, key)

        btn_export = QPushButton("Export…"); btn_export.setFixedWidth(80)

        rl1.addWidget(btn_compile)
        rl1.addWidget(btn_load)
        rl1.addWidget(btn_save)
        rl1.addWidget(lbl_status, 1)
        rl1.addWidget(QLabel("View:")); rl1.addWidget(combo_view)
        rl1.addWidget(btn_export)

        # ---- row 2: parameters ----------------------------------------
        row2 = QWidget()
        rl2  = QHBoxLayout(row2)
        rl2.setContentsMargins(0, 0, 0, 0); rl2.setSpacing(6)

        combo_ref = QComboBox()
        combo_ref.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        combo_ref.setMinimumWidth(100)
        combo_ref.setToolTip("Reference annotation class (PETH / Nearest)")

        spin_win = QDoubleSpinBox(); spin_win.setRange(1, 3600); spin_win.setValue(60)
        spin_win.setSuffix(" s"); spin_win.setDecimals(0); spin_win.setFixedWidth(80)
        spin_win.setToolTip("±window (seconds)")

        spin_bin = QDoubleSpinBox(); spin_bin.setRange(0.1, 120); spin_bin.setValue(2)
        spin_bin.setSuffix(" s"); spin_bin.setDecimals(1); spin_bin.setFixedWidth(72)
        spin_bin.setToolTip("Bin width (seconds)")

        spin_gap = QDoubleSpinBox(); spin_gap.setRange(0, 600); spin_gap.setValue(10)
        spin_gap.setSuffix(" s"); spin_gap.setDecimals(0); spin_gap.setFixedWidth(72)
        spin_gap.setToolTip("Gap between subjects in raster (seconds)")

        spin_flank = QDoubleSpinBox(); spin_flank.setRange(0, 3600); spin_flank.setValue(0)
        spin_flank.setSuffix(" s"); spin_flank.setDecimals(0); spin_flank.setFixedWidth(80)
        spin_flank.setToolTip("Expand each event by ±N seconds before overlap calculation")

        spin_maxdist = QDoubleSpinBox(); spin_maxdist.setRange(1, 86400); spin_maxdist.setValue(3600)
        spin_maxdist.setSuffix(" s"); spin_maxdist.setDecimals(0); spin_maxdist.setFixedWidth(88)
        spin_maxdist.setToolTip("Ignore nearest-neighbour / IEI values beyond this many seconds")

        combo_anchor = QComboBox(); combo_anchor.setFixedWidth(64)
        combo_anchor.addItem("Start", "start")
        combo_anchor.addItem("Mid",   "mid")
        combo_anchor.addItem("End",   "end")
        combo_anchor.setCurrentIndex(1)
        combo_anchor.setToolTip("Reference event anchor point")

        combo_tgt_mode = QComboBox(); combo_tgt_mode.setFixedWidth(110)
        combo_tgt_mode.addItem("Active span", "span")
        combo_tgt_mode.addItem("Onset",       "onset")
        combo_tgt_mode.setToolTip(
            "Active span: P(target covering lag t) — natural for epoch annotations\n"
            "Onset: rate of target start times at each lag — natural for point events")

        combo_nn_anchor = QComboBox(); combo_nn_anchor.setFixedWidth(64)
        combo_nn_anchor.addItem("Start", "start")
        combo_nn_anchor.addItem("Mid",   "mid")
        combo_nn_anchor.addItem("End",   "end")
        combo_nn_anchor.setCurrentIndex(1)
        combo_nn_anchor.setToolTip("Target event anchor point (nearest-neighbour)")

        combo_nn_mode = QComboBox(); combo_nn_mode.setFixedWidth(110)
        combo_nn_mode.addItem("Absolute", "absolute")
        combo_nn_mode.addItem("Leading",  "leading")
        combo_nn_mode.addItem("Lagging",  "lagging")
        combo_nn_mode.addItem("Signed avg", "signed")
        combo_nn_mode.setToolTip(
            "Absolute: nearest target regardless of order\n"
            "Leading: nearest target before the reference event\n"
            "Lagging: nearest target after the reference event\n"
            "Signed avg: nearest target by absolute distance, retaining sign"
        )

        lbl_anchor   = QLabel("Anchor:")
        lbl_tgt_mode = QLabel("Target:")
        lbl_nn_anchor = QLabel("Other:")
        lbl_nn_mode   = QLabel("Mode:")
        lbl_gap      = QLabel("Gap:")
        lbl_flank    = QLabel("Flank:")
        lbl_maxdist  = QLabel("Max:")

        lbl_ref = QLabel("Ref:")
        lbl_win = QLabel("±")
        lbl_bin = QLabel("Bin:")

        rl2.addWidget(lbl_ref); rl2.addWidget(combo_ref, 1)
        rl2.addWidget(lbl_win); rl2.addWidget(spin_win)
        rl2.addWidget(lbl_bin); rl2.addWidget(spin_bin)
        rl2.addWidget(lbl_anchor);   rl2.addWidget(combo_anchor)
        rl2.addWidget(lbl_tgt_mode); rl2.addWidget(combo_tgt_mode)
        rl2.addWidget(lbl_nn_anchor); rl2.addWidget(combo_nn_anchor)
        rl2.addWidget(lbl_nn_mode);   rl2.addWidget(combo_nn_mode)
        rl2.addWidget(lbl_gap);      rl2.addWidget(spin_gap)
        rl2.addWidget(lbl_flank);    rl2.addWidget(spin_flank)
        rl2.addWidget(lbl_maxdist);  rl2.addWidget(spin_maxdist)
        rl2.addStretch(1)

        # ---- class list (left) + canvas (right) -----------------------
        btn_toggle_all = QPushButton("Clear all")
        btn_toggle_all.setToolTip("Select or clear all annotation classes")
        btn_toggle_all.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        list_cls = QListWidget()
        list_cls.setMaximumWidth(260)
        list_cls.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        list_cls.setToolTip("Check/uncheck annotation classes to include")
        list_cls.itemChanged.connect(self._on_class_item_changed)

        list_host = QWidget()
        list_host.setMaximumWidth(260)
        list_layout = QVBoxLayout(list_host)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(6)
        list_layout.addWidget(btn_toggle_all)
        list_layout.addWidget(list_cls, 1)

        canvas_host = QFrame()
        canvas_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        canvas_host.setFrameShape(QFrame.NoFrame)
        canvas_host.setLayout(QVBoxLayout())
        canvas_host.layout().setContentsMargins(0, 0, 0, 0)
        canvas_host.layout().setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self._canvas_host = canvas_host

        canvas_scroll = QScrollArea()
        canvas_scroll.setFrameShape(QFrame.NoFrame)
        canvas_scroll.setWidgetResizable(True)
        canvas_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        canvas_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        canvas_scroll.setAlignment(Qt.AlignTop)
        canvas_scroll.setStyleSheet(
            "QScrollBar:vertical { background:#0d1117; width:12px; margin:0; }"
            "QScrollBar::handle:vertical { background:#4b5563; min-height:28px; border-radius:6px; }"
            "QScrollBar::handle:vertical:hover { background:#6b7280; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0px; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:#111827; }"
        )
        canvas_scroll.setWidget(canvas_host)
        self._canvas_scroll = canvas_scroll
        canvas_scroll.destroyed.connect(self._on_canvas_scroll_destroyed)
        canvas_scroll.viewport().installEventFilter(self)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(list_host)
        splitter.addWidget(canvas_scroll)
        splitter.setSizes([200, 1000])
        splitter.setStretchFactor(0, 0); splitter.setStretchFactor(1, 1)

        outer.addWidget(row1); outer.addWidget(row2); outer.addWidget(splitter, 1)

        # ---- store refs -----------------------------------------------
        self._root          = root
        self._lbl_status    = lbl_status
        self._combo_view    = combo_view
        self._combo_ref     = combo_ref
        self._spin_win      = spin_win
        self._spin_bin      = spin_bin
        self._spin_gap      = spin_gap
        self._spin_flank    = spin_flank
        self._spin_maxdist  = spin_maxdist
        self._combo_anchor  = combo_anchor
        self._combo_tgt_mode= combo_tgt_mode
        self._combo_nn_anchor = combo_nn_anchor
        self._combo_nn_mode   = combo_nn_mode
        self._lbl_ref       = lbl_ref
        self._lbl_win       = lbl_win
        self._lbl_bin       = lbl_bin
        self._lbl_anchor    = lbl_anchor
        self._lbl_tgt_mode  = lbl_tgt_mode
        self._lbl_nn_anchor = lbl_nn_anchor
        self._lbl_nn_mode   = lbl_nn_mode
        self._lbl_gap       = lbl_gap
        self._lbl_flank     = lbl_flank
        self._lbl_maxdist   = lbl_maxdist
        self._list_cls      = list_cls
        self._btn_toggle_all = btn_toggle_all

        # ---- wire signals ---------------------------------------------
        btn_compile.clicked.connect(self._compile)
        btn_load.clicked.connect(self._load_cache)
        btn_save.clicked.connect(self._save_cache)
        btn_export.clicked.connect(self._save_figure)
        btn_toggle_all.clicked.connect(self._toggle_all_annots)
        combo_view.currentIndexChanged.connect(self._on_view_changed)
        combo_ref.currentIndexChanged.connect(self._schedule_render)
        spin_win.valueChanged.connect(self._schedule_render)
        spin_bin.valueChanged.connect(self._schedule_render)
        spin_gap.valueChanged.connect(self._schedule_render)
        spin_flank.valueChanged.connect(self._schedule_render)
        spin_maxdist.valueChanged.connect(self._schedule_render)
        combo_anchor.currentIndexChanged.connect(self._schedule_render)
        combo_tgt_mode.currentIndexChanged.connect(self._schedule_render)
        combo_nn_anchor.currentIndexChanged.connect(self._schedule_render)
        combo_nn_mode.currentIndexChanged.connect(self._schedule_render)

        # Set initial visibility
        self._on_view_changed()

    def _set_canvas_height(self, nrows: int | None = None, min_height: int | None = None):
        """Set canvas height constraints.

        PETH (nrows > 1): canvas is pinned to a computed tall height so the
        scroll area activates and the user can scroll through all row panels.

        All other views: only a minimum height is set; the canvas is free to
        expand with the window (widgetResizable=True handles the stretching).
        """
        canvas = self._ensure_canvas()
        if canvas is None:
            return
        peth_scroll = (nrows is not None and nrows > 1)
        if min_height is None:
            if peth_scroll:
                min_height = 120 + (nrows * 260) + ((nrows - 1) * 24)
            else:
                min_height = 420
        canvas.setMinimumHeight(min_height)
        if self._canvas_host is not None:
            self._canvas_host.setMinimumHeight(min_height)
        if peth_scroll:
            # Pin height so the scroll area knows the content is taller than the viewport
            canvas.setMaximumHeight(min_height)
            if self._canvas_host is not None:
                self._canvas_host.setMaximumHeight(min_height)
        else:
            # Release the ceiling — let the canvas fill whatever vertical space is available
            canvas.setMaximumHeight(16_777_215)
            if self._canvas_host is not None:
                self._canvas_host.setMaximumHeight(16_777_215)
        self._sync_canvas_width()

    def _render_empty(self, msg: str = ""):
        self._set_canvas_height()
        super()._render_empty(msg)

    @staticmethod
    def _fmt_mean_secs(val: float) -> str:
        if not np.isfinite(val):
            return "n/a"
        sign = "-" if val < 0 else ""
        abs_v = abs(float(val))
        if abs_v >= 3600:
            return f"{sign}{abs_v/3600.0:.2f} h"
        if abs_v >= 60:
            return f"{sign}{abs_v/60.0:.2f} m"
        return f"{sign}{abs_v:.2f} s"

    def _add_mean_box(self, fig, items, *, title="Mean"):
        if not items:
            return
        lines = [title]
        for cls, mean_v in items:
            short = cls if len(cls) <= 18 else (cls[:17] + "…")
            lines.append(f"{short}: {self._fmt_mean_secs(float(mean_v))}")
        fig.text(
            0.80, 0.88, "\n".join(lines),
            ha="left", va="top", color=FG, fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#111827",
                      edgecolor=GRID, alpha=0.85)
        )

    def _update_toggle_all_button(self):
        btn = getattr(self, "_btn_toggle_all", None)
        lw = getattr(self, "_list_cls", None)
        if btn is None or lw is None:
            return
        n = lw.count()
        n_checked = sum(
            1 for i in range(n)
            if lw.item(i).checkState() == Qt.Checked
        )
        btn.setText("Clear all" if n_checked > 0 else "Select all")

    def _set_all_annots_checked(self, checked: bool):
        lw = self._list_cls
        state = Qt.Checked if checked else Qt.Unchecked
        lw.blockSignals(True)
        try:
            for i in range(lw.count()):
                item = lw.item(i)
                if item.checkState() != state:
                    item.setCheckState(state)
        finally:
            lw.blockSignals(False)
        self._update_toggle_all_button()
        self._schedule_render()

    def _toggle_all_annots(self):
        lw = self._list_cls
        any_checked = any(
            lw.item(i).checkState() == Qt.Checked
            for i in range(lw.count())
        )
        self._set_all_annots_checked(not any_checked)

    def _on_class_item_changed(self, *_):
        self._update_toggle_all_button()
        self._schedule_render()

    # ------------------------------------------------------------------
    # View-change: show/hide controls that are specific to certain views
    # ------------------------------------------------------------------

    def _on_view_changed(self, *_):
        view = self._combo_view.currentData()
        is_peth   = (view == "peth")
        is_nearest = (view == "nearest")
        is_raster = (view == "raster")
        is_overlap = (view == "overlap")
        is_occupancy = (view == "occupancy")
        is_dist = view in ("nearest", "iei")
        show_bin = is_peth or is_overlap or is_occupancy or is_nearest
        self._lbl_win.setVisible(is_peth)
        self._spin_win.setVisible(is_peth)
        self._lbl_bin.setVisible(show_bin)
        self._spin_bin.setVisible(show_bin)
        self._lbl_anchor.setVisible(is_peth or is_nearest)
        self._combo_anchor.setVisible(is_peth or is_nearest)
        self._lbl_tgt_mode.setVisible(is_peth)
        self._combo_tgt_mode.setVisible(is_peth)
        self._lbl_nn_anchor.setVisible(is_nearest)
        self._combo_nn_anchor.setVisible(is_nearest)
        self._lbl_nn_mode.setVisible(is_nearest)
        self._combo_nn_mode.setVisible(is_nearest)
        self._lbl_gap.setVisible(is_raster)
        self._spin_gap.setVisible(is_raster)
        self._lbl_flank.setVisible(is_overlap)
        self._spin_flank.setVisible(is_overlap)
        self._lbl_maxdist.setVisible(is_dist)
        self._spin_maxdist.setVisible(is_dist)
        self._schedule_render()

    # ------------------------------------------------------------------
    # Sample-list helpers
    # ------------------------------------------------------------------

    def _get_all_ids(self):
        try:
            df = self.ctrl.proj.sample_list()
            if df is None or df.empty:
                return []
            return df.iloc[:, 0].astype(str).tolist()
        except Exception:
            return []

    def _get_current_id(self):
        view = getattr(self.ctrl.ui, "tbl_slist", None)
        if view is None:
            return None
        idx = view.currentIndex()
        return idx.siblingAtColumn(0).data(Qt.DisplayRole) if idx.isValid() else None

    # ------------------------------------------------------------------
    # Save / load cache
    # ------------------------------------------------------------------

    def _save_cache(self):
        if not self._cohort:
            QtWidgets.QMessageBox.warning(self._root, "Annotation Explorer",
                                          "No data to save. Compile first.")
            return
        fn, _ = save_file_name(self._root, "Save Annotation Cache", "annot_cache.annot",
                               "Annotation cache (*.annot);;All files (*)")
        if fn:
            try:
                save_annex_cache(fn, self._cohort)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self._root, "Save error", str(e))

    def _load_cache(self):
        fn, _ = open_file_name(self._root, "Load Annotation Cache", "",
                               "Annotation cache (*.annot);;All files (*)")
        if not fn:
            return
        try:
            cohort = load_annex_cache(fn)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self._root, "Load error", str(e))
            return
        self._cohort = cohort
        self._post_compile()

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _compile(self):
        ids = self._get_all_ids()
        if not ids:
            QtWidgets.QMessageBox.warning(
                self._root, "Annotation Explorer",
                "No subjects in the sample list.")
            return
        n = len(ids)
        if not self._start_work(f"Compiling annotations from {n} subjects…"):
            return
        self._render_empty(
            f"Compiling annotations from {n} subjects…\n\nPlease wait.\n\n"
            "Tip: use  Save cache…  after compiling\n"
            "to speed up future loads."
        )
        self._saved_id = self._get_current_id()

        def _progress_cb(done, total):
            self._sig_progress.emit(done, total)

        fut = self.ctrl._exec.submit(
            compile_cohort, self.ctrl.proj, ids, None, _progress_cb)
        def _done(_f=fut):
            try:
                self._sig_ok.emit({"type": "compile", "result": _f.result()})
            except Exception:
                self._sig_err.emit(traceback.format_exc())
        fut.add_done_callback(_done)

    # ------------------------------------------------------------------
    # Analysis (background)
    # ------------------------------------------------------------------

    def _schedule_render(self, *_):
        if self._cohort is None:
            return
        self._render_timer.start()

    def _render_view(self):
        cohort = self._cohort
        if cohort is None:
            return
        checked = self._checked_classes()
        if not checked:
            self._render_empty("No annotation classes selected.")
            return
        if not self._start_work("Analysing…"):
            return

        view       = self._combo_view.currentData()
        ref        = self._combo_ref.currentText()
        window     = float(self._spin_win.value())
        bin_s      = float(self._spin_bin.value())
        gap        = float(self._spin_gap.value())
        flank_s    = float(self._spin_flank.value())
        max_dist_s = float(self._spin_maxdist.value())
        ref_anchor = self._combo_anchor.currentData()
        tgt_mode   = self._combo_tgt_mode.currentData()
        nn_anchor  = self._combo_nn_anchor.currentData()
        nn_mode    = self._combo_nn_mode.currentData()

        fut = self.ctrl._exec.submit(
            self._analyze_worker, cohort, view, checked, ref, window, bin_s, gap,
            flank_s, max_dist_s, ref_anchor, tgt_mode, nn_anchor, nn_mode)
        def _done(_f=fut):
            try:
                self._sig_ok.emit({"type": "render", "result": _f.result()})
            except Exception:
                self._sig_err.emit(traceback.format_exc())
        fut.add_done_callback(_done)

    @staticmethod
    def _analyze_worker(cohort, view, checked, ref, window, bin_s, gap,
                        flank_s, max_dist_s,
                        ref_anchor="mid", tgt_mode="span",
                        nn_anchor="mid", nn_mode="absolute"):
        colors = {
            cls: ANNOT_PALETTE[cohort["annot_classes"].index(cls) % len(ANNOT_PALETTE)]
            if cls in cohort["annot_classes"] else "#aaaaaa"
            for cls in checked
        }
        if view == "peth":
            # include ref_class itself last (auto-PETH / inter-event distribution)
            targets = [c for c in checked if c != ref] + ([ref] if ref in checked else [])
            data = peri_event_histogram(cohort, ref, targets, window, bin_s,
                                        ref_anchor=ref_anchor, target_mode=tgt_mode)
        elif view == "overlap":
            data = overlap_matrix(cohort, checked, bin_secs=bin_s, flank_secs=flank_s)
        elif view == "nearest":
            targets = [c for c in checked if c != ref]
            data = nearest_neighbor_distances(
                cohort, ref, targets, max_secs=max_dist_s,
                ref_anchor=ref_anchor, target_anchor=nn_anchor, direction=nn_mode
            )
        elif view == "raster":
            data = event_raster_data(cohort, checked, gap_secs=gap)
        elif view == "occupancy":
            data = temporal_occupancy(cohort, checked, bin_secs=bin_s)
        elif view == "duration":
            data = duration_stats(cohort, checked)
        elif view == "iei":
            data = inter_event_intervals(cohort, checked, max_secs=max_dist_s)
        else:
            data = {}
        return {"view": view, "data": data, "colors": colors,
                "checked": checked, "ref": ref, "window": window, "bin_s": bin_s,
                "flank_s": flank_s, "max_dist_s": max_dist_s,
                "ref_anchor": ref_anchor, "nn_anchor": nn_anchor, "nn_mode": nn_mode}

    # ------------------------------------------------------------------
    # Done callbacks
    # ------------------------------------------------------------------

    def _on_ok(self, payload):
        try:
            if payload["type"] == "compile":
                self._cohort = payload["result"]
                self._post_compile()
            elif payload["type"] == "render":
                self._do_render(payload["result"])
        except Exception:
            import traceback as tb; print(tb.format_exc(), flush=True)
        finally:
            self._end_work()

    def _on_err(self, tb_str):
        try:
            QtWidgets.QMessageBox.critical(
                self._root, "Annotation Explorer error", tb_str[:800])
        finally:
            self._end_work()

    def _on_progress(self, done, total):
        self._lbl_status.setStyleSheet("color:#888;")
        self._lbl_status.setText(f"Compiling…  {done} / {total}")

    # ------------------------------------------------------------------
    # Post-compile UI update
    # ------------------------------------------------------------------

    def _post_compile(self):
        cohort = self._cohort
        n_s  = cohort["n_subjects"]
        n_ev = cohort["total_events"]
        n_cl = len(cohort["annot_classes"])
        self._lbl_status.setStyleSheet("color:#06d6a0;")
        self._lbl_status.setText(
            f"{n_s} subjects · {n_ev:,} events · {n_cl} classes")

        self._list_cls.blockSignals(True)
        self._list_cls.clear()
        for i, cls in enumerate(cohort["annot_classes"]):
            item = QListWidgetItem(cls)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setForeground(QColor(ANNOT_PALETTE[i % len(ANNOT_PALETTE)]))
            self._list_cls.addItem(item)
        self._list_cls.blockSignals(False)
        self._update_toggle_all_button()

        blocker = QSignalBlocker(self._combo_ref)
        self._combo_ref.clear()
        self._combo_ref.addItems(cohort["annot_classes"])
        del blocker

        # Restore individual
        saved = getattr(self, "_saved_id", None)
        if saved:
            try:
                self.ctrl.p = self.ctrl.proj.inst(saved)
            except Exception:
                pass

        self._schedule_render()

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _checked_classes(self):
        lw = self._list_cls
        return [lw.item(i).text()
                for i in range(lw.count())
                if lw.item(i).checkState() == Qt.Checked]

    def _do_render(self, result):
        vm = result["view"]
        d  = result["data"]
        c  = result["colors"]
        ref = result["ref"]
        flank_s = result.get("flank_s", 0.0)
        max_dist_s = result.get("max_dist_s", 3600.0)
        bin_s = result.get("bin_s", 2.0)
        if vm == "peth":
            self._render_peth(d, c, ref)
        elif vm == "overlap":
            self._render_overlap(d, flank_s)
        elif vm == "nearest":
            self._render_nearest(d, c, ref, max_dist_s, bin_s)
        elif vm == "raster":
            self._render_raster(d, c)
        elif vm == "occupancy":
            self._render_occupancy(d)
        elif vm == "duration":
            self._render_duration(d, c)
        elif vm == "iei":
            self._render_iei(d, c, max_dist_s)

    # ------------------------------------------------------------------
    # Render: peri-event
    # ------------------------------------------------------------------

    def _render_peth(self, data, colors, ref_class):
        canvas = self._ensure_canvas()
        fig = canvas.figure; fig.clear(); fig.patch.set_facecolor(BG)
        targets    = data.get("target_classes", [])
        n_ref      = data.get("n_ref", 0)
        bins       = data.get("bins", np.array([]))
        density    = data.get("density", {})
        window     = data.get("window", 60)
        ref_anchor = data.get("ref_anchor", "mid")
        tgt_mode   = data.get("target_mode", "span")
        if not targets or n_ref == 0 or len(bins) == 0:
            ax = fig.add_subplot(111); ax.set_facecolor(BG); ax.set_axis_off()
            ax.text(0.5, 0.5, f"No reference events of  '{ref_class}'  found.",
                    color=FG, ha="center", va="center", fontsize=10,
                    transform=ax.transAxes)
            canvas.draw(); return
        ylabel = "P(active)" if tgt_mode == "span" else "events / ref / s"
        anchor_lbl = {"start": "onset", "mid": "mid", "end": "offset"}.get(ref_anchor, ref_anchor)
        n = len(targets)
        ncols = min(n, 3); nrows = int(np.ceil(n / ncols))
        self._set_canvas_height(nrows)
        axes = fig.subplots(nrows, ncols, squeeze=False)
        fig.subplots_adjust(hspace=0.45, wspace=0.35,
                            left=0.08, right=0.97, top=0.90, bottom=0.10)
        fig.suptitle(
            f"Peri-event  |  ref: {ref_class} @ {anchor_lbl}  ({n_ref:,} events)"
            f"  |  target: {tgt_mode}",
            color=FG, fontsize=10, y=0.97)
        for idx, cls in enumerate(targets):
            r, c_ = divmod(idx, ncols)
            ax = axes[r][c_]
            dens = density.get(cls, np.zeros_like(bins))
            col  = colors.get(cls, "#aaaaaa")
            is_self = (cls == ref_class)
            fill_alpha = 0.20 if is_self else 0.35
            ax.fill_between(bins, 0, dens, color=col, alpha=fill_alpha,
                            step="mid", hatch="////" if is_self else None,
                            edgecolor=col if is_self else "none")
            ax.step(bins, dens, where="mid", color=col,
                    linewidth=1.2, linestyle="--" if is_self else "-")
            ax.axvline(0, color="#ffffff", linewidth=0.7, linestyle="--", alpha=0.5)
            ax.set_xlim(-window, window)
            title = f"{cls}  (inter-event)" if is_self else cls
            self._style_ax(ax, title=title, xlabel="lag (s)", ylabel=ylabel)
        for idx in range(n, nrows * ncols):
            r, c_ = divmod(idx, ncols); axes[r][c_].set_visible(False)
        canvas.draw()

    # ------------------------------------------------------------------
    # Render: overlap matrix
    # ------------------------------------------------------------------

    def _render_overlap(self, data, flank_s=0.0):
        from matplotlib.colors import LinearSegmentedColormap
        canvas = self._ensure_canvas()
        fig = canvas.figure; fig.clear(); fig.patch.set_facecolor(BG)
        labels  = data.get("labels", [])
        jaccard = data.get("jaccard", np.zeros((0, 0)))
        directed= data.get("directed", np.zeros((0, 0)))
        n = len(labels)
        self._set_canvas_height(min_height=max(420, 220 + (28 * n)))
        if n < 2:
            ax = fig.add_subplot(111); ax.set_facecolor(BG); ax.set_axis_off()
            ax.text(0.5, 0.5, "Need ≥ 2 annotation classes.", color=FG,
                    ha="center", va="center", fontsize=10, transform=ax.transAxes)
            canvas.draw(); return
        cmap = LinearSegmentedColormap.from_list(
            "ah", ["#0d1117","#1a3a5c","#1e6091","#48cae4","#ffd166","#f9844a"], N=256)
        fig.subplots_adjust(left=0.18, right=0.92, top=0.88, bottom=0.18, wspace=0.5)
        ax1, ax2 = fig.subplots(1, 2)
        short = [lb[:12] + "…" if len(lb) > 13 else lb for lb in labels]
        def _hmap(ax, mat, title):
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto",
                           interpolation="nearest")
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7, color=FG)
            ax.set_yticklabels(short, fontsize=7, color=FG)
            ax.tick_params(colors=FG)
            for sp in ax.spines.values(): sp.set_edgecolor(GRID)
            for i in range(n):
                for j in range(n):
                    v = mat[i,j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5, color="#000" if v > 0.55 else FG)
            ax.set_facecolor(BG); ax.set_title(title, color=FG, fontsize=9, pad=6)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
                labelcolor=FG, labelsize=7)
        _hmap(ax1, jaccard, "Jaccard similarity")
        _hmap(ax2, directed, "P(col | row)")
        flank_lbl = f"  |  flank = +/-{flank_s:.0f} s" if flank_s > 0 else ""
        fig.suptitle(f"Annotation overlap matrix{flank_lbl}", color=FG, fontsize=10, y=0.97)
        canvas.draw()

    # ------------------------------------------------------------------
    # Render: nearest-neighbour CDFs
    # ------------------------------------------------------------------

    def _render_nearest(self, data, colors, ref_class, max_dist_s=3600.0, bin_s=2.0):
        canvas = self._ensure_canvas()
        self._set_canvas_height(min_height=420)
        fig = canvas.figure; fig.clear(); fig.patch.set_facecolor(BG)
        dists_by_class = data.get("distances", data)
        ref_anchor = data.get("ref_anchor", "mid")
        target_anchor = data.get("target_anchor", "mid")
        direction = data.get("direction", "absolute")
        bin_s = max(float(bin_s), 0.01)
        non_empty = {cls: arr for cls, arr in dists_by_class.items() if len(arr) > 0}
        if not non_empty:
            mode_lbl = {
                "absolute": "absolute",
                "leading": "leading",
                "lagging": "lagging",
                "signed": "signed",
            }.get(direction, direction)
            self._render_empty(
                f"No nearest-neighbour data for  '{ref_class}'.\n"
                f"Mode = {mode_lbl}, max = {max_dist_s:.0f} s."
            ); return
        ax = fig.add_subplot(111); self._style_ax(ax)
        ax.set_facecolor(BG)
        all_vals = np.concatenate(list(non_empty.values()))
        if direction == "signed":
            x_lo = min(float(np.percentile(all_vals, 2)), 0.0)
            x_hi = max(float(np.percentile(all_vals, 98)), 0.0)
            x_pad = max((x_hi - x_lo) * 0.04, 1.0)
            ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
            ax.axvline(0, color="#ffffff", linewidth=0.7, linestyle="--", alpha=0.5)
            x_start = x_lo - x_pad
            x_stop = x_hi + x_pad
            edges = np.arange(x_start, x_stop + bin_s, bin_s)
            if len(edges) < 2 or edges[-1] < x_stop:
                edges = np.append(edges, x_stop)
            centers = (edges[:-1] + edges[1:]) / 2.0
            bin_w = edges[1] - edges[0] if len(edges) > 1 else 1.0
            y_max = 0.0
            for cls, dists in non_empty.items():
                col = colors.get(cls, "#aaaaaa")
                counts, _ = np.histogram(dists, bins=edges)
                density = counts.astype(float) / max(len(dists) * bin_w, 1e-9)
                y_max = max(y_max, float(density.max()) if len(density) else 0.0)
                ax.step(centers, density, where="mid", color=col, linewidth=1.5, label=cls)
                ax.fill_between(centers, 0, density, step="mid", color=col, alpha=0.15)
                ax.axvline(float(np.mean(dists)), color=col, lw=0.8, ls=":", alpha=0.8)
            ax.set_ylim(0, y_max * 1.08 if y_max > 0 else 1.0)
        else:
            for cls, dists in non_empty.items():
                col = colors.get(cls, "#aaaaaa"); n = len(dists)
                dists_s = np.sort(dists)
                x_max = max(float(np.percentile(all_vals, 98)), 1.0)
                x = np.concatenate([[0], dists_s, [x_max * 1.1]])
                y = np.concatenate([[0], np.arange(1, n+1)/n, [1.0]])
                ax.step(x, y, where="post", color=col, linewidth=1.5, label=cls)
                ax.axvline(float(np.median(dists)), color=col, lw=0.6, ls=":", alpha=0.7)
            x_max = max(float(np.percentile(all_vals, 98)), 1.0)
            ax.set_xlim(0, x_max)
            ax.set_ylim(0, 1.02)
        ref_anchor_lbl = {"start": "start", "mid": "mid", "end": "end"}.get(ref_anchor, ref_anchor)
        target_anchor_lbl = {"start": "start", "mid": "mid", "end": "end"}.get(target_anchor, target_anchor)
        if direction == "leading":
            xlabel = f"Time to nearest leading target ({ref_anchor_lbl}->{target_anchor_lbl}, s)"
            mode_lbl = "leading"
        elif direction == "lagging":
            xlabel = f"Time to nearest lagging target ({ref_anchor_lbl}->{target_anchor_lbl}, s)"
            mode_lbl = "lagging"
        elif direction == "signed":
            xlabel = f"Signed nearest target lag ({ref_anchor_lbl}->{target_anchor_lbl}, s)"
            mode_lbl = "signed"
        else:
            xlabel = f"Distance to nearest target ({ref_anchor_lbl}<->{target_anchor_lbl}, s)"
            mode_lbl = "absolute"
        ax.set_xlabel(xlabel, color=FG, fontsize=9)
        ylabel = "Density" if direction == "signed" else "Cumulative fraction"
        ax.set_ylabel(ylabel, color=FG, fontsize=9)
        title_kind = "Nearest-neighbour lag histogram" if direction == "signed" else "Nearest-neighbour CDF"
        ax.set_title(
            f"{title_kind}  |  reference: {ref_class}  |  mode: {mode_lbl}"
            f"  |  max: {max_dist_s:.0f} s"
            f"{f'  |  bin: {bin_s:g} s' if direction == 'signed' else ''}\n"
            f"ref anchor: {ref_anchor_lbl}  |  target anchor: {target_anchor_lbl}",
            color=FG, fontsize=9, pad=6
        )
        ax.grid(True, color=GRID, lw=0.5)
        leg = ax.legend(fontsize=8, framealpha=0.3, facecolor="#1a1a1a", edgecolor=GRID)
        for t in leg.get_texts(): t.set_color(FG)
        mean_items = [(cls, np.mean(dists)) for cls, dists in non_empty.items()]
        self._add_mean_box(fig, mean_items, title="Mean")
        fig.subplots_adjust(left=0.10, right=0.78, top=0.86, bottom=0.12)
        canvas.draw()

    # ------------------------------------------------------------------
    # Render: raster
    # ------------------------------------------------------------------

    def _render_raster(self, data, colors):
        canvas = self._ensure_canvas()
        fig = canvas.figure; fig.clear(); fig.patch.set_facecolor(BG)
        by_class      = data.get("by_class", {})
        subject_bounds= data.get("subject_bounds", [])
        total_dur     = data.get("total_duration", 1.0)
        subject_ids   = data.get("subject_ids", [])
        cls_with_data = [cls for cls, ev in by_class.items() if ev]
        n_cls = len(cls_with_data)
        self._set_canvas_height(min_height=max(360, 180 + (22 * n_cls)))
        if not cls_with_data:
            self._render_empty("No events to display."); return
        ax = fig.add_subplot(111); ax.set_facecolor(BG)
        for row_idx, cls in enumerate(reversed(cls_with_data)):
            events = by_class[cls]
            if not events: continue
            positions = [(s + e) / 2.0 for s, e in events]
            ax.eventplot(positions, lineoffsets=row_idx, linelengths=0.7,
                         linewidths=0.8, colors=colors.get(cls, "#aaaaaa"), alpha=0.85)
        for i, (s0, s1) in enumerate(subject_bounds):
            if i % 2 == 0:
                ax.axvspan(s0, s1, color="#ffffff", alpha=0.03, linewidth=0)
            ax.axvline(s0, color=SEP, lw=0.4, alpha=0.5)
        ax.set_xlim(0, total_dur)
        ax.set_ylim(-0.5, n_cls - 0.5)
        ax.set_yticks(range(n_cls))
        ax.set_yticklabels([c[:14]+"…" if len(c)>15 else c
                            for c in reversed(cls_with_data)],
                           fontsize=7.5, color=FG)
        ax.tick_params(axis="x", colors=FG, labelsize=7)
        ax.tick_params(axis="y", colors=FG, labelsize=7, length=0)
        ax.set_xlabel("Pooled time (s)", color=FG, fontsize=9)
        ax.set_title(f"Event raster — {len(subject_ids)} subjects  (10 s gap)",
                     color=FG, fontsize=10, pad=6)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        fig.subplots_adjust(left=0.18, right=0.98, top=0.90, bottom=0.10)
        canvas.draw()

    # ------------------------------------------------------------------
    # Render: temporal occupancy heatmap
    # ------------------------------------------------------------------

    def _render_occupancy(self, data):
        from matplotlib.colors import LinearSegmentedColormap
        canvas = self._ensure_canvas()
        fig = canvas.figure; fig.clear(); fig.patch.set_facecolor(BG)

        bins      = data.get("bins", np.array([]))
        occupancy = data.get("occupancy", {})
        n_active  = data.get("n_active", np.array([]))
        n_subj    = data.get("n_subjects", 0)
        bin_secs  = data.get("bin_secs", 1.0)

        classes = [cls for cls in occupancy
                   if not np.all(np.isnan(occupancy.get(cls, np.array([np.nan]))))]
        if not classes or len(bins) == 0:
            self._render_empty("No occupancy data."); return

        n_cls = len(classes)
        self._set_canvas_height(min_height=max(360, 240 + (24 * n_cls)))
        t_max = float(bins[-1])

        # 2-D matrix: rows = classes, cols = time bins
        mat = np.vstack([occupancy[cls] for cls in classes])

        cmap = LinearSegmentedColormap.from_list(
            "occ", ["#0d1117", "#1a3a5c", "#1e6091", "#48cae4", "#ffd166", "#ffffff"], N=256)

        # Layout: heatmap (tall) + coverage strip (thin)
        heat_h = max(n_cls, 3)
        gs = fig.add_gridspec(2, 1, height_ratios=[heat_h, 1],
                              hspace=0.06, left=0.18, right=0.91,
                              top=0.91, bottom=0.10)
        ax_heat = fig.add_subplot(gs[0])
        ax_cov  = fig.add_subplot(gs[1], sharex=ax_heat)

        im = ax_heat.imshow(
            mat,
            aspect="auto",
            interpolation="nearest",
            extent=[0, t_max, -0.5, n_cls - 0.5],
            origin="lower",
            cmap=cmap,
            vmin=0, vmax=1,
        )
        ax_heat.set_facecolor(BG)
        ax_heat.set_yticks(range(n_cls))
        fs = max(5.0, min(9.0, 300.0 / n_cls))
        ax_heat.set_yticklabels(
            [c[:17] + "…" if len(c) > 18 else c for c in classes],
            fontsize=fs, color=FG,
        )
        ax_heat.tick_params(axis="x", labelbottom=False, length=0)
        ax_heat.tick_params(axis="y", length=0)
        for sp in ax_heat.spines.values(): sp.set_edgecolor(GRID)

        cb = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.01)
        cb.ax.tick_params(labelcolor=FG, labelsize=7)
        cb.set_label("P(active)", color=FG, fontsize=8)

        bin_label = (f"{bin_secs:.0f} s" if bin_secs >= 1 else f"{bin_secs:.2f} s")
        ax_heat.set_title(
            f"Temporal occupancy — {n_subj} subjects  ·  bin = {bin_label}",
            color=FG, fontsize=10, pad=5,
        )

        # Coverage strip
        ax_cov.fill_between(bins, 0, n_active, step="mid",
                            color="#4cc9f0", alpha=0.35, linewidth=0)
        ax_cov.step(bins, n_active, where="mid", color="#4cc9f0", linewidth=0.9)
        ax_cov.set_xlim(0, t_max)
        ax_cov.set_ylim(0, (n_active.max() * 1.2) if n_active.max() > 0 else 1)
        ax_cov.set_facecolor(BG)
        ax_cov.set_xlabel("Time (s)", color=FG, fontsize=9)
        ax_cov.set_ylabel("N", color=FG, fontsize=7, rotation=0, labelpad=10)
        ax_cov.tick_params(colors=FG, labelsize=7)
        for sp in ax_cov.spines.values(): sp.set_edgecolor(GRID)

        canvas.draw()

    # ------------------------------------------------------------------
    # Render: duration
    # ------------------------------------------------------------------

    def _render_duration(self, data, colors):
        from scipy.stats import gaussian_kde
        canvas = self._ensure_canvas()
        fig = canvas.figure; fig.clear(); fig.patch.set_facecolor(BG)
        if not data:
            self._render_empty("No duration data available."); return
        classes = list(data.keys()); n = len(classes)
        self._set_canvas_height(min_height=max(360, 180 + (24 * n)))
        ax = fig.add_subplot(111); ax.set_facecolor(BG)
        for i, cls in enumerate(reversed(classes)):
            vals = data[cls]
            if len(vals) == 0: continue
            col = colors.get(cls, "#aaaaaa")
            log_v = np.log10(np.clip(vals, 1e-4, None))
            if len(np.unique(log_v)) >= 2:
                try:
                    kde = gaussian_kde(log_v, bw_method=0.3)
                    xr  = np.linspace(log_v.min()-0.5, log_v.max()+0.5, 256)
                    dens= kde(xr); dens /= dens.max() * 2.5
                    ax.fill_between(10**xr, i-dens, i+dens, color=col, alpha=0.4)
                except Exception: pass
            p25, p50, p75 = np.percentile(vals, [25, 50, 75])
            ax.plot([p25, p75], [i, i], color=col, lw=2.0, solid_capstyle="round")
            ax.scatter([p50], [i], color="#ffffff", s=20, zorder=5)
        ax.set_xscale("log")
        ax.set_yticks(range(n))
        ax.set_yticklabels([c[:14]+"…" if len(c)>15 else c for c in reversed(classes)],
                           fontsize=8, color=FG)
        ax.tick_params(axis="x", colors=FG, labelsize=8)
        ax.tick_params(axis="y", colors=FG, labelsize=8, length=0)
        ax.set_xlabel("Duration (s)", color=FG, fontsize=9)
        ax.set_title("Duration distribution  (line=IQR, dot=median)",
                     color=FG, fontsize=10, pad=6)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.grid(True, axis="x", color=GRID, lw=0.5)
        ax.set_ylim(-0.7, n - 0.3)
        mean_items = [(cls, np.mean(vals)) for cls, vals in data.items() if len(vals) > 0]
        self._add_mean_box(fig, mean_items, title="Mean duration")
        fig.subplots_adjust(left=0.18, right=0.78, top=0.90, bottom=0.12)
        canvas.draw()

    # ------------------------------------------------------------------
    # Render: IEI
    # ------------------------------------------------------------------

    def _render_iei(self, data, colors, max_dist_s=3600.0):
        canvas = self._ensure_canvas()
        self._set_canvas_height(min_height=420)
        fig = canvas.figure; fig.clear(); fig.patch.set_facecolor(BG)
        non_empty = {cls: arr for cls, arr in data.items() if len(arr) > 0}
        if not non_empty:
            self._render_empty(
                f"No IEI data.\nEach class needs >=2 consecutive events within {max_dist_s:.0f} s."
            ); return
        ax = fig.add_subplot(111); self._style_ax(ax)
        for cls, ieis in non_empty.items():
            col = colors.get(cls, "#aaaaaa"); n = len(ieis)
            ieis_s = np.sort(ieis)
            x = np.concatenate([[ieis_s[0]*0.5], ieis_s])
            y = np.arange(n+1) / n
            ax.step(x, y, where="post", color=col, lw=1.5, label=f"{cls} (n={n:,})")
        ax.set_xscale("log"); ax.set_ylim(0, 1.02)
        ax.set_xlabel("Inter-event interval (s)", color=FG, fontsize=9)
        ax.set_ylabel("Cumulative fraction", color=FG, fontsize=9)
        ax.set_title(
            f"Inter-event interval CDF  |  max: {max_dist_s:.0f} s",
            color=FG, fontsize=10, pad=6
        )
        ax.grid(True, color=GRID, lw=0.5)
        leg = ax.legend(fontsize=8, framealpha=0.3, facecolor="#1a1a1a", edgecolor=GRID)
        for t in leg.get_texts(): t.set_color(FG)
        mean_items = [(cls, np.mean(ieis)) for cls, ieis in non_empty.items()]
        self._add_mean_box(fig, mean_items, title="Mean IEI")
        fig.subplots_adjust(left=0.10, right=0.78, top=0.90, bottom=0.12)
        canvas.draw()
