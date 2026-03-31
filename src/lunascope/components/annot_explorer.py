
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

"""Annotation Explorer dock: cohort-level annotation visualisation."""

import io
import traceback

import numpy as np

from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import QMetaObject, Qt, QSignalBlocker, Slot
from PySide6.QtGui import QColor, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QFileDialog,
)

from .annot_explorer_funcs import (
    ANNOT_PALETTE,
    compile_cohort,
    cross_correlogram,
    duration_stats,
    event_raster_data,
    inter_event_intervals,
    load_annex_cache,
    nearest_neighbor_distances,
    normalized_occupancy,
    overlap_matrix,
    peri_event_histogram,
    save_annex_cache,
)


# ---------------------------------------------------------------------------
# Dark-theme helpers
# ---------------------------------------------------------------------------

_BG = "#0d1117"       # figure / axes background
_FG = "#c9d1d9"       # text / axis labels
_GRID = "#21262d"     # grid lines
_SEP = "#30363d"      # subject separator colour


def _style_ax(ax, title="", xlabel="", ylabel=""):
    """Apply the dark theme to a matplotlib axes."""
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    if title:
        ax.set_title(title, color=_FG, fontsize=9, pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, color=_FG, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=_FG, fontsize=8)


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class AnnotExplorerMixin:
    """Dock for cohort-level annotation exploration."""

    _ANNEX_FLOAT_SIZE = (1400, 900)

    _ANNEX_VIEW_MODES = [
        ("peth",     "Peri-event (PETH)"),
        ("overlap",  "Overlap matrix"),
        ("nearest",  "Nearest-neighbour"),
        ("raster",   "Event raster"),
        ("occ_norm", "Occupancy (norm. night)"),
        ("duration", "Duration distribution"),
        ("iei",      "Inter-event intervals"),
    ]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_annot_explorer(self):
        self._annex_canvas = None
        self._annex_cohort = None
        self._annex_render_result = None
        self._annex_render_timer = QtCore.QTimer()
        self._annex_render_timer.setSingleShot(True)
        self._annex_render_timer.setInterval(250)
        self._annex_render_timer.timeout.connect(self._annex_render_view)

        # ---- dock shell -----------------------------------------------
        dock = QDockWidget("Annotation Explorer", self.ui)
        dock.setObjectName("dock_annex")
        dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        dock.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        dock.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        dock.visibilityChanged.connect(self._annex_on_visibility)

        # ---- root widget / outer layout -------------------------------
        root = QWidget(dock)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        # ---- Control row 1: compile / view / export -------------------
        row1 = QWidget(root)
        rl1 = QHBoxLayout(row1)
        rl1.setContentsMargins(0, 0, 0, 0)
        rl1.setSpacing(6)

        btn_compile = QPushButton("Compile All")
        btn_load = QPushButton("Load cache…")
        btn_save = QPushButton("Save cache…")
        btn_compile.setToolTip(
            "Load annotations from every subject in the sample list.\n"
            "Subjects are processed sequentially and the current individual "
            "is restored afterwards."
        )
        btn_compile.setFixedWidth(100)
        btn_load.setFixedWidth(100)
        btn_save.setFixedWidth(100)

        lbl_status = QLabel("No data compiled")
        lbl_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lbl_status.setStyleSheet("color: #888;")

        combo_view = QComboBox()
        combo_view.setToolTip("Select visualisation mode")
        combo_view.setMinimumWidth(180)
        for key, label in self._ANNEX_VIEW_MODES:
            combo_view.addItem(label, key)

        btn_export = QPushButton("Export…")
        btn_export.setFixedWidth(80)
        btn_export.setToolTip("Save current figure as PNG / SVG / PDF")

        rl1.addWidget(btn_compile)
        rl1.addWidget(btn_load)
        rl1.addWidget(btn_save)
        rl1.addWidget(lbl_status, 1)
        rl1.addWidget(QLabel("View:"))
        rl1.addWidget(combo_view)
        rl1.addWidget(btn_export)

        # ---- Control row 2: parameters --------------------------------
        row2 = QWidget(root)
        rl2 = QHBoxLayout(row2)
        rl2.setContentsMargins(0, 0, 0, 0)
        rl2.setSpacing(6)

        combo_ref = QComboBox()
        combo_ref.setToolTip("Reference annotation class (for PETH / Nearest-neighbour)")
        combo_ref.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        combo_ref.setMinimumWidth(100)

        spin_window = QDoubleSpinBox()
        spin_window.setRange(1.0, 3600.0)
        spin_window.setValue(60.0)
        spin_window.setSuffix(" s")
        spin_window.setDecimals(0)
        spin_window.setFixedWidth(80)
        spin_window.setToolTip("Peri-event window  (±seconds around reference event)")

        spin_bin = QDoubleSpinBox()
        spin_bin.setRange(0.1, 120.0)
        spin_bin.setValue(2.0)
        spin_bin.setSuffix(" s")
        spin_bin.setDecimals(1)
        spin_bin.setFixedWidth(72)
        spin_bin.setToolTip("Histogram bin width (seconds)")

        spin_gap = QDoubleSpinBox()
        spin_gap.setRange(0.0, 600.0)
        spin_gap.setValue(10.0)
        spin_gap.setSuffix(" s")
        spin_gap.setDecimals(0)
        spin_gap.setFixedWidth(72)
        spin_gap.setToolTip("Gap inserted between subjects in the raster view (seconds)")

        lbl_ref = QLabel("Ref:")
        lbl_window = QLabel("±")
        lbl_bin = QLabel("Bin:")
        lbl_gap = QLabel("Gap:")

        rl2.addWidget(lbl_ref)
        rl2.addWidget(combo_ref, 1)
        rl2.addWidget(lbl_window)
        rl2.addWidget(spin_window)
        rl2.addWidget(lbl_bin)
        rl2.addWidget(spin_bin)
        rl2.addWidget(lbl_gap)
        rl2.addWidget(spin_gap)
        rl2.addStretch(1)

        # ---- annotation class list (left) + canvas (right) ------------
        list_annots = QListWidget()
        list_annots.setMaximumWidth(230)
        list_annots.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        list_annots.setToolTip("Check / uncheck annotation classes to include in analysis")
        list_annots.itemChanged.connect(self._annex_schedule_render)

        canvas_host = QFrame()
        canvas_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas_host.setFrameShape(QFrame.NoFrame)
        canvas_host.setLayout(QVBoxLayout())
        canvas_host.layout().setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(list_annots)
        splitter.addWidget(canvas_host)
        splitter.setSizes([200, 1000])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # ---- assemble outer layout ------------------------------------
        outer.addWidget(row1)
        outer.addWidget(row2)
        outer.addWidget(splitter, 1)

        dock.setWidget(root)

        # ---- wire signals ---------------------------------------------
        btn_compile.clicked.connect(self._annex_compile)
        btn_load.clicked.connect(self._annex_load_cache)
        btn_save.clicked.connect(self._annex_save_cache)
        btn_export.clicked.connect(self._annex_save_figure)
        combo_view.currentIndexChanged.connect(self._annex_schedule_render)
        combo_ref.currentIndexChanged.connect(self._annex_schedule_render)
        spin_window.valueChanged.connect(self._annex_schedule_render)
        spin_bin.valueChanged.connect(self._annex_schedule_render)
        spin_gap.valueChanged.connect(self._annex_schedule_render)

        # ---- register on main window and store refs -------------------
        self.ui.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.ui.dock_annex = dock

        self._annex_dock = dock
        self._annex_combo_view = combo_view
        self._annex_combo_ref = combo_ref
        self._annex_spin_window = spin_window
        self._annex_spin_bin = spin_bin
        self._annex_spin_gap = spin_gap
        self._annex_lbl_ref = lbl_ref
        self._annex_lbl_window = lbl_window
        self._annex_lbl_bin = lbl_bin
        self._annex_lbl_gap = lbl_gap
        self._annex_lbl_status = lbl_status
        self._annex_list_annots = list_annots
        self._annex_canvas_host = canvas_host
        self._annex_update_controls()

    # ------------------------------------------------------------------
    # Canvas (lazy creation)
    # ------------------------------------------------------------------

    def _ensure_annex_canvas(self):
        if self._annex_canvas is not None:
            return self._annex_canvas

        from .mplcanvas import MplCanvas

        canvas = MplCanvas(self._annex_canvas_host)
        self._annex_canvas_host.layout().addWidget(canvas)
        canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        canvas.customContextMenuRequested.connect(self._annex_context_menu)
        self._annex_canvas = canvas
        return canvas

    # ------------------------------------------------------------------
    # Dock presentation
    # ------------------------------------------------------------------

    def _annex_on_visibility(self, visible):
        if visible:
            dock = self._annex_dock
            if not dock.isFloating():
                dock.setFloating(True)
            w, h = self._ANNEX_FLOAT_SIZE
            if dock.width() < w or dock.height() < h:
                dock.resize(w, h)
            try:
                pg = self.ui.frameGeometry()
                center = pg.center()
                rect = dock.frameGeometry()
                rect.moveCenter(center)
                dock.move(rect.topLeft())
            except Exception:
                pass
            if self._annex_cohort is not None:
                self._annex_schedule_render()

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _annex_get_current_id(self):
        """Return the ID of the currently selected subject (or None)."""
        view = getattr(self.ui, "tbl_slist", None)
        if view is None:
            return None
        idx = view.currentIndex()
        if not idx.isValid():
            return None
        return idx.siblingAtColumn(0).data(Qt.DisplayRole)

    def _annex_get_all_ids(self):
        """Return all subject IDs from the (unfiltered) sample list."""
        try:
            df = self.proj.sample_list()
            if df is None or df.empty:
                return []
            return df.iloc[:, 0].astype(str).tolist()
        except Exception:
            return []

    def _annex_compile(self):
        if getattr(self, "_busy", False):
            return

        ids = self._annex_get_all_ids()
        if not ids:
            QMessageBox.warning(
                self.ui, "Annotation Explorer",
                "No subjects in the sample list.\n\nLoad a sample list first."
            )
            return

        saved_id = self._annex_get_current_id()
        self._annex_saved_id = saved_id

        self._busy = True
        if hasattr(self, "_buttons"):
            self._buttons(False)
        self.sb_progress.setVisible(True)
        self.sb_progress.setRange(0, 0)
        self.sb_progress.setFormat(f"Compiling {len(ids)} subjects…")
        self.lock_ui(f"Compiling annotations from {len(ids)} subjects…")

        fut = self._exec.submit(compile_cohort, self.proj, ids)

        def _done(_f=fut):
            try:
                self._last_annex_result = _f.result()
                QMetaObject.invokeMethod(self, "_annex_compile_done_ok", Qt.QueuedConnection)
            except Exception as e:
                self._last_exc = e
                self._last_tb = traceback.format_exc()
                QMetaObject.invokeMethod(self, "_annex_compile_done_err", Qt.QueuedConnection)

        fut.add_done_callback(_done)

    def _annex_save_cache(self):
        cohort = self._annex_cohort
        if cohort is None or not cohort.get("subjects"):
            QMessageBox.warning(
                self.ui, "Annotation Explorer",
                "No data to save.\n\nCompile or load annotations first."
            )
            return

        fn, _ = QFileDialog.getSaveFileName(
            self.ui, "Save Annotation Cache", "annot_explorer.annot",
            "Luna annotation (*.annot);;All files (*)"
        )
        if not fn:
            return
        try:
            save_annex_cache(fn, cohort)
        except Exception as e:
            QMessageBox.critical(self.ui, "Save error", str(e))

    def _annex_load_cache(self):
        fn, _ = QFileDialog.getOpenFileName(
            self.ui, "Load Annotation Cache", "",
            "Luna annotation (*.annot);;All files (*)"
        )
        if not fn:
            return
        try:
            self._annex_cohort = load_annex_cache(fn)
        except Exception as e:
            QMessageBox.critical(self.ui, "Load error", str(e))
            return
        self._annex_post_compile()
        self._annex_lbl_status.setStyleSheet("color: #06d6a0;")
        self._annex_lbl_status.setText(
            f"{self._annex_cohort['n_subjects']} subjects loaded from cache"
        )

    @Slot()
    def _annex_compile_done_ok(self):
        try:
            self._annex_cohort = self._last_annex_result
            # Restore the individual that was loaded before compilation
            saved_id = getattr(self, "_annex_saved_id", None)
            if saved_id:
                try:
                    self.p = self.proj.inst(saved_id)
                except Exception:
                    pass
            self._annex_post_compile()
        finally:
            self._annex_compile_cleanup()

    @Slot()
    def _annex_compile_done_err(self):
        try:
            QMessageBox.critical(
                self.ui, "Annotation Explorer — compile error",
                getattr(self, "_last_tb", "Unknown error")
            )
        finally:
            self._annex_compile_cleanup()

    def _annex_compile_cleanup(self):
        self.unlock_ui()
        self._busy = False
        if hasattr(self, "_buttons"):
            self._buttons(True)
        self.sb_progress.setRange(0, 100)
        self.sb_progress.setValue(0)
        self.sb_progress.setVisible(False)

    def _annex_post_compile(self):
        """Update UI controls after a successful compilation."""
        cohort = self._annex_cohort
        n_subj = cohort["n_subjects"]
        n_ev = cohort["total_events"]
        n_cls = len(cohort["annot_classes"])

        self._annex_lbl_status.setStyleSheet("color: #06d6a0;")
        self._annex_lbl_status.setText(
            f"{n_subj} subjects · {n_ev:,} events · {n_cls} classes"
        )

        # Populate annotation class list
        self._annex_list_annots.blockSignals(True)
        self._annex_list_annots.clear()
        for i, cls in enumerate(cohort["annot_classes"]):
            item = QListWidgetItem(cls)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            color = QColor(ANNOT_PALETTE[i % len(ANNOT_PALETTE)])
            item.setForeground(color)
            self._annex_list_annots.addItem(item)
        self._annex_list_annots.blockSignals(False)

        # Populate reference class combo
        blocker = QSignalBlocker(self._annex_combo_ref)
        self._annex_combo_ref.clear()
        for cls in cohort["annot_classes"]:
            self._annex_combo_ref.addItem(cls)
        del blocker

        self._annex_schedule_render()

    # ------------------------------------------------------------------
    # Analysis + render pipeline
    # ------------------------------------------------------------------

    def _annex_schedule_render(self, *_):
        """Debounce rapid parameter changes before triggering a render."""
        self._annex_update_controls()
        if self._annex_cohort is None:
            return
        if not self._annex_dock.isVisible():
            return
        self._annex_render_timer.start()

    def _annex_update_controls(self):
        view_mode = self._annex_combo_view.currentData()
        is_pethish = view_mode in ("peth", "nearest")
        is_raster = view_mode == "raster"
        is_occ = view_mode == "occ_norm"

        self._annex_lbl_ref.setVisible(is_pethish)
        self._annex_combo_ref.setVisible(is_pethish)
        self._annex_lbl_window.setVisible(view_mode == "peth")
        self._annex_spin_window.setVisible(view_mode == "peth")
        self._annex_lbl_gap.setVisible(is_raster)
        self._annex_spin_gap.setVisible(is_raster)

        if is_occ:
            self._annex_lbl_bin.setText("Bin:")
            self._annex_spin_bin.setSuffix(" %")
            self._annex_spin_bin.setDecimals(2)
            self._annex_spin_bin.setRange(0.25, 25.0)
            self._annex_spin_bin.setToolTip(
                "Normalized bin width as percent of each subject's recording duration"
            )
        else:
            self._annex_lbl_bin.setText("Bin:")
            self._annex_spin_bin.setSuffix(" s")
            self._annex_spin_bin.setDecimals(1)
            self._annex_spin_bin.setRange(0.1, 120.0)
            self._annex_spin_bin.setToolTip("Histogram bin width (seconds)")

    def _annex_get_checked_classes(self):
        classes = []
        lw = self._annex_list_annots
        for i in range(lw.count()):
            item = lw.item(i)
            if item.checkState() == Qt.Checked:
                classes.append(item.text())
        return classes

    def _annex_get_class_colors(self, classes):
        all_classes = (
            self._annex_cohort["annot_classes"] if self._annex_cohort else classes
        )
        return {
            cls: ANNOT_PALETTE[all_classes.index(cls) % len(ANNOT_PALETTE)]
            if cls in all_classes
            else "#aaaaaa"
            for cls in classes
        }

    def _annex_render_view(self):
        """Gather current parameters, run analysis in background, render."""
        if getattr(self, "_busy", False):
            return
        if self._annex_cohort is None:
            self._annex_render_empty("No data compiled.\n\nClick  Compile All  to load annotations.")
            return

        checked = self._annex_get_checked_classes()
        if not checked:
            self._annex_render_empty("No annotation classes selected.\n\nCheck at least one class in the list.")
            return

        view_mode = self._annex_combo_view.currentData()
        ref_class = self._annex_combo_ref.currentText()
        window = float(self._annex_spin_window.value())
        bin_s = float(self._annex_spin_bin.value())
        gap = float(self._annex_spin_gap.value())
        cohort = self._annex_cohort
        colors = self._annex_get_class_colors(checked)

        self._busy = True
        if hasattr(self, "_buttons"):
            self._buttons(False)
        self.sb_progress.setVisible(True)
        self.sb_progress.setRange(0, 0)
        self.sb_progress.setFormat("Analysing…")

        fut = self._exec.submit(
            self._annex_analyze_worker,
            cohort, view_mode, checked, ref_class, window, bin_s, gap,
        )

        def _done(_f=fut):
            try:
                self._annex_render_result = _f.result()
                QMetaObject.invokeMethod(self, "_annex_render_done_ok", Qt.QueuedConnection)
            except Exception as e:
                self._last_exc = e
                self._last_tb = traceback.format_exc()
                QMetaObject.invokeMethod(self, "_annex_render_done_err", Qt.QueuedConnection)

        fut.add_done_callback(_done)

    @staticmethod
    def _annex_analyze_worker(cohort, view_mode, checked, ref_class, window, bin_s, gap):
        """Background thread: run the appropriate analysis function."""
        colors = {
            cls: ANNOT_PALETTE[cohort["annot_classes"].index(cls) % len(ANNOT_PALETTE)]
            if cls in cohort["annot_classes"] else "#aaaaaa"
            for cls in checked
        }

        if view_mode == "peth":
            targets = [c for c in checked if c != ref_class]
            data = peri_event_histogram(cohort, ref_class, targets, window, bin_s)
        elif view_mode == "overlap":
            data = overlap_matrix(cohort, checked, bin_secs=bin_s)
        elif view_mode == "nearest":
            targets = [c for c in checked if c != ref_class]
            data = nearest_neighbor_distances(cohort, ref_class, targets)
        elif view_mode == "raster":
            data = event_raster_data(cohort, checked, gap_secs=gap)
        elif view_mode == "occ_norm":
            data = normalized_occupancy(cohort, checked, bin_pct=bin_s)
        elif view_mode == "duration":
            data = duration_stats(cohort, checked)
        elif view_mode == "iei":
            data = inter_event_intervals(cohort, checked)
        else:
            data = {}

        return {
            "view_mode": view_mode,
            "data": data,
            "checked": checked,
            "colors": colors,
            "ref_class": ref_class,
            "window": window,
            "bin_s": bin_s,
            "gap": gap,
        }

    @Slot()
    def _annex_render_done_ok(self):
        try:
            result = self._annex_render_result
            if result:
                self._annex_do_render(result)
        except Exception:
            tb = traceback.format_exc()
            print(tb, flush=True)
            self._annex_render_empty(f"Render error:\n{tb[:300]}")
        finally:
            self._annex_render_cleanup()

    @Slot()
    def _annex_render_done_err(self):
        try:
            QMessageBox.critical(
                self.ui, "Annotation Explorer — analysis error",
                getattr(self, "_last_tb", "Unknown error")
            )
        finally:
            self._annex_render_cleanup()

    def _annex_render_cleanup(self):
        self._busy = False
        if hasattr(self, "_buttons"):
            self._buttons(True)
        self.sb_progress.setRange(0, 100)
        self.sb_progress.setValue(0)
        self.sb_progress.setVisible(False)

    # ------------------------------------------------------------------
    # Render dispatchers
    # ------------------------------------------------------------------

    def _annex_do_render(self, result):
        vm = result["view_mode"]
        d = result["data"]
        c = result["colors"]
        checked = result["checked"]
        ref = result["ref_class"]

        if vm == "peth":
            self._annex_render_peth(d, c, ref)
        elif vm == "overlap":
            self._annex_render_overlap(d)
        elif vm == "nearest":
            self._annex_render_nearest(d, c, ref)
        elif vm == "raster":
            self._annex_render_raster(d, c)
        elif vm == "occ_norm":
            self._annex_render_occ_norm(d, c)
        elif vm == "duration":
            self._annex_render_duration(d, c)
        elif vm == "iei":
            self._annex_render_iei(d, c)

    # ------------------------------------------------------------------
    # Peri-event time histogram
    # ------------------------------------------------------------------

    def _annex_render_peth(self, data, colors, ref_class):
        canvas = self._ensure_annex_canvas()
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(_BG)

        targets = data.get("target_classes", [])
        n_ref = data.get("n_ref", 0)
        bins = data.get("bins", np.array([]))
        density = data.get("density", {})
        window = data.get("window", 60)

        if not targets or n_ref == 0 or len(bins) == 0:
            ax = fig.add_subplot(111)
            ax.set_facecolor(_BG)
            msg = (f"No reference events of class  '{ref_class}'  found."
                   if n_ref == 0 else
                   "Select a reference class and at least one target class.")
            ax.text(0.5, 0.5, msg, color=_FG, ha="center", va="center",
                    fontsize=10, transform=ax.transAxes, wrap=True)
            ax.set_axis_off()
            fig.patch.set_facecolor(_BG)
            canvas.draw()
            return

        n = len(targets)
        ncols = min(n, 3)
        nrows = int(np.ceil(n / ncols))
        axes = fig.subplots(nrows, ncols, squeeze=False)
        fig.subplots_adjust(hspace=0.45, wspace=0.35,
                            left=0.08, right=0.97, top=0.90, bottom=0.10)

        title = f"Peri-event density  |  reference: {ref_class}  ({n_ref:,} events)"
        fig.suptitle(title, color=_FG, fontsize=10, y=0.97)

        for idx, cls in enumerate(targets):
            r, c_ = divmod(idx, ncols)
            ax = axes[r][c_]
            ax.set_facecolor(_BG)
            dens = density.get(cls, np.zeros_like(bins))
            col = colors.get(cls, "#aaaaaa")

            ax.fill_between(bins, 0, dens, color=col, alpha=0.35, step="mid")
            ax.step(bins, dens, where="mid", color=col, linewidth=1.2)
            ax.axvline(0, color="#ffffff", linewidth=0.7, linestyle="--", alpha=0.5)
            ax.set_xlim(-window, window)
            _style_ax(ax, title=cls, xlabel="lag (s)", ylabel="density")

        # Hide unused axes
        for idx in range(n, nrows * ncols):
            r, c_ = divmod(idx, ncols)
            axes[r][c_].set_visible(False)

        canvas.draw()

    # ------------------------------------------------------------------
    # Overlap matrix
    # ------------------------------------------------------------------

    def _annex_render_overlap(self, data):
        from matplotlib.colors import LinearSegmentedColormap

        canvas = self._ensure_annex_canvas()
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(_BG)

        labels = data.get("labels", [])
        jaccard = data.get("jaccard", np.zeros((0, 0)))
        directed = data.get("directed", np.zeros((0, 0)))
        event_rate = data.get("event_rate", {})

        if len(labels) < 2:
            ax = fig.add_subplot(111)
            ax.set_facecolor(_BG)
            ax.text(0.5, 0.5, "Need at least 2 annotation classes.",
                    color=_FG, ha="center", va="center", fontsize=10,
                    transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw()
            return

        n = len(labels)
        # Show two matrices side by side: Jaccard (symmetric) and directed overlap
        fig.subplots_adjust(left=0.18, right=0.92, top=0.88, bottom=0.18, wspace=0.5)
        ax1, ax2 = fig.subplots(1, 2)

        cmap = LinearSegmentedColormap.from_list(
            "annex_heat", ["#0d1117", "#1a3a5c", "#1e6091", "#48cae4", "#ffd166", "#f9844a"],
            N=256
        )

        def _draw_heatmap(ax, mat, title, fmt=".2f"):
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1,
                           aspect="auto", interpolation="nearest")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            short = [lb[:12] + "…" if len(lb) > 13 else lb for lb in labels]
            ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7, color=_FG)
            ax.set_yticklabels(short, fontsize=7, color=_FG)
            ax.tick_params(colors=_FG)
            for spine in ax.spines.values():
                spine.set_edgecolor(_GRID)
            # Annotate cells
            for i in range(n):
                for j in range(n):
                    v = mat[i, j]
                    txt_col = "#000000" if v > 0.55 else _FG
                    ax.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                            fontsize=6.5, color=txt_col)
            ax.set_facecolor(_BG)
            ax.set_title(title, color=_FG, fontsize=9, pad=6)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
                labelcolor=_FG, labelsize=7
            )

        _draw_heatmap(ax1, jaccard, "Jaccard similarity")
        _draw_heatmap(ax2, directed, "P(col | row)")

        # Add event-rate bar along diagonal (text annotation)
        fig.suptitle("Annotation overlap matrix", color=_FG, fontsize=10, y=0.97)
        canvas.draw()

    # ------------------------------------------------------------------
    # Nearest-neighbour CDFs
    # ------------------------------------------------------------------

    def _annex_render_nearest(self, data, colors, ref_class):
        canvas = self._ensure_annex_canvas()
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(_BG)

        non_empty = {cls: arr for cls, arr in data.items() if len(arr) > 0}
        if not non_empty:
            ax = fig.add_subplot(111)
            ax.set_facecolor(_BG)
            ax.text(0.5, 0.5,
                    f"No nearest-neighbour data found.\n\n"
                    f"Check that  '{ref_class}'  has events\n"
                    f"and target classes share subjects.",
                    color=_FG, ha="center", va="center", fontsize=10,
                    transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw()
            return

        ax = fig.add_subplot(111)
        ax.set_facecolor(_BG)

        all_vals = np.concatenate(list(non_empty.values()))
        x_max = np.percentile(all_vals, 98) if len(all_vals) else 100.0
        x_max = max(x_max, 1.0)

        for cls, dists in non_empty.items():
            col = colors.get(cls, "#aaaaaa")
            n = len(dists)
            x = np.concatenate([[0], dists, [x_max * 1.1]])
            y = np.concatenate([[0], np.arange(1, n + 1) / n, [1.0]])
            ax.step(x, y, where="post", color=col, linewidth=1.5, label=cls)
            # Median marker
            med = float(np.median(dists))
            ax.axvline(med, color=col, linewidth=0.6, linestyle=":", alpha=0.7)

        ax.set_xlim(0, x_max)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Distance to nearest event (s)", color=_FG, fontsize=9)
        ax.set_ylabel("Cumulative fraction", color=_FG, fontsize=9)
        ax.set_title(
            f"Nearest-neighbour CDF  |  reference: {ref_class}",
            color=_FG, fontsize=10, pad=6
        )
        ax.tick_params(colors=_FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)
        ax.grid(True, color=_GRID, linewidth=0.5)

        legend = ax.legend(fontsize=8, framealpha=0.3,
                           labelcolor=_FG, facecolor="#1a1a1a",
                           edgecolor=_GRID)
        for text in legend.get_texts():
            text.set_color(_FG)

        fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12)
        canvas.draw()

    # ------------------------------------------------------------------
    # Event raster
    # ------------------------------------------------------------------

    def _annex_render_raster(self, data, colors):
        canvas = self._ensure_annex_canvas()
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(_BG)

        by_class = data.get("by_class", {})
        subject_bounds = data.get("subject_bounds", [])
        total_dur = data.get("total_duration", 1.0)
        subject_ids = data.get("subject_ids", [])

        classes_with_data = [cls for cls, ev in by_class.items() if ev]
        if not classes_with_data:
            ax = fig.add_subplot(111)
            ax.set_facecolor(_BG)
            ax.text(0.5, 0.5, "No events to display.", color=_FG,
                    ha="center", va="center", fontsize=10, transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw()
            return

        n_cls = len(classes_with_data)
        ax = fig.add_subplot(111)
        ax.set_facecolor(_BG)

        for row_idx, cls in enumerate(reversed(classes_with_data)):
            events = by_class[cls]
            col = colors.get(cls, "#aaaaaa")
            if not events:
                continue
            # Pass a single dataset explicitly so style arrays stay aligned.
            positions = [(s + e) / 2.0 for s, e in events]
            ax.eventplot(
                [positions],
                lineoffsets=[row_idx],
                linelengths=[0.7],
                linewidths=[0.8],
                colors=[col],
                alpha=0.85,
            )

        # Subject boundary shading
        for i, (s_start, s_end) in enumerate(subject_bounds):
            if i % 2 == 0:
                ax.axvspan(s_start, s_end, color="#ffffff", alpha=0.03, linewidth=0)
            ax.axvline(s_start, color=_SEP, linewidth=0.4, alpha=0.5)

        ax.set_xlim(0, total_dur)
        ax.set_ylim(-0.5, n_cls - 0.5)
        ax.set_yticks(range(n_cls))
        ax.set_yticklabels(
            [cls[:14] + "…" if len(cls) > 15 else cls
             for cls in reversed(classes_with_data)],
            fontsize=7.5, color=_FG
        )
        ax.tick_params(axis="x", colors=_FG, labelsize=7)
        ax.tick_params(axis="y", colors=_FG, labelsize=7, length=0)
        ax.set_xlabel("Pooled time (s)", color=_FG, fontsize=9)
        ax.set_title(
            f"Event raster — {len(subject_ids)} subjects  "
            f"(10 s gap between subjects)",
            color=_FG, fontsize=10, pad=6
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)

        fig.subplots_adjust(left=0.18, right=0.98, top=0.90, bottom=0.10)
        canvas.draw()

    # ------------------------------------------------------------------
    # Duration distribution
    # ------------------------------------------------------------------

    def _annex_render_duration(self, data, colors):
        canvas = self._ensure_annex_canvas()
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(_BG)

        if not data:
            ax = fig.add_subplot(111)
            ax.set_facecolor(_BG)
            ax.text(0.5, 0.5, "No duration data available.", color=_FG,
                    ha="center", va="center", fontsize=10, transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw()
            return

        classes = list(data.keys())
        n = len(classes)
        ax = fig.add_subplot(111)
        ax.set_facecolor(_BG)

        for i, cls in enumerate(reversed(classes)):
            vals = data[cls]
            if len(vals) == 0:
                continue
            col = colors.get(cls, "#aaaaaa")

            # Violin-style plot via kernel density
            from scipy.stats import gaussian_kde
            log_vals = np.log10(np.clip(vals, 1e-4, None))
            if len(np.unique(log_vals)) < 2:
                ax.scatter([np.median(vals)], [i], color=col, s=20, zorder=3)
                continue
            try:
                kde = gaussian_kde(log_vals, bw_method=0.3)
                x_range = np.linspace(log_vals.min() - 0.5, log_vals.max() + 0.5, 256)
                dens = kde(x_range)
                dens = dens / dens.max() * 0.4
                ax.fill_between(10 ** x_range, i - dens, i + dens,
                                color=col, alpha=0.4)
                ax.plot(10 ** x_range, np.ones_like(x_range) * i,
                        color=col, linewidth=0.5, alpha=0.6)
            except Exception:
                pass

            # Median / IQR markers
            p25, p50, p75 = np.percentile(vals, [25, 50, 75])
            ax.plot([p25, p75], [i, i], color=col, linewidth=2.0, solid_capstyle="round")
            ax.scatter([p50], [i], color="#ffffff", s=20, zorder=5, linewidths=0)

        ax.set_xscale("log")
        ax.set_yticks(range(n))
        ax.set_yticklabels(
            [cls[:14] + "…" if len(cls) > 15 else cls for cls in reversed(classes)],
            fontsize=8, color=_FG
        )
        ax.tick_params(axis="x", colors=_FG, labelsize=8)
        ax.tick_params(axis="y", colors=_FG, labelsize=8, length=0)
        ax.set_xlabel("Duration (s)", color=_FG, fontsize=9)
        ax.set_title("Event duration distribution  (line = IQR, dot = median)",
                     color=_FG, fontsize=10, pad=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)
        ax.grid(True, axis="x", color=_GRID, linewidth=0.5)
        ax.set_ylim(-0.7, n - 0.3)

        fig.subplots_adjust(left=0.18, right=0.97, top=0.90, bottom=0.12)
        canvas.draw()

    # ------------------------------------------------------------------
    # Normalized occupancy
    # ------------------------------------------------------------------

    def _annex_render_occ_norm(self, data, colors):
        canvas = self._ensure_annex_canvas()
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(_BG)

        x = data.get("x", np.array([]))
        mean_occ = data.get("mean_occ", {})
        n_subjects = data.get("n_subjects_used", 0)
        bin_pct = data.get("bin_pct", 0.0)

        classes = [cls for cls, vals in mean_occ.items() if len(vals) and np.any(vals > 0)]
        if len(x) == 0 or not classes:
            ax = fig.add_subplot(111)
            ax.set_facecolor(_BG)
            ax.text(0.5, 0.5, "No occupancy data available.", color=_FG,
                    ha="center", va="center", fontsize=10, transform=ax.transAxes)
            ax.set_axis_off()
            canvas.draw()
            return

        ax = fig.add_subplot(111)
        ax.set_facecolor(_BG)

        x_pct = x * 100.0
        ys = [mean_occ[cls] for cls in classes]
        cols = [colors.get(cls, "#aaaaaa") for cls in classes]
        ax.stackplot(x_pct, ys, colors=cols, alpha=0.78, labels=classes, baseline="zero")

        total = np.sum(np.vstack(ys), axis=0)
        ax.plot(x_pct, total, color="#ffffff", linewidth=1.0, alpha=0.75)

        ax.set_xlim(0, 100)
        ax.set_xlabel("Normalized recording progress (%)", color=_FG, fontsize=9)
        ax.set_ylabel("Mean occupancy fraction", color=_FG, fontsize=9)
        ax.set_title(
            f"Normalized occupancy profile — {n_subjects} subjects  "
            f"(bin width {bin_pct:g}% of recording)",
            color=_FG, fontsize=10, pad=6
        )
        ax.tick_params(colors=_FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)
        ax.grid(True, axis="y", color=_GRID, linewidth=0.5)

        legend = ax.legend(fontsize=8, framealpha=0.3,
                           facecolor="#1a1a1a", edgecolor=_GRID, ncols=2)
        for text in legend.get_texts():
            text.set_color(_FG)

        note = (
            "Values are averaged over subjects after rescaling each recording to 0–100%. "
            "Overlapping classes can stack above 1.0."
        )
        fig.text(0.5, 0.015, note, ha="center", va="bottom", color="#8b949e", fontsize=8)
        fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12)
        canvas.draw()

    # ------------------------------------------------------------------
    # Inter-event intervals
    # ------------------------------------------------------------------

    def _annex_render_iei(self, data, colors):
        canvas = self._ensure_annex_canvas()
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(_BG)

        non_empty = {cls: arr for cls, arr in data.items() if len(arr) > 0}
        if not non_empty:
            ax = fig.add_subplot(111)
            ax.set_facecolor(_BG)
            ax.text(0.5, 0.5,
                    "No inter-event interval data.\n\n"
                    "Each selected class needs ≥2 consecutive events in at least one subject.",
                    color=_FG, ha="center", va="center", fontsize=10,
                    transform=ax.transAxes, wrap=True)
            ax.set_axis_off()
            canvas.draw()
            return

        ax = fig.add_subplot(111)
        ax.set_facecolor(_BG)

        for cls, ieis in non_empty.items():
            col = colors.get(cls, "#aaaaaa")
            n = len(ieis)
            ieis_s = np.sort(ieis)
            x = np.concatenate([[ieis_s[0] * 0.5], ieis_s])
            y = np.arange(n + 1) / n
            ax.step(x, y, where="post", color=col, linewidth=1.5, label=f"{cls} (n={n:,})")

        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Inter-event interval (s)", color=_FG, fontsize=9)
        ax.set_ylabel("Cumulative fraction", color=_FG, fontsize=9)
        ax.set_title("Inter-event interval CDF", color=_FG, fontsize=10, pad=6)
        ax.tick_params(colors=_FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)
        ax.grid(True, color=_GRID, linewidth=0.5)

        legend = ax.legend(fontsize=8, framealpha=0.3,
                           facecolor="#1a1a1a", edgecolor=_GRID)
        for text in legend.get_texts():
            text.set_color(_FG)

        fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12)
        canvas.draw()

    # ------------------------------------------------------------------
    # Empty / error state
    # ------------------------------------------------------------------

    def _annex_render_empty(self, message=""):
        canvas = self._ensure_annex_canvas()
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(_BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(_BG)
        ax.text(0.5, 0.5, message or "No data", color=_FG,
                ha="center", va="center", fontsize=10,
                transform=ax.transAxes, wrap=True,
                multialignment="center")
        ax.set_axis_off()
        canvas.draw()

    # ------------------------------------------------------------------
    # Export / context menu
    # ------------------------------------------------------------------

    def _annex_context_menu(self, pos):
        if self._annex_canvas is None:
            return
        menu = QMenu(self._annex_canvas)
        act_copy = menu.addAction("Copy to Clipboard")
        act_save = menu.addAction("Save Figure…")
        action = menu.exec(self._annex_canvas.mapToGlobal(pos))
        if action == act_copy:
            self._annex_copy_to_clipboard()
        elif action == act_save:
            self._annex_save_figure()

    def _annex_copy_to_clipboard(self):
        if self._annex_canvas is None:
            return
        buf = io.BytesIO()
        self._annex_canvas.figure.savefig(buf, format="png", bbox_inches="tight",
                                           facecolor=_BG)
        img = QtGui.QImage.fromData(buf.getvalue(), "PNG")
        QtWidgets.QApplication.clipboard().setImage(img)

    def _annex_save_figure(self):
        if self._annex_canvas is None:
            return
        fn, _ = QFileDialog.getSaveFileName(
            self.ui, "Save Figure", "annot_explorer",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )
        if fn:
            self._annex_canvas.figure.savefig(fn, bbox_inches="tight", facecolor=_BG)
