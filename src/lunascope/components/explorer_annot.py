
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
    QListWidget, QListWidgetItem, QPushButton, QSizePolicy, QSplitter,
    QVBoxLayout, QWidget,
)

from .explorer_base import BG, FG, GRID, SEP, _ExplorerTab
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

        combo_anchor = QComboBox(); combo_anchor.setFixedWidth(64)
        combo_anchor.addItem("Start", "start")
        combo_anchor.addItem("Mid",   "mid")
        combo_anchor.addItem("End",   "end")
        combo_anchor.setCurrentIndex(1)
        combo_anchor.setToolTip("Reference event anchor point (PETH)")

        combo_tgt_mode = QComboBox(); combo_tgt_mode.setFixedWidth(110)
        combo_tgt_mode.addItem("Active span", "span")
        combo_tgt_mode.addItem("Onset",       "onset")
        combo_tgt_mode.setToolTip(
            "Active span: P(target covering lag t) — natural for epoch annotations\n"
            "Onset: rate of target start times at each lag — natural for point events")

        lbl_anchor   = QLabel("Anchor:")
        lbl_tgt_mode = QLabel("Target:")
        lbl_gap      = QLabel("Gap:")

        rl2.addWidget(QLabel("Ref:")); rl2.addWidget(combo_ref, 1)
        rl2.addWidget(QLabel("±")); rl2.addWidget(spin_win)
        rl2.addWidget(QLabel("Bin:")); rl2.addWidget(spin_bin)
        rl2.addWidget(lbl_anchor);   rl2.addWidget(combo_anchor)
        rl2.addWidget(lbl_tgt_mode); rl2.addWidget(combo_tgt_mode)
        rl2.addWidget(lbl_gap);      rl2.addWidget(spin_gap)
        rl2.addStretch(1)

        # ---- class list (left) + canvas (right) -----------------------
        list_cls = QListWidget()
        list_cls.setMaximumWidth(220)
        list_cls.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        list_cls.setToolTip("Check/uncheck annotation classes to include")
        list_cls.itemChanged.connect(self._schedule_render)

        canvas_host = QFrame()
        canvas_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas_host.setFrameShape(QFrame.NoFrame)
        canvas_host.setLayout(QVBoxLayout())
        canvas_host.layout().setContentsMargins(0, 0, 0, 0)
        self._canvas_host = canvas_host

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(list_cls)
        splitter.addWidget(canvas_host)
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
        self._combo_anchor  = combo_anchor
        self._combo_tgt_mode= combo_tgt_mode
        self._lbl_anchor    = lbl_anchor
        self._lbl_tgt_mode  = lbl_tgt_mode
        self._lbl_gap       = lbl_gap
        self._list_cls      = list_cls

        # ---- wire signals ---------------------------------------------
        btn_compile.clicked.connect(self._compile)
        btn_load.clicked.connect(self._load_cache)
        btn_save.clicked.connect(self._save_cache)
        btn_export.clicked.connect(self._save_figure)
        combo_view.currentIndexChanged.connect(self._on_view_changed)
        combo_ref.currentIndexChanged.connect(self._schedule_render)
        spin_win.valueChanged.connect(self._schedule_render)
        spin_bin.valueChanged.connect(self._schedule_render)
        spin_gap.valueChanged.connect(self._schedule_render)
        combo_anchor.currentIndexChanged.connect(self._schedule_render)
        combo_tgt_mode.currentIndexChanged.connect(self._schedule_render)

        # Set initial visibility
        self._on_view_changed()

    # ------------------------------------------------------------------
    # View-change: show/hide controls that are specific to certain views
    # ------------------------------------------------------------------

    def _on_view_changed(self, *_):
        view = self._combo_view.currentData()
        is_peth   = (view == "peth")
        is_raster = (view == "raster")
        self._lbl_anchor.setVisible(is_peth)
        self._combo_anchor.setVisible(is_peth)
        self._lbl_tgt_mode.setVisible(is_peth)
        self._combo_tgt_mode.setVisible(is_peth)
        self._lbl_gap.setVisible(is_raster)
        self._spin_gap.setVisible(is_raster)
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
        fn, _ = QFileDialog.getSaveFileName(
            self._root, "Save Annotation Cache", "annot_cache.annot",
            "Annotation cache (*.annot);;All files (*)"
        )
        if fn:
            try:
                save_annex_cache(fn, self._cohort)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self._root, "Save error", str(e))

    def _load_cache(self):
        fn, _ = QFileDialog.getOpenFileName(
            self._root, "Load Annotation Cache", "",
            "Annotation cache (*.annot);;All files (*)"
        )
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
        ref_anchor = self._combo_anchor.currentData()
        tgt_mode   = self._combo_tgt_mode.currentData()

        fut = self.ctrl._exec.submit(
            self._analyze_worker, cohort, view, checked, ref, window, bin_s, gap,
            ref_anchor, tgt_mode)
        def _done(_f=fut):
            try:
                self._sig_ok.emit({"type": "render", "result": _f.result()})
            except Exception:
                self._sig_err.emit(traceback.format_exc())
        fut.add_done_callback(_done)

    @staticmethod
    def _analyze_worker(cohort, view, checked, ref, window, bin_s, gap,
                        ref_anchor="mid", tgt_mode="span"):
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
            data = overlap_matrix(cohort, checked, bin_secs=bin_s)
        elif view == "nearest":
            targets = [c for c in checked if c != ref]
            data = nearest_neighbor_distances(cohort, ref, targets)
        elif view == "raster":
            data = event_raster_data(cohort, checked, gap_secs=gap)
        elif view == "occupancy":
            data = temporal_occupancy(cohort, checked, bin_secs=bin_s)
        elif view == "duration":
            data = duration_stats(cohort, checked)
        elif view == "iei":
            data = inter_event_intervals(cohort, checked)
        else:
            data = {}
        return {"view": view, "data": data, "colors": colors,
                "checked": checked, "ref": ref, "window": window, "bin_s": bin_s}

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
        if vm == "peth":
            self._render_peth(d, c, ref)
        elif vm == "overlap":
            self._render_overlap(d)
        elif vm == "nearest":
            self._render_nearest(d, c, ref)
        elif vm == "raster":
            self._render_raster(d, c)
        elif vm == "occupancy":
            self._render_occupancy(d)
        elif vm == "duration":
            self._render_duration(d, c)
        elif vm == "iei":
            self._render_iei(d, c)

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

    def _render_overlap(self, data):
        from matplotlib.colors import LinearSegmentedColormap
        canvas = self._ensure_canvas()
        fig = canvas.figure; fig.clear(); fig.patch.set_facecolor(BG)
        labels  = data.get("labels", [])
        jaccard = data.get("jaccard", np.zeros((0, 0)))
        directed= data.get("directed", np.zeros((0, 0)))
        n = len(labels)
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
        fig.suptitle("Annotation overlap matrix", color=FG, fontsize=10, y=0.97)
        canvas.draw()

    # ------------------------------------------------------------------
    # Render: nearest-neighbour CDFs
    # ------------------------------------------------------------------

    def _render_nearest(self, data, colors, ref_class):
        canvas = self._ensure_canvas()
        fig = canvas.figure; fig.clear(); fig.patch.set_facecolor(BG)
        non_empty = {cls: arr for cls, arr in data.items() if len(arr) > 0}
        if not non_empty:
            self._render_empty(f"No nearest-neighbour data for  '{ref_class}'."); return
        ax = fig.add_subplot(111); self._style_ax(ax)
        ax.set_facecolor(BG)
        all_vals = np.concatenate(list(non_empty.values()))
        x_max = max(float(np.percentile(all_vals, 98)), 1.0)
        for cls, dists in non_empty.items():
            col = colors.get(cls, "#aaaaaa"); n = len(dists)
            x = np.concatenate([[0], dists, [x_max * 1.1]])
            y = np.concatenate([[0], np.arange(1, n+1)/n, [1.0]])
            ax.step(x, y, where="post", color=col, linewidth=1.5, label=cls)
            ax.axvline(float(np.median(dists)), color=col, lw=0.6, ls=":", alpha=0.7)
        ax.set_xlim(0, x_max); ax.set_ylim(0, 1.02)
        ax.set_xlabel("Distance to nearest event (s)", color=FG, fontsize=9)
        ax.set_ylabel("Cumulative fraction", color=FG, fontsize=9)
        ax.set_title(f"Nearest-neighbour CDF  |  ref: {ref_class}", color=FG, fontsize=10)
        ax.grid(True, color=GRID, lw=0.5)
        leg = ax.legend(fontsize=8, framealpha=0.3, facecolor="#1a1a1a", edgecolor=GRID)
        for t in leg.get_texts(): t.set_color(FG)
        fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12)
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
        if not cls_with_data:
            self._render_empty("No events to display."); return
        n_cls = len(cls_with_data)
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
        fig.subplots_adjust(left=0.18, right=0.97, top=0.90, bottom=0.12)
        canvas.draw()

    # ------------------------------------------------------------------
    # Render: IEI
    # ------------------------------------------------------------------

    def _render_iei(self, data, colors):
        canvas = self._ensure_canvas()
        fig = canvas.figure; fig.clear(); fig.patch.set_facecolor(BG)
        non_empty = {cls: arr for cls, arr in data.items() if len(arr) > 0}
        if not non_empty:
            self._render_empty("No IEI data.\nEach class needs ≥2 consecutive events."); return
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
        ax.set_title("Inter-event interval CDF", color=FG, fontsize=10, pad=6)
        ax.grid(True, color=GRID, lw=0.5)
        leg = ax.legend(fontsize=8, framealpha=0.3, facecolor="#1a1a1a", edgecolor=GRID)
        for t in leg.get_texts(): t.set_color(FG)
        fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.12)
        canvas.draw()
