
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
Explorer: Harmonizer tab.

Scans the current sample list for channel names (with SR/TRANS/PDIM) and
annotation labels, then lets the user interactively define remappings
(aliases) and a blacklist (ignore/drop), preview the harmonised view, and
export a Luna @param file.
"""

import threading
import traceback
import os

import numpy as np
import pandas as pd

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, QModelIndex, QSortFilterProxyModel, QTimer, Slot
from PySide6.QtGui import QColor, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
    QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QSplitter, QTabWidget, QTableView, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .explorer_base import BG, FG, GRID, SEP, _ExplorerTab
from ..file_dialogs import open_file_name, save_file_name
from .harmonizer_funcs import (
    ScanResult,
    annot_summary,
    annot_rare_cooccurrence_pairs,
    build_presence,
    channel_summary,
    coverage_stats,
    domain_assignments,
    load_cache,
    rare_cooccurrence_pairs,
    save_cache,
    write_param_file,
)


# ---------------------------------------------------------------------------
# Theme colours
# ---------------------------------------------------------------------------

_C_PRESENT = QColor("#1f6640")   # dark green: channel present
_C_ABSENT  = QColor("#111820")   # near-black: channel absent
_C_HDR     = QColor("#161b22")   # header background


# ---------------------------------------------------------------------------
# Domain options
# ---------------------------------------------------------------------------

_DOMAINS = ["", "EEG", "ECG", "EMG", "EOG", "RESP", "SpO2", "OTHER"]


# ---------------------------------------------------------------------------
# Presence matrix model
# ---------------------------------------------------------------------------

class _PresenceModel(QtCore.QAbstractTableModel):
    """Efficient virtual model backed by a numpy bool[n_rows, n_cols] array.

    Displays coloured cells (no text) for fast rendering of large cohorts.
    """

    def __init__(self, row_names, col_names, matrix, parent=None):
        super().__init__(parent)
        self._rows = row_names          # list[str] – channel / annot names
        self._cols = col_names          # list[str] – subject IDs
        self._mat  = matrix             # np.ndarray bool [n_rows × n_cols]

    # ---- mandatory overrides -------------------------------------------

    def rowCount(self, parent=QModelIndex()):
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return len(self._cols)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        if r >= len(self._rows) or c >= len(self._cols):
            return None

        if role == Qt.BackgroundRole:
            return _C_PRESENT if self._mat[r, c] else _C_ABSENT

        if role == Qt.ToolTipRole:
            state = "present" if self._mat[r, c] else "absent"
            return f"{self._rows[r]}  ·  {self._cols[c]}  →  {state}"

        # No text rendered in cells – pure colour grid.
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal:
            if section >= len(self._cols):
                return None
            full = self._cols[section]
            if role == Qt.DisplayRole:
                return full[:8] + "…" if len(full) > 8 else full
            if role == Qt.ToolTipRole:
                return full
            if role == Qt.UserRole:
                return full             # full ID for click-to-open
        else:
            if section >= len(self._rows):
                return None
            if role == Qt.DisplayRole:
                return self._rows[section]
            if role == Qt.ToolTipRole:
                return self._rows[section]
        return None

    def flags(self, index):
        return Qt.ItemIsEnabled


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _NumericSortProxyModel(QSortFilterProxyModel):
    """Proxy that sorts numeric columns as floats, strings otherwise."""
    def lessThan(self, left, right):
        lv = self.sourceModel().data(left, Qt.DisplayRole)
        rv = self.sourceModel().data(right, Qt.DisplayRole)
        try:
            return float(lv) < float(rv)
        except (TypeError, ValueError):
            return super().lessThan(left, right)


# ---------------------------------------------------------------------------
# Main tab
# ---------------------------------------------------------------------------

class HarmonizerTab(_ExplorerTab):
    """Cohort-level channel and annotation harmonizer tab."""

    _sig_ok       = QtCore.Signal(object)   # ScanResult
    _sig_err      = QtCore.Signal(str)       # traceback string
    _sig_progress = QtCore.Signal(int, int)  # (done, total)

    _DEFAULT_CELL = 16   # px

    def __init__(self, ctrl, parent=None):
        super().__init__(ctrl, parent)

        self._scan:     ScanResult | None = None
        self._domains:  dict = {}           # CH → domain string (user-editable)
        self._scanning: bool = False
        self._stop_flag = threading.Event()
        self._fut       = None

        # Debounce timer for filter text changes
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(300)
        self._filter_timer.timeout.connect(self._refresh_presence)

        self._sig_ok.connect(self._on_scan_ok,   Qt.QueuedConnection)
        self._sig_err.connect(self._on_scan_err,  Qt.QueuedConnection)
        self._sig_progress.connect(self._on_progress, Qt.QueuedConnection)

        self._build_widget()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_widget(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(3)

        outer.addWidget(self._build_toolbar())

        self._prog_bar = QProgressBar()
        self._prog_bar.setMaximumHeight(5)
        self._prog_bar.setTextVisible(False)
        self._prog_bar.setRange(0, 100)
        self._prog_bar.setValue(0)
        self._prog_bar.setVisible(False)
        outer.addWidget(self._prog_bar)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([275, 900])
        outer.addWidget(splitter, 1)   # stretch=1: fill all remaining vertical space

        self._root = root

    # ---- toolbar -------------------------------------------------------

    def _build_toolbar(self) -> QWidget:
        w = QWidget()
        lay = QGridLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self._chk_channels = QCheckBox("Channels")
        self._chk_channels.setChecked(True)
        self._chk_annots   = QCheckBox("Annots")
        self._chk_annots.setChecked(True)
        self._chk_edfplus  = QCheckBox("EDF+ annots (slow)")
        self._chk_edfplus.setChecked(False)
        self._chk_edfplus.setToolTip(
            "Run ANNOTS command to also detect embedded EDF+ annotation labels. "
            "Requires reading the full EDF and is substantially slower."
        )

        self._btn_scan = QPushButton("Scan All")
        self._btn_scan.setFixedWidth(80)
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setFixedWidth(55)
        self._btn_stop.setEnabled(False)

        self._btn_save_cache = QPushButton("Save cache…")
        self._btn_load_cache = QPushButton("Load cache…")

        self._lbl_status = QLabel("No scan")
        self._lbl_status.setStyleSheet(f"color:#888; font-size:11px;")
        self._lbl_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(self._chk_channels, 0, 0)
        lay.addWidget(self._chk_annots, 0, 1)
        lay.addWidget(self._chk_edfplus, 0, 2, 1, 2)
        lay.addWidget(self._btn_scan, 0, 4)
        lay.addWidget(self._btn_stop, 0, 5)
        lay.addWidget(self._btn_save_cache, 1, 0)
        lay.addWidget(self._btn_load_cache, 1, 1)
        lay.addWidget(self._lbl_status, 1, 2, 1, 4)
        lay.setColumnStretch(3, 1)

        self._btn_scan.clicked.connect(self._start_scan)
        self._btn_stop.clicked.connect(self._request_stop)
        self._btn_save_cache.clicked.connect(self._save_cache)
        self._btn_load_cache.clicked.connect(self._load_cache)
        return w

    # ---- left panel ----------------------------------------------------

    def _build_left_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(180)
        scroll.setMaximumWidth(240)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(6, 4, 8, 4)
        lay.setSpacing(6)

        # ---- View -------------------------------------------------
        lay.addWidget(_sec("View"))

        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        lbl_filt = QLabel("Filter:")
        lbl_filt.setFixedWidth(38)
        self._txt_filter = QLineEdit()
        self._txt_filter.setPlaceholderText("Search names…")
        row2.addWidget(lbl_filt); row2.addWidget(self._txt_filter)
        lay.addLayout(row2)

        self._chk_apply_remap = QCheckBox("Apply remapping to all views")
        self._chk_apply_remap.setChecked(False)
        lay.addWidget(self._chk_apply_remap)

        cell_row = QHBoxLayout()
        cell_row.setContentsMargins(0, 0, 0, 0)
        cell_row.addWidget(QLabel("Cell size:"))
        self._spin_cell = QSpinBox()
        self._spin_cell.setRange(6, 40)
        self._spin_cell.setValue(self._DEFAULT_CELL)
        self._spin_cell.setFixedWidth(55)
        self._spin_cell.setSuffix(" px")
        cell_row.addWidget(self._spin_cell)
        cell_row.addStretch()
        lay.addLayout(cell_row)

        lay.addWidget(_hsep())

        self._left_tabs = QTabWidget()
        self._left_tabs.addTab(self._build_left_channels_tab(inner), "Channels")
        self._left_tabs.addTab(self._build_left_annots_tab(inner), "Annotations")
        lay.addWidget(self._left_tabs)

        lay.addWidget(_hsep())
        lay.addWidget(_sec("Export @param File"))

        self._btn_exp_all = QPushButton("Export all…")
        self._btn_exp_all.clicked.connect(lambda: self._export_param(True,  True))
        lay.addWidget(self._btn_exp_all)

        lay.addStretch()
        inner.setLayout(lay)
        scroll.setWidget(inner)

        # ---- connect view-mode controls --------------------------------
        self._txt_filter.textChanged.connect(lambda: self._filter_timer.start())
        self._chk_apply_remap.stateChanged.connect(self._on_apply_remap_toggled)
        self._spin_cell.valueChanged.connect(self._apply_cell_size)
        for ck in (self._chk_show_sr, self._chk_show_trans, self._chk_show_pdim):
            ck.stateChanged.connect(
                lambda _: self._populate_if_active(1)   # channels tab idx=1
            )

        return scroll

    def _build_left_channels_tab(self, parent=None) -> QWidget:
        w = QWidget(parent)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self._chk_show_sr    = QCheckBox("Split rows by SR")
        self._chk_show_trans = QCheckBox("Split rows by TRANS")
        self._chk_show_pdim  = QCheckBox("Split rows by PDIM")
        for ck in (self._chk_show_sr, self._chk_show_trans, self._chk_show_pdim):
            ck.setChecked(False)
            lay.addWidget(ck)

        lay.addWidget(_hsep())
        lay.addWidget(_sec("Channel Remapping"))

        self._tbl_ch_remap = _make_remap_table(w)
        self._tbl_ch_remap.itemChanged.connect(self._on_remap_table_changed)
        lay.addWidget(self._tbl_ch_remap)
        lay.addLayout(self.__remap_buttons(
            self._tbl_ch_remap, self._load_luna_ch_aliases
        ))

        lay.addWidget(_hsep())
        lay.addWidget(_sec("Ignore / Blacklist"))
        lay.addWidget(QLabel("Channels:"))

        self._lst_ignore_ch = _make_ignore_list(w)
        lay.addWidget(self._lst_ignore_ch)
        lay.addLayout(self.__ignore_buttons(self._lst_ignore_ch))

        lay.addWidget(_hsep())
        self._btn_exp_ch  = QPushButton("Channel aliases…")
        self._btn_exp_ch.clicked.connect(lambda: self._export_param(True,  False))
        lay.addWidget(self._btn_exp_ch)
        lay.addStretch()
        return w

    def _build_left_annots_tab(self, parent=None) -> QWidget:
        w = QWidget(parent)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        lay.addWidget(_sec("Annotation Remapping"))

        self._tbl_ann_remap = _make_remap_table(w)
        self._tbl_ann_remap.itemChanged.connect(self._on_remap_table_changed)
        self._tbl_ann_remap.setMaximumHeight(120)
        lay.addWidget(self._tbl_ann_remap)
        lay.addLayout(self.__remap_buttons(
            self._tbl_ann_remap, self._load_luna_ann_aliases
        ))

        lay.addWidget(_hsep())
        lay.addWidget(_sec("Ignore / Blacklist"))
        lay.addWidget(QLabel("Annotations:"))

        self._lst_ignore_ann = _make_ignore_list(w)
        self._lst_ignore_ann.setMaximumHeight(80)
        lay.addWidget(self._lst_ignore_ann)
        lay.addLayout(self.__ignore_buttons(self._lst_ignore_ann))

        lay.addWidget(_hsep())
        self._btn_exp_ann = QPushButton("Annot aliases…")
        self._btn_exp_ann.clicked.connect(lambda: self._export_param(False, True))
        lay.addWidget(self._btn_exp_ann)
        lay.addStretch()
        return w

    def __remap_buttons(self, tbl, luna_fn) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(3)
        b_add   = QPushButton("+ Row")
        b_del   = QPushButton("− Del")
        b_luna  = QPushButton("← Luna")
        b_clear = QPushButton("Clear")
        for b in (b_add, b_del, b_luna, b_clear):
            b.setFixedWidth(52)
            row.addWidget(b)
        row.addStretch()
        b_add.clicked.connect(lambda: self._remap_add_row(tbl))
        b_del.clicked.connect(lambda: self._remap_del_row(tbl))
        b_luna.clicked.connect(luna_fn)
        b_clear.clicked.connect(lambda: tbl.setRowCount(0))
        tbl.model().rowsRemoved.connect(lambda *_: self._refresh_status_for(tbl))
        return row

    def __ignore_buttons(self, lst) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(3)
        b_add = QPushButton("+ Add")
        b_del = QPushButton("− Del")
        b_add.setFixedWidth(55)
        b_del.setFixedWidth(55)
        row.addWidget(b_add); row.addWidget(b_del); row.addStretch()
        b_add.clicked.connect(lambda: self._ignore_add(lst))
        b_del.clicked.connect(lambda: self._ignore_del(lst))
        lst.model().rowsInserted.connect(lambda *_: self._refresh_status_for(lst))
        lst.model().rowsRemoved.connect(lambda *_: self._refresh_status_for(lst))
        return row

    # ---- right panel ---------------------------------------------------

    def _build_right_panel(self) -> QWidget:
        # Vertical splitter: tabs on top, always-visible detail pane below.
        vsplit = QSplitter(Qt.Vertical)
        vsplit.setHandleWidth(4)

        self._right_tabs = QTabWidget()
        self._right_tabs.setTabPosition(QTabWidget.North)
        self._right_tabs.addTab(self._build_presence_tab(),  "Presence")
        self._right_tabs.addTab(self._build_channels_tab(),  "Channels")
        self._right_tabs.addTab(self._build_annots_tab(),    "Annotations")
        self._right_tabs.addTab(self._build_domains_tab(),   "Domains")
        self._right_tabs.addTab(self._build_coverage_tab(),  "Coverage")
        self._right_tabs.currentChanged.connect(self._on_right_tab_changed)
        vsplit.addWidget(self._right_tabs)

        vsplit.addWidget(self._build_detail_pane())
        vsplit.setStretchFactor(0, 3)
        vsplit.setStretchFactor(1, 1)
        vsplit.setSizes([560, 220])
        return vsplit

    def _build_presence_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        self._lbl_presence_info = QLabel("No data")
        self._lbl_presence_info.setStyleSheet("color:#888; font-size:11px;")
        self._lbl_presence_info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self._combo_view = QComboBox()
        self._combo_view.addItems(["Channels", "Annotations"])
        self._combo_view.currentTextChanged.connect(self._on_view_mode_changed)
        self._chk_sort_coverage = QCheckBox("Sort by coverage")
        self._chk_sort_coverage.setChecked(True)
        self._chk_sort_coverage.setToolTip(
            "Rows: most-common channels first\n"
            "Columns: most-complete subjects first"
        )
        self._chk_sort_coverage.toggled.connect(self._refresh_presence)
        btn_refresh = QPushButton("Refresh")
        btn_refresh.setFixedWidth(65)
        btn_refresh.clicked.connect(self._refresh_presence)
        top.addWidget(self._lbl_presence_info)
        top.addStretch()
        top.addWidget(QLabel("Mode:"))
        top.addWidget(self._combo_view)
        top.addWidget(self._chk_sort_coverage)
        top.addWidget(btn_refresh)
        lay.addLayout(top)

        self._view_presence = QTableView()
        self._view_presence.setShowGrid(False)
        self._view_presence.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._view_presence.setSelectionBehavior(QAbstractItemView.SelectRows)
        # Keep cell colours dominant; selection shows only as a subtle border
        self._view_presence.setStyleSheet(
            "QTableView { selection-background-color: transparent; }"
            "QTableView::item:selected { border: 1px solid #4cc9f0; }"
        )
        self._view_presence.setContextMenuPolicy(Qt.CustomContextMenu)
        self._view_presence.customContextMenuRequested.connect(
            self._presence_context_menu
        )
        hdr = self._view_presence.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Fixed)
        hdr.setMinimumSectionSize(6)
        hdr.setVisible(False)
        vhdr = self._view_presence.verticalHeader()
        vhdr.setSectionResizeMode(QHeaderView.Fixed)
        vhdr.setMinimumSectionSize(6)
        vhdr.sectionClicked.connect(self._on_presence_row_clicked)
        lay.addWidget(self._view_presence)
        return w

    def _build_channels_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(4)
        self._lbl_ch_info = QLabel("")
        self._lbl_ch_info.setStyleSheet("color:#888; font-size:11px;")
        self._lbl_ch_info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        lay.addWidget(self._lbl_ch_info)

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        self._view_channels = QTableView()
        self._view_channels.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._view_channels.setSortingEnabled(True)
        self._view_channels.setContextMenuPolicy(Qt.CustomContextMenu)
        self._view_channels.customContextMenuRequested.connect(
            self._channels_context_menu
        )
        self._view_channels.clicked.connect(self._on_channels_row_clicked)
        splitter.addWidget(self._view_channels)

        pairs_w = QWidget()
        pairs_lay = QVBoxLayout(pairs_w)
        pairs_lay.setContentsMargins(0, 2, 0, 0)
        pairs_lay.setSpacing(2)
        self._lbl_pairs_info = QLabel("")
        self._lbl_pairs_info.setStyleSheet("color:#888; font-size:11px;")
        self._lbl_pairs_info.setWordWrap(True)
        pairs_lay.addWidget(self._lbl_pairs_info)
        self._view_pairs = QTableView()
        self._view_pairs.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._view_pairs.setSortingEnabled(True)
        self._view_pairs.clicked.connect(self._on_pairs_row_clicked)
        pairs_lay.addWidget(self._view_pairs, 1)
        splitter.addWidget(pairs_w)

        splitter.setSizes([300, 130])
        lay.addWidget(splitter, 1)
        return w

    def _build_annots_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        self._lbl_ann_info = QLabel("")
        self._lbl_ann_info.setStyleSheet("color:#888; font-size:11px;")
        self._lbl_ann_info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        lay.addWidget(self._lbl_ann_info)

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        self._view_annots = QTableView()
        self._view_annots.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._view_annots.setSortingEnabled(True)
        self._view_annots.setContextMenuPolicy(Qt.CustomContextMenu)
        self._view_annots.customContextMenuRequested.connect(
            self._annots_context_menu
        )
        self._view_annots.clicked.connect(self._on_annots_row_clicked)
        splitter.addWidget(self._view_annots)

        ann_pairs_w = QWidget()
        ann_pairs_lay = QVBoxLayout(ann_pairs_w)
        ann_pairs_lay.setContentsMargins(0, 2, 0, 0)
        ann_pairs_lay.setSpacing(2)
        self._lbl_ann_pairs_info = QLabel("")
        self._lbl_ann_pairs_info.setStyleSheet("color:#888; font-size:11px;")
        self._lbl_ann_pairs_info.setWordWrap(True)
        ann_pairs_lay.addWidget(self._lbl_ann_pairs_info)
        self._view_ann_pairs = QTableView()
        self._view_ann_pairs.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._view_ann_pairs.setSortingEnabled(True)
        self._view_ann_pairs.clicked.connect(self._on_ann_pairs_row_clicked)
        ann_pairs_lay.addWidget(self._view_ann_pairs, 1)
        splitter.addWidget(ann_pairs_w)

        splitter.setSizes([300, 130])
        lay.addWidget(splitter, 1)
        return w

    def _build_domains_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(4)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel("Domain assignments (from TYPES cmd, editable)")
        lbl.setStyleSheet("color:#888; font-size:11px;")
        btn_save = QPushButton("Save…"); btn_save.setFixedWidth(52)
        btn_load = QPushButton("Load…"); btn_load.setFixedWidth(52)
        top.addWidget(lbl); top.addStretch()
        top.addWidget(btn_save); top.addWidget(btn_load)
        lay.addLayout(top)

        lbl_hint = QLabel(f"Domains: {', '.join(d for d in _DOMAINS if d)}")
        lbl_hint.setStyleSheet("color:#555; font-size:10px;")
        lay.addWidget(lbl_hint)

        self._view_domains = QTableView()
        self._view_domains.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._view_domains.setSortingEnabled(True)
        lay.addWidget(self._view_domains)

        btn_save.clicked.connect(self._save_domains)
        btn_load.clicked.connect(self._load_domains)
        return w

    def _build_coverage_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(4)
        self._lbl_coverage_summary = QLabel("")
        self._lbl_coverage_summary.setStyleSheet(
            f"color:{FG}; font-size:12px; font-weight:bold;"
        )
        self._lbl_coverage_summary.setWordWrap(True)
        lay.addWidget(self._lbl_coverage_summary)
        self._view_coverage = QTableView()
        self._view_coverage.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._view_coverage.setSortingEnabled(True)
        lay.addWidget(self._view_coverage)
        return w

    # ------------------------------------------------------------------
    # Scan lifecycle
    # ------------------------------------------------------------------

    def _start_scan(self):
        if self._scanning:
            return
        ids = self._get_sample_ids()
        if not ids:
            QMessageBox.warning(
                self._root, "Harmonizer",
                "No subjects in the sample list."
            )
            return

        self._scanning = True
        self._btn_scan.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._prog_bar.setRange(0, len(ids))
        self._prog_bar.setValue(0)
        self._prog_bar.setVisible(True)
        self._lbl_status.setText(f"Scanning 0 / {len(ids)}…")
        self._stop_flag.clear()

        sc  = self._chk_channels.isChecked()
        sa  = self._chk_annots.isChecked()
        sep = self._chk_edfplus.isChecked()

        self._fut = self.ctrl._exec.submit(
            self._do_scan, ids, sc, sa, sep
        )

    def _do_scan(self, ids, scan_channels, scan_annots, scan_edfplus):
        try:
            from .harmonizer_funcs import scan_cohort
            result = scan_cohort(
                self.ctrl.proj,
                ids,
                scan_channels=scan_channels,
                scan_annots=scan_annots,
                scan_edfplus=scan_edfplus,
                stop_flag=self._stop_flag,
                progress_cb=lambda d, t: self._sig_progress.emit(d, t),
            )
            self._sig_ok.emit(result)
        except Exception:
            self._sig_err.emit(traceback.format_exc())

    @Slot(int, int)
    def _on_progress(self, done: int, total: int):
        self._prog_bar.setValue(done)
        self._lbl_status.setText(f"Scanning {done} / {total}…")

    @Slot(object)
    def _on_scan_ok(self, result: ScanResult):
        self._scan = result
        self._end_scan_ui()
        n_ch  = result.channels_df['CH'].nunique() if not result.channels_df.empty else 0
        n_ann = result.annots_df['ANNOT'].nunique() if not result.annots_df.empty else 0
        n_sub = len(result.ids)
        stopped = n_sub < result.n_total
        note = f" (stopped at {n_sub})" if stopped else ""
        self._lbl_status.setText(
            f"Scanned {n_sub}/{result.n_total} subjects{note}  ·  "
            f"{n_ch} channels  ·  {n_ann} annot classes  ·  {result.scan_ts}"
        )
        self._on_right_tab_changed(self._right_tabs.currentIndex())

    @Slot(str)
    def _on_scan_err(self, tb: str):
        self._end_scan_ui()
        self._lbl_status.setText("Scan error — see console")
        print(f"[Harmonizer] scan error:\n{tb}")

    def _end_scan_ui(self):
        self._scanning = False
        self._btn_scan.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._prog_bar.setVisible(False)

    def _request_stop(self):
        self._stop_flag.set()
        self._lbl_status.setText("Stopping…")

    # ------------------------------------------------------------------
    # Right-panel refresh
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_right_tab_changed(self, idx: int):
        if self._scan is None:
            return
        if idx == 0:
            self._refresh_presence()
        elif idx == 1:
            self._populate_channels_tab()
        elif idx == 2:
            self._populate_annots_tab()
        elif idx == 3:
            self._populate_domains_tab()
        elif idx == 4:
            self._populate_coverage_tab()

    def _on_view_mode_changed(self, _=None):
        if self._right_tabs.currentIndex() == 0:
            self._refresh_presence()

    def _on_apply_remap_toggled(self, _=None):
        self._on_right_tab_changed(self._right_tabs.currentIndex())

    def _on_remap_table_changed(self, item=None):
        if self._chk_apply_remap.isChecked():
            self._on_right_tab_changed(self._right_tabs.currentIndex())
        else:
            tbl = item.tableWidget() if item else None
            self._refresh_status_for(tbl)

    def _refresh_status_for(self, widget=None):
        """Refresh the channels (1) or annots (2) summary tab after a remap/ignore change."""
        if not hasattr(self, '_right_tabs'):
            return
        cur = self._right_tabs.currentIndex()
        if widget is getattr(self, '_tbl_ch_remap', None) or \
                widget is getattr(self, '_lst_ignore_ch', None):
            if cur == 1:
                self._populate_channels_tab()
        elif widget is getattr(self, '_tbl_ann_remap', None) or \
                widget is getattr(self, '_lst_ignore_ann', None):
            if cur == 2:
                self._populate_annots_tab()
        else:
            if cur == 1:
                self._populate_channels_tab()
            elif cur == 2:
                self._populate_annots_tab()

    def _populate_if_active(self, tab_idx: int):
        if self._right_tabs.currentIndex() == tab_idx:
            self._on_right_tab_changed(tab_idx)

    # ---- Presence matrix -----------------------------------------------

    def _refresh_presence(self):
        if self._scan is None:
            return
        apply   = self._chk_apply_remap.isChecked()
        mode    = self._combo_view.currentText()
        filt    = self._txt_filter.text().strip().lower()
        cell_px = self._spin_cell.value()

        if mode == "Channels":
            remap  = self._get_ch_remap()  if apply else {}
            ignore = self._get_ch_ignore() if apply else set()
            row_names, col_ids, mat = build_presence(
                self._scan.channels_df, 'CH', 'ID',
                ordered_ids=self._scan.ids, remap=remap, ignore=ignore,
            )
        else:
            remap  = self._get_ann_remap()  if apply else {}
            ignore = self._get_ann_ignore() if apply else set()
            row_names, col_ids, mat = build_presence(
                self._scan.annots_df, 'ANNOT', 'ID',
                ordered_ids=self._scan.ids, remap=remap, ignore=ignore,
            )

        # Apply text filter to row names
        if filt and row_names:
            keep = [i for i, n in enumerate(row_names) if filt in n.lower()]
            row_names = [row_names[i] for i in keep]
            mat       = mat[keep, :] if len(keep) and mat.size else np.zeros((0, len(col_ids)), dtype=bool)

        # Sort rows by frequency desc, columns by coverage desc → staircase pattern
        if self._chk_sort_coverage.isChecked() and mat.size:
            row_order = np.argsort(-mat.sum(axis=1), kind='stable')
            col_order = np.argsort(-mat.sum(axis=0), kind='stable')
            mat       = mat[np.ix_(row_order, col_order)]
            row_names = [row_names[i] for i in row_order]
            col_ids   = [col_ids[i]   for i in col_order]

        self._presence_model = _PresenceModel(row_names, col_ids, mat)
        self._view_presence.setModel(self._presence_model)
        self._view_presence.selectionModel().selectionChanged.connect(
            self._on_presence_selection_changed
        )

        hdr = self._view_presence.horizontalHeader()
        hdr.setDefaultSectionSize(cell_px)
        vhdr = self._view_presence.verticalHeader()
        vhdr.setDefaultSectionSize(cell_px)
        self._apply_presence_view_width(len(col_ids), cell_px)

        n_rows = len(row_names)
        n_cols = len(col_ids)
        n_pres = int(mat.sum()) if mat.size else 0
        n_poss = n_rows * n_cols
        pct    = round(100 * n_pres / n_poss, 1) if n_poss else 0
        label  = "channels" if mode == "Channels" else "annots"
        self._lbl_presence_info.setText(
            f"{n_rows} {label} × {n_cols} subjects  ·  "
            f"{n_pres}/{n_poss} cells present ({pct}%)"
        )

    def _apply_cell_size(self, px: int):
        model = self._view_presence.model()
        if model is None:
            return
        hdr  = self._view_presence.horizontalHeader()
        vhdr = self._view_presence.verticalHeader()
        hdr.setDefaultSectionSize(px)
        vhdr.setDefaultSectionSize(px)
        self._apply_presence_view_width(model.columnCount(), px)

    def _apply_presence_view_width(self, n_cols: int, cell_px: int):
        vhdr = self._view_presence.verticalHeader()
        vhdr_w = vhdr.sizeHint().width()
        frame_w = self._view_presence.frameWidth() * 2
        scroll_w = self._view_presence.style().pixelMetric(
            QtWidgets.QStyle.PM_ScrollBarExtent, None, self._view_presence
        )
        min_visual_cols = 8
        # Keep the table scrollable instead of sizing the whole window to the
        # full cohort width, which can become enormous for large sample lists.
        max_visual_cols = 64
        visible_cols = max(min_visual_cols, min(int(n_cols), max_visual_cols))
        width = vhdr_w + (visible_cols * int(cell_px)) + scroll_w + frame_w + 6
        self._view_presence.setMinimumWidth(width)
        self._view_presence.setMaximumWidth(16777215)

    # ---- Channels tab --------------------------------------------------

    def _populate_channels_tab(self):
        if self._scan is None or self._scan.channels_df.empty:
            self._lbl_ch_info.setText("No channel data")
            self._lbl_pairs_info.setText("")
            return

        apply  = self._chk_apply_remap.isChecked()
        remap_full  = self._get_ch_remap()
        ignore_full = self._get_ch_ignore()
        remap  = remap_full if apply else {}
        ignore = ignore_full if apply else set()

        summary = channel_summary(
            self._scan.channels_df,
            remap,
            ignore,
            split_by_sr=self._chk_show_sr.isChecked(),
            split_by_trans=self._chk_show_trans.isChecked(),
            split_by_pdim=self._chk_show_pdim.isChecked(),
        )
        domains_df = domain_assignments(
            self._scan.channels_df,
            types_df=self._scan.types_df,
            remap=remap,
            ignore=ignore,
            user_domains=self._domains,
        )
        if not domains_df.empty:
            summary = summary.merge(domains_df[['CH', 'Domain']], on='CH', how='left')
        else:
            summary['Domain'] = ''

        reverse_alias = {}
        for orig, canon in remap_full.items():
            reverse_alias.setdefault(canon, []).append(orig)

        def _status_for_channel(name: str) -> str:
            if not apply:
                if name in ignore_full:
                    return 'Ignored'
                canon = remap_full.get(name, '')
                if canon and canon != name:
                    return f"Alias → {canon}"
                incoming = sorted(set(reverse_alias.get(name, [])), key=str.lower)
                if incoming:
                    return f"Canonical ({len(incoming)})"
                return ''
            incoming = sorted(set(reverse_alias.get(name, [])), key=str.lower)
            if incoming:
                return f"Aliases: {', '.join(incoming[:3])}" + (
                    f" +{len(incoming) - 3}" if len(incoming) > 3 else ""
                )
            return ''

        summary['Status'] = summary['CH'].map(_status_for_channel)

        cols = ['CH', 'N', 'Domain', 'SR', 'TRANS', 'PDIM', 'Status']
        summary = summary[[c for c in cols if c in summary.columns]]

        model = QStandardItemModel(len(summary), len(summary.columns))
        model.setHorizontalHeaderLabels(list(summary.columns))
        for r, (_, row) in enumerate(summary.iterrows()):
            status = str(row.get('Status', '') or '')
            for c, val in enumerate(row):
                it = QStandardItem(str(val) if not pd.isna(val) else '')
                it.setEditable(False)
                if status == 'Ignored':
                    it.setForeground(QColor("#777777"))
                elif status.startswith('Alias →'):
                    it.setForeground(QColor("#ffd166"))
                elif status.startswith('Canonical') or status.startswith('Aliases:'):
                    it.setForeground(QColor("#7cc7ff"))
                model.setItem(r, c, it)

        proxy = QSortFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        proxy.setFilterKeyColumn(-1)
        self._view_channels.setModel(proxy)
        self._view_channels.selectionModel().selectionChanged.connect(
            self._on_channels_selection_changed
        )
        self._view_channels.horizontalHeader().setStretchLastSection(True)
        self._view_channels.resizeColumnsToContents()

        self._lbl_ch_info.setText(f"{len(summary)} channels")

        pairs = rare_cooccurrence_pairs(
            self._scan.channels_df,
            types_df=self._scan.types_df,
            remap=remap,
            ignore=ignore,
            user_domains=self._domains,
        )

        pairs_model = QStandardItemModel(len(pairs), len(pairs.columns))
        pairs_model.setHorizontalHeaderLabels(list(pairs.columns))
        for r, (_, row) in enumerate(pairs.iterrows()):
            for c, val in enumerate(row):
                it = QStandardItem(str(val) if not pd.isna(val) else '')
                it.setEditable(False)
                pairs_model.setItem(r, c, it)

        pairs_proxy = _NumericSortProxyModel()
        pairs_proxy.setSourceModel(pairs_model)
        self._view_pairs.setModel(pairs_proxy)
        self._view_pairs.horizontalHeader().setStretchLastSection(True)
        self._view_pairs.resizeColumnsToContents()

        if pairs.empty:
            self._lbl_pairs_info.setText(
                "Rare co-occurrence candidates: none found with the current remap/ignore state."
            )
        else:
            self._lbl_pairs_info.setText(
                "Rare co-occurrence candidates: channel pairs from the same inferred domain "
                "that each recur across subjects but almost never appear together."
            )

    # ---- Annotations tab -----------------------------------------------

    def _populate_annots_tab(self):
        if self._scan is None or self._scan.annots_df.empty:
            self._lbl_ann_info.setText("No annotation data")
            return

        apply  = self._chk_apply_remap.isChecked()
        remap_full  = self._get_ann_remap()
        ignore_full = self._get_ann_ignore()
        remap  = remap_full if apply else {}
        ignore = ignore_full if apply else set()

        summary = annot_summary(self._scan.annots_df, remap, ignore)

        reverse_alias = {}
        for orig, canon in remap_full.items():
            reverse_alias.setdefault(canon, []).append(orig)

        def _status_for_annot(name: str) -> str:
            if not apply:
                if name in ignore_full:
                    return 'Ignored'
                canon = remap_full.get(name, '')
                if canon and canon != name:
                    return f"Alias → {canon}"
                incoming = sorted(set(reverse_alias.get(name, [])), key=str.lower)
                if incoming:
                    return f"Canonical ({len(incoming)})"
                return ''
            incoming = sorted(set(reverse_alias.get(name, [])), key=str.lower)
            if incoming:
                return f"Aliases: {', '.join(incoming[:3])}" + (
                    f" +{len(incoming) - 3}" if len(incoming) > 3 else ""
                )
            return ''

        summary['Status'] = summary['ANNOT'].map(_status_for_annot)

        model = QStandardItemModel(len(summary), 3)
        model.setHorizontalHeaderLabels(['ANNOT', 'N', 'Status'])
        for r, (_, row) in enumerate(summary.iterrows()):
            status = str(row.get('Status', '') or '')
            for c, val in enumerate(row):
                it = QStandardItem(str(val))
                it.setEditable(False)
                if status == 'Ignored':
                    it.setForeground(QColor("#777777"))
                elif status.startswith('Alias →'):
                    it.setForeground(QColor("#ffd166"))
                elif status.startswith('Canonical') or status.startswith('Aliases:'):
                    it.setForeground(QColor("#7cc7ff"))
                model.setItem(r, c, it)

        proxy = QSortFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        proxy.setFilterKeyColumn(-1)
        self._view_annots.setModel(proxy)
        self._view_annots.selectionModel().selectionChanged.connect(
            self._on_annots_selection_changed
        )
        self._view_annots.horizontalHeader().setStretchLastSection(True)
        self._view_annots.resizeColumnsToContents()

        self._lbl_ann_info.setText(f"{len(summary)} annotation classes")

        pairs = annot_rare_cooccurrence_pairs(
            self._scan.annots_df, remap=remap, ignore=ignore,
        )
        pairs_model = QStandardItemModel(len(pairs), len(pairs.columns))
        pairs_model.setHorizontalHeaderLabels(list(pairs.columns))
        for r, (_, row) in enumerate(pairs.iterrows()):
            for c, val in enumerate(row):
                it = QStandardItem(str(val) if not pd.isna(val) else '')
                it.setEditable(False)
                pairs_model.setItem(r, c, it)
        pairs_proxy = _NumericSortProxyModel()
        pairs_proxy.setSourceModel(pairs_model)
        self._view_ann_pairs.setModel(pairs_proxy)
        self._view_ann_pairs.horizontalHeader().setStretchLastSection(True)
        self._view_ann_pairs.resizeColumnsToContents()
        self._lbl_ann_pairs_info.setText(
            "Rare co-occurrence candidates: none found." if pairs.empty else
            "Rare co-occurrence candidates: annotation pairs that each recur across subjects "
            "but almost never appear together."
        )

    # ---- Domains tab ---------------------------------------------------

    def _populate_domains_tab(self):
        if self._scan is None or self._scan.channels_df.empty:
            return

        apply  = self._chk_apply_remap.isChecked()
        remap  = self._get_ch_remap()  if apply else {}
        ignore = self._get_ch_ignore() if apply else set()

        domains_df = domain_assignments(
            self._scan.channels_df,
            types_df=self._scan.types_df,
            remap=remap,
            ignore=ignore,
            user_domains=self._domains,
        )

        model = QStandardItemModel(len(domains_df), 2)
        model.setHorizontalHeaderLabels(['CH', 'Domain'])
        for r, (_, row) in enumerate(domains_df.iterrows()):
            name = str(row.get('CH', ''))
            it_ch = QStandardItem(name)
            it_ch.setEditable(False)
            domain = str(row.get('Domain', ''))
            it_dom = QStandardItem(domain)
            it_dom.setEditable(True)
            model.setItem(r, 0, it_ch)
            model.setItem(r, 1, it_dom)

        model.dataChanged.connect(self._on_domains_changed)

        proxy = QSortFilterProxyModel()
        proxy.setSourceModel(model)
        self._view_domains.setModel(proxy)
        self._view_domains.horizontalHeader().setStretchLastSection(True)
        self._view_domains.resizeColumnsToContents()

    def _on_domains_changed(self, tl, br, roles):
        model = self._view_domains.model()
        src = getattr(model, 'sourceModel', lambda: model)()
        for r in range(src.rowCount()):
            ch  = str(src.data(src.index(r, 0)) or '')
            dom = str(src.data(src.index(r, 1)) or '')
            if ch:
                self._domains[ch] = dom
        if self._right_tabs.currentIndex() == 1:
            self._populate_channels_tab()

    # ---- Coverage tab --------------------------------------------------

    def _populate_coverage_tab(self):
        if self._scan is None or self._scan.channels_df.empty:
            self._lbl_coverage_summary.setText("No channel scan data.")
            return

        apply  = self._chk_apply_remap.isChecked()
        remap  = self._get_ch_remap()  if apply else {}
        ignore = self._get_ch_ignore() if apply else set()

        df = coverage_stats(self._scan.channels_df, remap, ignore)
        if df.empty:
            self._lbl_coverage_summary.setText("No data.")
            return

        n_can  = df['N_canonical'].iloc[0] if not df.empty else 0
        n_full = (df['Pct'] == 100).sum()
        pct_full = round(100 * n_full / len(df), 1) if len(df) else 0
        self._lbl_coverage_summary.setText(
            f"{n_can} canonical channels  ·  "
            f"{n_full} / {len(df)} subjects ({pct_full}%) fully covered"
        )

        model = QStandardItemModel(len(df), 4)
        model.setHorizontalHeaderLabels(['ID', 'Present', 'Canonical', '%'])
        for r, (_, row) in enumerate(df.iterrows()):
            for c, val in enumerate(row):
                it = QStandardItem(str(val))
                it.setEditable(False)
                model.setItem(r, c, it)

        proxy = QSortFilterProxyModel()
        proxy.setSourceModel(model)
        self._view_coverage.setModel(proxy)
        self._view_coverage.horizontalHeader().setStretchLastSection(True)
        self._view_coverage.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Detail pane construction
    # ------------------------------------------------------------------

    def _build_detail_pane(self) -> QWidget:
        """Always-visible cross-talk panel.

        Row-header click  → channel / annotation detail (which subjects have it).
        Column-header click → subject detail (which channels / annots it has).
        Cell click         → subject detail for that cell's subject.
        Rows inside the detail table cross-link back.
        """
        w = QWidget()
        w.setMinimumHeight(80)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(3)

        # ---- header row ------------------------------------------------
        hdr = QHBoxLayout()
        hdr.setContentsMargins(0, 0, 0, 0)
        hdr.setSpacing(6)

        self._lbl_detail_icon  = QLabel("◈")
        self._lbl_detail_icon.setStyleSheet("color:#4cc9f0; font-size:14px;")
        self._lbl_detail_icon.setFixedWidth(18)

        self._lbl_detail_title = QLabel(
            "Click a channel/annot row  ·  or a subject column  ·  in the Presence matrix"
        )
        self._lbl_detail_title.setStyleSheet(f"color:{FG}; font-size:11px;")
        self._lbl_detail_title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self._chk_detail_all = QCheckBox("Show all rows")
        self._chk_detail_all.setChecked(False)
        self._chk_detail_all.setVisible(False)
        self._chk_detail_all.toggled.connect(self._on_detail_all_toggled)

        # Subject view: toggle channels ↔ annots
        self._btn_detail_toggle = QPushButton("⇄ Annots")
        self._btn_detail_toggle.setFixedWidth(88)
        self._btn_detail_toggle.setCheckable(True)
        self._btn_detail_toggle.setVisible(False)
        self._btn_detail_toggle.clicked.connect(self._toggle_subject_view)

        # Subject view: open in viewer
        self._btn_detail_open = QPushButton("Open →")
        self._btn_detail_open.setFixedWidth(65)
        self._btn_detail_open.setVisible(False)
        self._btn_detail_open.clicked.connect(self._detail_open_in_viewer)

        # Channel / annot view: quick remap + ignore
        self._btn_detail_remap  = QPushButton("Add to remap")
        self._btn_detail_remap.setFixedWidth(100)
        self._btn_detail_remap.setVisible(False)
        self._btn_detail_remap.clicked.connect(self._detail_add_to_remap)

        self._btn_detail_ignore = QPushButton("Ignore")
        self._btn_detail_ignore.setFixedWidth(58)
        self._btn_detail_ignore.setVisible(False)
        self._btn_detail_ignore.clicked.connect(self._detail_add_to_ignore)

        hdr.addWidget(self._lbl_detail_icon)
        hdr.addWidget(self._lbl_detail_title)
        hdr.addStretch()
        hdr.addWidget(self._chk_detail_all)
        hdr.addWidget(self._btn_detail_toggle)
        hdr.addWidget(self._btn_detail_remap)
        hdr.addWidget(self._btn_detail_ignore)
        hdr.addWidget(self._btn_detail_open)
        lay.addLayout(hdr)
        lay.addWidget(_hsep())

        self._view_detail = QTableView()
        self._view_detail.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._view_detail.setSortingEnabled(True)
        self._view_detail.clicked.connect(self._on_detail_row_clicked)
        lay.addWidget(self._view_detail)

        # State
        self._detail_mode            = None     # 'channel' | 'subject' | 'annot'
        self._detail_name            = None     # name of the selected item
        self._detail_subject_annots  = False    # subject view: show annots instead of channels
        self._detail_df              = None
        self._detail_cols            = None
        self._detail_color_col       = None

        return w

    # ------------------------------------------------------------------
    # Cross-talk: show detail for a channel / annot / subject
    # ------------------------------------------------------------------

    def _show_channel_detail(self, ch_name: str):
        """Show detail for a single channel (delegates to multi-channel)."""
        self._show_channels_detail([ch_name])

    def _show_channels_detail(self, ch_names: list | None = None, row_specs: list | None = None):
        """Populate detail pane for one or more channels: Subject × CH breakdown."""
        if self._scan is None or self._scan.channels_df.empty:
            return
        if row_specs:
            ch_names = [str(spec.get('CH', '')) for spec in row_specs if str(spec.get('CH', '')).strip()]
        if not ch_names:
            return
        self._detail_mode = 'channel'
        self._detail_name = ch_names[0] if len(ch_names) == 1 else None

        apply = self._chk_apply_remap.isChecked()
        remap = self._get_ch_remap() if apply else {}

        df = self._scan.channels_df.copy()
        df['CH'] = df['CH'].map(lambda x: remap.get(x, x))

        if row_specs:
            specs = []
            for spec in row_specs:
                canon = remap.get(spec.get('CH', ''), spec.get('CH', ''))
                item = {
                    'CH': canon,
                    'SR': str(spec.get('SR', '') or ''),
                    'TRANS': str(spec.get('TRANS', '') or ''),
                    'PDIM': str(spec.get('PDIM', '') or ''),
                }
                if item not in specs and item['CH']:
                    specs.append(item)
        else:
            specs = []
            for ch in ch_names:
                canon = remap.get(ch, ch)
                item = {'CH': canon, 'SR': '', 'TRANS': '', 'PDIM': ''}
                if item not in specs:
                    specs.append(item)

        rows = []
        for spec in specs:
            present = df[df['CH'] == spec['CH']]
            if spec['SR']:
                present = present[present['SR'].astype(str) == spec['SR']]
            if spec['TRANS']:
                present = present[present['TRANS'].astype(str) == spec['TRANS']]
            if spec['PDIM']:
                present = present[present['PDIM'].astype(str) == spec['PDIM']]
            present = present.drop_duplicates('ID').set_index('ID')
            for id_str in self._scan.ids:
                if id_str in present.index:
                    r = present.loc[id_str]
                    rows.append({
                        'Subject': id_str,
                        'CH':      spec['CH'],
                        'Present': '✓',
                        'SR':      str(r.get('SR',    '')),
                        'TRANS':   str(r.get('TRANS', '')),
                        'PDIM':    str(r.get('PDIM',  '')),
                    })
                else:
                    rows.append({'Subject': id_str, 'CH': spec['CH'],
                                 'Present': '—', 'SR': '', 'TRANS': '', 'PDIM': ''})

        if not rows:
            return

        full_df   = pd.DataFrame(rows)
        n_present = int((full_df['Present'] == '✓').sum())
        n_total   = len(rows)
        pct       = round(100 * n_present / n_total, 1) if n_total else 0

        if len(specs) == 1:
            spec = specs[0]
            ch = ch_names[0]
            canon = spec['CH']
            alias = f" → {canon}" if canon != ch else ""
            qualifiers = []
            if spec['SR']:
                qualifiers.append(f"SR={spec['SR']}")
            if spec['TRANS']:
                qualifiers.append(f"TRANS={spec['TRANS']}")
            if spec['PDIM']:
                qualifiers.append(f"PDIM={spec['PDIM']}")
            qual_txt = f"  ·  {'  ·  '.join(qualifiers)}" if qualifiers else ""
            title = (
                f"Channel: {ch}{alias}{qual_txt}   ·   "
                f"{n_present} / {len(self._scan.ids)} subjects ({pct}%)"
            )
        else:
            title = (
                f"{len(specs)} channels selected   ·   "
                f"{n_present} / {n_total} subject×channel pairs present ({pct}%)"
            )

        self._lbl_detail_icon.setText("▣")
        self._lbl_detail_icon.setStyleSheet("color:#06d6a0; font-size:14px;")
        self._lbl_detail_title.setText(title)
        self._btn_detail_toggle.setVisible(False)
        self._btn_detail_open.setVisible(False)
        self._btn_detail_remap.setVisible(len(ch_names) == 1)
        self._btn_detail_ignore.setVisible(len(ch_names) == 1)
        self._chk_detail_all.setVisible(True)

        cols = ['Subject', 'CH', 'Present', 'SR', 'TRANS', 'PDIM']
        self._populate_detail_table(full_df, cols=cols, color_col='Present')

    def _show_annot_detail(self, ann_name: str):
        """Populate detail pane with all subjects for this annotation class."""
        if self._scan is None or self._scan.annots_df.empty:
            return
        self._detail_mode = 'annot'
        self._detail_name = ann_name

        apply     = self._chk_apply_remap.isChecked()
        remap     = self._get_ann_remap() if apply else {}
        canonical = remap.get(ann_name, ann_name)

        df = self._scan.annots_df.copy()
        df['ANNOT'] = df['ANNOT'].map(lambda x: remap.get(x, x))
        present_ids = set(df[df['ANNOT'] == canonical]['ID'].unique())

        rows = [
            {'Subject': iid, 'Present': '✓' if iid in present_ids else '—'}
            for iid in self._scan.ids
        ]

        n_present = len(present_ids)
        n_total   = len(self._scan.ids)
        pct       = round(100 * n_present / n_total, 1) if n_total else 0
        alias     = f" → {canonical}" if canonical != ann_name else ""

        self._lbl_detail_icon.setText("▸")
        self._lbl_detail_icon.setStyleSheet("color:#ffd166; font-size:14px;")
        self._lbl_detail_title.setText(
            f"Annotation: {ann_name}{alias}   ·   {n_present} / {n_total} subjects ({pct}%)"
        )
        self._btn_detail_toggle.setVisible(False)
        self._btn_detail_open.setVisible(False)
        self._btn_detail_remap.setVisible(True)
        self._btn_detail_ignore.setVisible(True)
        self._chk_detail_all.setVisible(True)

        self._populate_detail_table(
            pd.DataFrame(rows),
            cols=['Subject', 'Present'],
            color_col='Present',
        )

    def _show_annots_detail(self, ann_names: list):
        """Detail pane for two annotation classes: Subject × ANNOT breakdown."""
        if not ann_names:
            return
        if len(ann_names) == 1:
            self._show_annot_detail(ann_names[0])
            return
        if self._scan is None or self._scan.annots_df.empty:
            return

        self._detail_mode = 'annot'
        self._detail_name = None

        apply = self._chk_apply_remap.isChecked()
        remap = self._get_ann_remap() if apply else {}

        df = self._scan.annots_df.copy()
        df['ANNOT'] = df['ANNOT'].map(lambda x: remap.get(x, x))

        rows = []
        for ann in ann_names:
            canon = remap.get(ann, ann)
            present_ids = set(df[df['ANNOT'] == canon]['ID'].unique())
            for iid in self._scan.ids:
                rows.append({
                    'Subject': iid,
                    'ANNOT':   canon,
                    'Present': '✓' if iid in present_ids else '—',
                })

        full_df   = pd.DataFrame(rows)
        n_present = int((full_df['Present'] == '✓').sum())
        n_total   = len(rows)
        pct       = round(100 * n_present / n_total, 1) if n_total else 0

        self._lbl_detail_icon.setText("▸")
        self._lbl_detail_icon.setStyleSheet("color:#ffd166; font-size:14px;")
        self._lbl_detail_title.setText(
            f"Annotations: {' / '.join(ann_names)}   ·   "
            f"{n_present} / {n_total} subject×annot pairs present ({pct}%)"
        )
        self._btn_detail_toggle.setVisible(False)
        self._btn_detail_open.setVisible(False)
        self._btn_detail_remap.setVisible(False)
        self._btn_detail_ignore.setVisible(False)
        self._chk_detail_all.setVisible(True)

        self._populate_detail_table(full_df, cols=['Subject', 'ANNOT', 'Present'],
                                    color_col='Present')

    def _show_subject_detail(self, id_str: str, show_annots: bool | None = None):
        """Populate detail pane with all channels (or annots) for this subject."""
        if self._scan is None:
            return
        self._detail_mode = 'subject'
        self._detail_name = id_str
        if show_annots is not None:
            self._detail_subject_annots = show_annots

        apply  = self._chk_apply_remap.isChecked()

        if not self._detail_subject_annots:
            # ---- channels view -----------------------------------------
            remap  = self._get_ch_remap()  if apply else {}
            ignore = self._get_ch_ignore() if apply else set()

            df = self._scan.channels_df.copy()
            df['CH'] = df['CH'].map(lambda x: remap.get(x, x))
            df = df[~df['CH'].isin(ignore)]
            all_ch = sorted(df['CH'].dropna().astype(str).unique())
            sub_present = (
                df[df['ID'] == id_str][['CH', 'SR', 'TRANS', 'PDIM']]
                .drop_duplicates('CH')
                .set_index('CH')
            )
            rows = []
            for ch in all_ch:
                if ch in sub_present.index:
                    r = sub_present.loc[ch]
                    rows.append({
                        'CH': ch,
                        'Present': '✓',
                        'SR': str(r.get('SR', '')),
                        'TRANS': str(r.get('TRANS', '')),
                        'PDIM': str(r.get('PDIM', '')),
                    })
                else:
                    rows.append({
                        'CH': ch,
                        'Present': '—',
                        'SR': '',
                        'TRANS': '',
                        'PDIM': '',
                    })
            sub = pd.DataFrame(rows)

            n_ch  = int((sub['Present'] == '✓').sum()) if not sub.empty else 0
            n_ann = self._scan.annots_df['ANNOT'].nunique() \
                    if not self._scan.annots_df.empty else 0
            has_annots = not self._scan.annots_df.empty

            self._lbl_detail_icon.setText("◉")
            self._lbl_detail_icon.setStyleSheet("color:#4cc9f0; font-size:14px;")
            self._lbl_detail_title.setText(
                f"Subject: {id_str}   ·   {n_ch} channels"
                + (f"  ·  {n_ann} annot classes" if has_annots else "")
            )
            self._populate_detail_table(sub, cols=['CH', 'Present', 'SR', 'TRANS', 'PDIM'],
                                        color_col='Present')

        else:
            # ---- annotations view --------------------------------------
            remap  = self._get_ann_remap()  if apply else {}
            ignore = self._get_ann_ignore() if apply else set()

            df = self._scan.annots_df.copy()
            df['ANNOT'] = df['ANNOT'].map(lambda x: remap.get(x, x))
            df = df[~df['ANNOT'].isin(ignore)]
            all_ann = sorted(df['ANNOT'].dropna().astype(str).unique())
            sub_present = set(df[df['ID'] == id_str]['ANNOT'].dropna().astype(str).unique())
            sub = pd.DataFrame([
                {'ANNOT': ann, 'Present': '✓' if ann in sub_present else '—'}
                for ann in all_ann
            ])

            n_ch  = self._scan.channels_df[self._scan.channels_df['ID'] == id_str]['CH'].nunique() \
                    if not self._scan.channels_df.empty else 0

            self._lbl_detail_icon.setText("◉")
            self._lbl_detail_icon.setStyleSheet("color:#4cc9f0; font-size:14px;")
            self._lbl_detail_title.setText(
                f"Subject: {id_str}   ·   {n_ch} channels"
                + f"  ·  {int((sub['Present'] == '✓').sum()) if not sub.empty else 0} annot classes"
            )
            self._populate_detail_table(sub, cols=['ANNOT', 'Present'],
                                        color_col='Present')

        # Update toggle button label
        toggle_label = "⇄ Channels" if self._detail_subject_annots else "⇄ Annots"
        self._btn_detail_toggle.setText(toggle_label)
        self._btn_detail_toggle.setChecked(self._detail_subject_annots)
        has_ann = not (self._scan.annots_df.empty if self._scan else True)
        self._chk_detail_all.setVisible(True)
        self._btn_detail_toggle.setVisible(has_ann)
        self._btn_detail_open.setVisible(True)
        self._btn_detail_remap.setVisible(False)
        self._btn_detail_ignore.setVisible(False)

    def _populate_detail_table(self, df: pd.DataFrame,
                               cols: list | None = None,
                               color_col: str | None = None):
        self._detail_df = df.copy()
        self._detail_cols = list(cols) if cols else None
        self._detail_color_col = color_col
        self._refresh_detail_table()

    @Slot(bool)
    def _on_detail_all_toggled(self, checked: bool):
        self._refresh_detail_table()

    def _refresh_detail_table(self):
        df = self._detail_df.copy() if isinstance(self._detail_df, pd.DataFrame) else None
        if df is None:
            return
        cols = self._detail_cols
        color_col = self._detail_color_col
        if cols:
            df = df[[c for c in cols if c in df.columns]]
        if 'Present' in df.columns and not self._chk_detail_all.isChecked():
            df = df[df['Present'] == '✓']
        df = df.reset_index(drop=True)

        model = QStandardItemModel(len(df), len(df.columns))
        model.setHorizontalHeaderLabels(list(df.columns))

        cc_idx = list(df.columns).index(color_col) if color_col and color_col in df.columns else -1

        for r, row in df.iterrows():
            for c, val in enumerate(row):
                it = QStandardItem(str(val) if not pd.isna(val) else '')
                it.setEditable(False)
                if c == cc_idx:
                    if str(val) == '✓':
                        it.setForeground(QColor("#4ade80"))
                    elif str(val) == '—':
                        it.setForeground(QColor("#444"))
                model.setItem(r, c, it)

        proxy = QSortFilterProxyModel()
        proxy.setSourceModel(model)
        self._view_detail.setModel(proxy)
        self._view_detail.horizontalHeader().setStretchLastSection(True)
        self._view_detail.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Detail pane: row clicks cross-link back
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_detail_row_clicked(self, index):
        model = self._view_detail.model()
        if model is None:
            return
        val = str(model.data(model.index(index.row(), 0), Qt.DisplayRole) or '').strip()
        if not val or val == '—' or val == '✓':
            return

        if self._detail_mode in ('channel', 'annot'):
            # Rows are subjects → drill into that subject
            self._show_subject_detail(val)
        elif self._detail_mode == 'subject':
            # Rows are channels or annots
            if self._detail_subject_annots:
                self._show_annot_detail(val)
            else:
                self._show_channel_detail(val)

    # ------------------------------------------------------------------
    # Detail pane: action buttons
    # ------------------------------------------------------------------

    def _toggle_subject_view(self):
        if self._detail_mode == 'subject' and self._detail_name:
            self._show_subject_detail(
                self._detail_name,
                show_annots=not self._detail_subject_annots,
            )

    def _detail_open_in_viewer(self):
        if self._detail_mode == 'subject' and self._detail_name:
            self._open_subject_in_viewer(self._detail_name)

    def _detail_add_to_remap(self):
        if self._detail_mode == 'channel' and self._detail_name:
            self._prompt_add_to_remap('channel', self._detail_name)
        elif self._detail_mode == 'annot' and self._detail_name:
            self._prompt_add_to_remap('annot', self._detail_name)

    def _detail_add_to_ignore(self):
        if self._detail_mode == 'channel' and self._detail_name:
            self._ignore_add_item(self._lst_ignore_ch, self._detail_name)
        elif self._detail_mode == 'annot' and self._detail_name:
            self._ignore_add_item(self._lst_ignore_ann, self._detail_name)

    # ------------------------------------------------------------------
    # Presence-matrix interactions
    # ------------------------------------------------------------------

    @Slot(int)
    def _on_presence_col_clicked(self, section: int):
        """Column header = subject → show subject detail + open in viewer."""
        model = self._view_presence.model()
        if model is None:
            return
        id_str = model.headerData(section, Qt.Horizontal, Qt.UserRole)
        if id_str:
            self._show_subject_detail(str(id_str))
            self._open_subject_in_viewer(str(id_str))

    @Slot(int)
    def _on_presence_row_clicked(self, section: int):
        """Row header = channel / annot → show channel or annot detail."""
        model = self._view_presence.model()
        if model is None:
            return
        name = model.headerData(section, Qt.Vertical, Qt.DisplayRole)
        if not name:
            return
        if self._combo_view.currentText() == 'Channels':
            self._show_channel_detail(str(name))
        else:
            self._show_annot_detail(str(name))

    def _on_presence_selection_changed(self, selected, deselected):
        """Row selection changed → update detail pane for all selected rows."""
        model = self._view_presence.model()
        if model is None:
            return
        rows = sorted({idx.row() for idx in self._view_presence.selectedIndexes()})
        if not rows:
            return
        mode = self._combo_view.currentText()
        if mode == 'Channels':
            names = []
            for r in rows:
                n = model.headerData(r, Qt.Vertical, Qt.DisplayRole)
                if n and n not in names:
                    names.append(str(n))
            if names:
                self._show_channels_detail(names)
        else:
            # For annots only single-annot detail makes sense; use first selected
            name = model.headerData(rows[0], Qt.Vertical, Qt.DisplayRole)
            if name:
                self._show_annot_detail(str(name))

    def _on_channels_row_clicked(self, index):
        """Channels-tab row → show channel detail (kept for direct mouse clicks)."""
        self._on_channels_selection_changed(None, None)

    def _on_channels_selection_changed(self, selected, deselected):
        """Channel tab selection changed → update detail pane."""
        model = self._view_channels.model()
        if model is None:
            return
        rows = sorted({idx.row() for idx in self._view_channels.selectedIndexes()})
        if not rows:
            return
        names = []
        specs = []
        headers = {
            str(model.headerData(c, Qt.Horizontal, Qt.DisplayRole) or ''): c
            for c in range(model.columnCount())
        }
        for r in rows:
            ch = str(model.data(model.index(r, 0), Qt.DisplayRole) or '').strip()
            if ch and ch not in names:
                names.append(ch)
            spec = {'CH': ch}
            if self._chk_show_sr.isChecked() and 'SR' in headers:
                spec['SR'] = str(model.data(model.index(r, headers['SR']), Qt.DisplayRole) or '').strip()
            if self._chk_show_trans.isChecked() and 'TRANS' in headers:
                spec['TRANS'] = str(model.data(model.index(r, headers['TRANS']), Qt.DisplayRole) or '').strip()
            if self._chk_show_pdim.isChecked() and 'PDIM' in headers:
                spec['PDIM'] = str(model.data(model.index(r, headers['PDIM']), Qt.DisplayRole) or '').strip()
            if ch:
                specs.append(spec)
        if specs:
            self._show_channels_detail(names, row_specs=specs)

    def _on_annots_row_clicked(self, index):
        """Annotations-tab row → show annot detail (kept for direct mouse clicks)."""
        self._on_annots_selection_changed(None, None)

    def _on_annots_selection_changed(self, selected, deselected):
        """Annots tab selection changed → update detail pane."""
        model = self._view_annots.model()
        if model is None:
            return
        rows = sorted({idx.row() for idx in self._view_annots.selectedIndexes()})
        if not rows:
            return
        ann = str(model.data(model.index(rows[0], 0), Qt.DisplayRole) or '').strip()
        if ann:
            self._show_annot_detail(ann)

    def _on_pairs_row_clicked(self, index):
        """Rare-pairs tab row → show the two-channel detail breakdown."""
        model = self._view_pairs.model()
        if model is None:
            return
        ch_a = str(model.data(model.index(index.row(), 0), Qt.DisplayRole) or '').strip()
        ch_b = str(model.data(model.index(index.row(), 1), Qt.DisplayRole) or '').strip()
        names = [ch for ch in (ch_a, ch_b) if ch]
        if names:
            self._show_channels_detail(names)

    def _on_ann_pairs_row_clicked(self, index):
        model = self._view_ann_pairs.model()
        if model is None:
            return
        ann_a = str(model.data(model.index(index.row(), 0), Qt.DisplayRole) or '').strip()
        ann_b = str(model.data(model.index(index.row(), 1), Qt.DisplayRole) or '').strip()
        names = [a for a in (ann_a, ann_b) if a]
        if names:
            self._show_annots_detail(names)

    def _presence_context_menu(self, pos):
        model = self._view_presence.model()
        if model is None:
            return
        idx = self._view_presence.indexAt(pos)
        if not idx.isValid():
            return

        ch_name = model.headerData(idx.row(), Qt.Vertical, Qt.DisplayRole) or ''
        id_str  = model.headerData(idx.column(), Qt.Horizontal, Qt.UserRole) or ''

        menu = QMenu(self._view_presence)
        if ch_name:
            a_remap  = menu.addAction(f"Add '{ch_name}' to remap (Original)")
            a_ignore = menu.addAction(f"Add '{ch_name}' to ignore")
        if id_str:
            a_open = menu.addAction(f"Open '{id_str}' in viewer")
        menu.addSeparator()
        a_refresh = menu.addAction("Refresh")

        action = menu.exec(self._view_presence.viewport().mapToGlobal(pos))
        if not action:
            return
        if ch_name:
            if action == a_remap:
                self._prompt_add_to_remap('channel', ch_name)
            elif action == a_ignore:
                self._ignore_add_item(self._lst_ignore_ch, ch_name)
        if id_str and action == a_open:
            self._open_subject_in_viewer(id_str)
        if action == a_refresh:
            self._refresh_presence()

    # ---- Channels table context menu -----------------------------------

    def _channels_context_menu(self, pos):
        idx = self._view_channels.indexAt(pos)
        if not idx.isValid():
            return
        model = self._view_channels.model()
        # column 0 is CH name (proxy might sort)
        ch_idx = model.index(idx.row(), 0)
        ch_name = str(model.data(ch_idx, Qt.DisplayRole) or '')
        if not ch_name:
            return

        menu = QMenu(self._view_channels)
        a_remap  = menu.addAction(f"Add '{ch_name}' to remap (Original)")
        a_ignore = menu.addAction(f"Add '{ch_name}' to ignore")
        action = menu.exec(self._view_channels.viewport().mapToGlobal(pos))
        if action == a_remap:
            self._prompt_add_to_remap('channel', ch_name)
        elif action == a_ignore:
            self._ignore_add_item(self._lst_ignore_ch, ch_name)

    def _annots_context_menu(self, pos):
        idx = self._view_annots.indexAt(pos)
        if not idx.isValid():
            return
        model = self._view_annots.model()
        ann_name = str(model.data(model.index(idx.row(), 0), Qt.DisplayRole) or '')
        if not ann_name:
            return

        menu = QMenu(self._view_annots)
        a_remap  = menu.addAction(f"Add '{ann_name}' to remap (Original)")
        a_ignore = menu.addAction(f"Add '{ann_name}' to ignore")
        action = menu.exec(self._view_annots.viewport().mapToGlobal(pos))
        if action == a_remap:
            self._prompt_add_to_remap('annot', ann_name)
        elif action == a_ignore:
            self._ignore_add_item(self._lst_ignore_ann, ann_name)

    # ------------------------------------------------------------------
    # Open subject in main viewer
    # ------------------------------------------------------------------

    def _open_subject_in_viewer(self, id_str: str):
        try:
            tbl = self.ctrl.ui.tbl_slist
            model = tbl.model()
            src = getattr(model, 'sourceModel', lambda: model)()
            for r in range(src.rowCount()):
                val = src.data(src.index(r, 0), Qt.DisplayRole)
                if str(val) == id_str:
                    src_idx   = src.index(r, 0)
                    proxy_idx = (model.mapFromSource(src_idx)
                                 if hasattr(model, 'mapFromSource')
                                 else src_idx)
                    tbl.scrollTo(proxy_idx)
                    tbl.setCurrentIndex(proxy_idx)
                    tbl.doubleClicked.emit(proxy_idx)
                    return
        except Exception as exc:
            print(f"[Harmonizer] open_subject {id_str!r}: {exc}")

    # ------------------------------------------------------------------
    # Remap / ignore helpers
    # ------------------------------------------------------------------

    def _remap_add_row(self, tbl: QTableWidget, orig: str = '', canon: str = ''):
        r = tbl.rowCount()
        tbl.insertRow(r)
        tbl.setItem(r, 0, QTableWidgetItem(orig))
        tbl.setItem(r, 1, QTableWidgetItem(canon))
        tbl.setCurrentCell(r, 1 if canon == '' else 0)
        tbl.editItem(tbl.item(r, 1 if canon == '' else 0))
        tbl.scrollToBottom()

    def _canonical_terms(self, kind: str) -> list[str]:
        tbl = self._tbl_ch_remap if kind == 'channel' else self._tbl_ann_remap
        terms = set()
        for r in range(tbl.rowCount()):
            it = tbl.item(r, 1)
            if it:
                text = it.text().strip()
                if text:
                    terms.add(text)
        return sorted(terms, key=str.lower)

    def _prompt_add_to_remap(self, kind: str, orig: str):
        terms = self._canonical_terms(kind)
        title = "Add channel remap" if kind == 'channel' else "Add annotation remap"
        label = f"Canonical {'channel' if kind == 'channel' else 'annotation'}:"
        canon = ""
        if terms:
            canon, ok = QInputDialog.getItem(
                self._root, title, label, terms, 0, True
            )
            if not ok:
                return
            canon = str(canon).strip()
        else:
            canon, ok = QInputDialog.getText(self._root, title, label, text=orig)
            if not ok:
                return
            canon = canon.strip()
        if not canon:
            return
        tbl = self._tbl_ch_remap if kind == 'channel' else self._tbl_ann_remap
        self._remap_add_row(tbl, orig, canon)

    def _remap_del_row(self, tbl: QTableWidget):
        rows = sorted({idx.row() for idx in tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            tbl.removeRow(r)

    def _ignore_add(self, lst: QListWidget):
        text, ok = QInputDialog.getText(
            self._root, "Add to ignore", "Channel / annotation name:"
        )
        if ok and text.strip():
            self._ignore_add_item(lst, text.strip())

    def _ignore_add_item(self, lst: QListWidget, name: str):
        existing = [lst.item(i).text() for i in range(lst.count())]
        if name not in existing:
            lst.addItem(QListWidgetItem(name))

    def _ignore_del(self, lst: QListWidget):
        for item in lst.selectedItems():
            lst.takeItem(lst.row(item))

    # ---- remap dict getters ----------------------------------------

    def _get_ch_remap(self) -> dict:
        return _read_remap_table(self._tbl_ch_remap)

    def _get_ch_ignore(self) -> set:
        return _read_ignore_list(self._lst_ignore_ch)

    def _get_ann_remap(self) -> dict:
        return _read_remap_table(self._tbl_ann_remap)

    def _get_ann_ignore(self) -> set:
        return _read_ignore_list(self._lst_ignore_ann)

    # ------------------------------------------------------------------
    # Load aliases from Luna engine
    # ------------------------------------------------------------------

    def _load_luna_ch_aliases(self):
        try:
            aliases = self.ctrl.proj.eng.aliases()
        except Exception as exc:
            QMessageBox.warning(self._root, "Harmonizer", f"Cannot load aliases: {exc}")
            return
        # aliases: list of [Type, Primary, Secondary]
        count = 0
        for row in aliases:
            if len(row) >= 3 and str(row[0]).upper() not in ('ANNOT',):
                orig  = str(row[2])
                canon = str(row[1])
                if orig != canon:
                    self._remap_add_row(self._tbl_ch_remap, orig, canon)
                    count += 1
        self._lbl_status.setText(
            self._lbl_status.text() + f"  ·  loaded {count} Luna channel aliases"
        )

    def _load_luna_ann_aliases(self):
        try:
            aliases = self.ctrl.proj.eng.aliases()
        except Exception as exc:
            QMessageBox.warning(self._root, "Harmonizer", f"Cannot load aliases: {exc}")
            return
        count = 0
        for row in aliases:
            if len(row) >= 3 and str(row[0]).upper() == 'ANNOT':
                orig  = str(row[2])
                canon = str(row[1])
                if orig != canon:
                    self._remap_add_row(self._tbl_ann_remap, orig, canon)
                    count += 1
        self._lbl_status.setText(
            self._lbl_status.text() + f"  ·  loaded {count} Luna annot aliases"
        )

    # ------------------------------------------------------------------
    # Domains save / load
    # ------------------------------------------------------------------

    def _save_domains(self):
        path, _ = save_file_name(
            self._root, "Save Domains", "domains",
            "TSV (*.tsv);;All Files (*)"
        )
        if not path:
            return
        if not path.lower().endswith('.tsv'):
            path += '.tsv'
        model = self._view_domains.model()
        src = getattr(model, 'sourceModel', lambda: model)()
        lines = ["CH\tDomain"]
        for r in range(src.rowCount()):
            ch  = str(src.data(src.index(r, 0)) or '')
            dom = str(src.data(src.index(r, 1)) or '')
            if ch:
                lines.append(f"{ch}\t{dom}")
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _load_domains(self):
        path, _ = open_file_name(
            self._root, "Load Domains", "",
            "TSV (*.tsv);;All Files (*)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path, sep='\t', dtype=str).fillna('')
            self._domains = {}
            for _, row in df.iterrows():
                ch = str(row.get('CH', '')).strip()
                dom = str(row.get('Domain', '')).strip()
                if ch:
                    self._domains[ch] = dom
            self._populate_domains_tab()
        except Exception as exc:
            QMessageBox.critical(self._root, "Harmonizer", f"Load domains error: {exc}")

    # ------------------------------------------------------------------
    # Export @param
    # ------------------------------------------------------------------

    def _export_param(self, channels: bool, annots: bool):
        if not channels and not annots:
            return
        path, _ = save_file_name(
            self._root, "Export @param file", "harmonizer",
            "Param files (*.param);;Text files (*.txt);;All Files (*)"
        )
        if not path:
            return

        sig_names = []
        annot_names = []
        if self._scan is not None:
            if channels and not self._scan.channels_df.empty:
                sig_names = channel_summary(
                    self._scan.channels_df,
                    remap=self._get_ch_remap(),
                    ignore=self._get_ch_ignore(),
                )['CH'].astype(str).tolist()
            if annots and not self._scan.annots_df.empty:
                annot_names = annot_summary(
                    self._scan.annots_df,
                    remap=self._get_ann_remap(),
                    ignore=self._get_ann_ignore(),
                )['ANNOT'].astype(str).tolist()

        try:
            write_param_file(
                path,
                remap_ch   = self._get_ch_remap()   if channels else {},
                ignore_ch  = self._get_ch_ignore()  if channels else set(),
                remap_ann  = self._get_ann_remap()  if annots   else {},
                ignore_ann = self._get_ann_ignore() if annots   else set(),
                sig_names  = sig_names,
                annot_names = annot_names,
            )
            self._lbl_status.setText(
                self._lbl_status.text() + f"  ·  exported {path}"
            )
        except Exception as exc:
            QMessageBox.critical(self._root, "Harmonizer", f"Export error: {exc}")

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _save_cache(self):
        if self._scan is None:
            QMessageBox.information(self._root, "Harmonizer", "No scan to save.")
            return
        path, selected_filter = save_file_name(
            self._root, "Save Scan Cache", "harmonizer_cache",
            "Pickle (*.pkl);;All Files (*)"
        )
        if not path:
            return
        if "*.pkl" in selected_filter and not path.lower().endswith(".pkl"):
            path = f"{path}.pkl"
        try:
            save_cache(path, self._scan)
        except Exception as exc:
            QMessageBox.critical(self._root, "Harmonizer", f"Save error: {exc}")

    def _load_cache(self):
        path, _ = open_file_name(
            self._root, "Load Scan Cache", "",
            "Pickle (*.pkl);;All Files (*)"
        )
        if not path:
            return
        try:
            scan = load_cache(path)
            self._scan = scan
            n_ch  = scan.channels_df['CH'].nunique() if not scan.channels_df.empty else 0
            n_ann = scan.annots_df['ANNOT'].nunique() if not scan.annots_df.empty else 0
            self._lbl_status.setText(
                f"Cache loaded  ·  {len(scan.ids)}/{scan.n_total} subjects  ·  "
                f"{n_ch} channels  ·  {n_ann} annot classes  ·  {scan.scan_ts}"
            )
            self._on_right_tab_changed(self._right_tabs.currentIndex())
        except Exception as exc:
            QMessageBox.critical(self._root, "Harmonizer", f"Load cache error: {exc}")

    # ------------------------------------------------------------------
    # Sample list helper
    # ------------------------------------------------------------------

    def _get_sample_ids(self) -> list:
        try:
            tbl = self.ctrl.ui.tbl_slist
            model = tbl.model()
            src = getattr(model, 'sourceModel', lambda: model)()
            ids = []
            for r in range(src.rowCount()):
                val = src.data(src.index(r, 0), Qt.DisplayRole)
                if val:
                    ids.append(str(val))
            return ids
        except Exception:
            return []

    # ------------------------------------------------------------------
    # ExplorerMixin interface
    # ------------------------------------------------------------------

    def refresh_controls(self):
        """Called when this tab becomes visible in the Explorer dock."""
        if self._scan is not None:
            self._on_right_tab_changed(self._right_tabs.currentIndex())


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _sec(label: str) -> QLabel:
    lbl = QLabel(label)
    lbl.setStyleSheet(
        f"color:{FG}; font-size:11px; font-weight:bold;"
        f" padding-bottom:1px;"
    )
    return lbl


def _hsep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setFrameShadow(QFrame.Sunken)
    sep.setStyleSheet(f"color:{SEP};")
    return sep


def _vsep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.VLine)
    sep.setFrameShadow(QFrame.Sunken)
    return sep


def _make_remap_table(parent=None) -> QTableWidget:
    tbl = QTableWidget(0, 2, parent)
    tbl.setHorizontalHeaderLabels(["Original", "Canonical"])
    tbl.horizontalHeader().setStretchLastSection(True)
    tbl.verticalHeader().setVisible(False)
    tbl.setMaximumHeight(150)
    tbl.setMinimumHeight(70)
    tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
    tbl.setEditTriggers(
        QAbstractItemView.DoubleClicked
        | QAbstractItemView.SelectedClicked
        | QAbstractItemView.EditKeyPressed
        | QAbstractItemView.AnyKeyPressed
    )
    return tbl


def _make_ignore_list(parent=None) -> QListWidget:
    lst = QListWidget(parent)
    lst.setMaximumHeight(100)
    lst.setMinimumHeight(50)
    return lst


def _read_remap_table(tbl: QTableWidget) -> dict:
    remap = {}
    for r in range(tbl.rowCount()):
        it_orig  = tbl.item(r, 0)
        it_canon = tbl.item(r, 1)
        if it_orig and it_canon:
            orig  = it_orig.text().strip()
            canon = it_canon.text().strip()
            if orig and canon and orig != canon:
                remap[orig] = canon
    return remap


def _read_ignore_list(lst: QListWidget) -> set:
    return {lst.item(i).text() for i in range(lst.count())}
