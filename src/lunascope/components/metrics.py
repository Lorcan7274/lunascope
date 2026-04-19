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

import pandas as pd
import numpy as np
import lunapi as lp

from typing import Callable, Iterable, List, Optional

from scipy.signal import butter, sosfilt

from PySide6.QtWidgets import QHeaderView, QAbstractItemView, QTableView, QMessageBox
from PySide6.QtGui import QStandardItemModel, QStandardItem, QColor
from PySide6.QtCore import Qt, QSortFilterProxyModel, QRegularExpression, QModelIndex, QSignalBlocker
from PySide6.QtCore import QTimer
        
from ..helpers import sort_df_by_list
from .tbl_funcs import add_combo_column, add_check_column, attach_comma_filter


# drop-in: minimal changes to make combo editors persist across filtering

from PySide6.QtWidgets import QStyledItemDelegate, QComboBox, QStyleOptionViewItem, QStyle
from PySide6.QtCore import Qt

class _ComboDelegate(QStyledItemDelegate):
    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.items = items

    def createEditor(self, parent, option, index):
        cb = QComboBox(parent)
        cb.addItems(self.items)
        cb.setEditable(True)
        cb.setFrame(False)
        cb.setMaxVisibleItems(len(self.items))
        le = cb.lineEdit()
        if le is not None:
            le.setReadOnly(True)
            le.setFrame(False)
        cb.setStyleSheet(
            "QComboBox{padding:0 12px 0 1px;margin:0;border:0;background:transparent;}"
            "QComboBox::drop-down{subcontrol-origin:padding;subcontrol-position:top right;width:11px;border:0;}"
            "QComboBox::down-arrow{image:none;width:0px;height:0px;"
            "border-left:4px solid transparent;border-right:4px solid transparent;"
            "border-top:6px solid #c7c7c7;margin-right:2px;}"
            "QComboBox::drop-down{background:rgba(255,255,255,0.08);}"
            "QComboBox QAbstractItemView{selection-background-color:#4a4a4a;}"
            "QLineEdit{padding:0;margin:0;border:0;background:transparent;}"
        )
        def _commit_later():
            # defer commit until the editor is fully attached to the view
            QTimer.singleShot(0, lambda: (
                self.commitData.emit(cb),
                self.closeEditor.emit(cb, QStyledItemDelegate.NoHint)
            ))
        cb.currentIndexChanged.connect(_commit_later)
        return cb


    def setEditorData(self, editor, index):
        v = index.data(Qt.EditRole) or index.data(Qt.DisplayRole) or "None"
        i = editor.findText(v)
        editor.setCurrentIndex(max(0, i))

    def setModelData(self, editor, model, index):
        t = editor.currentText()
        model.setData(index, t, Qt.EditRole)
        model.setData(index, t, Qt.DisplayRole)

    # avoid “double text” under the combo
    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""  # suppress underlying text
        if option.widget:
            option.widget.style().drawControl(QStyle.CE_ItemViewItem, opt, painter, option.widget)
        else:
            super().paint(painter, opt, index)



            

class MetricsMixin:

    def _apply_compact_dock_styles(self):
        button_names = ("butt_sig", "butt_annot")
        for name in button_names:
            widget = getattr(self.ui, name, None)
            if widget is None:
                continue
            widget.setStyleSheet(
                "font-size: 10px; min-height: 26px; padding: 2px 4px;"
            )

        line_edit_names = ("txt_signals", "txt_annots", "txt_events")
        for name in line_edit_names:
            widget = getattr(self.ui, name, None)
            if widget is None:
                continue
            widget.setStyleSheet(
                "font-size: 10px; min-height: 18px; max-height: 18px; padding: 0 4px;"
            )

        table_names = ("tbl_desc_signals", "tbl_desc_annots", "tbl_desc_events")
        for name in table_names:
            view = getattr(self.ui, name, None)
            if view is None:
                continue
            view.setStyleSheet(
                """
                QTableView {
                    font-size: 10px;
                }
                QTableView::item {
                    padding: 0 2px;
                }
                QHeaderView::section {
                    font-size: 9px;
                    padding: 0 2px;
                    min-height: 16px;
                }
                QComboBox {
                    font-size: 10px;
                    padding: 0 1px;
                    margin: 0;
                    min-height: 14px;
                }
                QComboBox::down-arrow {
                    image: none;
                    width: 0px;
                }
                QComboBox::drop-down {
                    width: 10px;
                    border: 0;
                }
                """
            )

    def _configure_dense_table(
        self,
        view: QTableView,
        *,
        row_height: int = 18,
        header_height: int = 16,
    ):
        h = view.horizontalHeader()
        v = view.verticalHeader()
        h.setMinimumSectionSize(18)
        h.setFixedHeight(header_height)
        v.setDefaultSectionSize(row_height)
        v.setMinimumSectionSize(row_height)
        v.setVisible(False)
        view.setShowGrid(False)

    def _init_metrics(self):
        self._apply_compact_dock_styles()
        self._pending_instance_annots = []
        self._instances_update_dirty = False
        self._instances_update_timer = QTimer(self.ui)
        self._instances_update_timer.setSingleShot(True)
        self._instances_update_timer.timeout.connect(self._flush_instances_update)
        if hasattr(self.ui, "dock_annots"):
            self.ui.dock_annots.visibilityChanged.connect(self._on_instances_dock_visibility_changed)
        
        # signal table
        view = self.ui.tbl_desc_signals

        view.setSortingEnabled(False)
        h = view.horizontalHeader()
        h.setMinimumSectionSize(20)   	
        h.setStretchLastSection(False)
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        view.resizeColumnsToContents()
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        QTimer.singleShot(0, lambda: h.setSectionResizeMode(QHeaderView.Interactive))
        view.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.setSelectionMode(QAbstractItemView.SingleSelection)
        view.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._configure_dense_table(view)
        
        # annots table

        view = self.ui.tbl_desc_annots
        view.setSortingEnabled(False)
        h = view.horizontalHeader()
        h.setMinimumSectionSize(20)
        h.setStretchLastSection(False)
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        view.resizeColumnsToContents()
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        QTimer.singleShot(0, lambda: h.setSectionResizeMode(QHeaderView.Interactive))
        view.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.setSelectionMode(QAbstractItemView.SingleSelection)
        view.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._configure_dense_table(view)

        # wiring
        self.ui.butt_sig.clicked.connect( self._toggle_sigs )
        self.ui.butt_annot.clicked.connect( self._toggle_annots )

        
    def _toggle_sigs(self):
        if not hasattr(self.ui.tbl_desc_signals, "checked_visible"):
            return
        n = len(self.ui.tbl_desc_signals.checked_visible())
        if n == 0:
            self.ui.tbl_desc_signals.select_all_checks()
        else:
            self.ui.tbl_desc_signals.select_none_checks()

    def _toggle_annots(self):
        if not hasattr(self.ui.tbl_desc_annots, "checked_visible"):
            return
        n = len(self.ui.tbl_desc_annots.checked_visible())
        if n == 0:
            self.ui.tbl_desc_annots.select_all_checks()
        else:
            self.ui.tbl_desc_annots.select_none_checks()
        
    def all_labels(self,view):
        m = view.model()
        return [m.data(m.index(r, 0)) for r in range(m.rowCount())]

        
    # ------------------------------------------------------------
    # Attach EDF

    def _update_metrics(self):

        # ------------------------------------------------------------
        # EDF header metrics --> status bar
        
        try:
            self.p.silent_proc( 'EPOCH' )
        except RuntimeError as e:
            import sys
            print(f"[lunascope] note: EPOCH command raised RuntimeError ({e}) — refreshing EDF", file=sys.stderr)
            self._refresh()
            return
        df_raw = self.p.table( 'EPOCH' )
        try:
            edf_ne_raw = int( df_raw.iloc[0, df_raw.columns.get_loc('NE')] )
        except KeyError:
            QMessageBox.critical(self.ui, "Problem", "Likely no unmasked epochs left\nGoing to refresh the EDF" )
            self._refresh()
            return

        self.p.silent_proc( 'HEADERS & EPOCH align' )
        df_align = self.p.table( 'EPOCH' )
        edf_ne_align = int( df_align.iloc[0, df_align.columns.get_loc('NE')] ) \
            if 'NE' in df_align.columns else 0

        if edf_ne_align == edf_ne_raw:
            epoch_str = str( edf_ne_raw )
        else:
            epoch_str = f"{edf_ne_raw}({edf_ne_align})"

        df = self.p.table( 'HEADERS' )
        edf_id = self.p.id()
        rec_dur_hms = df.iloc[0, df.columns.get_loc('REC_DUR_HMS')]
        tot_dur_hms = df.iloc[0, df.columns.get_loc('TOT_DUR_HMS')]
        edf_type = df.iloc[0, df.columns.get_loc('EDF_TYPE')]
        edf_na = self.p.annots().size
        edf_ns = df.iloc[0, df.columns.get_loc('NS')]
        edf_starttime = df.iloc[0, df.columns.get_loc('START_TIME')]
        edf_startdate = df.iloc[0, df.columns.get_loc('START_DATE')]


        self.sb_id.setText( f"{edf_type}: {edf_id}" )
        self.sb_start.setText( f"Start time: {edf_starttime} date: {edf_startdate}" )
        self.sb_dur.setText( f"Duration: {rec_dur_hms} / {tot_dur_hms} / {epoch_str} epochs" )
        self.sb_ns.setText( f"{edf_ns} signals, {edf_na} annotations" )

        
        # --------------------------------------------------------------------------------
        # get units (for plot labels) and sample rates (for filters)

        hdr = self.p.headers()

        if hdr is not None:
            self.units = dict( zip( hdr.CH , hdr.PDIM ) )
            self.srs   = dict( zip( hdr.CH , hdr.SR ) )
        else:
            self.units = None
            self.srs = None
        
            
        # ------------------------------------------------------------
        # populate signal box


        df = self.p.table('HEADERS', 'CH')
        if len(df.index) > 0:
            df = df[['CH', 'PDIM', 'SR']]
        else:
            df = pd.DataFrame(columns=["CH", "PDIM", "SR"])

        if self.cmap_list:
            df = sort_df_by_list(df, 0, self.cmap_list)

        # SOURCE model from your DataFrame
        src_sig = self.df_to_std_model(df)  # needs insertColumn / add_check_column

        # add filter proxy
        self.signals_table_proxy = attach_comma_filter(
            self.ui.tbl_desc_signals,
            self.ui.txt_signals
        )
        self.signals_table_proxy.setSourceModel(src_sig)

        # Put proxy on the view
        view = self.ui.tbl_desc_signals
        view.setModel(self.signals_table_proxy)
        
        # View config
        view.setSortingEnabled(False)
        h = view.horizontalHeader()
        h.setMinimumSectionSize(20)                 
        h.setStretchLastSection(False)              
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        view.resizeColumnsToContents()              
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        QTimer.singleShot(0, lambda: h.setSectionResizeMode(QHeaderView.Interactive))
        view.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.setSelectionMode(QAbstractItemView.SingleSelection)
        view.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._configure_dense_table(view)


        
        # Add virtual checkbox column (unchanged)
        add_check_column(
            view,
            channel_col_before_insert=0,
            header_text="Sel",
            initial_checked=[],
            on_change=lambda _: (self._clear_pg1(), self._update_scaling() ),
        )

        # --- minimal change start: make a real model column for the combo and bind a delegate ---

        # Fixed schema after checkbox insert:
        # Sel(0), CH(1), PDIM(2), SR(3)  -> insert Filter at source col 3? You used 2 originally.
        # Keep your original: insert at source col 2, which shifts PDIM->3, SR->4
        SRC_COL_FILTER = 2
        SRC_COL_CH     = 1
        SRC_COL_SR     = 4
        PROXY_COL_FILTER = 2  # columns/order fixed per your note

        # ensure Filter column exists on SOURCE with default values
        src_sig.insertColumn(SRC_COL_FILTER)
        src_sig.setHeaderData(SRC_COL_FILTER, Qt.Horizontal, "Filter")
        for r in range(src_sig.rowCount()):
            idx = src_sig.index(r, SRC_COL_FILTER)
            if not (idx.data(Qt.DisplayRole) or idx.data(Qt.EditRole)):
                src_sig.setData(idx, "None", Qt.EditRole)
                src_sig.setData(idx, "None", Qt.DisplayRole)

        # bind delegate on the PROXY column and reopen editors after proxy changes
        filt_items = ["None", "0.3-35Hz", "Slow", "Delta", "Theta", "Alpha", "Sigma", "Beta", "Gamma", "User"]
        view.setItemDelegateForColumn(PROXY_COL_FILTER, _ComboDelegate(filt_items, view))

        def _open_all():
            proxy = self.signals_table_proxy
            for r in range(proxy.rowCount()):
                view.openPersistentEditor(proxy.index(r, PROXY_COL_FILTER))

        _open_all()
        self.signals_table_proxy.modelReset.connect(_open_all)
        self.signals_table_proxy.layoutChanged.connect(_open_all)
        self.signals_table_proxy.rowsInserted.connect(lambda *a: _open_all())

        # widths
        proxy = self.signals_table_proxy
        view.setColumnWidth(PROXY_COL_FILTER, 76)
        view.setColumnWidth(0, 24)
        view.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        view.horizontalHeader().setSectionResizeMode(PROXY_COL_FILTER, QHeaderView.Fixed)
        for col_name, width in {"PDIM": 38, "SR": 34}.items():
            for c in range(proxy.columnCount()):
                header = str(proxy.headerData(c, Qt.Horizontal) or "")
                if header == col_name:
                    view.setColumnWidth(c, width)
                    view.horizontalHeader().setSectionResizeMode(c, QHeaderView.Fixed)
                    break
    
        # inside your setup method
        self._signals_proxy = self.signals_table_proxy
        self._signals_view  = self.ui.tbl_desc_signals
        PROXY_COL_FILTER = 2

        def _reopen_all():
            for r in range(self._signals_proxy.rowCount()):
                idx = self._signals_proxy.index(r, PROXY_COL_FILTER)
                if idx.isValid():
                    self._signals_view.openPersistentEditor(idx)
                    
        def _reopen_all_later():
            QTimer.singleShot(0, _reopen_all)
            
        # store on self so they can be reused / disconnected cleanly later
        self._reopen_all_filters = _reopen_all
        self._reopen_all_filters_later = _reopen_all_later

        p = self._signals_proxy
        p.modelReset.connect(_reopen_all_later)
        p.layoutChanged.connect(_reopen_all_later)
        p.rowsInserted.connect(lambda *_: _reopen_all_later())
        p.rowsRemoved.connect(lambda *_: _reopen_all_later())
        p.rowsMoved.connect(lambda *_: _reopen_all_later())
        p.dataChanged.connect(lambda *_: _reopen_all_later())
        
        self.ui.txt_signals.textChanged.connect(lambda *_: _reopen_all_later())
        _reopen_all_later()

        
        # --- minimal change end ---

        proxy = view.model()
        src_sig = getattr(proxy, "sourceModel", None) and proxy.sourceModel() or proxy

        # hardcode target_src_col: we know it's the Filter column at 2; CH at 1; SR at 4
        target_src_col = SRC_COL_FILTER
        CH_SRC_COL = SRC_COL_CH

        def on_sig_changed(top_left, bottom_right, roles, *,
                           src=src_sig, target_col=target_src_col, ch_col=CH_SRC_COL):
            if not (top_left.column() <= target_col <= bottom_right.column()):
                return
            can_update_backend = hasattr(self, "ss") and getattr(self, "rendered", False)
            for r in range(top_left.row(), bottom_right.row() + 1):
                val = src.index(r, target_col).data(Qt.EditRole) or 'None'
                ch_label = src.index(r, ch_col).data(Qt.DisplayRole)
                sr = src.index(r, SRC_COL_SR).data(Qt.DisplayRole)

                if val == 'None':
                    self.fmap.pop(ch_label, None)
                    if can_update_backend:
                        self.ss.clear_filter(ch_label)
                    continue

                self.fmap[ch_label] = val

                if val == 'User':
                    if ch_label in self.user_fmap_frqs:
                        frqs = list(self.user_fmap_frqs[ch_label])
                    else:
                        frqs = []
                else:
                    frqs = self.fmap_frqs[val]

                sr = float(sr)
                # Use throttled SR when rendered: segsrv stores channels at
                # min(orig_sr, input_throttle_sr) after populate_lunascope.
                throttle_sr = getattr(self, '_segsrv_input_throttle_sr', None)
                eff_sr = min(sr, float(throttle_sr)) if (can_update_backend and throttle_sr) else sr
                valid_band = (
                    len(frqs) == 2 and
                    frqs[0] < frqs[1] and
                    frqs[0] >= 0 and
                    frqs[1] <= eff_sr / 2
                )
                if valid_band:
                    if can_update_backend:
                        order = 2
                        sos = butter(order, frqs, btype='band', fs=eff_sr, output='sos')
                        self.ss.apply_filter(ch_label, sos.reshape(-1))
                else:
                    self.fmap.pop(ch_label, None)
                    if can_update_backend:
                        self.ss.clear_filter(ch_label)

            self._clear_pg1()
            self._update_scaling() # calls _update_pg1() 


        # add wiring
        src_sig.dataChanged.connect(on_sig_changed)


        
        # --------------------------------------------------------------------------------
        # populate annotations box


        # SOURCE model
        df = self.p.annots()
        if not df.empty:
            df = df[df["Annotations"] != "SleepStage"]
        
        # re-order channels based on a cmap?                                                                                             
        if self.cmap_list:
            df = sort_df_by_list( df , 0 , self.cmap_list )
        
        src = self.df_to_std_model(df)  # needs add_check_column
        
        # add filter proxy
        self.annots_table_proxy = attach_comma_filter(
            self.ui.tbl_desc_annots,
            self.ui.txt_annots            
        )

        self.annots_table_proxy.setSourceModel(src)

        # View + proxy
        view = self.ui.tbl_desc_annots
        view.setModel(self.annots_table_proxy)

        # View config
        view.setSortingEnabled(False)
        h = view.horizontalHeader()
        h.setMinimumSectionSize(20)
        h.setStretchLastSection(False)
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        view.resizeColumnsToContents()
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        QTimer.singleShot(0, lambda: h.setSectionResizeMode(QHeaderView.Interactive))
        view.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.setSelectionMode(QAbstractItemView.SingleSelection)
        view.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._configure_dense_table(view)

        # Add checkbox column; index is SOURCE column before insertion
        add_check_column(
            view,
            channel_col_before_insert=0,  
            header_text="Sel",
            initial_checked=[],
            on_change=lambda anns: (
                self._update_instances(anns),
                self._clear_pg1(),
                self._update_scaling(),
                self._schedule_actigraphy_update() if hasattr(self, "_schedule_actigraphy_update") else None
            ),
        )
        view.setColumnWidth(0, 24)
        view.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        view.resizeColumnsToContents()



        
        # --------------------------------------------------------------------------------
        # redo original population of ssa

        # track all original annots (to keep the same y-axes)
        self.ssa_anns = self.p.edf.annots()
        self.ssa_anns = [s for s in self.ssa_anns if s != "SleepStage"]
        self.ssa_anns_lookup = {v: i for i, v in enumerate(self.ssa_anns)}
        
        # but initialize a separate ss for annotations only
        # for lookups (event instance listing)
        self.ssa = lp.segsrv( self.p )
        self.ssa.populate( chs = [ ] , anns = self.ssa_anns )
        self.ssa.set_annot_format6( False )  # pyqtgraph vs plotly
        self.ssa.set_clip_xaxes( False )
        self.ssa.window(self.last_x1, self.last_x2) 
        
        # populate here, as used by plot_simple (prior to render)
        self.ss_anns = self.ui.tbl_desc_annots.checked()
        self.ss_chs = self.ui.tbl_desc_signals.checked()

        # update palette
        self.set_palette()
        if hasattr(self, "_annotator_refresh_classes"):
            self._annotator_refresh_classes()



    # --------------------------------------------------------------------------------
    # populate annotation instances (updated when annots selected)

    def _mark_instances_dirty(self, anns):
        self._pending_instance_annots = list(anns or [])
        self._instances_update_dirty = True

    def _update_instances(self, anns):
        self._pending_instance_annots = list(anns or [])
        if not getattr(self.ui, "dock_annots", None) or not self.ui.dock_annots.isVisible():
            self._instances_update_dirty = True
            return
        self._instances_update_dirty = False
        self._instances_update_timer.start(0)

    def _on_instances_dock_visibility_changed(self, visible: bool):
        if visible and (self._instances_update_dirty or self._pending_instance_annots):
            self._instances_update_timer.start(0)

    def _flush_instances_update(self):
        if not getattr(self.ui, "dock_annots", None) or not self.ui.dock_annots.isVisible():
            self._instances_update_dirty = True
            return
        self._instances_update_dirty = False
        self._rebuild_instances_table(list(self._pending_instance_annots))

    def _rebuild_instances_table(self, anns):

        # request w/ hms=True; new API returns 9 cols:
        # [class, label, hms, start_sec, dur, start_tp, stop_tp, inst_id, ch_str]
        rows = self.ssa.get_all_annots_with_inst_ids(anns, True)
        df = pd.DataFrame(rows, columns=[
            "class", "label", "hms", "start", "dur",
            "start_tp", "stop_tp", "inst_id", "ch_str",
        ])
        df["inst"] = df["inst_id"]
        df["stop"] = (df["start"].astype(float) + df["dur"].astype(float)).round(3)

        # Fetch per-event metadata via lunapi's full annotation accessor.
        # Newer lunapi builds can preserve keys as "k=v;k2=v2" with
        # add_keys=True, which is much faster than running the ANNOTS command.
        # Build (class, start, stop) -> Meta from that result.
        meta_lookup = {}
        try:
            full_annots = self.p.fetch_fulls_annots(anns, add_keys=True)
            if full_annots is not None and not full_annots.empty:
                for row in full_annots.itertuples(index=False):
                    raw = row.Meta if hasattr(row, "Meta") else None
                    if raw is None or str(raw) in (".", "", "nan", "None"):
                        val = ""
                    else:
                        val = str(raw).replace(";", "; ")
                    key = (
                        str(row.Class),
                        round(float(row.Start), 3),
                        round(float(row.Stop), 3),
                    )
                    meta_lookup[key] = val
        except TypeError:
            try:
                self.p.silent_proc("ANNOTS")
                ann_tbl = self.p.table("ANNOTS", "ANNOT_INST_T1_T2")
                if ann_tbl is not None and not ann_tbl.empty:
                    for row in ann_tbl.itertuples(index=False):
                        raw = row.VAL if hasattr(row, "VAL") else None
                        if raw is None or str(raw) in (".", "", "nan", "None"):
                            val = ""
                        else:
                            val = str(raw).replace(";", "; ")
                        key = (
                            str(row.ANNOT),
                            round(float(row.START), 3),
                            round(float(row.STOP), 3),
                        )
                        meta_lookup[key] = val
            except Exception:
                pass
        except Exception:
            pass

        def _get_meta(r):
            start = round(float(r["start"]), 3)
            stop  = round(float(r["stop"]),  3)
            return meta_lookup.get((str(r["class"]), start, stop), "")

        df["meta"] = df.apply(_get_meta, axis=1)

        # store identity cols for the annotation editor (keyed by source row)
        # use int(float(...)) for tp cols: pandas may read uint64 strings as float64
        # (e.g. "22277570000000" → 22277570000000.0 → str gives "22277570000000.0")
        self._events_identity = [
            {
                "aclass":    str(row["class"]),
                "inst_id":   str(row["inst_id"]),
                "start_tp":  str(int(float(row["start_tp"]))),
                "stop_tp":   str(int(float(row["stop_tp"]))),
                "ch_str":    str(row["ch_str"]),
                "start_sec": str(round(float(row["start"]), 3)),
                "stop_sec":  str(round(float(row["stop"]),  3)),
                "meta":      str(row["meta"]),
            }
            for _, row in df.iterrows()
        ]

        # prefix class names for queued edits/deletes (visual indicator only;
        # _events_identity above already holds the clean aclass for navigation)
        queued_del = getattr(self, "_queued_deletes", set())
        queued_edit = getattr(self, "_queued_edits", set())
        if queued_del or queued_edit:
            def _class_disp(row):
                key = (
                    str(row["class"]),
                    str(int(float(row["start_tp"]))),
                    str(int(float(row["stop_tp"]))),
                    str(row["inst_id"]),
                    str(row["ch_str"]),
                )
                if key in queued_del:
                    return "(X) " + str(row["class"])
                if key in queued_edit:
                    return "(E) " + str(row["class"])
                return str(row["class"])
            df = df.copy()
            df["class"] = df.apply(_class_disp, axis=1)

        df = df[["class", "hms", "start", "dur", "inst", "meta"]]
        self.events_model = self.df_to_model(df)

        view = self.ui.tbl_desc_events
        self.events_table_proxy = attach_comma_filter(self.ui.tbl_desc_events, self.ui.txt_events)
        self.events_table_proxy.setSourceModel(self.events_model)
        view.setModel(self.events_table_proxy)
        
        h = view.horizontalHeader()
        h.setStretchLastSection(True)
        h.setSectionResizeMode(QHeaderView.Interactive)
        
        self._configure_dense_table(view)
        view.resizeColumnsToContents()
        view.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.setSelectionMode(QAbstractItemView.SingleSelection)
        view.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        for col_name, width in {"inst": 32, "hms": 58, "start": 44, "dur": 38}.items():
            for c in range(self.events_table_proxy.columnCount()):
                header = str(self.events_table_proxy.headerData(c, Qt.Horizontal) or "")
                if header == col_name:
                    view.setColumnWidth(c, width)
                    h.setSectionResizeMode(c, QHeaderView.Interactive)
                    break
        
        sel = view.selectionModel()
        sel.currentRowChanged.connect(self._on_row_changed)


    # ------------------------------------------------------------    
    # events table: allow filtering of events

    def _on_events_filter_text(self, text: str):
        # split on commas and trim
        parts = [s.strip() for s in text.split(',') if s.strip()]
        if not parts:
            self.events_table_proxy.setFilterRegularExpression(QRegularExpression())  # clear filter
            return

        # build an OR regex safely escaped
        escaped = [QRegularExpression.escape(p) for p in parts]
        pattern = "(" + "|".join(escaped) + ")"
        rx = QRegularExpression(pattern)
        rx.setPatternOptions(QRegularExpression.CaseInsensitiveOption)
        self.events_table_proxy.setFilterRegularExpression(rx)

    

    # ------------------------------------------------------------    
    # events table: row-change callback

    def _on_row_changed(self, curr: QModelIndex, _prev: QModelIndex):
        if not curr.isValid():
            return

        model = curr.model()
        src_idx = curr
        while isinstance(model, QSortFilterProxyModel):
            src_idx = model.mapToSource(src_idx)
            if not src_idx.isValid():
                return
            model = model.sourceModel()

        if model is not self.events_model:
            return

        src_row = src_idx.row()
        headers = [
            str(self.events_model.headerData(c, Qt.Horizontal) or "")
            for c in range(self.events_model.columnCount())
        ]
        try:
            start_col = headers.index("start")
            dur_col = headers.index("dur")
        except ValueError:
            return

        # get interval
        left_data = self.events_model.data(self.events_model.index(src_row, start_col))
        dur_data = self.events_model.data(self.events_model.index(src_row, dur_col))
        if left_data is None or dur_data is None:
            return

        try:
            left = float(left_data)
            right = left + float(dur_data)
        except (TypeError, ValueError):
            return

        # expand?
        fixed_w = getattr(self.ui, "spin_jump_width", None)
        fixed_w = fixed_w.value() if fixed_w is not None else 0.0
        left , right = expand_interval( left, right, fixed_width=fixed_w )

        # set range and this should(?) update the plot
        if not getattr(self, "_annot_select_no_zoom", False):
            self.sel.setRange( left , right )

            # update plot
            if self.rendered: self.on_window_range( left , right )

        # feed annotation editor form
        identity_list = getattr(self, "_events_identity", None)
        if identity_list is not None and src_row < len(identity_list):
            if hasattr(self, "_annot_editor_from_instance"):
                self._annot_editor_from_instance(identity_list[src_row])
        


        
#------------------------------------------------------------------
# helper functions


def expand_interval(left, right, *, factor=2.0, point_width=10.0,
                    min_left=0.0, fixed_width=0.0):
    """
    Expand [left, right] to a wider interval centered on it.

    fixed_width > 0: fixed window of that many seconds centered on the event
        midpoint.  If the event itself is wider, event width wins (fixed_width
        acts as a minimum / floor).
    fixed_width == 0 (default / 'auto'): expand by factor on each side.

    Other params:
    - point_width: window used when left == right (zero-duration event)
    - min_left: clamp so L >= min_left, shifting right without changing width
    """
    a, b = sorted((float(left), float(right)))

    if fixed_width > 0:
        mid = (a + b) / 2.0
        w   = max(b - a, float(fixed_width))
        L   = mid - w / 2.0
        R   = mid + w / 2.0
        if L < min_left:
            shift = min_left - L
            L += shift
            R += shift
        return L, R

    # ── auto mode: factor-based ──────────────────────────────────────
    if a == b:
        half = point_width / 2.0
        L = max(min_left, a - half)
        R = L + point_width
        return L, R

    if factor <= 0:
        raise ValueError("factor must be > 0")

    w = b - a
    new_w = w * factor
    pad = 0.5 * (new_w - w)

    L = a - pad
    R = b + pad

    if L < min_left:
        shift = min_left - L
        L += shift
        R += shift
    return L, R
