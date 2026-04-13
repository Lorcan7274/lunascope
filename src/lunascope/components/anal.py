
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

import sys, traceback, os, threading, time
import pandas as pd
from typing import List, Tuple

from concurrent.futures import ThreadPoolExecutor

from  ..helpers import clear_rows
from .tbl_funcs import attach_comma_filter, copy_selection, save_table_as_tsv
from .slist import NumericSortFilterProxy

from PySide6.QtWidgets import QPlainTextEdit, QFileDialog, QMessageBox
from PySide6.QtCore import QMetaObject, Qt, Slot
from PySide6.QtCore import Qt, QItemSelection, QSortFilterProxyModel, QRegularExpression
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QAbstractItemView, QHeaderView
from PySide6.QtGui import QTextCursor

from PySide6.QtGui import QKeySequence, QGuiApplication, QShortcut

from PySide6.QtGui import QAction
from ..file_dialogs import open_file_name, save_file_name


def _diag_log(message: str) -> None:
    sys.stderr.write(f"[lunascope] {message}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# TEMPORARY VERBOSE INSTRUMENTATION — remove when no longer needed
# ---------------------------------------------------------------------------
try:
    import psutil as _psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

def _vlog(tag: str, extra: str = "") -> None:
    """Timestamped diagnostic log with thread ID and optional memory snapshot."""
    ts = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
    tid = threading.get_ident()
    if _PSUTIL:
        try:
            proc = _psutil.Process(os.getpid())
            mem_mb = proc.memory_info().rss / 1_048_576
            mem_str = f"  mem={mem_mb:.1f}MB"
        except Exception:
            mem_str = ""
    else:
        mem_str = ""
    line = f"[VERBOSE {ts}] tid={tid}  {tag}"
    if extra:
        line += f"  | {extra}"
    line += mem_str
    sys.stderr.write(line + "\n")
    sys.stderr.flush()
# ---------------------------------------------------------------------------


def _append_selected_extension(filename: str, selected_filter: str, allowed_exts: tuple[str, ...]) -> str:
    lower = filename.lower()
    if any(lower.endswith(ext) for ext in allowed_exts):
        return filename

    filt = (selected_filter or "").lower()
    for ext in allowed_exts:
        if f"*{ext}" in filt:
            return filename + ext

    return filename + allowed_exts[0]



class AnalMixin:

    # ------------------------------------------------------------
    # Initiate analysis tab

    def _init_anal(self):

        self.ui.butt_anal_exec.clicked.connect( self._exec_single_luna )

        self.ui.butt_anal_load.clicked.connect( self._load_luna )

        self.ui.butt_anal_save.clicked.connect( self._save_luna )

        self.ui.butt_anal_clear.clicked.connect( self._clear_luna )
        
        self.ui.radio_transpose.toggled.connect( self._on_radio_transpose_changed)
        
        # tree 'destrat' view

        m = QStandardItemModel(self)
        m.setHorizontalHeaderLabels(["Command", "Strata"])
        self._anal_model = m        
        tv = self.ui.anal_tables
        tv.setModel(m)
        tv.setUniformRowHeights(True)
        tv.header().setStretchLastSection(True)

        # store info on selecting rows of destrat
        self._tree_sel = None
        self.ui.anal_tables.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ui.anal_tables.setSelectionMode(QAbstractItemView.SingleSelection)

        view = self.ui.anal_table

        # --- Copy action ---
        copy_action = QAction("Copy", view)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        copy_action.triggered.connect(lambda: copy_selection(view,self))
        view.addAction(copy_action)

        # --- Save-as-TSV action ---
        tsv_action = QAction("Save as TSV…", view)
        tsv_action.triggered.connect(lambda: save_table_as_tsv(view,self))
        view.addAction(tsv_action)

        view.setContextMenuPolicy(Qt.ActionsContextMenu)
   
        
        # whether single-sample or whole-project mode
        self.project_mode = False
        self._project_results_mode = False
        self._proj_cancel_event = threading.Event()
        self._proj_cancel_requested = False
        self._proj_cancel_action = QAction("Stop project eval after current record", self.ui)
        self._proj_cancel_action.setShortcut(QKeySequence("Ctrl+."))
        self._proj_cancel_action.setShortcutContext(Qt.ApplicationShortcut)
        self._proj_cancel_action.triggered.connect(self._request_project_eval_cancel)
        self.ui.addAction(self._proj_cancel_action)
        self.sig_proj_eval_stream.connect(self._proj_eval_append_stream, Qt.QueuedConnection)
        self.sig_proj_eval_progress.connect(self._proj_eval_update_progress, Qt.QueuedConnection)
        self.sig_proj_eval_finished.connect(self._proj_eval_done_ok, Qt.QueuedConnection)
        self.sig_proj_eval_failed.connect(self._proj_eval_done_err, Qt.QueuedConnection)



    # ------------------------------------------------------------
    # Run a Luna command in non-project mode

    def _exec_single_luna(self):
        self.project_mode = False
        self._project_results_mode = False
        self._exec_luna()
        
    # ------------------------------------------------------------
    # Run a Luna command

    def _exec_luna(self):

        # nothing attached
        if not hasattr(self, "p"):
            QMessageBox.critical( self.ui , "Error", "No instance attached" )
            return

        # if already running.
        if self._busy:
            return  # or show a status message

        # clear any old output
        if not self.project_mode:
            self._project_results_mode = False
            clear_rows( self.ui.anal_tables )
            clear_rows( self.ui.anal_table )
        
        # note that we're busy
        self._busy = True

        # and do not let other jobs be run
        self._buttons( False )
        
        # get input
        cmd = self.ui.txt_inp.toPlainText()

        # save currents channels/annots selections
        self.curr_chs = self.ui.tbl_desc_signals.checked()                   
        self.curr_anns = self.ui.tbl_desc_annots.checked()
        
        # get/set parameters
        self.proj.clear_vars()
        self.proj.reinit()
        self.p.refresh_channel_vars()
        self.proj.silence( False )
        param = self._parse_tab_pairs( self.ui.txt_param )
        for p in param:
            self.proj.var( p[0] , p[1] )
   
        
        # ------------------------------------------------------------
        # execute command string 'cmd' in a separate thread

        self.sb_progress.setVisible(True)
        self.sb_progress.setRange(0, 0)
        self.sb_progress.setFormat("Running…")
        self.lock_ui()

        # TEMPORARY INSTRUMENTATION
        _vlog("submit: eval_lunascope", f"cmd={cmd!r:.120}")
        _t_submit = time.monotonic()

        fut = self._exec.submit(self.p.eval_lunascope, cmd)  # returns str

        def done(_f=fut, _t0=_t_submit):
            # TEMPORARY: this callback runs on the thread-pool thread
            _vlog("done-callback: entered")
            try:
                exc = _f.exception()
                if exc is None:
                    result = _f.result()  # cheap; already completed
                    _vlog("done-callback: eval OK",
                          f"elapsed={time.monotonic()-_t0:.3f}s  result_len={len(result) if result else 0}")
                    self._last_result = result
                    _vlog("done-callback: invoking _eval_done_ok on GUI thread")
                    QMetaObject.invokeMethod(self, "_eval_done_ok", Qt.QueuedConnection)
                else:
                    _vlog("done-callback: eval raised exception",
                          f"elapsed={time.monotonic()-_t0:.3f}s  exc={type(exc).__name__}: {exc}")
                    self._last_exc = exc
                    self._last_tb = f"{type(exc).__name__}: {exc}"
                    QMetaObject.invokeMethod(self, "_eval_done_err", Qt.QueuedConnection)
            except Exception as cb_exc:
                # guard against exceptions in the callback itself
                _vlog("done-callback: EXCEPTION IN CALLBACK", traceback.format_exc().strip())
                self._last_exc = cb_exc
                self._last_tb = f"{type(cb_exc).__name__}: {cb_exc}"
                QMetaObject.invokeMethod(self, "_eval_done_err", Qt.QueuedConnection)
            _vlog("done-callback: exiting")

        fut.add_done_callback(done)


    @Slot()
    def _eval_done_ok(self):
        # TEMPORARY INSTRUMENTATION
        _vlog("_eval_done_ok: entered (GUI thread)")
        _t0 = time.monotonic()
        try:
            # --- step 1: write result text to console widget ---
            _vlog("_eval_done_ok: step 1 — writing output to console",
                  f"project_mode={getattr(self,'project_mode',False)}")
            try:
                if self.project_mode:
                    out = self.ui.txt_out
                    out.moveCursor(QTextCursor.End)
                    out.insertPlainText(self._last_result)
                else:
                    self.ui.txt_out.setPlainText(self._last_result)
                _vlog("_eval_done_ok: step 1 OK")
            except Exception:
                _vlog("_eval_done_ok: step 1 FAILED", traceback.format_exc().strip())
                raise

            # --- step 2: fetch strata from luna ---
            _vlog("_eval_done_ok: step 2 — p.strata()")
            try:
                tbls = self.p.strata()
                _vlog("_eval_done_ok: step 2 OK",
                      f"strata shape={tbls.shape if hasattr(tbls,'shape') else type(tbls)}  "
                      f"commands={tbls['Command'].tolist() if hasattr(tbls,'columns') and 'Command' in tbls.columns else '?'}")
            except Exception:
                _vlog("_eval_done_ok: step 2 FAILED", traceback.format_exc().strip())
                raise

            if self.project_mode:
                # --- step 3a: project-mode accumulation ---
                _vlog("_eval_done_ok: step 3a — _accumulate_project_results")
                try:
                    self._accumulate_project_results(tbls)
                    _vlog("_eval_done_ok: step 3a OK")
                except Exception:
                    _vlog("_eval_done_ok: step 3a FAILED", traceback.format_exc().strip())
                    raise
            else:
                # --- step 3b: render result tables ---
                _vlog("_eval_done_ok: step 3b — _render_tables")
                try:
                    self._render_tables(tbls)
                    _vlog("_eval_done_ok: step 3b OK")
                except Exception:
                    _vlog("_eval_done_ok: step 3b FAILED", traceback.format_exc().strip())
                    raise

                # --- step 4: full hypnogram redraw ---
                if hasattr(self, "_render_hypnogram"):
                    _vlog("_eval_done_ok: step 4 — _render_hypnogram")
                    try:
                        self._render_hypnogram()
                        _vlog("_eval_done_ok: step 4 OK")
                    except Exception:
                        _vlog("_eval_done_ok: step 4 FAILED", traceback.format_exc().strip())
                        raise

                # --- step 5: hypnogram mask/epoch overlay update ---
                if hasattr(self, "_update_hypnogram"):
                    _vlog("_eval_done_ok: step 5 — _update_hypnogram")
                    try:
                        self._update_hypnogram()
                        _vlog("_eval_done_ok: step 5 OK")
                    except Exception:
                        _vlog("_eval_done_ok: step 5 FAILED", traceback.format_exc().strip())
                        raise

            _vlog("_eval_done_ok: all steps completed",
                  f"total elapsed={time.monotonic()-_t0:.3f}s")

        except Exception:
            self._last_tb = traceback.format_exc().strip()
            _diag_log("_eval_done_ok: unhandled exception\n" + self._last_tb)
            _vlog("_eval_done_ok: unhandled exception", self._last_tb)
            try:
                QMessageBox.critical(self.ui, "Evaluation error", self._last_tb)
            except Exception:
                pass

        finally:
            # --- step 6: UI unlock / cleanup ---
            _vlog("_eval_done_ok: step 6 — unlock_ui")
            try:
                self.unlock_ui()
                _vlog("_eval_done_ok: unlock_ui OK")
            except Exception:
                _vlog("_eval_done_ok: unlock_ui FAILED", traceback.format_exc().strip())

            _vlog("_eval_done_ok: step 6b — _busy=False, _buttons(True)")
            try:
                self._busy = False
                self._buttons(True)
                _vlog("_eval_done_ok: step 6b OK")
            except Exception:
                _vlog("_eval_done_ok: step 6b FAILED", traceback.format_exc().strip())

            _vlog("_eval_done_ok: step 6c — _set_render_status")
            try:
                self._set_render_status(self.rendered, False)
                _vlog("_eval_done_ok: step 6c OK")
            except Exception:
                _vlog("_eval_done_ok: step 6c FAILED", traceback.format_exc().strip())

            _vlog("_eval_done_ok: step 6d — progress bar hide")
            try:
                self.sb_progress.setRange(0, 100)
                self.sb_progress.setValue(0)
                self.sb_progress.setVisible(False)
                _vlog("_eval_done_ok: step 6d OK")
            except Exception:
                _vlog("_eval_done_ok: step 6d FAILED", traceback.format_exc().strip())

            if getattr(self, 'project_mode', False) and getattr(self, '_proj_n', 0) > 0:
                _vlog("_eval_done_ok: step 6e — _proj_eval_next")
                try:
                    self._proj_i += 1
                    self._proj_eval_next()
                    _vlog("_eval_done_ok: step 6e OK")
                except Exception:
                    _vlog("_eval_done_ok: step 6e FAILED", traceback.format_exc().strip())

            _vlog("_eval_done_ok: finally block complete")
            
    @Slot()
    def _eval_done_err(self):
        # TEMPORARY INSTRUMENTATION
        _vlog("_eval_done_err: entered (GUI thread)",
              f"last_tb={getattr(self,'_last_tb','<none>')[:200]}")
        try:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self.ui, "Evaluation error", self._last_tb)
        finally:
            _vlog("_eval_done_err: cleanup")
            self.unlock_ui()
            self._busy = False
            self._buttons(True)
            self._set_render_status(self.rendered, False)
            self.sb_progress.setRange(0, 100); self.sb_progress.setValue(0)
            self.sb_progress.setVisible(False)
            # turn off any prior REPORT hides (allow that 'problem' flag may be set)
            try: self.p.silent_proc('REPORT show-all')
            except RuntimeError: pass
            if getattr(self, 'project_mode', False):
                self.project_mode = False
                self._proj_n = 0
            _vlog("_eval_done_err: cleanup complete")

    def _buttons( self, status ):
        stage_tools_enabled = status and not getattr(self, 'multiday_mode', False)
        self.ui.butt_anal_exec.setEnabled(status)
        self.ui.butt_spectrogram.setEnabled(status)
        self.ui.butt_hjorth.setEnabled(status)
        self.ui.butt_calc_hypnostats.setEnabled(stage_tools_enabled)
        self.ui.butt_soap.setEnabled(stage_tools_enabled)
        self.ui.butt_pops.setEnabled(stage_tools_enabled)
        self.ui.butt_render.setEnabled(status)
        self.ui.butt_refresh.setEnabled(status)
        self.ui.butt_load_slist.setEnabled(status)
        self.ui.butt_build_slist.setEnabled(status)
        self.ui.butt_load_edf.setEnabled(status)

            
    def _render_tables(self, tbls):

        # did we add any annotations? if so, updating ssa needed 
        # (as this is where events table pulls from)
        annots = [x for x in self.p.edf.annots() if x != "SleepStage" ]
        self.ssa.populate( chs = [ ] , anns = annots )

        # some commands don't return output
        if tbls is not None:
        
            # update strata list and rewire to show
            # data table on selection
            self.set_tree_from_df( tbls )

            # save, i.e. as internal results will be overwritten
            # by the HEADERS command run implicit in the updates below
            self.results = dict()
            for row in tbls.itertuples(index=True):
                v = "_".join( [ row.Command , row.Strata ] )
                self.results[ v ] = self.p.table( row.Command, row.Strata )
            self.sig_results_changed.emit()

        # we're now finished w/ the internal Luna tables: run this command
        # just in case the user run REPORT hide of some flavor, e.g. to
        # make sure the silent_proc() calls work as expected, e.g. used
        # used below

        try: self.p.silent_proc( 'REPORT show-all' )
        except RuntimeError: pass
        
            
        # update main metrics tables (i.e. if new things added)
        self._update_metrics()
        self._update_spectrogram_list()
        self._update_actigraphy_list()
        self._update_mask_list()
        self._update_soap_list()

        # reset any prior selections
        self.ui.tbl_desc_signals.set_checked_by_labels( self.curr_chs )
        if hasattr(self.ui.tbl_desc_annots, "set_checked_by_labels_silent"):
            self.ui.tbl_desc_annots.set_checked_by_labels_silent( self.curr_anns )
        else:
            self.ui.tbl_desc_annots.set_checked_by_labels( self.curr_anns )
        if hasattr(self, "_mark_instances_dirty"):
            self._mark_instances_dirty( self.curr_anns )


    # ------------------------------------------------------------
    # aggregate tbls (project mode)

    def _accumulate_project_results(self, tbls):
        if tbls is None:
            return

        # 1) Accumulate only Command/Strata for the tree
        #    (no ID / Observation in this DF)
        if not hasattr(self, "_proj_tbls"):
            self._proj_tbls = []
        self._proj_tbls.append(tbls[["Command", "Strata"]].copy())

        # 2) Aggregate tables by Command/Strata key
        if not hasattr(self, "_proj_results"):
            self._proj_results = {}

        for row in tbls.itertuples(index=False):
            key = f"{row.Command}_{row.Strata}"
            df = self._normalize_project_result_table(
                self.p.table(row.Command, row.Strata),
                getattr(self.p, "id", None),
            )

            if key in self._proj_results:
                self._proj_results[key] = pd.concat(
                    [self._proj_results[key], df],
                    ignore_index=True,
                )
            else:
                self._proj_results[key] = df

        # Keep the REPORT state sane per record if needed
        try:
            self.p.silent_proc("REPORT show-all")
        except RuntimeError:
            pass

    def _normalize_project_result_table(self, df, record_id):
        if df is None:
            return None

        out = df.copy()
        record_id = "" if record_id is None else str(record_id)

        if "ID" not in out.columns:
            out.insert(0, "ID", record_id)
            return out

        try:
            id_col = out["ID"]
            missing = id_col.isna()
            if hasattr(id_col, "astype"):
                missing = missing | id_col.astype(str).str.strip().eq("")
            if missing.any():
                out.loc[missing, "ID"] = record_id
        except Exception:
            pass

        cols = ["ID"] + [c for c in out.columns if c != "ID"]
        return out.loc[:, cols]

        
    # ------------------------------------------------------------
    # clear luna script box

    def _clear_luna(self):
        self.ui.txt_inp.clear() 


    # ------------------------------------------------------------
    # load a luna script
        
    def _load_luna(self):
        txt_file, _ = open_file_name(
            self.ui,
            "Open Luna script",
            "",
            "Luna Scripts (*.txt *.cmd *);;All Files (*)"
        )
        if txt_file:
            try:
                text = open(txt_file, "r", encoding="utf-8").read()
                self.ui.txt_inp.setPlainText(text)
            except (UnicodeDecodeError, OSError) as e:
                QMessageBox.critical(
                    self.ui,
                    "Error opening Luna script",
                    f"Could not load {txt_file}\nException: {type(e).__name__}: {e}"
                )

            
    # ------------------------------------------------------------
    # save a luna script

    def _save_luna(self):

        new_file = self.ui.txt_inp.toPlainText()

        filename, selected_filter = save_file_name(
            self.ui,
            "Save Luna Script",
            "",
            "Luna Scripts (*.txt *.cmd *);;All Files (*)"
        )

        if filename:
            filename = _append_selected_extension(filename, selected_filter, (".txt", ".cmd"))
                
            with open(filename, "w", encoding="utf-8") as f:
                f.write(new_file)


            
    # ------------------------------------------------------------
    # handle output tables
                
    def _update_table(self, cmd , stratum ):
        
        tbl = self.results[ "_".join( [ cmd , stratum ] ) ]

        # Keep Luna's original row order unless an ID column is present.
        # If ID exists, sort stably by ID only.
        if "ID" in tbl.columns:
            try:
                tbl = tbl.sort_values(
                    ["ID"],
                    na_position="last",
                    kind="stable",
                )
            except Exception:
                pass

        if not self.project_mode and not self._project_results_mode:
            tbl = tbl.drop(columns=["ID"])

        # transpose?
        if self.ui.radio_transpose.isChecked():
            # first coerce, otherwise this step will be missed by df_to_model()
            tbl = self.coerce_numeric_df( tbl )
            tbl = tbl.T.reset_index()
            tbl.rename(columns={"index": "VAR"}, inplace=True)
            tbl.columns = ["VAR"] + [f"row{i}" for i in range(1, tbl.shape[1])]
        
        self.anal_model = self.df_to_model( tbl )

        # single proxy handles both numeric sort and comma filter
        self.anal_table_proxy = NumericSortFilterProxy(self)
        self.anal_table_proxy.setSourceModel( self.anal_model )

        view = self.ui.anal_table
        view.setSortingEnabled(False)
        view.setModel(self.anal_table_proxy)

        # pass existing proxy so attach_comma_filter wires the filter without wrapping again
        self.ui.flt_table.clear()
        self.events_table_proxy = attach_comma_filter( self.ui.anal_table , self.ui.flt_table , proxy=self.anal_table_proxy )

        h = view.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.Interactive)  # user-resizable
        h.setStretchLastSection(False)                   # no auto-stretch fighting you
        h.setResizeContentsPrecision(50)                 # sample first 50 rows only
        view.resizeColumnsToContents()
        if "ID" in tbl.columns:
            try:
                id_col = list(tbl.columns).index("ID")
                view.setSortingEnabled(True)
                view.sortByColumn(id_col, Qt.AscendingOrder)
            except Exception:
                view.setSortingEnabled(False)

        
    def _on_anal_filter_text(self, text: str):
        rx = QRegularExpression(QRegularExpression.escape(text))
        rx.setPatternOptions(QRegularExpression.CaseInsensitiveOption)
        self.anal_table_proxy.setFilterRegularExpression(rx)
        


    
    # ------------------------------------------------------------
    # tree helpers

    def set_tree_from_df(self, df):
        m = QStandardItemModel(self)
        m.setHorizontalHeaderLabels(["Key", "Values"])
        root = m.invisibleRootItem()

        # Empty or None: just show headers
        if df is None or getattr(df, "empty", True):
            self.ui.anal_tables.setModel(m)
            self._anal_model = m
            self._wire_tree_selection()
            self.ui.anal_tables.resizeColumnToContents(0)
            self.ui.anal_tables.resizeColumnToContents(1)
            return

        # Ensure we have up to two columns
        sub = df.iloc[:, :2].copy()
        if sub.shape[1] == 1:
            sub.insert(1, "_val", "")

        # Build rows
        keys = sub.iloc[:, 0].astype(str)
        vals = sub.iloc[:, 1]

        for key, val in zip(keys, vals):
            parts = [] if pd.isna(val) else [p for p in str(val).split("_") if p]
            root.appendRow([
                QStandardItem(key),
                QStandardItem(", ".join(parts))
            ])

        self.ui.anal_tables.setModel(m)
        self._anal_model = m
        self._wire_tree_selection()
        self.ui.anal_tables.resizeColumnToContents(0)
        self.ui.anal_tables.resizeColumnToContents(1)

           
    def _wire_tree_selection(self):
        tv = self.ui.anal_tables
        # disconnect old selection model if present
        if self._tree_sel is not None:
            try: self._tree_sel.selectionChanged.disconnect(self._on_tree_sel)
            except TypeError: pass
        self._tree_sel = tv.selectionModel()
        # avoid duplicate connects if this gets called often
        try:
            self._tree_sel.selectionChanged.connect(self._on_tree_sel, Qt.UniqueConnection)
        except TypeError:
            self._tree_sel.selectionChanged.connect(self._on_tree_sel)


    # refactored  _on_tree_sel() 

    def _current_key_vals(self):
        sm = self.ui.anal_tables.selectionModel()
        if not sm:
            return None
        ix = sm.currentIndex()
        if not ix.isValid():
            return None
        r = ix.row()
        key  = ix.sibling(r, 0).data()
        vals = ix.sibling(r, 1).data()
        return key, vals
        
    def _on_tree_sel(self, selected, _):
        kv = self._current_key_vals()
        if not kv:
            return
        key, vals = kv
        self._update_table(key, vals.replace(", ", "_"))

    def _on_radio_transpose_changed(self, checked):
        # call on any toggle, or guard if you only care about checked=True
        kv = self._current_key_vals()
        if not kv:
            return
        key, vals = kv
        self._update_table(key, vals.replace(", ", "_"))


    # ------------------------------------------------------------
    # helper - parse parameter file
    

    def _tokenize_pair_line(self, line: str, keep_quotes: bool = True) -> list[str]:
        out, buf, q, esc = [], [], None, False
        for ch in line:
            if esc:
                buf.append(ch); esc = False; continue
            if q:
                buf.append(ch)
                if ch == '\\': esc = True
                elif ch == q:  q = None
                continue
            if ch in ('"', "'"):
                q = ch; buf.append(ch); continue
            if ch in (' ', '\t', '=') and not out:
                out.append(''.join(buf).strip())
                buf = []  # start capturing right side fresh
                continue
            buf.append(ch)
        if buf:
            out.append(''.join(buf).strip())
        # remove leading = or whitespace on right side
        if len(out) == 2:
            out[1] = out[1].lstrip('= \t')
        if not keep_quotes and len(out) == 2:
            v = out[1]
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                out[1] = v[1:-1]
        return out


    def _parse_tab_pairs(self, edit: QPlainTextEdit) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for raw in edit.toPlainText().splitlines():
            line = raw.strip()
            if not line or line.startswith('%'):
                continue
            toks = self._tokenize_pair_line(line)
            if len(toks) != 2:
                continue
            a, b = toks[0].strip(), toks[1].strip()
            if a == '' and b == '':
                continue
            pairs.append((a, b))
        return pairs



    # ------------------------------------------------------------
    # project-level eval

    def _proj_eval(self):
        if self._busy:
            if self.project_mode:
                self._request_project_eval_cancel()
            return

        view = self.ui.tbl_slist
        model = view.model()
        if not model:
            return
        n = model.rowCount()
        if n == 0:
            return

        cmd = self.ui.txt_inp.toPlainText()
        param = self._parse_tab_pairs(self.ui.txt_param)
        records = []
        for row in range(n):
            idx = model.index(row, 0)
            label = str(model.data(idx) or "")
            if label:
                records.append((label, label))
        if not records:
            return

        self.project_mode = True
        self._project_results_mode = False
        self._proj_cancel_event.clear()
        self._proj_cancel_requested = False

        clear_rows(self.ui.anal_tables)
        clear_rows(self.ui.anal_table)
        self.ui.txt_out.clear()
        self._busy = True
        self._buttons(False)
        self._set_project_eval_action_state(running=True, cancel_requested=False)
        self.sb_progress.setVisible(True)
        self.sb_progress.setRange(0, len(records))
        self.sb_progress.setValue(0)
        self.sb_progress.setFormat(f"0 / {len(records)}")
        self.lock_ui("Processing...\n\nPress Ctrl+. to stop after this record")

        fut = self._exec.submit(self._project_eval_worker, records, cmd, param)

        def _done(_f=fut):
            try:
                self.sig_proj_eval_finished.emit(_f.result())
            except Exception as e:
                self._last_exc = e
                self._last_tb = f"{type(e).__name__}: {e}"
                self.sig_proj_eval_failed.emit(self._last_tb)

        fut.add_done_callback(_done)

    def _project_eval_worker(self, records, cmd, param):
        proj_tbls = []
        proj_results = {}
        cancelled = False

        for i, (id_str, label) in enumerate(records, start=1):
            if self._proj_cancel_event.is_set():
                cancelled = True
                self.sig_proj_eval_stream.emit("\nInterrupted.\n")
                break

            header = (
                "\n\n------------------------------------------------------------------\n"
                f"Processing: {label} (#{i})\n"
            )
            for chunk in header.splitlines(True):
                self.sig_proj_eval_stream.emit(chunk)

            stderr_txt = ""
            try:
                self.proj.clear_vars()
                self.proj.reinit()
                for a, b in param:
                    self.proj.var(a, b)

                p = self.proj.inst(id_str)
                stderr_txt = p.eval_lunascope(cmd) or ""
                if stderr_txt:
                    for chunk in stderr_txt.splitlines(True):
                        self.sig_proj_eval_stream.emit(chunk)

                tbls = p.strata()
                if tbls is not None:
                    proj_tbls.append(tbls[["Command", "Strata"]].copy())
                    for row in tbls.itertuples(index=False):
                        key = f"{row.Command}_{row.Strata}"
                        df = self._normalize_project_result_table(
                            p.table(row.Command, row.Strata),
                            id_str,
                        )
                        if key in proj_results:
                            proj_results[key] = pd.concat([proj_results[key], df], ignore_index=True)
                        else:
                            proj_results[key] = df

                try:
                    p.silent_proc("REPORT show-all")
                except RuntimeError:
                    pass
            except Exception as e:
                if stderr_txt:
                    self.sig_proj_eval_stream.emit("\n")
                raise RuntimeError(f"{label}: {type(e).__name__}: {e}") from e

            self.sig_proj_eval_progress.emit(i, len(records))

        all_tbls = None
        if proj_tbls:
            all_tbls = pd.concat(proj_tbls, ignore_index=True)
            all_tbls = all_tbls.drop_duplicates(subset=["Command", "Strata"])

        return {"tbls": all_tbls, "results": proj_results, "cancelled": cancelled}

    def _request_project_eval_cancel(self):
        if self._proj_cancel_requested:
            return
        self._proj_cancel_requested = True
        self._proj_cancel_event.set()
        self._set_project_eval_action_state(running=True, cancel_requested=True)

    def _set_project_eval_action_state(self, running=False, cancel_requested=False):
        act = getattr(self, "_act_proj_eval", None)
        if act is None:
            return
        if not running:
            act.setText("Evaluate (project)")
            return
        if cancel_requested:
            act.setText("Stopping project eval...")
        else:
            act.setText("Stop after current record")

    @Slot(str)
    def _proj_eval_append_stream(self, text):
        out = self.ui.txt_out
        out.moveCursor(QTextCursor.End)
        out.insertPlainText(text)
        out.moveCursor(QTextCursor.End)

    @Slot(int, int)
    def _proj_eval_update_progress(self, done, total):
        self.sb_progress.setRange(0, total)
        self.sb_progress.setValue(done)
        suffix = " (stopping)" if self._proj_cancel_requested else ""
        self.sb_progress.setFormat(f"{done} / {total}{suffix}")

    @Slot(object)
    def _proj_eval_done_ok(self, payload):
        try:
            tbls = payload.get("tbls")
            self.results = payload.get("results", {})
            self._project_results_mode = True
            if tbls is not None:
                self._render_project_results(tbls)
            self._detach_inst_preserve_analysis()
        finally:
            self.unlock_ui()
            self._busy = False
            self._proj_cancel_event.clear()
            self._proj_cancel_requested = False
            self._set_project_eval_action_state(running=False)
            self._buttons(True)
            self.sb_progress.setRange(0, 100)
            self.sb_progress.setValue(0)
            self.sb_progress.setVisible(False)
            self.project_mode = False

    @Slot(str)
    def _proj_eval_done_err(self, msg):
        try:
            self._project_results_mode = bool(getattr(self, "results", {}))
            QMessageBox.critical(self.ui, "Project evaluation error", msg)
            self._detach_inst_preserve_analysis()
        finally:
            self.unlock_ui()
            self._busy = False
            self._proj_cancel_event.clear()
            self._proj_cancel_requested = False
            self._set_project_eval_action_state(running=False)
            self._buttons(True)
            self.sb_progress.setRange(0, 100)
            self.sb_progress.setValue(0)
            self.sb_progress.setVisible(False)
            self.project_mode = False

        

    def _render_project_results(self, tbls):
        # build the tree from the *aggregate* strata DF
        if tbls is not None:            
            self.set_tree_from_df(tbls)



    
        
