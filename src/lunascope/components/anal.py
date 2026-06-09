
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

import os, sys, traceback, threading, multiprocessing, queue
import pandas as pd
from typing import List, Tuple

from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

from  ..helpers import clear_rows
from .tbl_funcs import attach_comma_filter, copy_selection, save_table_as_tsv
from .slist import NumericSortFilterProxy

from PySide6.QtWidgets import QPlainTextEdit, QFileDialog, QMessageBox
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLabel, QSpinBox, QVBoxLayout
from PySide6.QtCore import QEvent, QMetaObject, QSettings, Qt, Slot
from PySide6.QtCore import Qt, QItemSelection, QSortFilterProxyModel, QRegularExpression
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QAbstractItemView, QHeaderView
from PySide6.QtGui import QTextCursor

from PySide6.QtGui import QKeySequence, QGuiApplication, QShortcut

from PySide6.QtGui import QAction
from ..file_dialogs import open_file_name, save_file_name


_PROJECT_EVAL_WORKERS_KEY = "analysis/project_eval_workers"
_PROJECT_EVAL_CHILD_PROJ = None
_OUTPUT_TABLE_AUTOSORT_CELL_LIMIT = 250_000
_OUTPUT_TABLE_AUTORESIZE_CELL_LIMIT = 25_000


def _project_eval_settings() -> QSettings:
    return QSettings("Lunascope", "Lunascope")


def _diag_log(message: str) -> None:
    sys.stderr.write(f"[lunascope] {message}\n")
    sys.stderr.flush()


def _default_project_eval_workers(cpu_count=None) -> int:
    if cpu_count is None:
        cpu_count = os.cpu_count()
    try:
        cpu_count = int(cpu_count)
    except (TypeError, ValueError):
        cpu_count = 1
    return min(10, max(1, cpu_count // 2))


def _clamp_project_eval_workers(value, total_records=None) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = _default_project_eval_workers()
    value = min(10, max(1, value))
    if total_records is not None:
        value = min(value, max(1, int(total_records)))
    return value


def _normalize_project_result_table(df, record_id):
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


def _project_eval_run_record(proj, record):
    ordinal = record["ordinal"]
    sample_row = record["sample_row"]
    label = record["label"]
    cmd = record["cmd"]
    param = record["param"]
    id_str = str(sample_row[0] or "").strip()

    stdout_txt = ""
    try:
        proj.clear_vars()
        proj.reinit()
        for a, b in param:
            proj.var(a, b)

        p = proj.inst(id_str)
        stdout_txt = p.eval_lunascope(cmd) or ""

        tbls = p.strata()
        tree_tbls = None
        results = {}
        if tbls is not None:
            tree_tbls = tbls[["Command", "Strata"]].copy()
            for row in tbls.itertuples(index=False):
                key = f"{row.Command}_{row.Strata}"
                df = _normalize_project_result_table(
                    p.table(row.Command, row.Strata),
                    id_str,
                )
                if df is not None:
                    results[key] = df

        try:
            p.silent_proc("REPORT show-all")
        except RuntimeError:
            pass

        return {
            "ordinal": ordinal,
            "label": label,
            "id": id_str,
            "stdout": stdout_txt,
            "tbls": tree_tbls,
            "results": results,
            "error": None,
        }
    except Exception as e:
        return {
            "ordinal": ordinal,
            "label": label,
            "id": id_str,
            "stdout": stdout_txt,
            "tbls": None,
            "results": {},
            "error": f"{type(e).__name__}: {e}",
        }


def _project_eval_slice_worker(task):
    global _PROJECT_EVAL_CHILD_PROJ

    if _PROJECT_EVAL_CHILD_PROJ is None:
        _init_project_eval_child()
    proj = _PROJECT_EVAL_CHILD_PROJ

    rows = task["rows"]
    records = task["records"]
    result_queue = task.get("result_queue")
    try:
        proj.clear()
        proj.eng.set_sample_list(rows)
        results = []
        for record in records:
            result = _project_eval_run_record(proj, record)
            if result_queue is not None:
                result_queue.put(result)
            else:
                results.append(result)
        return {
            "slice_index": task["slice_index"],
            "start_ordinal": task["start_ordinal"],
            "end_ordinal": task["end_ordinal"],
            "results": results,
        }
    finally:
        try:
            proj.clear()
        except Exception:
            pass


def _project_eval_slices(tasks, workers):
    if not tasks:
        return []
    workers = _clamp_project_eval_workers(workers, len(tasks))
    chunk_size = (len(tasks) + workers - 1) // workers
    chunk_size = max(1, chunk_size)
    chunks = []
    for chunk_index, start in enumerate(range(0, len(tasks), chunk_size), start=1):
        records = tasks[start:start + chunk_size]
        if not records:
            continue
        chunks.append({
            "slice_index": chunk_index,
            "start_ordinal": records[0]["ordinal"],
            "end_ordinal": records[-1]["ordinal"],
            "rows": [list(record["sample_row"]) for record in records],
            "records": records,
        })
    return chunks


def _init_project_eval_child():
    global _PROJECT_EVAL_CHILD_PROJ
    import lunapi as lp

    _PROJECT_EVAL_CHILD_PROJ = lp.proj()
    _PROJECT_EVAL_CHILD_PROJ.silence(True)


def _terminate_process_pool(executor) -> bool:
    if hasattr(executor, "terminate_workers"):
        try:
            executor.terminate_workers()
            return True
        except Exception:
            return False

    processes = getattr(executor, "_processes", None)
    if not processes:
        return False
    for proc in list(processes.values()):
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass
    return False



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
        sc_exec = QShortcut(QKeySequence("Ctrl+Return"), self.ui)
        sc_exec.setContext(Qt.ApplicationShortcut)
        sc_exec.activated.connect(self._exec_single_luna)

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
        self._proj_cancel_action = QAction("Stop queued project eval records", self.ui)
        self._proj_cancel_action.setShortcuts(self._project_eval_cancel_shortcuts())
        self._proj_cancel_action.setShortcutContext(Qt.ApplicationShortcut)
        self._proj_cancel_action.triggered.connect(self._request_project_eval_cancel)
        self.ui.addAction(self._proj_cancel_action)
        app = QGuiApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self.sig_proj_eval_stream.connect(self._proj_eval_append_stream, Qt.QueuedConnection)
        self.sig_proj_eval_progress.connect(self._proj_eval_update_progress, Qt.QueuedConnection)
        self.sig_proj_eval_finished.connect(self._proj_eval_done_ok, Qt.QueuedConnection)
        self.sig_proj_eval_failed.connect(self._proj_eval_done_err, Qt.QueuedConnection)


    def _project_eval_config_dialog(self, total_records):
        total_records = max(1, int(total_records))
        default_workers = _default_project_eval_workers()
        settings = _project_eval_settings()
        saved_workers = _clamp_project_eval_workers(
            settings.value(_PROJECT_EVAL_WORKERS_KEY, default_workers),
            total_records,
        )

        dlg = QDialog(self.ui)
        dlg.setWindowTitle("Evaluate Project")
        layout = QVBoxLayout(dlg)
        form = QFormLayout()

        obs_label = QLabel(str(total_records))
        jobs_per_process_label = QLabel("")

        spin_workers = QSpinBox(dlg)
        spin_workers.setRange(1, min(10, total_records))
        spin_workers.setValue(saved_workers)
        spin_workers.setToolTip("Number of parallel worker processes.")

        def _sync_jobs_per_process_label(value=None):
            workers = _clamp_project_eval_workers(
                spin_workers.value() if value is None else value,
                total_records,
            )
            jobs_per_process = (total_records + workers - 1) // workers
            jobs_per_process_label.setText(str(jobs_per_process))

        form.addRow("# of cores:", spin_workers)
        form.addRow("# of obs:", obs_label)
        form.addRow("# jobs per process:", jobs_per_process_label)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        ok_button = buttons.button(QDialogButtonBox.Ok)
        if ok_button is not None:
            ok_button.setDefault(True)
            ok_button.setAutoDefault(True)
        spin_workers.valueChanged.connect(_sync_jobs_per_process_label)
        _sync_jobs_per_process_label()
        spin_workers.selectAll()
        spin_workers.setFocus(Qt.OtherFocusReason)

        if dlg.exec() != QDialog.Accepted:
            return None

        workers = _clamp_project_eval_workers(spin_workers.value(), total_records)
        settings.setValue(_PROJECT_EVAL_WORKERS_KEY, workers)
        return {"workers": workers}


    def _project_eval_cancel_shortcuts(self):
        shortcuts = [QKeySequence("Ctrl+.")]
        if sys.platform == "darwin":
            shortcuts.append(QKeySequence("Meta+."))
        return shortcuts

    def _project_eval_cancel_shortcut_label(self):
        labels = [
            seq.toString(QKeySequence.NativeText)
            for seq in self._project_eval_cancel_shortcuts()
            if not seq.isEmpty()
        ]
        return " or ".join(labels)

    def _is_project_eval_cancel_key_event(self, event):
        if event.type() not in (QEvent.KeyPress, QEvent.ShortcutOverride):
            return False
        if event.key() != Qt.Key_Period:
            return False
        mods = event.modifiers()
        if sys.platform == "darwin":
            return bool(mods & (Qt.ControlModifier | Qt.MetaModifier))
        return bool(mods & Qt.ControlModifier)


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

        fut = self._exec.submit(self.p.eval_lunascope, cmd)  # returns str

        def done(_f=fut):
            try:
                exc = _f.exception()
                if exc is None:
                    self._last_result = _f.result()  # cheap; already completed
                    QMetaObject.invokeMethod(self, "_eval_done_ok", Qt.QueuedConnection)
                else:
                    self._last_exc = exc
                    self._last_tb = f"{type(exc).__name__}: {exc}"
                    QMetaObject.invokeMethod(self, "_eval_done_err", Qt.QueuedConnection)
            except Exception as cb_exc:
                self._last_exc = cb_exc
                self._last_tb = f"{type(cb_exc).__name__}: {cb_exc}"
                QMetaObject.invokeMethod(self, "_eval_done_err", Qt.QueuedConnection)

        fut.add_done_callback(done)


    @Slot()
    def _eval_done_ok(self):
        try:
            # --- step 1: write result text to console widget ---
            if self.project_mode:
                out = self.ui.txt_out
                out.moveCursor(QTextCursor.End)
                out.insertPlainText(self._last_result)
            else:
                self.ui.txt_out.setPlainText(self._last_result)

            # --- step 2: fetch strata from luna ---
            tbls = self.p.strata()

            if self.project_mode:
                self._accumulate_project_results(tbls)
            else:
                if hasattr(self, "_invalidate_spec_data_cache"):
                    self._invalidate_spec_data_cache()
                self._render_tables(tbls)

                rendered_ss = getattr(self, "ss", None) if getattr(self, "rendered", False) else None
                if hasattr(self, "_render_hypnogram"):
                    self._render_hypnogram()
                if rendered_ss is not None:
                    self.ss = rendered_ss

                if hasattr(self, "_update_hypnogram"):
                    self._update_hypnogram()

        except Exception:
            self._last_tb = traceback.format_exc().strip()
            _diag_log("_eval_done_ok: unhandled exception\n" + self._last_tb)
            try:
                QMessageBox.critical(self.ui, "Evaluation error", self._last_tb)
            except Exception:
                pass

        finally:
            self.unlock_ui()
            self._busy = False
            self._buttons(True)
            self._set_render_status(self.rendered, False)
            self.sb_progress.setRange(0, 100)
            self.sb_progress.setValue(0)
            self.sb_progress.setVisible(False)

            if getattr(self, 'project_mode', False) and getattr(self, '_proj_n', 0) > 0:
                self._proj_i += 1
                self._proj_eval_next()
            
    @Slot()
    def _eval_done_err(self):
        try:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self.ui, "Evaluation error", self._last_tb)
        finally:
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
        return _normalize_project_result_table(df, record_id)

        
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

        if not self.project_mode and not self._project_results_mode:
            tbl = tbl.drop(columns=["ID"])

        # transpose?
        if self.ui.radio_transpose.isChecked():
            # first coerce, otherwise this step will be missed by df_to_model()
            tbl = self.coerce_numeric_df( tbl )
            tbl = tbl.T.reset_index()
            tbl.rename(columns={"index": "VAR"}, inplace=True)
            tbl.columns = ["VAR"] + [f"row{i}" for i in range(1, tbl.shape[1])]

        cell_count = len(tbl) * max(1, len(tbl.columns))
        is_large_output = cell_count > _OUTPUT_TABLE_AUTORESIZE_CELL_LIMIT
        
        self.anal_model = self.df_to_model(
            tbl,
            coerce_numeric=False,
            build_row_text=False,
        )

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
        if is_large_output:
            default_w = max(70, min(120, h.defaultSectionSize()))
            h.setDefaultSectionSize(default_w)
        else:
            h.setResizeContentsPrecision(50)             # sample first 50 rows only
            view.resizeColumnsToContents()
        if (
            "ID" in tbl.columns
            and cell_count <= _OUTPUT_TABLE_AUTOSORT_CELL_LIMIT
        ):
            try:
                id_col = list(tbl.columns).index("ID")
                view.setSortingEnabled(True)
                view.sortByColumn(id_col, Qt.AscendingOrder)
            except Exception:
                view.setSortingEnabled(False)
        else:
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
            try:
                self._tree_sel.selectionChanged.disconnect(self._on_tree_sel)
            except TypeError:
                pass
        self._tree_sel = tv.selectionModel()
        if self._tree_sel is not None:
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
        all_rows = self._sample_rows_from_source_model()
        if not all_rows:
            return
        records = []
        for row in range(n):
            idx = model.index(row, 0)
            sample_row = self._sample_row_from_index(idx)
            if not sample_row:
                continue
            id_str = str(sample_row[0] or "").strip()
            if id_str:
                records.append((sample_row, id_str))
        if not records:
            return

        eval_config = self._project_eval_config_dialog(len(records))
        if eval_config is None:
            return
        workers = eval_config["workers"]

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
        shortcut_label = self._project_eval_cancel_shortcut_label()
        worker_word = "process" if workers == 1 else "processes"
        self.lock_ui(
            f"Processing with {workers} {worker_word}...\n\n"
            f"Press {shortcut_label} to stop queued records"
        )

        fut = self._exec.submit(
            self._project_eval_worker,
            records,
            all_rows,
            cmd,
            param,
            workers,
        )

        def _done(_f=fut):
            try:
                self.sig_proj_eval_finished.emit(_f.result())
            except Exception as e:
                self._last_exc = e
                self._last_tb = f"{type(e).__name__}: {e}"
                self.sig_proj_eval_failed.emit(self._last_tb)

        fut.add_done_callback(_done)

    def _project_eval_worker(self, records, all_rows, cmd, param, workers):
        total = len(records)
        workers = _clamp_project_eval_workers(workers, total)
        tasks = [
            {
                "ordinal": i,
                "sample_row": list(sample_row),
                "label": label,
                "cmd": cmd,
                "param": list(param),
            }
            for i, (sample_row, label) in enumerate(records, start=1)
        ]
        slices = _project_eval_slices(tasks, workers)
        completed = []
        completed_ordinals = set()
        errors = []
        cancelled = False
        done = 0

        executor = None
        manager = None
        result_queue = None
        pending = {}

        def _handle_record_result(result):
            nonlocal done
            ordinal = result.get("ordinal")
            if ordinal in completed_ordinals:
                return
            completed_ordinals.add(ordinal)
            done += 1
            completed.append(result)
            header = (
                "\n\n------------------------------------------------------------------\n"
                f"Finished: {result['label']} (#{result['ordinal']})\n"
            )
            self.sig_proj_eval_stream.emit(header)
            stdout_txt = result.get("stdout") or ""
            if stdout_txt:
                for chunk in stdout_txt.splitlines(True):
                    self.sig_proj_eval_stream.emit(chunk)

            err = result.get("error")
            if err:
                msg = f"{result['label']}: {err}"
                errors.append(msg)
                self.sig_proj_eval_stream.emit(f"\nERROR: {msg}\n")

            self.sig_proj_eval_progress.emit(done, total)

        def _drain_result_queue():
            if result_queue is None:
                return
            while True:
                try:
                    result = result_queue.get_nowait()
                except (queue.Empty, EOFError, OSError):
                    break
                _handle_record_result(result)

        try:
            mp_context = multiprocessing.get_context("spawn")
            manager = mp_context.Manager()
            result_queue = manager.Queue()
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp_context,
                initializer=_init_project_eval_child,
            )
            for task_slice in slices:
                if self._proj_cancel_event.is_set():
                    cancelled = True
                    break
                submitted_slice = dict(task_slice)
                submitted_slice["result_queue"] = result_queue
                fut = executor.submit(_project_eval_slice_worker, submitted_slice)
                pending[fut] = task_slice
                header = (
                    "\n\n------------------------------------------------------------------\n"
                    f"Queued slice {task_slice['slice_index']}: "
                    f"records #{task_slice['start_ordinal']}-{task_slice['end_ordinal']}\n"
                )
                self.sig_proj_eval_stream.emit(header)

            while pending:
                _drain_result_queue()
                if self._proj_cancel_event.is_set():
                    cancelled = True
                    self.sig_proj_eval_stream.emit("\nInterrupted: stopping project eval slices.\n")
                    for fut in pending:
                        fut.cancel()
                    break

                done_futs, _ = wait(
                    pending.keys(),
                    timeout=0.2,
                    return_when=FIRST_COMPLETED,
                )
                if not done_futs:
                    continue

                _drain_result_queue()
                for fut in done_futs:
                    task_slice = pending.pop(fut)
                    try:
                        slice_result = fut.result()
                    except Exception as e:
                        slice_result = {
                            "slice_index": task_slice["slice_index"],
                            "results": [
                                {
                                    "ordinal": record["ordinal"],
                                    "label": record["label"],
                                    "id": str(record["sample_row"][0] or "").strip(),
                                    "stdout": "",
                                    "tbls": None,
                                    "results": {},
                                    "error": f"{type(e).__name__}: {e}",
                                }
                                for record in task_slice["records"]
                            ],
                        }
                    for result in slice_result.get("results", []):
                        _handle_record_result(result)
        finally:
            _drain_result_queue()
            if executor is not None:
                if cancelled:
                    terminated = _terminate_process_pool(executor)
                    if not terminated:
                        executor.shutdown(wait=False, cancel_futures=True)
                else:
                    executor.shutdown(wait=True, cancel_futures=True)
            if manager is not None:
                manager.shutdown()
            self.proj.clear()
            if all_rows:
                self.proj.eng.set_sample_list(all_rows)

        completed.sort(key=lambda item: item.get("ordinal", 0))
        proj_tbls = []
        result_parts = {}
        for result in completed:
            tbls = result.get("tbls")
            if tbls is not None:
                proj_tbls.append(tbls)
            for key, df in (result.get("results") or {}).items():
                if df is not None:
                    result_parts.setdefault(key, []).append(df)

        all_tbls = None
        if proj_tbls:
            all_tbls = pd.concat(proj_tbls, ignore_index=True)
            all_tbls = all_tbls.drop_duplicates(subset=["Command", "Strata"])

        proj_results = {
            key: pd.concat(parts, ignore_index=True)
            for key, parts in result_parts.items()
            if parts
        }

        return {
            "tbls": all_tbls,
            "results": proj_results,
            "cancelled": cancelled,
            "errors": errors,
            "workers": workers,
        }

    def _request_project_eval_cancel(self):
        if not (getattr(self, "_busy", False) and getattr(self, "project_mode", False)):
            return
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
            act.setText("Stopping queued records...")
        else:
            act.setText("Stop project eval")

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
            errors = payload.get("errors") or []
            if errors:
                shown = "\n".join(errors[:12])
                more = "" if len(errors) <= 12 else f"\n...and {len(errors) - 12} more."
                QMessageBox.warning(
                    self.ui,
                    "Project evaluation completed with errors",
                    f"{len(errors)} record(s) failed; successful results were kept.\n\n"
                    f"{shown}{more}",
                )
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



    
        
