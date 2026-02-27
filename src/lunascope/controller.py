
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

from . import __version__

import lunapi as lp
import pandas as pd

import os, sys, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from PySide6.QtCore import QModelIndex, QObject, Signal, Qt, QSortFilterProxyModel
from PySide6.QtGui import QAction, QStandardItemModel
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QLabel, QFrame, QSizePolicy, QMessageBox, QLayout
from PySide6.QtWidgets import QMainWindow, QProgressBar, QTableView, QAbstractItemView
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget
from PySide6.QtGui import QKeySequence, QGuiApplication

import pyqtgraph as pg

from  .helpers import clear_rows, add_dock_shortcuts, pick_two_colors, override_colors, random_darkbg_colors, Blocker
from .components.tbl_funcs import add_combo_column, add_check_column

from .components.slist import SListMixin
from .components.metrics import MetricsMixin
from .components.hypno import HypnoMixin
from .components.anal import AnalMixin
from .components.signals import SignalsMixin
from .components.settings import SettingsMixin
from .components.masks import MasksMixin
from .components.ctree import CTreeMixin
from .components.spectrogram import SpecMixin
from .components.soappops import SoapPopsMixin
from .components.cmaps import CMapsMixin
from .gui_help import apply_gui_help, set_render_button_help
from .session_state import save_session_file, load_session_file


# ------------------------------------------------------------
# main GUI controller class

from PySide6.QtCore import QObject


class Controller( QObject, CMapsMixin, 
                  SListMixin , MetricsMixin ,
                  HypnoMixin , SoapPopsMixin, 
                  AnalMixin , SignalsMixin, 
                  SettingsMixin, CTreeMixin ,
                  SpecMixin , MasksMixin ):

    def __init__(self, ui, proj):

        super().__init__()

        # GUI
        self.ui = ui

        # Luna
        self.proj = proj
        
        # set up threading for compute funcs
        self._exec = ThreadPoolExecutor(max_workers=1)
        self._busy = False
        self.blocker = Blocker(self.ui, "...Processing...\n...please wait...", alpha=120)

        # initiate each component
        self._init_colors()
        self._init_cmaps()
        self._init_slist()
        self._init_metrics()
        self._init_hypno()
        self._init_anal()
        self._init_signals()
        self._init_settings()
        self._init_ctree()
        self._init_spec()
        self._init_soap_pops()
        self._init_masks()
        
        # for the tables added above, ensure all are read-only
        for v in self.ui.findChildren(QTableView):
            v.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        # set up menu items: open projects
        act_load_slist = QAction("Load S-List", self)
        act_build_slist = QAction("Build S-List", self)
        act_load_edf = QAction("Load EDF", self)
        act_load_annot = QAction("Load Annotations", self)
        act_refresh = QAction("Refresh", self)
        act_proj_eval = QAction("Evaluate (project)", self)
        act_save_session = QAction("Save Session...", self)
        act_load_session = QAction("Load Session...", self)
        
        # connect to same slots as buttons
        act_load_slist.triggered.connect(self.open_file)
        act_build_slist.triggered.connect(self.open_folder)
        act_load_edf.triggered.connect(self.open_edf)
        act_load_annot.triggered.connect(self.open_annot)
        act_refresh.triggered.connect(self._refresh)
        act_proj_eval.triggered.connect(self._proj_eval)
        act_save_session.triggered.connect(self._save_session_state)
        act_load_session.triggered.connect(self._load_session_state)

        self.ui.menuProject.addAction(act_load_slist)
        self.ui.menuProject.addAction(act_build_slist)
        self.ui.menuProject.addSeparator()
        self.ui.menuProject.addAction(act_load_edf)
        self.ui.menuProject.addAction(act_load_annot)
        self.ui.menuProject.addSeparator()
        self.ui.menuProject.addAction(act_refresh)
        self.ui.menuProject.addSeparator()
        self.ui.menuProject.addAction(act_proj_eval)
        self.ui.menuProject.addSeparator()
        self.ui.menuProject.addAction(act_save_session)
        self.ui.menuProject.addAction(act_load_session)

        # set up menu items: viewing
        self.ui.menuView.addAction(self.ui.dock_slist.toggleViewAction())
        self.ui.menuView.addAction(self.ui.dock_settings.toggleViewAction())
        self.ui.menuView.addSeparator()
        self.ui.menuView.addAction(self.ui.dock_sig.toggleViewAction())
        self.ui.menuView.addAction(self.ui.dock_annot.toggleViewAction())
        self.ui.menuView.addAction(self.ui.dock_annots.toggleViewAction())
        self.ui.menuView.addSeparator()
        self.ui.menuView.addAction(self.ui.dock_spectrogram.toggleViewAction())
        self.ui.menuView.addAction(self.ui.dock_hypno.toggleViewAction())
        self.ui.menuView.addSeparator()
        self.ui.menuView.addAction(self.ui.dock_mask.toggleViewAction())
        self.ui.menuView.addAction(self.ui.dock_console.toggleViewAction())
        self.ui.menuView.addAction(self.ui.dock_outputs.toggleViewAction())
        self.ui.menuView.addSeparator()
#        self.ui.menuView.addAction(self.ui.dock_help.toggleViewAction())

        # set up menu: about
        act_about = QAction("Help", self)

        act_about.triggered.connect( self.show_about )
        
        # palette menu
        act_pal_spectrum = QAction("Spectrum", self)
        act_pal_white    = QAction("White", self)
        act_pal_muted    = QAction("Muted", self)
        act_pal_black    = QAction("Black", self)
        act_pal_random   = QAction("Random", self)
#        act_pal_load     = QAction("Bespoke (load)", self)
        act_pal_bespoke  = QAction("Bespoke (apply)", self)
        act_pal_user     = QAction("Pick", self)

        act_pal_spectrum.triggered.connect(self._set_spectrum_palette)
        act_pal_white.triggered.connect(self._set_white_palette)
        act_pal_muted.triggered.connect(self._set_muted_palette)
        act_pal_black.triggered.connect(self._set_black_palette)
        act_pal_random.triggered.connect(self._set_random_palette)
#        act_pal_load.triggered.connect(self._load_palette)
        act_pal_bespoke.triggered.connect(self._set_bespoke_palette)
        act_pal_user.triggered.connect(self._select_user_palette)

        self._help_actions = {
            "project_load_slist": act_load_slist,
            "project_build_slist": act_build_slist,
            "project_load_edf": act_load_edf,
            "project_load_annot": act_load_annot,
            "project_refresh": act_refresh,
            "project_eval": act_proj_eval,
            "project_save_session": act_save_session,
            "project_load_session": act_load_session,
            "about_help": act_about,
            "palette_spectrum": act_pal_spectrum,
            "palette_white": act_pal_white,
            "palette_muted": act_pal_muted,
            "palette_black": act_pal_black,
            "palette_random": act_pal_random,
            "palette_pick": act_pal_user,
            "palette_bespoke": act_pal_bespoke,
        }
        
        self.ui.menuPalettes.addAction(act_pal_spectrum)
        self.ui.menuPalettes.addAction(act_pal_white)
        self.ui.menuPalettes.addAction(act_pal_muted)
        self.ui.menuPalettes.addAction(act_pal_black)
        self.ui.menuPalettes.addAction(act_pal_random)
        self.ui.menuPalettes.addSeparator()
        self.ui.menuPalettes.addAction(act_pal_user)
        self.ui.menuPalettes.addSeparator()
#        self.ui.menuPalettes.addAction(act_pal_load)
        self.ui.menuPalettes.addAction(act_pal_bespoke)
        
        # about menu
        self.ui.menuAbout.addAction(act_about)   

        # window title
        self.ui.setWindowTitle(f"Lunascope v{__version__}")

        # add QSplitter for console
        container = self.ui.console_splitter
        layout = container.layout()  # that's your console_layout
        splitter = QSplitter(Qt.Vertical)
        self.ui.txt_out.setParent(None)
        self.ui.txt_inp.setParent(None)
        splitter = QSplitter(Qt.Vertical, container)
        splitter.addWidget(self.ui.txt_out)
        splitter.addWidget(self.ui.txt_inp)
        layout.addWidget(splitter)
        
        # add QSplitter for output
        container2 = self.ui.anal_out_frame
        layout2 = container2.layout()  # that's your console_layout
        self.ui.anal_tables.setParent(None)
        self.ui.anal_right_table.setParent(None)
        splitter2 = QSplitter(Qt.Horizontal, container2)
        splitter2.addWidget(self.ui.anal_tables)
        splitter2.addWidget(self.ui.anal_right_table)
        layout2.addWidget(splitter2)

        # short keyboard cuts
        add_dock_shortcuts( self.ui, self.ui.menuView )

        # arrange docks: hide some docks
        self.ui.dock_help.hide()
        self.ui.dock_console.hide()
        self.ui.dock_outputs.hide()
        self.ui.dock_mask.hide()

        self.ui.dock_hypno.show()
        self.ui.dock_spectrogram.show()

        # arrange docks: lock and resize
        self.ui.setCorner(Qt.TopRightCorner,    Qt.RightDockWidgetArea)
        self.ui.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        
        # arrange docks: lower docks (console, outputs)
        w = self.ui.width()
        self.ui.resizeDocks(
            [self.ui.dock_console, self.ui.dock_outputs],
            [int(w * 0.6), int(w * 0.4)],
            Qt.Horizontal
        )

        # arrange docks: left docks (samples, settings)
        self.ui.resizeDocks(
            [self.ui.dock_slist, self.ui.dock_settings],
            [int(w * 0.5), int(w * 0.5)],
            Qt.Vertical
        )

        # arrange docks: stack spectrogram and hypnogram
        self.ui.tabifyDockWidget(self.ui.dock_spectrogram, self.ui.dock_hypno)
        self.ui.dock_spectrogram.raise_()

        # arrange docks: right docks (signals, annotations, events)
        h = self.ui.height()
        self.ui.resizeDocks(
            [self.ui.dock_sig, self.ui.dock_annot, self.ui.dock_annots, self.ui.dock_mask],
            [int(h * 0.35), int(h * 0.25), int(h * 0.1), int(h * 0.1)],
            Qt.Vertical
        )

        # adjust overall left vs right width
        w_right = 720
        self.ui.resizeDocks(
            [self.ui.dock_slist, self.ui.dock_sig],
            [self.ui.width() - w_right, w_right],
            Qt.Horizontal
        )

        # general layout policies
        cw = self.ui.centralWidget()
        cw.setMinimumWidth(0)
        cw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # hide these after resizing
        self.ui.dock_hypno.hide()
        self.ui.dock_spectrogram.hide()
        self.ui.dock_spectrogram.widget().setMinimumHeight(240)
        
        # ------------------------------------------------------------
        # set up status bar

        # ID | EDF-type start time/date | hms(act) / hms(tot) / epochs | # sigs / # annots | progress bar

        def mk_section(text):
            lab = QLabel(text)
            lab.setAlignment(Qt.AlignLeft)
            lab.setFrameShape(QFrame.StyledPanel)
            lab.setFrameShadow(QFrame.Sunken)
            lab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            return lab

        def vsep():
            s = QFrame(); s.setFrameShape(QFrame.VLine); s.setFrameShadow(QFrame.Sunken)
            return s

        sb = self.ui.statusbar

        sb.setSizeGripEnabled(True)
        
        self.sb_id     = mk_section( "" ); 
        self.sb_start  = mk_section( "" ); 
        self.sb_dur    = mk_section( "" );
        self.sb_ns     = mk_section( "" );
        self.sb_progress = QProgressBar()
        self.sb_progress.setRange(0, 100)
        self.sb_progress.setValue(0)

        sb.addPermanentWidget(self.sb_id ,1)
        sb.addPermanentWidget(vsep(),0)
        sb.addPermanentWidget(self.sb_start,1)
        sb.addPermanentWidget(vsep(),0)
        sb.addPermanentWidget(self.sb_dur,1)
        sb.addPermanentWidget(vsep(),0)
        sb.addPermanentWidget(self.sb_ns,1)
        sb.addPermanentWidget(vsep(),0)
        sb.addPermanentWidget(self.sb_progress,1)
        sb.addPermanentWidget(vsep(),0)


        # ------------------------------------------------------------
        # size overall app window
        
        self.ui.resize(1200, 800)
        apply_gui_help(self.ui, self._help_actions)
        set_render_button_help(self.ui, rendered=False, current=False)


    # ------------------------------------------------------------
    # blockers
    # ------------------------------------------------------------

    def lock_ui(self, msg="Processing...\n\n...please wait"):
        self.blocker.show_block(msg)

    def unlock_ui(self):
        self.blocker.hide_block()

    # ------------------------------------------------------------
    # clear all, i.e. drop current record
    # ------------------------------------------------------------

    def _drop_inst(self):
        # clear existing stuff
        self._clear_all()
        self.proj.clear_vars()
        self.proj.reinit()
            
    # ------------------------------------------------------------
    # attach a new record
    # ------------------------------------------------------------

    def _attach_inst(self, current: QModelIndex, _):

        # get ID from (possibly filtered) table
        if not current.isValid():
            return
        
        # clear existing stuff
        self._clear_all()

        # get/set parameters
        self.proj.clear_vars()
        self.proj.reinit()
        param = self._parse_tab_pairs( self.ui.txt_param )
        for p in param:
            self.proj.var( p[0] , p[1] )
            
        # attach the individual by ID (i.e. as list may be filtered)
        id_str = current.siblingAtColumn(0).data(Qt.DisplayRole)
        
        # attach EDF
        try:
            self.p = self.proj.inst( id_str )
        except Exception as e:
            QMessageBox.critical(
                self.ui,
                "Error",
                f"Problem attaching individual {id_str}\nError:\n{e}",
            )
            return

        # check for weird EDF record sizes
        rec_size = self.p.edf.stat()['rs']
        if not rec_size.is_integer():

            edf_file = self.p.edf.stat()['edf_file']
            base, ext = os.path.splitext(edf_file)
            if ext.lower() == ".edf":
                edf_file = f"{base}-edit.edf"
            else:
                edf_file = f"{path}-edit.edf"

            reply = QMessageBox.question(
                self.ui,
                "Fractional EDF record size warning",
                f"Non-integer EDF record size ({rec_size}).\n\nNot an error, but can cause problems.\n\n"                
                f"Would you like to generate a new EDF with standard 1-second EDF records?\n\n{edf_file}",
                QMessageBox.Yes | QMessageBox.No )        

            if reply == QMessageBox.Yes:
                try:
                    self.p.eval( 'RECORD-SIZE dur=1 no-problem edf=' + edf_file[:-4] )
                except Exception as e:
                    QMessageBox.critical(
                        self.ui,
                        "Error",
                        f"Problem generating new EDF\nError:\n{e}",
                    )
                    return
                finally:
                    QMessageBox.information(
                        self.ui,
                        "Reload EDF",
                        "Done - now reload the new EDF (or make a new sample list)" )
                return
        
        # initiate graphs
        self.curves = [ ]
        self.y0_curves = [ ]
        self.y_curves = [ ] 
        self.sigmod_curves = [ ] 
        self.annot_curves = [ ] 
        
        # and update things that need updating
        self._update_metrics()
        self._render_hypnogram()
        self._update_spectrogram_list()
        self._update_mask_list()
        self._update_soap_list()
        self._update_params()

        # get/set cmaps (done automatically via updates) 
        self._apply_cmaps()

        # initially, no signals rendered / not rendered / not current
        self._set_render_status( False , False )

        # draw
        self._render_signals_simple()

        # hypnogram + stats if available
        self._calc_hypnostats()

        
    # ------------------------------------------------------------
    #
    # clear for a new record
    #
    # ------------------------------------------------------------

    def _clear_all(self):

        if getattr(self, "events_table_proxy", None) is not None:
            clear_rows( self.events_table_proxy )

        if getattr(self, "anal_table_proxy", None) is not None:
            clear_rows( self.anal_table_proxy , keep_headers = False )

        if getattr(self, "signals_table_proxy", None) is not None:
            clear_rows( self.signals_table_proxy )

        if getattr(self, "annots_table_proxy", None) is not None:
            clear_rows( self.annots_table_proxy )

        clear_rows( self.ui.anal_tables ) 

        self.ui.combo_spectrogram.clear()
        self.ui.combo_pops.clear()
        self.ui.combo_soap.clear()

        if not getattr(self, "project_mode", False):
            self.ui.txt_out.clear()
        
        self.spectrogramcanvas.ax.cla()
        self.spectrogramcanvas.figure.canvas.draw_idle()

        self.hypnocanvas.ax.cla()
        self.hypnocanvas.figure.canvas.draw_idle()

        self.soapcanvas.ax.cla()
        self.soapcanvas.figure.canvas.draw_idle()

        self.popscanvas.ax.cla()
        self.popscanvas.figure.canvas.draw_idle()

        # POPS results
        self.pops_df = pd.DataFrame()
        
        # filters: chennels -> filters
        self.fmap = { }

        # filter label -> frqs
        self.fmap_frqs = {
            "0.3-35Hz": [0.3,35] ,
            "Slow": [0.5,1] ,
            "Delta": [1,4],
            "Theta": [4,8],
            "Alpha": [8,11],
            "Sigma": [11,15],
            "Beta": [15,30] ,
            "Gamma": [30,50] ,
            "User": [ ] } 

        # user-speific filter map: { ch : [ lwr , upr ] } 
        self.user_fmap_frqs = { } 

        # SR + label --> butterworth model
        self.fmap_flts = { } 

    #
    # helper to handle render button
    #


    def _set_render_status(self, rendered , current ):
        # three modes:
        #   initial (pg1_simple)     not rendered (ignore changed) --> red
        #   post render              render and not changed        --> green
        #   post render, post Exec   render and changed            --> amber
        
        self.rendered = rendered
        self.current  = current

        if self.rendered:
            if self.current:
                self.ui.butt_render.setStyleSheet("background-color: #2E8B57; color: #FFFFFF;")
            else:
                self.ui.butt_render.setStyleSheet("background-color: #FFC107; color: #5C0000;")
        else:
            self.ui.butt_render.setStyleSheet("background-color: #F8F8F8; color: #8B0000;")

        # set empiric false to allow fixed scale in un-rendered
        self.ui.radio_empiric.setChecked( self.rendered )
        self.ui.radio_empiric.setEnabled( self.rendered )
        self.ui.radio_clip.setEnabled( self.rendered )
        self.ui.spin_scale.setEnabled( self.rendered )
        self.ui.spin_spacing.setEnabled( self.rendered )
        self.ui.label_spacing.setEnabled( self.rendered )
        self.ui.label_scale.setEnabled( self.rendered )
        self.ui.radio_fixedscale.setEnabled( self.rendered )
        set_render_button_help(self.ui, rendered=self.rendered, current=self.current)
                        
        
    def show_about(self):
        box = QMessageBox(self.ui)
        box.setWindowTitle("About Lunascope")
        box.setIcon(QMessageBox.Information)
        box.setTextFormat(Qt.RichText)

        # compute versions
        x = lp.version()  # { lunapi:ver, luna:ver }
        box.setText(
            f"<p>Lunascope v{__version__}</p>"
            f"<p>Lunapi {x['lunapi']}</p>"
            f"<p>Luna {x['luna']}</p>"
            "<p>Documentation:<br> <a href='http://zzz-luna.org/lunascope'>"
            "http://zzz-luna.org/lunascope</a></p>"
            "<p>Created by Shaun Purcell</p>"
            "<p>Developed and maintained by Lorcan Purcell</p>"
        )

        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        box.layout().setSizeConstraint(QLayout.SetMinimumSize)

        lbl = box.findChild(QLabel)
        if lbl:
            lbl.setOpenExternalLinks(True)

        box.exec()

    def _save_session_state(self):
        filename, _ = QFileDialog.getSaveFileName(
            self.ui,
            "Save Session",
            "",
            "Lunascope Session (*.lss);;JSON (*.json);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not filename:
            return

        app_meta = {
            "lunascope_version": __version__,
        }
        try:
            x = lp.version()
            app_meta["lunapi_version"] = x.get("lunapi")
            app_meta["luna_version"] = x.get("luna")
        except Exception:
            pass

        try:
            session_meta = {}
            slabel = self.ui.lbl_slist.text().strip() if hasattr(self.ui, "lbl_slist") else ""
            if slabel and slabel not in {"(none)", "<internal>"}:
                session_meta["slist_path"] = slabel

            # Save current displayed sample rows so internal EDF-based sessions can be restored.
            model = self.ui.tbl_slist.model() if hasattr(self.ui, "tbl_slist") else None
            rows = []
            if model is not None:
                for r in range(model.rowCount()):
                    row = []
                    for c in range(3):
                        idx = model.index(r, c)
                        row.append(str(model.data(idx, Qt.DisplayRole) or ""))
                    rows.append(row)
            if rows:
                session_meta["sample_rows"] = rows

            sel = self.ui.tbl_slist.selectionModel() if hasattr(self.ui, "tbl_slist") else None
            if sel and sel.currentIndex().isValid():
                session_meta["selected_row"] = int(sel.currentIndex().row())

            save_session_file(
                filename,
                self.ui,
                app_meta=app_meta,
                session_meta=session_meta,
            )
        except Exception as e:
            QMessageBox.critical(
                self.ui,
                "Save Session Error",
                f"Could not save session.\n\n{type(e).__name__}: {e}",
            )
            return

    def _load_session_state(self):
        filename, _ = QFileDialog.getOpenFileName(
            self.ui,
            "Load Session",
            "",
            "Lunascope Session (*.lss);;JSON (*.json);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not filename:
            return

        self.load_session_state_file(filename)

    def load_session_state_file(self, filename: str):
        slist_load_note = None
        try:
            res = load_session_file(filename, self.ui)
        except Exception as e:
            QMessageBox.critical(
                self.ui,
                "Load Session Error",
                f"Could not load session.\n\n{type(e).__name__}: {e}",
            )
            return

        # Optional Phase 1 project pointer restore: re-load sample list path if provided.
        try:
            smeta = res.get("state", {}).get("session", {})
            slist_path = smeta.get("slist_path")
            selected_row = int(smeta.get("selected_row", 0))
            if slist_path:
                p = Path(str(slist_path)).expanduser()
                if p.is_file():
                    folder_path = str(p.parent) + os.sep
                    self.proj.var("path", folder_path)
                    self._read_slist_from_file(str(p))
                    model = self.ui.tbl_slist.model()
                    if model and model.rowCount() > 0:
                        row = max(0, min(selected_row, model.rowCount() - 1))
                        self.ui.tbl_slist.setCurrentIndex(model.index(row, 0))
                        self.ui.tbl_slist.selectRow(row)
                    slist_load_note = f"slist loaded: {p}"
                else:
                    slist_load_note = f"slist missing: {p}"
            else:
                sample_rows = smeta.get("sample_rows")
                if isinstance(sample_rows, list) and sample_rows:
                    self.proj.clear()
                    self.proj.eng.set_sample_list(sample_rows)
                    df = self.proj.sample_list()
                    model = self.df_to_model(df)
                    self._proxy.setSourceModel(model)
                    self.ui.lbl_slist.setText("<internal>")
                    view = self.ui.tbl_slist
                    h = view.horizontalHeader()
                    h.setSectionResizeMode(QHeaderView.Interactive)
                    h.setStretchLastSection(False)
                    view.resizeColumnsToContents()
                    view.setSelectionBehavior(QAbstractItemView.SelectRows)
                    view.setSelectionMode(QAbstractItemView.SingleSelection)
                    view.verticalHeader().setVisible(True)
                    if model.rowCount() > 0:
                        row = max(0, min(selected_row, model.rowCount() - 1))
                        self.ui.tbl_slist.setCurrentIndex(model.index(row, 0))
                        self.ui.tbl_slist.selectRow(row)
                    slist_load_note = "internal sample list restored"
        except Exception as e:
            slist_load_note = f"slist restore error: {type(e).__name__}: {e}"

        rep = res["report"]
        has_issues = bool(rep.get("deferred") or rep.get("skipped") or rep.get("missing"))
        details = []
        if slist_load_note and ("missing:" in slist_load_note or "error:" in slist_load_note):
            details.append(f"Session project context: {slist_load_note}")
        if rep.get("deferred_items"):
            details.append("Deferred items:")
            details.extend(f"  - {x}" for x in rep["deferred_items"][:20])
        if rep.get("skipped_items"):
            details.append("Skipped items:")
            details.extend(f"  - {x}" for x in rep["skipped_items"][:20])
        if rep.get("missing_items"):
            details.append("Missing items:")
            details.extend(f"  - {x}" for x in rep["missing_items"][:20])
        details_text = ("\n\n" + "\n".join(details)) if details else ""

        if has_issues or details:
            QMessageBox.information(
                self.ui,
                "Session Loaded",
                f"Loaded session:\n{res['path']}\n\n"
                f"Restored: {rep['restored']}\n"
                f"Deferred: {rep.get('deferred', 0)}\n"
                f"Skipped: {rep['skipped']}\n"
                f"Missing: {rep['missing']}"
                f"{details_text}",
            )
