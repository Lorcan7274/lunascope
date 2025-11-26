
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

        # connect to same slots as buttons
        act_load_slist.triggered.connect(self.open_file)
        act_build_slist.triggered.connect(self.open_folder)
        act_load_edf.triggered.connect(self.open_edf)
        act_load_annot.triggered.connect(self.open_annot)
        act_refresh.triggered.connect(self._refresh)

        self.ui.menuProject.addAction(act_load_slist)
        self.ui.menuProject.addAction(act_build_slist)
        self.ui.menuProject.addSeparator()
        self.ui.menuProject.addAction(act_load_edf)
        self.ui.menuProject.addAction(act_load_annot)
        self.ui.menuProject.addSeparator()
        self.ui.menuProject.addAction(act_refresh)

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
        self.ui.menuView.addAction(self.ui.dock_help.toggleViewAction())

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
            [int(w * 0.7), int(w * 0.3)],
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


    # ------------------------------------------------------------
    # blockers
    # ------------------------------------------------------------

    def lock_ui(self, msg="Processing...\n\n...please wait"):
        self.blocker.show_block(msg)

    def unlock_ui(self):
        self.blocker.hide_block()

            
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

