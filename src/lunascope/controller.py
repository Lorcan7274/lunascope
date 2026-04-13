
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

from PySide6.QtCore import QModelIndex, QObject, Signal, Qt, QSortFilterProxyModel, QEvent, QTimer
from PySide6.QtGui import QAction, QStandardItemModel
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QLabel, QFrame, QSizePolicy, QMessageBox, QLayout
from PySide6.QtWidgets import QMainWindow, QProgressBar, QTableView, QAbstractItemView
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QSplitter, QVBoxLayout, QWidget
from PySide6.QtGui import QKeySequence, QGuiApplication
from .file_dialogs import open_file_name, save_file_name

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
from .components.actigraphy import ActigraphyMixin
from .components.soappops import SoapPopsMixin
from .components.tutorial import TutorialMixin
from .components.save_edf import SaveEDFMixin
from .components.drop_signals import DropSignalsMixin
from .components.psd_overlay import PSDOverlayMixin
from .components.cmaps import CMapsMixin
from .components.results_io import ResultsIOMixin
from .components.moonbeam_dock import MoonbeamMixin
from .components.explorer_dock import ExplorerMixin
from .components.annotator import AnnotatorMixin
from .gui_help import apply_gui_help, set_render_button_help
from .session_state import save_session_file, load_session_file, save_geometry_file, load_geometry_file
from .runtime_paths import app_state_file


# ------------------------------------------------------------
# main GUI controller class

from PySide6.QtCore import QObject


class Controller( QObject, CMapsMixin, ResultsIOMixin,
                  SListMixin , MetricsMixin ,
                  HypnoMixin , SoapPopsMixin,
                  AnalMixin , SignalsMixin,
                  SettingsMixin, CTreeMixin ,
                  SpecMixin , ActigraphyMixin, MasksMixin,
                  MoonbeamMixin, ExplorerMixin, AnnotatorMixin, TutorialMixin,
                  SaveEDFMixin, DropSignalsMixin,
                  PSDOverlayMixin ):

    sig_results_changed = Signal()   # emitted whenever self.results is repopulated
    sig_proj_eval_stream = Signal(str)
    sig_proj_eval_progress = Signal(int, int)
    sig_proj_eval_finished = Signal(object)
    sig_proj_eval_failed = Signal(str)

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
        self._geometry_cache_saved = False
        self.ui.installEventFilter(self)

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
        self._init_actigraphy()
        self._init_soap_pops()
        self._init_masks()
        self._init_results_io()
        self._init_moonbeam()
        self._init_explorer()
        self._init_annotator()
        self._init_psd_overlay()
        
        # for the tables added above, ensure all are read-only
        for v in self.ui.findChildren(QTableView):
            v.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        # set up menu items: open projects
        act_load_slist = QAction("Load S-List", self)
        act_build_slist = QAction("Build S-List", self)
        act_load_edf = QAction("Load EDF", self)
        act_load_annot = QAction("Load Annotations", self)
        act_save_edf = QAction("Export EDF + Annotations…", self)
        act_drop_signals = QAction("Drop channels / annotations…", self)
        act_refresh = QAction("Refresh", self)
        act_proj_eval = QAction("Evaluate (project)", self)
        act_save_session = QAction("Save Session...", self)
        act_load_session = QAction("Load Session...", self)
        act_download_pops = QAction("Download POPS Resources...", self)
        act_download_tutorial = QAction("Download Tutorial...", self)

        # connect to same slots as buttons
        act_load_slist.triggered.connect(self.open_file)
        act_build_slist.triggered.connect(self.open_folder)
        act_load_edf.triggered.connect(self.open_edf)
        act_load_annot.triggered.connect(self.open_annot)
        act_save_edf.triggered.connect(self._save_edf_annots)
        act_drop_signals.triggered.connect(self._drop_signals_annots)
        act_refresh.triggered.connect(self._refresh)
        act_proj_eval.triggered.connect(self._proj_eval)
        act_save_session.triggered.connect(self._save_session_state)
        act_load_session.triggered.connect(self._load_session_state)
        act_download_pops.triggered.connect(self._download_pops_resources)
        act_download_tutorial.triggered.connect(self._download_tutorial)
        self._act_proj_eval = act_proj_eval

        self.ui.menuProject.addAction(act_load_slist)
        self.ui.menuProject.addAction(act_build_slist)
        self.ui.menuProject.addSeparator()
        self.ui.menuProject.addAction(act_load_edf)
        self.ui.menuProject.addAction(act_load_annot)
        self.ui.menuProject.addAction(act_save_edf)
        self.ui.menuProject.addAction(act_drop_signals)
        self.ui.menuProject.addSeparator()
        self.ui.menuProject.addAction(act_refresh)
        self.ui.menuProject.addSeparator()
        self.ui.menuProject.addAction(act_proj_eval)
        self.ui.menuProject.addSeparator()
        self.ui.menuProject.addAction(act_save_session)
        self.ui.menuProject.addAction(act_load_session)
        self.ui.menuProject.addAction(act_download_pops)
        self.ui.menuProject.addAction(act_download_tutorial)

        # set up menu items: viewing
        self.ui.menuView.addAction(self.ui.dock_slist.toggleViewAction())
        self.ui.menuView.addAction(self.ui.dock_settings.toggleViewAction())
        self.ui.menuView.addSeparator()
        self.ui.menuView.addAction(self.ui.dock_sig.toggleViewAction())
        self.ui.menuView.addAction(self.ui.dock_annot.toggleViewAction())
        self.ui.menuView.addAction(self.ui.dock_annots.toggleViewAction())
        self.ui.menuView.addSeparator()
        self.ui.menuView.addAction(self.ui.dock_spectrogram.toggleViewAction())
        act_hypno      = self.ui.dock_hypno.toggleViewAction()
        act_actigraphy = self.ui.dock_actigraphy.toggleViewAction()
        self.ui.menuView.addAction(act_hypno)
        self.ui.menuView.addAction(act_actigraphy)
        self._act_hypno_menu      = act_hypno
        self._act_actigraphy_menu = act_actigraphy

        # Single Ctrl+7 dispatcher – toggles whichever dock is active for the mode
        act_7 = QAction("Hypno/Actigraphy (Ctrl+7)", self)
        act_7.setShortcut(QKeySequence("Ctrl+7"))
        act_7.setShortcutContext(Qt.ApplicationShortcut)
        act_7.triggered.connect(self._toggle_hypno_or_actigraphy)
        self.ui.addAction(act_7)   # attach to window so it fires globally
        self.ui.menuView.addSeparator()
        self.ui.menuView.addAction(self.ui.dock_mask.toggleViewAction())
        self.ui.menuView.addAction(self.ui.dock_console.toggleViewAction())
        self.ui.menuView.addAction(self.ui.dock_outputs.toggleViewAction())
        self.ui.menuView.addSeparator()
        act_moonbeam = self.ui.dock_moonbeam.toggleViewAction()
        act_moonbeam.setShortcut(QKeySequence("Ctrl+M"))
        self.ui.menuView.addAction(act_moonbeam)
        act_annex = self.ui.dock_explorer.toggleViewAction()
        act_annex.setShortcut(QKeySequence("Ctrl+E"))
        act_annex.setText("Explorer (Ctrl+E)")
        self.ui.menuView.addAction(act_annex)
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

        self._help_actions = {
            "project_load_slist": act_load_slist,
            "project_build_slist": act_build_slist,
            "project_load_edf": act_load_edf,
            "project_load_annot": act_load_annot,
            "project_refresh": act_refresh,
            "project_eval": act_proj_eval,
            "project_save_session": act_save_session,
            "project_load_session": act_load_session,
            "project_download_pops": act_download_pops,
            "project_download_tutorial": act_download_tutorial,
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
        add_dock_shortcuts( self.ui, self.ui.menuView, self._toggle_signals_only_or_default )

        # size overall app window – cap to available screen space.
        # 1440 wide allows the banner's PSD panel to show at default on screens
        # >= 1440 px logical width; it hides naturally on narrower displays.
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            win_w = min(1440, int(avail.width()  * 0.92))
            win_h = min(900,  int(avail.height() * 0.92))
        else:
            win_w, win_h = 1440, 900
        self.ui.resize(win_w, win_h)

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

        # Use relative defaults so the center signal viewer keeps a similar
        # proportion across lower-resolution Windows and higher-resolution macOS displays.
        w = self.ui.width()
        h = self.ui.height()
        left_w  = max(220, int(w * 0.20))
        right_w = max(220, int(w * 0.20))
        bottom_left_w  = max(220, int(w * 0.60))
        bottom_right_w = max(180, w - bottom_left_w)

        # arrange docks: lower docks (console, outputs)
        self.ui.resizeDocks(
            [self.ui.dock_console, self.ui.dock_outputs],
            [bottom_left_w, bottom_right_w],
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
        self.ui.resizeDocks(
            [self.ui.dock_sig, self.ui.dock_annot, self.ui.dock_annots, self.ui.dock_mask],
            [int(h * 0.35), int(h * 0.25), int(h * 0.1), int(h * 0.1)],
            Qt.Vertical
        )

        # adjust overall left vs right width
        self.ui.resizeDocks(
            [self.ui.dock_slist, self.ui.dock_sig],
            [left_w, right_w],
            Qt.Horizontal
        )

        # general layout policies
        cw = self.ui.centralWidget()
        cw.setMinimumWidth(0)
        cw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # hide these after resizing
        self.ui.dock_hypno.hide()
        self.ui.dock_spectrogram.hide()
        self.ui.dock_actigraphy.hide()
        self._detach_actigraphy_dock_from_main_layout()
        self.ui.dock_explorer.hide()
        self.ui.dock_explorer.setFloating(True)
        self.annotator.hide()
        self.annotator.setFloating(True)
        self.ui.dock_spectrogram.widget().setMinimumHeight(240)
        self._capture_default_dock_layout()
        QTimer.singleShot(0, self._set_initial_focus)
        
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
        self.sb_mode   = mk_section( "" );
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
        sb.addPermanentWidget(self.sb_mode,1)
        sb.addPermanentWidget(vsep(),0)
        sb.addPermanentWidget(self.sb_progress,1)
        sb.addPermanentWidget(vsep(),0)

        self.sb_mode.setMinimumWidth(120)
        self._update_mode_badge()

        # On Windows set a modest monospace font size for the text-edit panels
        # so they don't appear oversized on lower-resolution / non-HiDPI screens.
        if sys.platform == "win32":
            from PySide6.QtGui import QFont
            mono = QFont("Courier New", 9)
            for _name in ("txt_param", "txt_cmap", "txt_out", "txt_inp"):
                _w = getattr(self.ui, _name, None)
                if _w is not None:
                    _w.setFont(mono)

        apply_gui_help(self.ui, self._help_actions)
        set_render_button_help(self.ui, rendered=False, current=False)

    def _set_initial_focus(self):
        target = getattr(self.ui, "pg1", None)
        if target is None:
            return
        try:
            target.setFocus(Qt.OtherFocusReason)
        except Exception:
            try:
                target.setFocus()
            except Exception:
                pass


    # ------------------------------------------------------------
    # blockers
    # ------------------------------------------------------------

    def lock_ui(self, msg="Processing...\n\n...please wait"):
        self.blocker.show_block(msg)
        QGuiApplication.processEvents()

    def unlock_ui(self):
        self.blocker.hide_block()

    def _toggle_hypno_or_actigraphy(self):
        if getattr(self, "multiday_mode", False):
            dock = self.ui.dock_actigraphy
        else:
            dock = self.ui.dock_hypno
        dock.setVisible(not dock.isVisible())

    def _update_mode_badge(self):
        multiday = getattr(self, "multiday_mode", False)
        if multiday:
            self.sb_mode.setText("Multiple days")
            self.sb_mode.setStyleSheet(
                "QLabel { color: #f2d48f; background: rgba(88,64,18,0.35); border: 1px solid rgba(222,184,82,0.35); }"
            )
        else:
            self.sb_mode.setText("Single night")
            self.sb_mode.setStyleSheet(
                "QLabel { color: #b8c4d6; background: rgba(33,44,62,0.28); border: 1px solid rgba(120,140,170,0.2); }"
            )
        # Gray out whichever Ctrl+7 dock is not relevant for the current mode
        if hasattr(self, "_act_hypno_menu"):
            self._act_hypno_menu.setEnabled(not multiday)
        if hasattr(self, "_act_actigraphy_menu"):
            self._act_actigraphy_menu.setEnabled(multiday)

    # ------------------------------------------------------------
    # clear all, i.e. drop current record
    # ------------------------------------------------------------

    def _drop_inst(self):
        # clear existing stuff
        self._clear_all()
        self.proj.clear_vars()
        self.proj.reinit()
        if hasattr(self, "p"):
            del self.p

    def _detach_inst_preserve_analysis(self):
        if getattr(self, "events_table_proxy", None) is not None:
            clear_rows(self.events_table_proxy)
        if getattr(self, "signals_table_proxy", None) is not None:
            clear_rows(self.signals_table_proxy)
        if getattr(self, "annots_table_proxy", None) is not None:
            clear_rows(self.annots_table_proxy)

        self.ui.combo_spectrogram.clear()
        self.ui.combo_actigraphy.clear()
        self.ui.combo_pops.clear()
        self.ui.combo_soap.clear()

        if getattr(self, "spectrogramcanvas", None) is not None:
            self.spectrogramcanvas.ax.cla()
            self.spectrogramcanvas.figure.canvas.draw_idle()
        if getattr(self, "hypnocanvas", None) is not None:
            self.hypnocanvas.ax.cla()
            self.hypnocanvas.figure.canvas.draw_idle()
        if getattr(self, "actigraphycanvas", None) is not None:
            self.actigraphycanvas.ax.cla()
            self.actigraphycanvas.figure.canvas.draw_idle()
        if getattr(self, "soapcanvas", None) is not None:
            self.soapcanvas.ax.cla()
            self.soapcanvas.figure.canvas.draw_idle()

        self.multiday_mode = False
        if hasattr(self, "_update_mode_badge"):
            self._update_mode_badge()
        if hasattr(self, "_sync_multiday_actigraphy_dock"):
            self._sync_multiday_actigraphy_dock()
        self._detach_actigraphy_dock_from_main_layout()
        self._set_render_status(False, False)

        self.proj.clear_vars()
        self.proj.reinit()
        if hasattr(self, "p"):
            del self.p

        try:
            self.ui.tbl_slist.clearSelection()
            self.ui.tbl_slist.setCurrentIndex(QModelIndex())
        except Exception:
            pass
            
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
                edf_file = f"{base}-edit.edf"

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
        self.sigmod_curve_colors = [ ]
        self.annot_curves = [ ] 
        
        # and update things that need updating
        self._update_metrics()
        self._render_hypnogram()
        self._update_spectrogram_list()
        self._update_actigraphy_list()
        self._update_mask_list()
        self._update_soap_list()
        self._update_params()

        # get/set cmaps (done automatically via updates) 
        self._apply_cmaps()

        # initially, no signals rendered / not rendered / not current
        self._set_render_status( False , False )

        # draw
        self._render_signals_simple()

        # multiday records default to the actigraphy dock; otherwise keep HYPNO
        if getattr(self, "multiday_mode", False):
            self._sync_multiday_actigraphy_dock()
        else:
            self._sync_multiday_actigraphy_dock()
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
        self.ui.combo_actigraphy.clear()
        self.ui.combo_pops.clear()
        self.ui.combo_soap.clear()

        if not getattr(self, "project_mode", False):
            self.ui.txt_out.clear()
        
        if getattr(self, "spectrogramcanvas", None) is not None:
            self.spectrogramcanvas.ax.cla()
            self.spectrogramcanvas.figure.canvas.draw_idle()

        if getattr(self, "hypnocanvas", None) is not None:
            self.hypnocanvas.ax.cla()
            self.hypnocanvas.figure.canvas.draw_idle()

        if getattr(self, "actigraphycanvas", None) is not None:
            self.actigraphycanvas.ax.cla()
            self.actigraphycanvas.figure.canvas.draw_idle()
        if hasattr(self, "_update_actigraphy_summary"):
            self._update_actigraphy_summary()
        self.multiday_mode = False
        if hasattr(self, "_update_mode_badge"):
            self._update_mode_badge()
        if hasattr(self, "_sync_multiday_actigraphy_dock"):
            self._sync_multiday_actigraphy_dock()
        self._detach_actigraphy_dock_from_main_layout()

        if getattr(self, "soapcanvas", None) is not None:
            self.soapcanvas.ax.cla()
            self.soapcanvas.figure.canvas.draw_idle()

        if getattr(self, "popscanvas", None) is not None:
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

        # In non-render (epoch) mode the view is fixed at 30 s, so a larger
        # jump window would just snap back.  Cap the spinbox and clamp value.
        jump = self.ui.spin_jump_width
        if self.rendered:
            jump.setMaximum(3600.0)
        else:
            jump.setMaximum(30.0)
            if jump.value() > 30.0:
                jump.setValue(30.0)

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
        filename, _ = save_file_name(
            self.ui,
            "Save Session",
            "",
            "Lunascope Session (*.lss);;JSON (*.json);;All Files (*)",
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
        filename, _ = open_file_name(
            self.ui,
            "Load Session",
            "",
            "Lunascope Session (*.lss);;JSON (*.json);;All Files (*)",
        )
        if not filename:
            return

        self.load_session_state_file(filename)

    def _detach_actigraphy_dock_from_main_layout(self):
        dock = getattr(self.ui, "dock_actigraphy", None)
        if dock is None:
            return

        was_visible = dock.isVisible()
        if not dock.isFloating():
            dock.setFloating(True)

        if was_visible:
            if hasattr(self, "_present_actigraphy_dock"):
                self._present_actigraphy_dock()
            dock.raise_()
        else:
            dock.hide()

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
                    self._configure_slist_view()
                    if model.rowCount() > 0:
                        row = max(0, min(selected_row, model.rowCount() - 1))
                        self.ui.tbl_slist.setCurrentIndex(model.index(row, 0))
                        self.ui.tbl_slist.selectRow(row)
                    slist_load_note = "internal sample list restored"
        except Exception as e:
            slist_load_note = f"slist restore error: {type(e).__name__}: {e}"

        self._detach_actigraphy_dock_from_main_layout()

        if slist_load_note and ("missing:" in slist_load_note or "error:" in slist_load_note):
            QMessageBox.warning(
                self.ui,
                "Session Loaded With Warning",
                f"Loaded session, but project context was not fully restored.\n\n{slist_load_note}",
            )

    def _geometry_cache_path(self) -> Path:
        return app_state_file("window_geometry.json")

    def _capture_default_dock_layout(self):
        try:
            self._default_window_geometry_b64 = self.ui.saveGeometry()
        except Exception:
            self._default_window_geometry_b64 = None
        try:
            self._default_window_state_b64 = self.ui.saveState()
        except Exception:
            self._default_window_state_b64 = None

    def _signals_only_mode_active(self):
        try:
            return (
                not self.ui.dock_slist.isVisible()
                and not self.ui.dock_settings.isVisible()
                and not self.ui.dock_sig.isVisible()
                and not self.ui.dock_annot.isVisible()
                and not self.ui.dock_annots.isVisible()
                and not self.annotator.isVisible()
            )
        except Exception:
            return False

    def _show_signals_only_layout(self):
        for dock in (
            self.ui.dock_slist,
            self.ui.dock_settings,
            self.ui.dock_sig,
            self.ui.dock_annot,
            self.ui.dock_annots,
            self.ui.dock_spectrogram,
            self.ui.dock_hypno,
            self.ui.dock_actigraphy,
            self.ui.dock_mask,
            self.ui.dock_console,
            self.ui.dock_outputs,
            self.ui.dock_help,
        ):
            dock.hide()
        self.ui.dock_explorer.hide()
        self.annotator.hide()

    def _restore_default_dock_layout(self):
        try:
            if getattr(self, "_default_window_geometry_b64", None) is not None:
                self.ui.restoreGeometry(self._default_window_geometry_b64)
        except Exception:
            pass
        try:
            if getattr(self, "_default_window_state_b64", None) is not None:
                self.ui.restoreState(self._default_window_state_b64)
        except Exception:
            pass

        for dock in (
            self.ui.dock_slist,
            self.ui.dock_settings,
            self.ui.dock_sig,
            self.ui.dock_annot,
            self.ui.dock_annots,
        ):
            dock.show()

        for dock in (
            self.ui.dock_spectrogram,
            self.ui.dock_hypno,
            self.ui.dock_actigraphy,
            self.ui.dock_mask,
            self.ui.dock_console,
            self.ui.dock_outputs,
            self.ui.dock_help,
        ):
            dock.hide()
        self.annotator.hide()

        self.ui.dock_explorer.hide()
        self.ui.dock_explorer.setFloating(True)
        self._detach_actigraphy_dock_from_main_layout()

    def _toggle_signals_only_or_default(self):
        if self._signals_only_mode_active():
            self._restore_default_dock_layout()
        else:
            self._show_signals_only_layout()

    def save_geometry_cache_silently(self):
        try:
            save_geometry_file(
                self._geometry_cache_path(),
                self.ui,
                app_meta={"lunascope_version": __version__},
            )
            self._geometry_cache_saved = True
        except Exception:
            pass

    def load_geometry_cache_silently(self):
        try:
            p = self._geometry_cache_path()
            if p.is_file():
                load_geometry_file(p, self.ui)
                self._detach_actigraphy_dock_from_main_layout()
        except Exception:
            pass

    def eventFilter(self, obj, event):
        try:
            if obj is self.ui and event.type() == QEvent.Close and not self._geometry_cache_saved:
                self.save_geometry_cache_silently()
        except RuntimeError:
            return False

        try:
            return super().eventFilter(obj, event)
        except RuntimeError:
            return False
