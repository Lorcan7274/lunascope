
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
#  Luna / Lunascope  —  Explorer dock (outer shell)
#  --------------------------------------------------------------------

"""
Ctrl+E  →  floating "Explorer" dock with tabbed panels:

    1  Annotations  – cohort-level annotation explorer (PETH, overlap, …)
    2  Hypnoscope   – staging grid across all subjects aligned by time
    3  Waveforms    – peri-event signal traces for the current record
    4  Plotter      – generic scatter / line / bar / histogram for output tables
    5  Assoc        – GPA / group-level association analyses
    6  Topo         – EEG topographic maps (Results + Live animated player)
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QTabWidget

from ..helpers import screen_clamp, AuxiliaryWindow


class ExplorerMixin:
    """Mixin that creates and owns the tabbed Explorer dock."""

    _EXPLORER_FLOAT_SIZE = (1120, 840)

    # ------------------------------------------------------------------
    # Initialisation (called from Controller.__init__)
    # ------------------------------------------------------------------

    def _init_explorer(self):
        # Late imports avoid circular dependencies at module load
        from .explorer_annot      import AnnotTab
        from .explorer_hypnoscope import HypnoscopeTab
        from .explorer_waveform   import WaveformTab
        from .explorer_event_decomp import EventDecompTab
        from .explorer_two_channel_dynamics import TwoChannelDynamicsTab
        from .explorer_plotter    import PlotterTab
        from .explorer_gpa        import GPATab
        from .explorer_filter_design import FilterDesignTab
        from .explorer_topo       import TopoTab
        from .explorer_harmonizer import HarmonizerTab

        # ---- dock shell -----------------------------------------------
        dock = AuxiliaryWindow("Explorer", self.ui)
        dock.setObjectName("dock_explorer")
        dock.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        dock.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        dock.visibilityChanged.connect(self._explorer_on_visibility)

        # ---- tab widget -----------------------------------------------
        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.North)
        tabs.setDocumentMode(True)

        # ---- instantiate tabs (each holds its own widgets + logic) ----
        self._tab_annot  = AnnotTab(self)
        self._tab_hscope = HypnoscopeTab(self)
        self._tab_wave   = WaveformTab(self)
        self._tab_event_decomp = EventDecompTab(self)
        self._tab_two_channel = TwoChannelDynamicsTab(self)
        self._tab_plot   = PlotterTab(self)
        self._tab_gpa    = GPATab(self)
        self._tab_filter_design = FilterDesignTab(self)
        self._tab_topo   = TopoTab(self)
        self._tab_harm   = HarmonizerTab(self)

        tabs.addTab(self._tab_harm.widget(),       "Harmonizer")   # 0
        tabs.addTab(self._tab_annot.widget(),      "Annotations")  # 1
        tabs.addTab(self._tab_hscope.widget(),     "Hypnoscope")   # 2
        tabs.addTab(self._tab_wave.widget(),       "Waveforms")    # 3
        tabs.addTab(self._tab_two_channel.widget(), "Player")      # 4  (hidden)
        tabs.addTab(self._tab_plot.widget(),       "Plotter")      # 5
        tabs.addTab(self._tab_gpa.widget(),        "Assoc")        # 6
        tabs.addTab(self._tab_topo.widget(),       "Topo (Experimental)")  # 7

        tabs.setTabVisible(4, False)   # Player

        tabs.currentChanged.connect(self._explorer_tab_changed)

        dock.setWidget(tabs)

        # Make accessible from controller.ui for View-menu toggle
        self.ui.dock_explorer = dock

        self._explorer_dock = dock
        self._explorer_tabs = tabs
        self._explorer_has_positioned = False

        # Auto-refresh results/annotation lists whenever a command completes
        self.sig_results_changed.connect(self._tab_plot.refresh_tables)
        self.sig_results_changed.connect(self._tab_topo.refresh_tables)
        self.sig_results_changed.connect(self._tab_wave._refresh_ann_ch)

    # ------------------------------------------------------------------
    # Visibility / tab-switch callbacks
    # ------------------------------------------------------------------

    def _explorer_on_visibility(self, visible):
        if not visible:
            try:
                from lunapi import gpa_clear_cache
                gpa_clear_cache()
            except Exception:
                pass
            return
        dock = self._explorer_dock
        if not self._explorer_has_positioned:
            w, h = screen_clamp(*self._EXPLORER_FLOAT_SIZE)
            dock.resize(w, h)
        try:
            if not self._explorer_has_positioned:
                pg  = self.ui.frameGeometry()
                ctr = pg.center()
                rect = dock.frameGeometry()
                rect.moveCenter(ctr)
                top_left = rect.topLeft()
                if top_left.y() < pg.top():
                    top_left.setY(pg.top())
                dock.move(top_left)
        except Exception:
            pass
        self._explorer_has_positioned = True

    def _explorer_tab_changed(self, idx):
        """Refresh context-sensitive controls when switching tabs."""
        if idx != 7:
            self._tab_topo.pause_live_playback()
        if idx == 0:     # Harmonizer tab
            self._tab_harm.refresh_controls()
        elif idx == 1:   # Annotations tab: reload from Dock 4 unless cache is pinned
            self._tab_annot.refresh_controls()
        elif idx == 3:   # Waveforms tab: reload channels/annotations
            self._tab_wave.refresh_controls()
        elif idx == 4:   # Two-channel tab (hidden): reload channel list
            self._tab_two_channel.refresh_controls()
        elif idx == 5:   # Plotter tab: reload available result tables
            self._tab_plot.refresh_tables()
        elif idx == 7:   # Topo tab: sync results table list
            self._tab_topo.refresh_tables()
