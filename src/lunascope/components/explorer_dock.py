
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
Ctrl+E  →  floating "Explorer" dock with four tabbed panels:

    1  Annotations  – cohort-level annotation explorer (PETH, overlap, …)
    2  Hypnoscope   – staging grid across all subjects aligned by time
    3  Waveforms    – peri-event signal traces for the current record
    4  Plotter      – generic scatter / line / bar / histogram for output tables
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QTabWidget

from ..helpers import screen_clamp


class ExplorerMixin:
    """Mixin that creates and owns the tabbed Explorer dock."""

    _EXPLORER_FLOAT_SIZE = (1320, 840)

    # ------------------------------------------------------------------
    # Initialisation (called from Controller.__init__)
    # ------------------------------------------------------------------

    def _init_explorer(self):
        # Late imports avoid circular dependencies at module load
        from .explorer_annot      import AnnotTab
        from .explorer_hypnoscope import HypnoscopeTab
        from .explorer_waveform   import WaveformTab
        from .explorer_event_decomp import EventDecompTab
        from .explorer_plotter    import PlotterTab

        # ---- dock shell -----------------------------------------------
        dock = QDockWidget("Explorer", self.ui)
        dock.setObjectName("dock_explorer")
        dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea
        )
        from PySide6.QtWidgets import QDockWidget as _DW
        dock.setFeatures(
            _DW.DockWidgetMovable | _DW.DockWidgetFloatable | _DW.DockWidgetClosable
        )
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
        self._tab_plot   = PlotterTab(self)

        tabs.addTab(self._tab_annot.widget(),  "Annotations")
        tabs.addTab(self._tab_hscope.widget(), "Hypnoscope")
        tabs.addTab(self._tab_wave.widget(),   "Waveforms")
        tabs.addTab(self._tab_plot.widget(),   "Plotter")

        tabs.currentChanged.connect(self._explorer_tab_changed)

        dock.setWidget(tabs)
        self.ui.addDockWidget(Qt.RightDockWidgetArea, dock)

        # Make accessible from controller.ui for View-menu toggle
        self.ui.dock_explorer = dock

        self._explorer_dock = dock
        self._explorer_tabs = tabs

        # Auto-refresh plotter whenever results are repopulated
        self.sig_results_changed.connect(self._tab_plot.refresh_tables)

    # ------------------------------------------------------------------
    # Visibility / tab-switch callbacks
    # ------------------------------------------------------------------

    def _explorer_on_visibility(self, visible):
        if not visible:
            return
        dock = self._explorer_dock
        if not dock.isFloating():
            dock.setFloating(True)
        w, h = screen_clamp(*self._EXPLORER_FLOAT_SIZE)
        if dock.width() < w or dock.height() < h:
            dock.resize(w, h)
        try:
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

    def _explorer_tab_changed(self, idx):
        """Refresh context-sensitive controls when switching tabs."""
        if idx == 2:   # Waveforms tab: reload channels/annotations
            self._tab_wave.refresh_controls()
        elif idx == 3: # Plotter tab: reload available result tables
            self._tab_plot.refresh_tables()
