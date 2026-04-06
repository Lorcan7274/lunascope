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

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, dpi=100):
        fig = Figure(dpi=dpi, facecolor="black")         # figure background
        super().__init__(fig)
        if parent is not None:
            self.setParent(parent)
        self.ax = fig.add_axes([0,0,1,1])                # full-bleed axes
        self.ax.set_facecolor("black")                   # axes background
        self.ax.set_axis_off()                           # no ticks, spines, grid
        self.ax.grid(False)
        fig.patch.set_facecolor("black")                 # extra safety
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumSize(320, 240)
        self.draw()

    def sizeHint(self):
        w, h = self.get_width_height()
        return QSize(max(320, w), max(240, h))
