
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

import lunapi as lp
import io
import numpy as np

from PySide6.QtWidgets import QVBoxLayout
from PySide6 import QtCore, QtWidgets, QtGui

from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import QMetaObject, Q_ARG, Qt, Slot

from .mplcanvas import MplCanvas
from .plts import plot_hjorth, plot_spec

class SpecMixin:

    def _init_spec(self):

        self.ui.host_spectrogram.setLayout(QVBoxLayout())
        self.spectrogramcanvas = MplCanvas(self.ui.host_spectrogram)
        self.ui.host_spectrogram.layout().setContentsMargins(0,0,0,0)
        self.ui.host_spectrogram.layout().addWidget( self.spectrogramcanvas )

        # wiring
        self.ui.butt_spectrogram.clicked.connect( self._calc_spectrogram )
        self.ui.butt_hjorth.clicked.connect( self._calc_hjorth )

        # context menu
        self.spectrogramcanvas.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.spectrogramcanvas.customContextMenuRequested.connect(self._spec_context_menu)


    # ------------------------------------------------------------    
    # right-click menus to save/copy images

    def _spec_context_menu(self, pos):
        menu = QtWidgets.QMenu(self.spectrogramcanvas)
        act_copy = menu.addAction("Copy to Clipboard")
        act_save = menu.addAction("Save Figure…")
        action = menu.exec(self.spectrogramcanvas.mapToGlobal(pos))
        if action == act_copy:
            self._spec_copy_to_clipboard()
        elif action == act_save:
            self._spec_save_figure()
            
    def _spec_copy_to_clipboard(self):
        buf = io.BytesIO()
        self.spectrogramcanvas.figure.savefig(buf, format="png", bbox_inches="tight")
        img = QtGui.QImage.fromData(buf.getvalue(), "PNG")
        QtWidgets.QApplication.clipboard().setImage(img)
        
    def _spec_save_figure(self):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.spectrogramcanvas,
            "Save Figure",
            "spectrogram",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )
        if not fn:
            return
        self.spectrogramcanvas.figure.savefig(fn, bbox_inches="tight")

        
    # ------------------------------------------------------------
    # Update list of signals (req. 32 Hz or more)
        
    def _update_spectrogram_list(self):

        # clear first
        self.ui.combo_spectrogram.clear()

        df = self.p.headers()
        
        if df is not None:
            chs = df.loc[df['SR'] >= 32, 'CH'].tolist()
        else:
            chs = [ ] 
        
        self.ui.combo_spectrogram.addItems( chs )
        

    # ------------------------------------------------------------
    # Caclculate a spectrogram
    
    def _calc_spectrogram(self):

        # requires attached individal
        if not hasattr(self, "p"):
            QMessageBox.critical( self.ui , "Error", "No instance attached" )
            return

        # requires 1+ channel
        count = self.ui.combo_spectrogram.model().rowCount()
        if count == 0:
            QMessageBox.critical( self.ui , "Error", "No suitable signal for a spectrogram" )
            return

        # channel must exist in EDF (should always be the case)
        ch = self.ui.combo_spectrogram.currentText()
        if ch not in self.p.edf.channels():
            return

        # UI busy
        self._busy = True
        self._buttons(False)
        self.sb_progress.setVisible(True)
        self.sb_progress.setRange(0, 0)
        self.sb_progress.setFormat("Running…")
        self.lock_ui()

        # submit worker
        fut_spec = self._exec.submit(
            self._derive_spectrogram,
            self.p,
            ch,
            float(self.ui.spin_lwrfrq.value()),
            float(self.ui.spin_uprfrq.value()),
            float(self.ui.spin_win.value()),
            int(getattr(self, "ne", 0)),
            float(getattr(self, "ns", 0.0))
        )


        # done callback runs in worker thread -> hop to GUI
        def _done( _f = fut_spec ):
            try:
                self._last_result = _f.result()  # (xi, yi, zi)
                # enqueue a call that runs in 'self' thread
                QMetaObject.invokeMethod(self,"_spectrogram_done_ok",Qt.QueuedConnection)
            except Exception as e:
                self._last_exc = e
                self._last_tb = f"{type(e).__name__}: {e}"
                QMetaObject.invokeMethod(self, "_spectrogram_done_err", Qt.QueuedConnection)

        fut_spec.add_done_callback(_done)

    @Slot()
    def _spectrogram_done_ok(self):
        try:
            xi, yi, zi = self._last_result 
            self._complete_spectrogram(xi, yi, zi)
        finally:
            self.unlock_ui()
            self._busy = False
            self._buttons(True)
            self.sb_progress.setRange(0, 100)
            self.sb_progress.setValue(0)
            self.sb_progress.setVisible(False)

    @Slot()
    def _spectrogram_done_err(self):
        try:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error deriving spectrogram", self._last_tb)
        finally:
            self.unlock_ui()
            self._busy = False
            self._buttons(True)
            self.sb_progress.setRange(0, 100)
            self.sb_progress.setValue(0)
            self.sb_progress.setVisible(False)
     
            
    def _derive_spectrogram(self, p, ch, minf, maxf, w, total_epochs=0, total_seconds=0.0):
        # worker thread: do not touch GUI,
        # return numpy arrays (by ref)

        res = p.silent_proc_lunascope(
            "EPOCH dur=30 verbose & PSD min-sr=32 epoch-spectrum dB sig="
            + ch
            + " min="
            + str(minf)
            + " max="
            + str(maxf)
        )
        df = res['PSD: CH_E_F']
        dt = res.get('EPOCH: E')

        # Use Luna's epoch mapping directly (E -> START), without constructing
        # an alternate epoch indexing scheme in the UI layer.
        x = None
        if dt is not None and 'START' in dt.columns and 'E' in dt.columns and 'E' in df.columns:
            dx = df[['E']].merge(dt[['E', 'START']], on='E', how='left')
            if dx['START'].notna().any():
                x = dx['START'].to_numpy(dtype=float)
        if x is None:
            x = df['E'].to_numpy(dtype=float)

        y = df['F'].to_numpy(dtype=float)
        z = df[ 'PSD' ].to_numpy(dtype=float)

        incl = np.zeros(len(df), dtype=bool)
        incl[ (y >= minf) & (y <= maxf) ] = True
        x = x[ incl ]
        y = y[ incl ]
        z = z[ incl ]
        z = lp.winsorize( z , limits=[w, w] )

        if x.size == 0 or y.size == 0:
            return np.array([]), np.array([]), np.array([])

        # Use full epoch timeline bounds; prefer known full-record bounds
        # so masked runs keep the same temporal resolution.
        x0 = float(np.min(x))
        x1 = float(np.max(x))
        xn = int(np.unique(x).size)
        if total_epochs is not None and int(total_epochs) > 0 and total_seconds is not None and float(total_seconds) > 0:
            x0 = 0.0
            x1 = float(total_seconds)
            xn = int(total_epochs)
        elif dt is not None and 'START' in dt.columns and len(dt) > 0:
            xt = np.sort(np.unique(dt['START'].to_numpy(dtype=float)))
            if xt.size > 0:
                step = 1.0
                if xt.size > 1:
                    d = np.diff(xt)
                    d = d[d > 0]
                    if d.size > 0:
                        step = float(np.median(d))
                x0 = float(xt[0])
                x1 = float(xt[-1] + step)
                xn = int(xt.size)

        yn = np.unique(y).size
        if xn < 1 or yn < 1:
            return np.array([]), np.array([]), np.array([])
        if not np.isfinite(x0) or not np.isfinite(x1) or x1 <= x0:
            return np.array([]), np.array([]), np.array([])
        zi, yi, xi = np.histogram2d(
            y, x, bins=(yn, xn), range=((minf, maxf), (x0, x1)), weights=z, density=False
        )
        counts, _, _ = np.histogram2d(
            y, x, bins=(yn, xn), range=((minf, maxf), (x0, x1))
        )
        with np.errstate(divide='ignore', invalid='ignore'):
            zi = zi / counts
            zi = np.ma.masked_invalid(zi)

        return xi, yi, zi


    def _complete_spectrogram(self,xi,yi,zi):
        # we can now touch the GUI
        ch = self.ui.combo_spectrogram.currentText()
        minf = self.ui.spin_lwrfrq.value() 
        maxf = self.ui.spin_uprfrq.value()
                
        plot_spec( xi,yi,zi, ch, minf, maxf, ax=self.spectrogramcanvas.ax , gui = self.ui )
        if hasattr(self, "ns") and self.ns is not None and self.ns > 0:
            self.spectrogramcanvas.ax.set_xlim(0, float(self.ns))

        self.spectrogramcanvas.draw_idle()

        
        
    # ------------------------------------------------------------
    # Caclculate a Hjorth plot        

    def _calc_hjorth(self):
        
        # requires attached individal
        if not hasattr(self, "p"):
            QMessageBox.critical( self.ui , "Error", "No instance attached" )
            return

        # requires 1+ channel
        count = self.ui.combo_spectrogram.model().rowCount()
        if count == 0:
            QMessageBox.critical( self.ui , "Error", "No suitable signal for a Hjorth-plot" )
            return

        # get channel
        ch = self.ui.combo_spectrogram.currentText()

        # check it still exists in the in-memory EDF                                          
        if ch not in self.p.edf.channels():
            return

        # do plot
        plot_hjorth( ch , ax=self.spectrogramcanvas.ax , p = self.p , gui = self.ui )
        if hasattr(self, "ns") and self.ns is not None and self.ns > 0:
            self.spectrogramcanvas.ax.set_xlim(0, float(self.ns))

        self.spectrogramcanvas.draw_idle()
