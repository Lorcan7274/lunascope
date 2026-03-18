
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
from lunascope.helpers import winsorize_array

from PySide6.QtWidgets import QVBoxLayout, QMessageBox
from PySide6 import QtCore, QtWidgets, QtGui

from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import QMetaObject, Q_ARG, Qt, Slot

class SpecMixin:

    def _ensure_spectrogram_canvas(self, *_args):
        if getattr(self, "spectrogramcanvas", None) is not None:
            return self.spectrogramcanvas

        layout = self.ui.host_spectrogram.layout()
        if layout is None:
            layout = QVBoxLayout()
            self.ui.host_spectrogram.setLayout(layout)
        layout.setContentsMargins(0,0,0,0)

        from .mplcanvas import MplCanvas
        self.spectrogramcanvas = MplCanvas(self.ui.host_spectrogram)
        layout.addWidget(self.spectrogramcanvas)
        self.spectrogramcanvas.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.spectrogramcanvas.customContextMenuRequested.connect(self._spec_context_menu)
        return self.spectrogramcanvas

    # Epoch durations (seconds) offered in the combo, in display order.
    _EPOCH_STEPS = [
        (5,    "5 s"),
        (10,   "10 s"),
        (15,   "15 s"),
        (20,   "20 s"),
        (30,   "30 s"),
        (60,   "1 min"),
        (120,  "2 min"),
        (300,  "5 min"),
        (600,  "10 min"),
        (900,  "15 min"),
        (1800, "30 min"),
        (3600, "60 min"),
    ]

    def _init_spec(self):

        self.spectrogramcanvas = None
        if self.ui.host_spectrogram.layout() is None:
            self.ui.host_spectrogram.setLayout(QVBoxLayout())
        self.ui.host_spectrogram.layout().setContentsMargins(0,0,0,0)

        # populate epoch combo
        for secs, label in self._EPOCH_STEPS:
            self.ui.combo_epoch.addItem(label, secs)
        self._set_epoch_default(multiday=False)

        # wiring
        self.ui.butt_spectrogram.clicked.connect( self._calc_spectrogram )
        self.ui.butt_hjorth.clicked.connect( self._calc_hjorth )
        self.ui.combo_spectrogram.currentIndexChanged.connect( self._on_spec_channel_changed )
        self.ui.combo_epoch.currentIndexChanged.connect( self._on_spec_channel_changed )

    def _set_epoch_default(self, multiday: bool):
        target = 1800 if multiday else 30
        for i in range(self.ui.combo_epoch.count()):
            if self.ui.combo_epoch.itemData(i) == target:
                self.ui.combo_epoch.setCurrentIndex(i)
                return

    def _get_epoch_dur(self) -> int:
        v = self.ui.combo_epoch.currentData()
        return int(v) if v else 30

    def _on_spec_channel_changed(self, *_):
        """Auto-set frequency spin boxes to sensible limits for the selected channel's SR."""
        if not hasattr(self, 'p'):
            return
        ch = self.ui.combo_spectrogram.currentText()
        if not ch:
            return
        df = self.p.headers()
        if df is None:
            return
        row = df.loc[df['CH'] == ch]
        if row.empty:
            return
        sr = float(row['SR'].iloc[0])
        nyquist = sr / 2.0

        # For normal-SR channels keep standard EEG defaults (0.5 / 20 Hz).
        # Only auto-derive limits for low-SR signals (actigraphy etc.) where the
        # freq range is completely different and the defaults are meaningless.
        if sr >= 1.0:
            for spin in (self.ui.spin_lwrfrq, self.ui.spin_uprfrq):
                spin.setDecimals(2)
                spin.setMinimum(0.0)
                spin.setMaximum(nyquist)
            self.ui.spin_lwrfrq.setSingleStep(0.5)
            self.ui.spin_lwrfrq.setValue(0.5)
            self.ui.spin_uprfrq.setSingleStep(1.0)
            self.ui.spin_uprfrq.setValue(min(20.0, nyquist))
        else:
            epoch_dur = self._get_epoch_dur()
            min_f = round(1.0 / epoch_dur, 6) if epoch_dur > 0 else 0.01
            decimals = max(2, min(6, -int(round(min_f)) + 4) if min_f < 0.01 else 3)
            for spin in (self.ui.spin_lwrfrq, self.ui.spin_uprfrq):
                spin.setDecimals(decimals)
                spin.setMinimum(0.0)
                spin.setMaximum(nyquist)
            self.ui.spin_lwrfrq.setSingleStep(max(0.001, round(min_f, 6)))
            self.ui.spin_lwrfrq.setValue(min_f)
            self.ui.spin_uprfrq.setSingleStep(max(0.001, round(nyquist / 20, 6)))
            self.ui.spin_uprfrq.setValue(nyquist)


    # ------------------------------------------------------------    
    # right-click menus to save/copy images

    def _spec_context_menu(self, pos):
        self._ensure_spectrogram_canvas()
        menu = QtWidgets.QMenu(self.spectrogramcanvas)
        act_copy = menu.addAction("Copy to Clipboard")
        act_save = menu.addAction("Save Figure…")
        action = menu.exec(self.spectrogramcanvas.mapToGlobal(pos))
        if action == act_copy:
            self._spec_copy_to_clipboard()
        elif action == act_save:
            self._spec_save_figure()
            
    def _spec_copy_to_clipboard(self):
        self._ensure_spectrogram_canvas()
        buf = io.BytesIO()
        self.spectrogramcanvas.figure.savefig(buf, format="png", bbox_inches="tight")
        img = QtGui.QImage.fromData(buf.getvalue(), "PNG")
        QtWidgets.QApplication.clipboard().setImage(img)
        
    def _spec_save_figure(self):
        self._ensure_spectrogram_canvas()
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
            if getattr(self, 'multiday_mode', False):
                chs = df['CH'].tolist()
            else:
                chs = df.loc[df['SR'] >= 32, 'CH'].tolist()
        else:
            chs = [ ]
        
        self.ui.combo_spectrogram.addItems( chs )
        self._on_spec_channel_changed()
        

    # ------------------------------------------------------------
    # Caclculate a spectrogram
    
    def _calc_spectrogram(self):
        self._ensure_spectrogram_canvas()

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
        epoch_dur = self._get_epoch_dur()
        ns = float(getattr(self, "ns", 0.0))
        sr = 0.0
        _hdr = self.p.headers()
        if _hdr is not None:
            _row = _hdr.loc[_hdr['CH'] == ch]
            if not _row.empty:
                sr = float(_row['SR'].iloc[0])
        fut_spec = self._exec.submit(
            self._derive_spectrogram,
            self.p,
            ch,
            float(self.ui.spin_lwrfrq.value()),
            float(self.ui.spin_uprfrq.value()),
            float(self.ui.spin_win.value()),
            int(ns / epoch_dur) if epoch_dur > 0 else int(getattr(self, "ne", 0)),
            ns,
            epoch_dur,
            sr,
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
            QMessageBox.critical(self.ui, "Error deriving spectrogram", self._last_tb)
        finally:
            self.unlock_ui()
            self._busy = False
            self._buttons(True)
            self.sb_progress.setRange(0, 100)
            self.sb_progress.setValue(0)
            self.sb_progress.setVisible(False)
     
            
    def _derive_spectrogram(self, p, ch, minf, maxf, w, total_epochs=0, total_seconds=0.0, epoch_dur=30, sr=0.0):
        # worker thread: do not touch GUI,
        # return numpy arrays (by ref)

        # Override Welch segment params only when needed:
        #   - low SR: default 4s window would have < 16 samples (e.g. actigraphy)
        #   - short epoch: epoch shorter than 8s, window must not exceed epoch
        # Otherwise use Luna's fast default (4s window / 2s increment).
        if (sr > 0 and sr * 4.0 < 16) or epoch_dur < 8:
            seg_extra = f" segment-sec={epoch_dur} segment-inc={epoch_dur}"
        else:
            seg_extra = ""

        cmd = (
            f"EPOCH dur={epoch_dur} verbose & PSD min-sr=0 epoch-spectrum dB sig={ch}"
            f" min={minf} max={maxf}{seg_extra}"
        )
        res = p.silent_proc_lunascope(cmd)
        df = res.get('PSD: CH_E_F')
        if df is None or df.empty:
            return np.array([]), np.array([]), np.array([])
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
        z = winsorize_array(z, w)

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
        self._ensure_spectrogram_canvas()
        # we can now touch the GUI
        ch = self.ui.combo_spectrogram.currentText()
        minf = self.ui.spin_lwrfrq.value() 
        maxf = self.ui.spin_uprfrq.value()
        from .plts import plot_spec
        plot_spec( xi,yi,zi, ch, minf, maxf, ax=self.spectrogramcanvas.ax , gui = self.ui )
        if hasattr(self, "ns") and self.ns is not None and self.ns > 0:
            self.spectrogramcanvas.ax.set_xlim(0, float(self.ns))

        self.spectrogramcanvas.draw_idle()

        
        
    # ------------------------------------------------------------
    # Caclculate a Hjorth plot        

    def _calc_hjorth(self):
        self._ensure_spectrogram_canvas()
        
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
        from .plts import plot_hjorth
        plot_hjorth( ch , ax=self.spectrogramcanvas.ax , p = self.p , gui = self.ui ,
                     epoch_dur=self._get_epoch_dur() )
        if hasattr(self, "ns") and self.ns is not None and self.ns > 0:
            self.spectrogramcanvas.ax.set_xlim(0, float(self.ns))

        self.spectrogramcanvas.draw_idle()
