
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

import io
import numpy as np
from lunascope.helpers import winsorize_array

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QCheckBox,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
)
from PySide6 import QtCore, QtWidgets, QtGui
from ..file_dialogs import save_file_name
from .soappops import MultiSelectComboBox

from PySide6.QtCore import QMetaObject, Qt, Slot

class SpecMixin:
    _SPEC_EPOCH_DUR = 30
    _MTM_ZOOM_MAX_SPAN = 30.0
    _MTM_ZOOM_CELL_CAP = 500_000

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
        self.spectrogramcanvas.setMinimumSize(0, 0)
        self.spectrogramcanvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        layout.addWidget(self.spectrogramcanvas)
        self.spectrogramcanvas.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.spectrogramcanvas.customContextMenuRequested.connect(self._spec_context_menu)
        return self.spectrogramcanvas

    def _init_spec(self):

        self.spectrogramcanvas = None
        self._spec_plot_kind = None
        self._spec_plot_cache = None
        self._spec_cache = {}
        self._spec_data_version = 0
        self._spec_job_token = 0
        self._spec_nonmeans_winsor = None
        self._spec_zoom_timer = QtCore.QTimer(self.ui)
        self._spec_zoom_timer.setSingleShot(True)
        self._spec_zoom_timer.setInterval(400)
        self._spec_zoom_timer.timeout.connect(self._spec_on_zoom_timer)
        if self.ui.host_spectrogram.layout() is None:
            self.ui.host_spectrogram.setLayout(QVBoxLayout())
        self.ui.host_spectrogram.layout().setContentsMargins(0,0,0,0)

        self._spec_rebuild_controls()
        self.ui.butt_spectrogram.clicked.connect(self._run_timefreq_active)
        self.ui.combo_spectrogram.currentIndexChanged.connect(self._on_spec_channel_changed)
        self.ui.combo_mean_channels.selectionChanged.connect(self._on_mean_channels_changed)
        self.ui.combo_mean_band.currentIndexChanged.connect(self._invalidate_spec_cache)
        self.ui.check_mean_relative.stateChanged.connect(self._on_means_display_changed)
        self.ui.check_mean_log.stateChanged.connect(self._on_means_display_changed)
        if hasattr(self.ui, "check_spec_legend"):
            self.ui.check_spec_legend.stateChanged.connect(self._on_spec_legend_changed)
        if hasattr(self, "sig_window_range_changed"):
            self.sig_window_range_changed.connect(self._on_spec_window_range_changed)

    def _set_epoch_default(self, multiday: bool):
        return

    def _get_epoch_dur(self) -> int:
        return self._SPEC_EPOCH_DUR

    def _spec_rebuild_controls(self):
        layout = self.ui.frame_3.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().setParent(None)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(4)
        for col in range(12):
            layout.setColumnStretch(col, 0)
            layout.setColumnMinimumWidth(col, 0)
        layout.setColumnStretch(7, 1)

        if hasattr(self.ui, "frame_epoch"):
            self.ui.frame_epoch.hide()
        if hasattr(self.ui, "butt_hjorth"):
            self.ui.butt_hjorth.hide()

        tabs = QTabWidget(self.ui.frame_3)
        tabs.setObjectName("tab_timefreq")
        tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        tabs.addTab(QtWidgets.QWidget(), "Welch")
        tabs.addTab(QtWidgets.QWidget(), "Hjorth")
        tabs.addTab(QtWidgets.QWidget(), "Multitaper")
        tabs.addTab(QtWidgets.QWidget(), "IRASA")
        tabs.addTab(QtWidgets.QWidget(), "Means")
        tabs.currentChanged.connect(self._on_timefreq_tab_changed)
        self.ui.tab_timefreq = tabs

        self.ui.butt_spectrogram.setText("Run")
        self.ui.butt_spectrogram.setFixedWidth(110)

        self.ui.combo_mtm_mode = QComboBox(self.ui.frame_3)
        self.ui.combo_mtm_mode.setObjectName("combo_mtm_mode")
        self.ui.combo_mtm_mode.addItem("Whole Night", "whole")
        self.ui.combo_mtm_mode.addItem("Zoom <=30s", "zoom")
        self.ui.combo_mtm_mode.setFixedWidth(120)

        self.ui.spin_mtm_nw = QDoubleSpinBox(self.ui.frame_3)
        self.ui.spin_mtm_nw.setObjectName("spin_mtm_nw")
        self.ui.spin_mtm_nw.setRange(1.0, 20.0)
        self.ui.spin_mtm_nw.setSingleStep(0.5)
        self.ui.spin_mtm_nw.setFixedWidth(80)

        self.ui.spin_mtm_t = QSpinBox(self.ui.frame_3)
        self.ui.spin_mtm_t.setObjectName("spin_mtm_t")
        self.ui.spin_mtm_t.setRange(0, 99)
        self.ui.spin_mtm_t.setValue(0)
        self.ui.spin_mtm_t.setSpecialValueText("auto")
        self.ui.spin_mtm_t.setFixedWidth(80)

        self.ui.spin_mtm_segment = QDoubleSpinBox(self.ui.frame_3)
        self.ui.spin_mtm_segment.setObjectName("spin_mtm_segment")
        self.ui.spin_mtm_segment.setRange(0.25, 30.0)
        self.ui.spin_mtm_segment.setSingleStep(0.25)
        self.ui.spin_mtm_segment.setFixedWidth(80)

        self.ui.spin_mtm_inc = QDoubleSpinBox(self.ui.frame_3)
        self.ui.spin_mtm_inc.setObjectName("spin_mtm_inc")
        self.ui.spin_mtm_inc.setRange(0.02, 30.0)
        self.ui.spin_mtm_inc.setSingleStep(0.02)
        self.ui.spin_mtm_inc.setFixedWidth(80)

        self.ui.combo_irasa_component = QComboBox(self.ui.frame_3)
        self.ui.combo_irasa_component.setObjectName("combo_irasa_component")
        self.ui.combo_irasa_component.addItem("Aperiodic", "APER")
        self.ui.combo_irasa_component.addItem("Periodic", "PER")
        self.ui.combo_irasa_component.setFixedWidth(110)

        self.ui.combo_mean_channels = MultiSelectComboBox(self.ui.frame_3)
        self.ui.combo_mean_channels.setObjectName("combo_mean_channels")
        self.ui.combo_mean_channels.setFixedWidth(210)
        self.ui.combo_mean_channels._popup_max_rows = 18
        self.ui.combo_mean_channels.setToolTip(
            "Select one or more channels for epoch-wise mean traces. "
            "All EDF channels are available here, including low-rate signals."
        )
        self.ui.combo_mean_band = QComboBox(self.ui.frame_3)
        self.ui.combo_mean_band.setObjectName("combo_mean_band")
        self.ui.combo_mean_band.addItem("Mean only", "mean")
        self.ui.combo_mean_band.addItem("Mean ± SE", "se")
        self.ui.combo_mean_band.addItem("Mean ± 95% CI", "ci")
        self.ui.combo_mean_band.addItem("Mean + IQR", "iqr")
        self.ui.combo_mean_band.addItem("Mean + min/max", "minmax")
        self.ui.combo_mean_band.setFixedWidth(125)
        self.ui.check_mean_relative = QCheckBox("Relative Y", self.ui.frame_3)
        self.ui.check_mean_relative.setObjectName("check_mean_relative")
        self.ui.check_mean_relative.setToolTip(
            "Use a separate 0–1 relative amplitude scale for each channel; "
            "otherwise all channels share their original Y-axis units."
        )
        self.ui.check_mean_log = QCheckBox("Log Y", self.ui.frame_3)
        self.ui.check_mean_log.setObjectName("check_mean_log")
        self.ui.check_mean_log.setToolTip("Use a logarithmic Y-axis (requires positive values).")

        self.ui.label_timefreq_status = QLabel("", self.ui.frame_3)
        self.ui.label_timefreq_status.setObjectName("label_timefreq_status")
        self.ui.label_timefreq_status.setMinimumWidth(0)

        self.ui.combo_spectrogram.setFixedWidth(130)
        self.ui.spin_lwrfrq.setFixedWidth(80)
        self.ui.spin_uprfrq.setFixedWidth(80)
        self.ui.spin_win.setFixedWidth(80)
        self.ui.check_spec_legend.setFixedWidth(90)

        def label(text):
            lab = QLabel(text, self.ui.frame_3)
            lab.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            lab.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            return lab

        layout.addWidget(tabs, 0, 0, 1, 8)
        self.ui.label_spec_channel = label("Channel")
        layout.addWidget(self.ui.label_spec_channel, 0, 8)
        layout.addWidget(self.ui.combo_spectrogram, 0, 9)
        layout.addWidget(self.ui.butt_spectrogram, 0, 10)
        layout.addWidget(self.ui.check_spec_legend, 0, 11)

        self.ui.label_spec_lower = label("Lower Hz")
        self.ui.label_spec_upper = label("Upper Hz")
        self.ui.label_spec_winsor = label("Winsor")
        layout.addWidget(self.ui.label_spec_lower, 1, 0)
        layout.addWidget(self.ui.spin_lwrfrq, 1, 1)
        layout.addWidget(self.ui.label_spec_upper, 1, 2)
        layout.addWidget(self.ui.spin_uprfrq, 1, 3)
        layout.addWidget(self.ui.label_spec_winsor, 1, 4)
        layout.addWidget(self.ui.spin_win, 1, 5)
        layout.addWidget(self.ui.label_timefreq_status, 1, 6, 1, 6)

        self.ui.label_mean_channels = label("Channels")
        self.ui.label_mean_band = label("Band")
        self.ui.panel_mean_controls = QtWidgets.QWidget(self.ui.frame_3)
        mean_layout = QtWidgets.QHBoxLayout(self.ui.panel_mean_controls)
        mean_layout.setContentsMargins(0, 0, 0, 0)
        mean_layout.setSpacing(4)
        mean_layout.addWidget(self.ui.label_mean_channels)
        mean_layout.addWidget(self.ui.combo_mean_channels)
        mean_layout.addWidget(self.ui.label_mean_band)
        mean_layout.addWidget(self.ui.combo_mean_band)
        mean_layout.addWidget(self.ui.check_mean_relative)
        mean_layout.addWidget(self.ui.check_mean_log)
        mean_layout.addStretch(1)
        layout.addWidget(self.ui.panel_mean_controls, 2, 0, 1, 12)

        self.ui.label_mtm_mode = label("MTM Mode")
        self.ui.label_mtm_nw = label("NW")
        self.ui.label_mtm_t = label("Tapers")
        self.ui.label_mtm_segment = label("Segment")
        self.ui.label_mtm_inc = label("Inc")
        self.ui.label_irasa_component = label("IRASA")

        self.ui.panel_mtm_controls = QtWidgets.QWidget(self.ui.frame_3)
        mtm_layout = QtWidgets.QHBoxLayout(self.ui.panel_mtm_controls)
        mtm_layout.setContentsMargins(0, 0, 0, 0)
        mtm_layout.setSpacing(4)
        for w in (
            self.ui.label_mtm_mode, self.ui.combo_mtm_mode,
            self.ui.label_mtm_nw, self.ui.spin_mtm_nw,
            self.ui.label_mtm_t, self.ui.spin_mtm_t,
            self.ui.label_mtm_segment, self.ui.spin_mtm_segment,
            self.ui.label_mtm_inc, self.ui.spin_mtm_inc,
        ):
            mtm_layout.addWidget(w)
        mtm_layout.addStretch(1)
        layout.addWidget(self.ui.panel_mtm_controls, 2, 0, 1, 12)

        self.ui.panel_irasa_controls = QtWidgets.QWidget(self.ui.frame_3)
        irasa_layout = QtWidgets.QHBoxLayout(self.ui.panel_irasa_controls)
        irasa_layout.setContentsMargins(0, 0, 0, 0)
        irasa_layout.setSpacing(4)
        irasa_layout.addWidget(self.ui.label_irasa_component)
        irasa_layout.addWidget(self.ui.combo_irasa_component)
        irasa_layout.addStretch(1)
        layout.addWidget(self.ui.panel_irasa_controls, 2, 0, 1, 12)

        for w in (
            self.ui.combo_mtm_mode, self.ui.spin_mtm_nw, self.ui.spin_mtm_t,
            self.ui.spin_mtm_segment, self.ui.spin_mtm_inc,
        ):
            try:
                w.currentIndexChanged.connect(self._invalidate_spec_cache)
            except AttributeError:
                w.valueChanged.connect(self._invalidate_spec_cache)
        for w in (self.ui.spin_lwrfrq, self.ui.spin_uprfrq, self.ui.spin_win):
            if w is self.ui.spin_win:
                w.valueChanged.connect(self._on_spec_winsor_changed)
            else:
                w.valueChanged.connect(self._invalidate_spec_cache)
        self.ui.combo_irasa_component.currentIndexChanged.connect(self._on_irasa_component_changed)
        self.ui.combo_mtm_mode.currentIndexChanged.connect(self._on_mtm_mode_changed)
        self._last_mtm_mode = None
        self._apply_mtm_mode_defaults(force=True)
        self._on_timefreq_tab_changed()


    def _timefreq_mode(self):
        idx = self.ui.tab_timefreq.currentIndex()
        return ["welch", "hjorth", "mtm", "irasa", "means"][max(0, min(idx, 4))]

    def _spec_record_ready(self):
        return (
            hasattr(self, "p")
            and float(getattr(self, "ns", 0.0) or 0.0) > 0
            and getattr(self, "last_x1", None) is not None
            and getattr(self, "last_x2", None) is not None
        )

    def _spec_channel_state_ready(self):
        return hasattr(self, "p") and float(getattr(self, "ns", 0.0) or 0.0) > 0

    def _on_mtm_mode_changed(self, *_):
        self._apply_mtm_mode_defaults()
        self._on_timefreq_tab_changed()

    def _apply_mtm_mode_defaults(self, force=False):
        mtm_mode = self.ui.combo_mtm_mode.currentData()
        if not force and mtm_mode == getattr(self, "_last_mtm_mode", None):
            return
        self._last_mtm_mode = mtm_mode
        if mtm_mode == "whole":
            values = (
                (self.ui.spin_mtm_segment, 30.0),
                (self.ui.spin_mtm_inc, 30.0),
                (self.ui.spin_mtm_nw, 15.0),
            )
        else:
            values = (
                (self.ui.spin_mtm_segment, 2.0),
                (self.ui.spin_mtm_inc, 0.1),
                (self.ui.spin_mtm_nw, 3.0),
            )
        for spin, value in values:
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self._invalidate_spec_cache()

    def _on_timefreq_tab_changed(self, *_):
        mode = self._timefreq_mode()
        is_mtm = mode == "mtm"
        is_irasa = mode == "irasa"
        is_means = mode == "means"
        current_kind = getattr(self, "_spec_plot_kind", None)
        if is_means and self._spec_nonmeans_winsor is None:
            self._spec_nonmeans_winsor = float(self.ui.spin_win.value())
            self.ui.spin_win.blockSignals(True)
            self.ui.spin_win.setValue(0.0)
            self.ui.spin_win.blockSignals(False)
        elif not is_means and self._spec_nonmeans_winsor is not None:
            self.ui.spin_win.blockSignals(True)
            self.ui.spin_win.setValue(self._spec_nonmeans_winsor)
            self.ui.spin_win.blockSignals(False)
            self._spec_nonmeans_winsor = None
        self.ui.panel_mtm_controls.setVisible(is_mtm)
        self.ui.panel_irasa_controls.setVisible(is_irasa)
        for widget in (
            self.ui.label_spec_channel, self.ui.combo_spectrogram,
        ):
            widget.setVisible(not is_means)
        for widget in (
            self.ui.label_spec_lower, self.ui.spin_lwrfrq,
            self.ui.label_spec_upper, self.ui.spin_uprfrq,
        ):
            widget.setVisible(not is_means)
        self.ui.label_spec_winsor.setVisible(True)
        self.ui.spin_win.setVisible(True)
        self.ui.panel_mean_controls.setVisible(is_means)
        self._set_timefreq_status("")
        self.ui.butt_spectrogram.setText("Run Hjorth" if mode == "hjorth" else "Run")

        if current_kind != mode:
            if self._restore_cached_timefreq_plot(mode):
                return
            self._clear_spectrogram_plot()

        if is_mtm and self.ui.combo_mtm_mode.currentData() == "zoom" and self._spec_record_ready():
            self._spec_zoom_timer.start()

    def _restore_cached_timefreq_plot(self, mode):
        if not hasattr(self, "p"):
            return False
        if mode == "means":
            channels = self._mean_channels()
            if not channels:
                return False
            params = self._means_params(channels)
            key = self._spec_cache_key(mode, params)
            result = self._spec_cache.get(key)
            if result is None:
                return False
            self._complete_timefreq(mode, result, cache_key=key)
            return True
        if self.ui.combo_spectrogram.model().rowCount() == 0:
            return False
        ch = self._current_channel()
        if not ch or ch not in self.p.edf.channels():
            return False
        if mode == "hjorth":
            params = self._hjorth_params(ch)
            key = self._spec_cache_key(mode, params)
            result = self._spec_cache.get(key)
            if result is None:
                return False
            self._complete_hjorth(result, cache_key=key)
            return True
        params = self._spec_params(mode, ch)
        if params is None:
            return False
        if not self._should_cache_timefreq(mode, params):
            return False
        key = self._spec_cache_key(mode, params)
        result = self._spec_cache.get(key)
        if result is None:
            return False
        self._complete_timefreq(mode, result, cache_key=key)
        return True

    def _clear_spectrogram_plot(self):
        self._spec_plot_kind = None
        self._spec_plot_cache = None
        if getattr(self, "spectrogramcanvas", None) is None:
            return
        ax = self.spectrogramcanvas.ax
        fig = ax.figure
        for extra_ax in list(fig.axes):
            if extra_ax is not ax:
                extra_ax.remove()
        ax.clear()
        ax.set_facecolor("black")
        ax.set_axis_off()
        fig.patch.set_facecolor("black")
        self.spectrogramcanvas.draw_idle()

    def _on_spec_channel_changed(self, *_):
        """Auto-set frequency spin boxes to sensible limits for the selected channel's SR."""
        if not self._spec_channel_state_ready():
            return
        ch = self._current_channel()
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
        self._invalidate_spec_cache()

    def _invalidate_spec_cache(self, *_):
        self._spec_cache = {}

    def _invalidate_spec_data_cache(self, *_):
        self._spec_data_version = int(getattr(self, "_spec_data_version", 0) or 0) + 1
        self._invalidate_spec_cache()

    # ------------------------------------------------------------    
    # right-click menus to save/copy images

    def _spec_context_menu(self, pos):
        self._ensure_spectrogram_canvas()
        menu = QtWidgets.QMenu(self.spectrogramcanvas)
        act_copy = menu.addAction("Copy to Clipboard")
        act_save = menu.addAction("Save As...")
        action = menu.exec(self.spectrogramcanvas.mapToGlobal(pos))
        if action == act_copy:
            self._spec_copy_to_clipboard()
        elif action == act_save:
            self._spec_save_figure()

    def _show_spec_legend(self):
        return bool(
            hasattr(self.ui, "check_spec_legend")
            and self.ui.check_spec_legend.isChecked()
        )

    def _on_spec_legend_changed(self, *_):
        if getattr(self, "spectrogramcanvas", None) is None:
            return
        kind = getattr(self, "_spec_plot_kind", None)
        if kind == "means" and getattr(self, "_spec_plot_cache", None):
            self._draw_means_plot(self._spec_plot_cache)
        elif kind in ("spectrogram", "welch", "mtm", "irasa") and getattr(self, "_spec_plot_cache", None):
            cache = self._spec_plot_cache
            self._draw_heatmap_cache(self._irasa_component_result(cache) if kind == "irasa" else cache)
        elif kind == "hjorth":
            self._draw_hjorth_plot()
            
    def _spec_copy_to_clipboard(self):
        self._ensure_spectrogram_canvas()
        buf = io.BytesIO()
        self.spectrogramcanvas.figure.savefig(buf, format="png", bbox_inches="tight")
        img = QtGui.QImage.fromData(buf.getvalue(), "PNG")
        QtWidgets.QApplication.clipboard().setImage(img)
        
    def _spec_save_figure(self):
        self._ensure_spectrogram_canvas()
        fn, _ = save_file_name(
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
        self.ui.combo_spectrogram.blockSignals(True)
        self.ui.combo_spectrogram.clear()

        try:
            df = self.p.headers()

            if df is not None:
                all_chs = df['CH'].tolist()
                labels = [self._format_channel_label(ch) for ch in all_chs]
                self.ui.combo_mean_channels.set_items(labels, labels[:1])
                if getattr(self, 'multiday_mode', False):
                    chs = all_chs
                else:
                    chs = df.loc[df['SR'] >= 32, 'CH'].tolist()
            else:
                chs = [ ]

            for ch in chs:
                label = self._format_channel_label(ch)
                self.ui.combo_spectrogram.addItem(label, ch)
        finally:
            self.ui.combo_spectrogram.blockSignals(False)
        self._on_spec_channel_changed()

    def _mean_channels(self):
        return [str(label).split(" (", 1)[0] for label in self.ui.combo_mean_channels.checked_items()]

    def _on_mean_channels_changed(self):
        self._invalidate_spec_cache()

    def _on_means_display_changed(self, *_):
        if getattr(self, "_spec_plot_kind", None) == "means" and getattr(self, "_spec_plot_cache", None):
            self._draw_means_plot(self._spec_plot_cache)

    def _on_spec_winsor_changed(self, *_):
        if getattr(self, "_spec_plot_kind", None) == "means":
            self._on_means_display_changed()
        else:
            self._invalidate_spec_cache()
        
    def _format_channel_label(self, ch):
        sr = self._channel_sr(ch)
        if sr is None:
            return str(ch)
        return f"{ch} ({sr:g} Hz)"

    def _current_channel(self, combo=None):
        combo = combo or self.ui.combo_spectrogram
        data = combo.currentData()
        return str(data) if data else combo.currentText().split(" (", 1)[0]

    def _channel_sr(self, ch):
        if not hasattr(self, "p") or not ch:
            return None
        try:
            df = self.p.headers()
        except Exception:
            return None
        if df is None or df.empty or "CH" not in df.columns or "SR" not in df.columns:
            return None
        row = df.loc[df["CH"] == ch]
        if row.empty:
            return None
        try:
            return float(row["SR"].iloc[0])
        except Exception:
            return None

    def _run_timefreq_active(self):
        mode = self._timefreq_mode()
        if mode == "hjorth":
            self._calc_hjorth()
            return
        if mode == "means":
            self._calc_means()
            return
        self._calc_timefreq(mode)

    def _begin_spec_job(self):
        self._busy = True
        self._buttons(False)
        self.sb_progress.setVisible(True)
        self.sb_progress.setRange(0, 0)
        self.sb_progress.setFormat("Running...")
        self.lock_ui()

    def _end_spec_job(self):
        self.unlock_ui()
        self._busy = False
        self._buttons(True)
        self.sb_progress.setRange(0, 100)
        self.sb_progress.setValue(0)
        self.sb_progress.setVisible(False)

    def _calc_timefreq(self, mode):
        self._ensure_spectrogram_canvas()
        if not hasattr(self, "p"):
            QMessageBox.critical(self.ui, "Error", "No instance attached")
            return
        if self.ui.combo_spectrogram.model().rowCount() == 0:
            QMessageBox.critical(self.ui, "Error", "No suitable signal")
            return
        ch = self._current_channel()
        if ch not in self.p.edf.channels():
            return
        params = self._spec_params(mode, ch)
        if params is None:
            return
        use_cache = self._should_cache_timefreq(mode, params)
        key = self._spec_cache_key(mode, params) if use_cache else None
        if use_cache and key in self._spec_cache:
            self._complete_timefreq(mode, self._spec_cache[key], cache_key=key)
            return

        self._begin_spec_job()
        self._spec_job_token += 1
        token = self._spec_job_token
        fut = self._exec.submit(self._derive_timefreq, self.p, mode, params)

        def _done(_f=fut):
            try:
                self._last_result = (token, mode, key, _f.result())
                QMetaObject.invokeMethod(self, "_timefreq_done_ok", Qt.QueuedConnection)
            except Exception as e:
                self._last_exc = e
                self._last_tb = f"{type(e).__name__}: {e}"
                QMetaObject.invokeMethod(self, "_spectrogram_done_err", Qt.QueuedConnection)

        fut.add_done_callback(_done)

    def _calc_means(self):
        self._ensure_spectrogram_canvas()
        if not hasattr(self, "p"):
            QMessageBox.critical(self.ui, "Error", "No instance attached")
            return
        channels = self._mean_channels()
        if not channels:
            QMessageBox.critical(self.ui, "Error", "Select at least one channel")
            return
        channels = [ch for ch in channels if ch in self.p.edf.channels()]
        if not channels:
            QMessageBox.critical(self.ui, "Error", "No selected channels are available")
            return
        params = self._means_params(channels)
        key = self._spec_cache_key("means", params)
        if key in self._spec_cache:
            self._complete_timefreq("means", self._spec_cache[key], cache_key=key)
            return

        self._begin_spec_job()
        self._spec_job_token += 1
        token = self._spec_job_token
        fut = self._exec.submit(self._derive_timefreq, self.p, "means", params)

        def _done(_f=fut):
            try:
                self._last_result = (token, "means", key, _f.result())
                QMetaObject.invokeMethod(self, "_timefreq_done_ok", Qt.QueuedConnection)
            except Exception as e:
                self._last_exc = e
                self._last_tb = f"{type(e).__name__}: {e}"
                QMetaObject.invokeMethod(self, "_spectrogram_done_err", Qt.QueuedConnection)

        fut.add_done_callback(_done)

    def _spec_params(self, mode, ch):
        minf = float(self.ui.spin_lwrfrq.value())
        maxf = float(self.ui.spin_uprfrq.value())
        if maxf <= minf:
            QMessageBox.critical(self.ui, "Error", "Upper frequency must exceed lower frequency")
            return None
        params = {
            "ch": ch,
            "minf": minf,
            "maxf": maxf,
            "winsor": float(self.ui.spin_win.value()),
            "ns": float(getattr(self, "ns", 0.0) or 0.0),
            "ne": int(getattr(self, "ne", 0) or 0),
            "sr": float(self._channel_sr(ch) or 0.0),
        }
        if mode == "mtm":
            seg = float(self.ui.spin_mtm_segment.value())
            inc = float(self.ui.spin_mtm_inc.value())
            zoom = self.ui.combo_mtm_mode.currentData() == "zoom"
            lo = float(getattr(self, "last_x1", 0.0) or 0.0)
            hi = float(getattr(self, "last_x2", lo) or lo)
            if zoom and hi - lo > self._MTM_ZOOM_MAX_SPAN + 1e-6:
                self._set_timefreq_status("Zoom MTM requires <=30s window")
                return None
            params.update({
                "mtm_mode": "zoom" if zoom else "whole",
                "nw": float(self.ui.spin_mtm_nw.value()),
                "t": int(self.ui.spin_mtm_t.value()),
                "segment": seg,
                "inc": inc,
                "lo": lo,
                "hi": hi,
            })
            if zoom:
                cells = self._estimate_mtm_cells(max(0.0, hi - lo), minf, maxf, seg, inc)
                if cells > self._MTM_ZOOM_CELL_CAP:
                    self._set_timefreq_status(
                        f"Zoom MTM estimate {cells:,} cells; increase segment/inc or narrow Hz"
                    )
                    return None
        return params

    def _means_params(self, channels):
        return {
            "channels": tuple(channels),
            "epoch_dur": float(self._get_epoch_dur()),
            "band": str(self.ui.combo_mean_band.currentData() or "mean"),
            "relative": bool(self.ui.check_mean_relative.isChecked()),
            "winsor": float(self.ui.spin_win.value()),
            "ns": float(getattr(self, "ns", 0.0) or 0.0),
        }

    def _estimate_mtm_cells(self, span, minf, maxf, segment, inc):
        if span <= 0 or segment <= 0 or inc <= 0 or maxf <= minf:
            return 0
        segs = max(1, int(np.floor(max(0.0, span - segment) / inc) + 1))
        bins = max(1, int(np.ceil((maxf - minf) * segment)) + 1)
        return int(segs * bins)

    def _spec_cache_key(self, mode, params):
        pieces = [
            int(getattr(self, "_spec_data_version", 0) or 0),
            mode,
            params.get("ch"),
            params.get("minf"),
            params.get("maxf"),
            params.get("winsor"),
        ]
        if mode == "means":
            pieces.extend([
                params.get("channels"), params.get("band"), params.get("epoch_dur"),
                params.get("relative"),
            ])
            return tuple(pieces)
        if mode == "mtm":
            pieces.extend([
                params.get("mtm_mode"), params.get("nw"), params.get("t"),
                params.get("segment"), params.get("inc"),
            ])
            if params.get("mtm_mode") == "zoom":
                pieces.extend([round(float(params.get("lo", 0.0)), 3), round(float(params.get("hi", 0.0)), 3)])
        elif mode == "hjorth":
            pieces.append(params.get("epoch_dur"))
        return tuple(pieces)

    def _should_cache_timefreq(self, mode, params):
        return not (mode == "mtm" and params.get("mtm_mode") == "zoom")

    def _hjorth_params(self, ch):
        return {
            "ch": ch,
            "minf": None,
            "maxf": None,
            "winsor": float(self.ui.spin_win.value()),
            "epoch_dur": self._get_epoch_dur(),
        }

    def _derive_timefreq(self, p, mode, params):
        if mode == "means":
            return self._derive_means(p, params)
        if mode == "welch":
            xi, yi, zi = self._derive_spectrogram(
                p, params["ch"], params["minf"], params["maxf"], params["winsor"],
                params["ne"], params["ns"], self._SPEC_EPOCH_DUR, params["sr"],
            )
            return {"xi": xi, "yi": yi, "zi": zi, "title": f"Welch {params['ch']}",
                    "cbar": "PSD (dB)", "ylabel": "Frequency (Hz)"}
        if mode == "mtm":
            return self._derive_mtm(p, params)
        if mode == "irasa":
            return self._derive_irasa(p, params)
        raise ValueError(f"Unknown mode: {mode}")

    @Slot()
    def _timefreq_done_ok(self):
        try:
            token, mode, key, result = self._last_result
            if token != self._spec_job_token:
                return
            if key is not None:
                self._spec_cache[key] = result
            self._complete_timefreq(mode, result, cache_key=key)
        finally:
            self._end_spec_job()

    def _complete_timefreq(self, mode, result, cache_key=None):
        self._spec_plot_kind = mode
        self._spec_plot_cache = result
        if mode == "means":
            self._draw_means_plot(result)
            if result.get("status"):
                self._set_timefreq_status(result["status"])
            return
        draw_result = self._irasa_component_result(result) if mode == "irasa" else result
        self._draw_heatmap_cache(draw_result)
        if draw_result.get("status"):
            self._set_timefreq_status(draw_result["status"])

    def _on_irasa_component_changed(self, *_):
        if getattr(self, "_spec_plot_kind", None) != "irasa":
            return
        result = getattr(self, "_spec_plot_cache", None)
        if not result:
            return
        self._draw_heatmap_cache(self._irasa_component_result(result))

    def _irasa_component_result(self, result):
        component = "APER"
        if hasattr(self.ui, "combo_irasa_component"):
            component = str(self.ui.combo_irasa_component.currentData() or "APER")
        components = result.get("components", {})
        return components.get(component) or components.get("APER") or result

    def _draw_heatmap_cache(self, result):
        self._ensure_spectrogram_canvas()
        from .plts import plot_tf_heatmap
        plot_tf_heatmap(
            result.get("xi", np.array([])),
            result.get("yi", np.array([])),
            result.get("zi", np.array([])),
            result.get("title", ""),
            self.spectrogramcanvas.ax,
            y_label=result.get("ylabel", "Frequency (Hz)"),
            cbar_label=result.get("cbar", "Value"),
            y_ticklabels=result.get("yticklabels"),
            show_legend=self._show_spec_legend(),
            cmap=result.get("cmap", "turbo"),
            center_zero=bool(result.get("center_zero", False)),
        )
        if result.get("xlim") is not None:
            self.spectrogramcanvas.ax.set_xlim(*result["xlim"])
        else:
            ns = getattr(self, "ns", None)
            if ns is not None and ns > 0:
                self.spectrogramcanvas.ax.set_xlim(0, float(ns))
        self.spectrogramcanvas.draw_idle()

    def _derive_means(self, p, params):
        """Calculate raw-sample summaries for each fixed-length epoch."""
        # STATS is Luna's canonical implementation for epoch-wise mean and
        # min/max.  It also respects Luna's epoch/mask model.  Use the raw
        # path below for bands (SE/CI/IQR), which STATS does not provide.
        if params.get("band", "mean") in ("mean", "minmax"):
            luna_result = self._derive_means_from_luna(p, params)
            if luna_result is not None:
                return luna_result

        epoch_dur = float(params["epoch_dur"])
        total_seconds = float(params.get("ns", 0.0) or 0.0)
        if epoch_dur <= 0 or total_seconds <= 0:
            return {"x": np.array([]), "series": [], "epoch_dur": epoch_dur}

        n_epochs = max(1, int(np.ceil(total_seconds / epoch_dur)))
        x = np.arange(n_epochs, dtype=float) * epoch_dur
        series = []
        idx = p.s2i([(0.0, total_seconds)])
        for ch in params["channels"]:
            raw = p.slice(idx, chs=ch, time=True)
            arr = raw[1] if isinstance(raw, (tuple, list)) and len(raw) > 1 else raw
            arr = np.asarray(arr) if arr is not None else np.empty((0, 2))
            if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] == 0:
                values = np.full(n_epochs, np.nan)
                lower = np.full(n_epochs, np.nan)
                upper = np.full(n_epochs, np.nan)
            else:
                times = np.asarray(arr[:, 0], dtype=float)
                samples = np.asarray(arr[:, 1], dtype=float)
                values = np.full(n_epochs, np.nan)
                lower = np.full(n_epochs, np.nan)
                upper = np.full(n_epochs, np.nan)
                for epoch in range(n_epochs):
                    mask = (
                        (times >= epoch * epoch_dur)
                        & (times < min(total_seconds, (epoch + 1) * epoch_dur))
                        & np.isfinite(samples)
                    )
                    vals = samples[mask]
                    if vals.size == 0:
                        continue
                    values[epoch] = np.mean(vals)
                    band = params.get("band", "mean")
                    if band == "se":
                        lower[epoch] = upper[epoch] = np.std(vals, ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
                    elif band == "ci":
                        se = np.std(vals, ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
                        lower[epoch] = upper[epoch] = 1.96 * se
                    elif band == "iqr":
                        lower[epoch], upper[epoch] = np.percentile(vals, [25, 75])
                    elif band == "minmax":
                        lower[epoch], upper[epoch] = np.min(vals), np.max(vals)
            series.append({"channel": ch, "mean": values, "lower": lower, "upper": upper})
        return {
            "x": x, "series": series, "epoch_dur": epoch_dur,
            "band": params.get("band", "mean"), "relative": bool(params.get("relative")),
        }

    def _derive_means_from_luna(self, p, params):
        """Return STATS epoch output for mean/min-max plots, if available."""
        epoch_dur = float(params["epoch_dur"])
        total_seconds = float(params.get("ns", 0.0) or 0.0)
        channels = list(params.get("channels", ()))
        if epoch_dur <= 0 or total_seconds <= 0 or not channels:
            return None
        try:
            cmd = (
                f"EPOCH dur={epoch_dur:g} verbose & "
                f"STATS epoch pct=F sig={','.join(channels)}"
            )
            res = p.silent_proc_lunascope(cmd)
            df = res.get("STATS: CH_E")
            dt = res.get("EPOCH: E")
            if df is None or dt is None or df.empty or dt.empty:
                return None
            required = {"E", "MEAN"}
            if not required.issubset(df.columns) or "START" not in dt.columns or "E" not in dt.columns:
                return None

            timeline = dt.sort_values("E").reset_index(drop=True)
            x = timeline["START"].to_numpy(dtype=float)
            epoch_index = {value: i for i, value in enumerate(timeline["E"].tolist())}
            has_channel = "CH" in df.columns
            if len(channels) > 1 and not has_channel:
                return None
            series = []
            for ch in channels:
                rows = df[df["CH"] == ch] if has_channel else df
                if rows.empty:
                    return None
                mean = np.full(len(x), np.nan)
                lower = np.full(len(x), np.nan)
                upper = np.full(len(x), np.nan)
                for row in rows.itertuples(index=False):
                    pos = epoch_index.get(getattr(row, "E", None))
                    if pos is None:
                        continue
                    mean[pos] = float(getattr(row, "MEAN"))
                    if params.get("band") == "minmax":
                        if "MIN" not in df.columns or "MAX" not in df.columns:
                            return None
                        lower[pos] = float(getattr(row, "MIN"))
                        upper[pos] = float(getattr(row, "MAX"))
                series.append({"channel": ch, "mean": mean, "lower": lower, "upper": upper})
            return {
                "x": x, "series": series, "epoch_dur": epoch_dur,
                "band": params.get("band", "mean"), "relative": bool(params.get("relative")),
            }
        except Exception:
            # Keep the Means tab usable across Luna versions/output schemas.
            return None

    def _draw_means_plot(self, result):
        self._ensure_spectrogram_canvas()
        ax = self.spectrogramcanvas.ax
        fig = ax.figure
        for extra_ax in list(fig.axes):
            if extra_ax is not ax:
                extra_ax.remove()
        ax.clear()
        show_legend = self._show_spec_legend()
        bg = "white" if show_legend else "black"
        fig.patch.set_facecolor(bg)
        ax.set_facecolor(bg)
        from matplotlib import colormaps
        colors = list(colormaps["tab10"].colors)
        band = result.get("band", "mean")
        band_label = {
            "se": "± SE", "ci": "± 95% CI", "iqr": "IQR", "minmax": "min/max"
        }.get(band, "")
        for i, item in enumerate(result.get("series", [])):
            color = colors[i % len(colors)]
            mean = np.asarray(item["mean"], dtype=float)
            lower = np.asarray(item["lower"], dtype=float)
            upper = np.asarray(item["upper"], dtype=float)
            winsor = float(self.ui.spin_win.value())
            mean = winsorize_array(mean, winsor)
            lower = winsorize_array(lower, winsor)
            upper = winsorize_array(upper, winsor)
            if result.get("relative"):
                good = np.isfinite(mean)
                if np.any(good):
                    base_lo = float(np.nanmin(mean[good]))
                    base_hi = float(np.nanmax(mean[good]))
                    scale = base_hi - base_lo
                    if not np.isfinite(scale) or scale <= 1e-12:
                        scale = 1.0
                    mean = (mean - base_lo) / scale
                    if band in ("se", "ci"):
                        lower = lower / scale
                        upper = upper / scale
                    else:
                        lower = (lower - base_lo) / scale
                        upper = (upper - base_lo) / scale
            line, = ax.plot(result["x"], mean, color=color, linewidth=1.0,
                            label=item["channel"])
            if band != "mean":
                ax.fill_between(result["x"], mean - lower if band in ("se", "ci") else lower,
                                mean + upper if band in ("se", "ci") else upper,
                                color=color, alpha=0.18, linewidth=0)
        ax.set_title(f"Epoch-wise means{(' ' + band_label) if band_label else ''}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean amplitude")
        if result.get("relative"):
            ax.set_ylabel("Relative mean amplitude (per channel)")
            ax.set_ylim(-0.05, 1.05)
        ax.set_axis_on()
        ax.margins(x=0.01, y=0.05)
        log_y = bool(self.ui.check_mean_log.isChecked())
        if log_y:
            ax.set_yscale("log")
            if result.get("relative"):
                ax.set_ylim(0.001, 1.05)
            if any(np.any(np.asarray(item["mean"], dtype=float) <= 0) for item in result.get("series", [])):
                self._set_timefreq_status("Log Y hides non-positive epoch means")
        else:
            ax.set_yscale("linear")
        ax.grid(True, alpha=0.2)
        if bg == "black":
            ax.tick_params(colors="#d7e3f4")
            ax.xaxis.label.set_color("#d7e3f4")
            ax.yaxis.label.set_color("#d7e3f4")
            ax.title.set_color("#d7e3f4")
            for spine in ax.spines.values():
                spine.set_color("#666666")
        if show_legend and result.get("series"):
            legend = ax.legend(loc="upper right", framealpha=0.75)
            if bg == "black":
                for text in legend.get_texts():
                    text.set_color("white")
        ns = getattr(self, "ns", None)
        if ns is not None and ns > 0:
            ax.set_xlim(0, float(ns))
        self.spectrogramcanvas.draw_idle()

    def _set_timefreq_status(self, text):
        if hasattr(self.ui, "label_timefreq_status"):
            self.ui.label_timefreq_status.setText(str(text or ""))

    def _edges_from_centers(self, centers, default_step=1.0):
        centers = np.asarray(centers, dtype=float)
        centers = np.sort(np.unique(centers))
        if centers.size == 0:
            return np.array([])
        if centers.size == 1:
            step = float(default_step)
            return np.array([centers[0] - step / 2.0, centers[0] + step / 2.0])
        mids = 0.5 * (centers[:-1] + centers[1:])
        first = centers[0] - (mids[0] - centers[0])
        last = centers[-1] + (centers[-1] - mids[-1])
        return np.concatenate([[first], mids, [last]])

    def _grid_from_points(self, x, y, z, *, x_edges=None, y_edges=None, default_x_step=30.0, winsor=0.0):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = winsorize_array(np.asarray(z, dtype=float), float(winsor or 0.0))
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[ok], y[ok], z[ok]
        if x.size == 0 or y.size == 0:
            return np.array([]), np.array([]), np.array([])
        xs = np.sort(np.unique(x))
        ys = np.sort(np.unique(y))
        xi = np.asarray(x_edges, dtype=float) if x_edges is not None else self._edges_from_centers(xs, default_x_step)
        yi = np.asarray(y_edges, dtype=float) if y_edges is not None else self._edges_from_centers(ys, 1.0)
        if xi.size < 2 or yi.size < 2:
            return np.array([]), np.array([]), np.array([])
        if x_edges is None:
            x_lookup = {v: i for i, v in enumerate(xs)}
            x_bins = np.array([x_lookup[v] for v in x], dtype=int)
            xn = xs.size
        else:
            x_bins = np.searchsorted(xi, x, side="right") - 1
            x_bins[x == xi[-1]] = xi.size - 2
            xn = xi.size - 1
        y_index = {v: i for i, v in enumerate(ys)}
        zi = np.full((ys.size, xn), np.nan, dtype=float)
        for xb, yv, zv in zip(x_bins, y, z):
            if 0 <= xb < xn:
                zi[y_index[yv], xb] = zv
        return xi, yi, np.ma.masked_invalid(zi)

    def _grid_elapsed_points(
        self,
        x,
        y,
        z,
        minf,
        maxf,
        w,
        *,
        total_epochs=0,
        total_seconds=0.0,
        timeline_starts=None,
    ):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (y >= minf) & (y <= maxf)
        x = x[ok]
        y = y[ok]
        z = winsorize_array(z[ok], w)
        if x.size == 0 or y.size == 0:
            return np.array([]), np.array([]), np.array([])

        x0 = float(np.min(x))
        x1 = float(np.max(x))
        xn = int(np.unique(x).size)
        if total_epochs is not None and int(total_epochs) > 0 and total_seconds is not None and float(total_seconds) > 0:
            x0 = 0.0
            x1 = float(total_seconds)
            xn = int(total_epochs)
        elif timeline_starts is not None:
            xt = np.sort(np.unique(np.asarray(timeline_starts, dtype=float)))
            xt = xt[np.isfinite(xt)]
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
        with np.errstate(divide="ignore", invalid="ignore"):
            zi = zi / counts
            zi = np.ma.masked_invalid(zi)
        return xi, yi, zi

    def _epoch_starts(self, res):
        dt = res.get("EPOCH: E")
        if dt is None or dt.empty or "E" not in dt.columns or "START" not in dt.columns:
            return None
        return dt[["E", "START"]].copy()

    def _derive_mtm(self, p, params):
        t = f" t={params['t']}" if int(params.get("t", 0)) > 0 else ""
        base = (
            f"MTM sig={params['ch']} dB segment-sec={params['segment']}"
            f" segment-inc={params['inc']} nw={params['nw']}{t}"
            f" min={params['minf']} max={params['maxf']}"
        )
        if params["mtm_mode"] == "whole":
            cmd = f"EPOCH dur=30 verbose & {base} epoch-output epoch-strata epoch-spectra"
            res = p.silent_proc_lunascope(cmd)
            df = res.get("MTM: CH_E_F")
            starts = self._epoch_starts(res)
            if df is None or df.empty or starts is None:
                return {"xi": np.array([]), "yi": np.array([]), "zi": np.array([]), "title": "MTM"}
            merged = df.merge(starts, on="E", how="left")
            xi, yi, zi = self._grid_elapsed_points(
                merged["START"].to_numpy(float),
                merged["F"].to_numpy(float),
                merged["MTM"].to_numpy(float),
                params["minf"],
                params["maxf"],
                params["winsor"],
                total_epochs=params.get("ne", 0),
                total_seconds=params.get("ns", 0.0),
                timeline_starts=starts["START"].to_numpy(float),
            )
            freqs = np.sort(np.unique(merged["F"].to_numpy(float)))
            fres = float(np.nanmedian(np.diff(freqs))) if freqs.size > 1 else np.nan
            ne = len(starts)
            status = (
                f"MTM whole-night: {ne} epochs; freq res={fres:g}Hz; "
                f"seg={params['segment']:g}s inc={params['inc']:g}s nw={params['nw']:g}"
            )
            return {
                "xi": xi, "yi": yi, "zi": zi, "ylabel": "Frequency (Hz)",
                "title": f"MTM whole night {params['ch']}", "cbar": "MTM (dB)", "status": status,
            }

        lo, hi = float(params["lo"]), float(params["hi"])
        half_seg = max(0.0, float(params["segment"]) / 2.0)
        req_lo = max(0.0, lo - half_seg)
        req_hi = hi + half_seg
        ns = float(params.get("ns", 0.0) or 0.0)
        if ns > 0:
            req_hi = min(ns, req_hi)
        cmd = f"{base} segment-spectra start={req_lo:.6g} stop={req_hi:.6g}"
        res = p.silent_proc_lunascope(cmd)
        df = res.get("MTM: CH_F_SEG")
        seg = res.get("MTM: CH_SEG")
        if df is None or df.empty or seg is None or seg.empty:
            return {"xi": np.array([]), "yi": np.array([]), "zi": np.array([]), "title": "MTM zoom"}
        seg = seg[["SEG", "START", "STOP"]].copy()
        merged = df.merge(seg, on="SEG", how="left")
        merged = merged[(merged["STOP"].astype(float) >= lo) & (merged["START"].astype(float) <= hi)]
        if merged.empty:
            return {"xi": np.array([]), "yi": np.array([]), "zi": np.array([]), "title": "MTM zoom"}
        x_centers = 0.5 * (merged["START"].to_numpy(float) + merged["STOP"].to_numpy(float))
        edge_tol = max(1e-9, float(params["inc"]) * 1e-6)
        in_view = (x_centers >= lo - edge_tol) & (x_centers <= hi + edge_tol)
        merged = merged.loc[in_view].copy()
        x_centers = x_centers[in_view]
        if merged.empty:
            return {"xi": np.array([]), "yi": np.array([]), "zi": np.array([]), "title": "MTM zoom"}
        xi, yi, zi = self._grid_from_points(
            x_centers, merged["F"].to_numpy(float), merged["MTM"].to_numpy(float),
            default_x_step=float(params["inc"]), winsor=params["winsor"],
        )
        freqs = np.sort(np.unique(merged["F"].to_numpy(float)))
        fres = float(np.nanmedian(np.diff(freqs))) if freqs.size > 1 else np.nan
        status = (
            f"MTM zoom: {len(np.unique(merged['SEG']))} segments; "
            f"freq res={fres:g}Hz; seg={params['segment']:g}s inc={params['inc']:g}s"
        )
        return {
            "xi": xi, "yi": yi, "zi": zi, "title": f"MTM zoom {params['ch']}",
            "cbar": "MTM (dB)", "status": status, "xlim": (lo, hi),
        }

    def _derive_irasa(self, p, params):
        cmd = (
            f"EPOCH dur=30 verbose & IRASA sig={params['ch']} epoch dB"
            f" min={params['minf']} max={params['maxf']}"
        )
        res = p.silent_proc_lunascope(cmd)
        df = res.get("IRASA: CH_E_F")
        starts = self._epoch_starts(res)
        if df is None or df.empty or starts is None or "APER" not in df.columns or "PER" not in df.columns:
            return {"xi": np.array([]), "yi": np.array([]), "zi": np.array([]), "title": "IRASA"}
        merged = df.merge(starts, on="E", how="left")
        components = {}
        for value_col, label in (("APER", "Aperiodic"), ("PER", "Periodic")):
            xi, yi, zi = self._grid_elapsed_points(
                merged["START"].to_numpy(float),
                merged["F"].to_numpy(float),
                merged[value_col].to_numpy(float),
                params["minf"],
                params["maxf"],
                params["winsor"],
                total_epochs=params.get("ne", 0),
                total_seconds=params.get("ns", 0.0),
                timeline_starts=starts["START"].to_numpy(float),
            )
            components[value_col] = {
                "xi": xi, "yi": yi, "zi": zi, "title": f"IRASA {label} {params['ch']}",
                "cbar": f"{label} (dB)",
            }
        return {"title": f"IRASA {params['ch']}", "components": components}

    def _on_spec_window_range_changed(self, lo, hi):
        if (
            self._spec_record_ready()
            and self._timefreq_mode() == "mtm"
            and self.ui.combo_mtm_mode.currentData() == "zoom"
        ):
            self._spec_zoom_timer.start()

    def _spec_on_zoom_timer(self):
        if (
            self._spec_record_ready()
            and self._timefreq_mode() == "mtm"
            and self.ui.combo_mtm_mode.currentData() == "zoom"
        ):
            self._calc_timefreq("mtm")


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
        ch = self._current_channel()
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

        timeline_starts = None
        if dt is not None and 'START' in dt.columns and len(dt) > 0:
            timeline_starts = dt['START'].to_numpy(dtype=float)

        return self._grid_elapsed_points(
            x,
            df['F'].to_numpy(dtype=float),
            df['PSD'].to_numpy(dtype=float),
            minf,
            maxf,
            w,
            total_epochs=total_epochs,
            total_seconds=total_seconds,
            timeline_starts=timeline_starts,
        )


    def _complete_spectrogram(self,xi,yi,zi):
        self._ensure_spectrogram_canvas()
        # we can now touch the GUI
        ch = self._current_channel()
        minf = self.ui.spin_lwrfrq.value() 
        maxf = self.ui.spin_uprfrq.value()
        self._spec_plot_kind = "spectrogram"
        self._spec_plot_cache = {
            "xi": xi, "yi": yi, "zi": zi, "title": f"Welch {ch}", "cbar": "PSD (dB)",
        }
        self._draw_heatmap_cache(self._spec_plot_cache)

    def _draw_spectrogram_plot(self, xi, yi, zi, ch, minf, maxf):
        from .plts import plot_spec
        plot_spec(
            xi, yi, zi, ch, minf, maxf,
            ax=self.spectrogramcanvas.ax,
            gui=self.ui,
            show_legend=self._show_spec_legend(),
        )
        ns = getattr(self, "ns", None)
        if ns is not None and ns > 0:
            self.spectrogramcanvas.ax.set_xlim(0, float(ns))

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
        ch = self._current_channel()

        # check it still exists in the in-memory EDF                                          
        if ch not in self.p.edf.channels():
            return

        params = self._hjorth_params(ch)
        key = self._spec_cache_key("hjorth", params)
        result = self._spec_cache.get(key)
        if result is None:
            from .plts import derive_hjorth_data
            result = derive_hjorth_data(
                ch,
                self.p,
                winsor=params["winsor"],
                epoch_dur=params["epoch_dur"],
            )
            if result is not None:
                self._spec_cache[key] = result
        self._complete_hjorth(result, cache_key=key)

    def _complete_hjorth(self, result, cache_key=None):
        self._spec_plot_kind = "hjorth"
        self._spec_plot_cache = result
        self._draw_hjorth_plot(result)

    def _draw_hjorth_plot(self, result=None):
        if not hasattr(self, "p"):
            return
        ch = self._current_channel()
        if not ch:
            return
        if result is None:
            result = getattr(self, "_spec_plot_cache", None)
        if result is None:
            params = self._hjorth_params(ch)
            from .plts import derive_hjorth_data
            result = derive_hjorth_data(
                ch,
                self.p,
                winsor=params["winsor"],
                epoch_dur=params["epoch_dur"],
            )
        from .plts import draw_hjorth_data
        draw_hjorth_data(result, ax=self.spectrogramcanvas.ax,
                         show_legend=self._show_spec_legend())
        ns = getattr(self, "ns", None)
        if ns is not None and ns > 0:
            self.spectrogramcanvas.ax.set_xlim(0, float(ns))

        self.spectrogramcanvas.draw_idle()
