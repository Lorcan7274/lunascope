
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

import pandas as pd
import numpy as np
from collections import defaultdict

from scipy.signal import butter, sosfilt

from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import QMetaObject, Qt, Slot

import pyqtgraph as pg
from PySide6.QtWidgets import (
    QProgressBar, QMessageBox, QDoubleSpinBox, QLabel, QHBoxLayout, QWidget,
    QVBoxLayout, QGridLayout, QFrame, QSizePolicy,
)
from PySide6.QtCore import QSignalBlocker, QEvent, QObject


class _BannerResizeFilter(QObject):
    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self._callback = callback

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Resize:
            self._callback()
        return False

class SignalsMixin:
    def _annotator_interval_capture_active(self):
        return bool(
            hasattr(self, "annotator")
            and self.annotator is not None
            and self.annotator.isVisible()
            and self.annotator.check_enabled.isChecked()
            and self.annotator.current_mode() == "interval"
        )

    def _init_pg1_annot_selector(self):
        if getattr(self, "annot_sel", None) is not None:
            try:
                self.annot_sel.dispose()
            except Exception:
                pass
            self.annot_sel = None

        self.annot_sel = XRangeSelector(
            self.ui.pg1,
            bounds=(0, self.ns),
            integer=False,
            click_span=30.0,
            min_span=0.01,
            step=1,
            big_step=10,
            active_predicate=self._annotator_interval_capture_active,
        )
        self.annot_sel.owner = self

    def _navigator_stage_query_classes(self):
        # Include both detailed staging and generic sleep/wake aliases.
        return ['N1', 'N2', 'N3', 'R', 'W', 'SP', 'WP', '?', 'L']

    def _navigator_stage_mode(self, stage_values):
        vals = set(stage_values)
        has_detailed = any(s in vals for s in ('N1', 'N2', 'N3', 'R'))
        has_sw = any(s in vals for s in ('SP', 'WP'))
        if has_detailed:
            return 'detailed'
        if has_sw:
            return 'sw'
        return 'other'

    def _filter_navigator_stage_df(self, df, class_col):
        if df is None or len(df) == 0:
            return df
        mode = self._navigator_stage_mode(df[class_col].tolist())
        if mode == 'detailed':
            keep = {'N1', 'N2', 'N3', 'R', 'W', '?', 'L'}
        elif mode == 'sw':
            keep = {'SP', 'WP', '?', 'L'}
        else:
            keep = {'W', '?', 'L'}
        return df[df[class_col].isin(keep)].copy()

    def _init_signals(self):

        # hypnogram / navigator
        h = self.ui.pgh
        h.showAxis('left', False)
        h.showAxis('bottom', False)
        h.setMenuEnabled(False)
        h.setMouseEnabled(x=False, y=False)
        
        # pg1 - main signals
        # pgh - hypnogram, controls view on pg1
        
        self.ui.butt_render.clicked.connect( self._render_signals )
        self._init_line_weight_control()

        # pyqtgraph config options
        pg.setConfigOptions(useOpenGL=False, antialias=False)  
        
        # pg1 properties
        pw = self.ui.pg1   
        pw.setXRange(0, 1, padding=0)   
        pw.setYRange(0, 1, padding=0)
        pw.showAxis('left', False)
        pw.showAxis('bottom', False)
        vb = pw.getViewBox()
        vb.enableAutoRange('x', False)
        vb.enableAutoRange('y', False)

        # disable mouse pan/zoom
        vb.setMouseEnabled(x=False, y=False)   # disables drag + wheel zoom
        vb.wheelEvent = lambda ev: None        # belt-and-suspenders on some styles
        vb.mouseDragEvent = lambda ev: None
        vb.setMenuEnabled(False)               # optional: no context menu

        pi = pw.getPlotItem()
        pi.enableAutoRange('xy', False)   # or: pi.disableAutoRange()
        pi.autoBtn.hide()                 # prevents UI trigger
        pi.disableAutoRange()
        pi.hideButtons()          # use this, not pi.autoBtn.hide()
        

        self.ui.spin_spacing.valueChanged.connect( self._update_scaling )
        self.ui.spin_scale.valueChanged.connect( self._update_scaling )

        self.ui.spin_fixed_max.valueChanged.connect( self._update_scaling )
        self.ui.spin_fixed_min.valueChanged.connect( self._update_scaling )

        self.ui.radio_fixedscale.clicked.connect( self._update_scaling )
        self.ui.radio_clip.clicked.connect( self._update_scaling )
        self.ui.radio_empiric.clicked.connect( self._update_scaling )
        self.ui.check_labels.clicked.connect( self._update_labels )
        
        self.last_x1 = 0
        self.last_x2 = 30
        self.multiday_mode = False
        self._record_start_tod_secs = 0
        self._day_lines_pgh = []
        self._day_lines_pg1 = []

        # safe defaults so window events don't crash before first render
        self.pg1_header_height = 0.05
        self.pg1_footer_height = 0.025
        self.pg1_annot_height  = 0
        self._pg1_channel_cache = []
        self._pg1_probe_line = None
        self._pg1_probe_start_sample_line = None
        self._pg1_probe_sample_line = None
        self._pg1_probe_span_line = None
        self._pg1_probe_label = None
        self._pg1_probe_legend = None
        self._pg1_probe_band_lines = []
        self._pg1_probe_grid_lines = None
        self._pg1_probe_zero_lines = None
        self._pg1_probe_peak_max = None
        self._pg1_probe_peak_min = None
        self._pg1_probe_keys = set()
        self._pg1_probe_grid_steps = [None, 1, 5, 10, 30, 60, 300, 600, 1800, 3600]
        self._pg1_probe_grid_idx = 0
        self._pg1_probe_pinned = False
        self._pg1_probe = MainTraceProbe(self.ui.pg1, self)
        self._pg1_nav_proxy = MainTraceNavProxy(self.ui.pg1, self)
        self._style_control_banner()

    def _init_line_weight_control(self):
        if getattr(self, "_line_weight_widget", None) is not None:
            return

        parent = self.ui.butt_render.parentWidget()
        if parent is None:
            return
        layout = parent.layout()
        if layout is None:
            return

        holder = QWidget(parent)
        hlay = QHBoxLayout(holder)
        hlay.setContentsMargins(0, 0, 0, 0)
        hlay.setSpacing(3)

        lab = QLabel("Weight", holder)
        spin = QDoubleSpinBox(holder)
        spin.setDecimals(0)
        spin.setSingleStep(1)
        spin.setRange(1, 8)
        spin.setValue(float(getattr(self, "cfg_line_weight", 1.0)))
        spin.setToolTip("Trace line width")
        spin.setFixedWidth(34)

        hlay.addWidget(lab)
        hlay.addWidget(spin)
        hlay.addStretch(1)

        # Place at row 1, col 12 (below Render — col 12 row 0 only in .ui)
        layout.addWidget(holder, 1, 12)

        spin.valueChanged.connect(self._on_line_weight_changed)
        self._line_weight_widget = holder
        self._line_weight_spin = spin

    def _banner_section(self, title: str, object_name: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame(self.ui.ctrframe)
        frame.setObjectName(object_name)
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(3)
        lay.setAlignment(Qt.AlignTop)
        if title:
            hdr = QLabel(title, frame)
            hdr.setObjectName("banner_section_title")
            hdr.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            hdr.setFixedHeight(14)
            lay.addWidget(hdr, 0, Qt.AlignTop)
        return frame, lay

    def _banner_field_row(self, left, right, object_name: str | None = None):
        row = QWidget(self.ui.ctrframe)
        if object_name:
            row.setObjectName(object_name)
        lay = QHBoxLayout(row)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        lay.addWidget(left)
        if right is not None:
            lay.addWidget(right)
        lay.addStretch(1)
        return row

    def _style_control_banner(self):
        host = getattr(self.ui, "ctrframe", None)
        if host is None:
            return
        grid = host.layout()
        if grid is None:
            return

        if getattr(self, "_banner_root", None) is None:
            self._banner_root = QWidget(host)
            self._banner_root.setObjectName("banner_root")
            root = QVBoxLayout(self._banner_root)
            root.setContentsMargins(4, 4, 4, 4)
            root.setSpacing(4)

            top = QHBoxLayout()
            top.setContentsMargins(0, 0, 0, 0)
            top.setSpacing(6)
            self._banner_top_row = QWidget(self._banner_root)
            self._banner_top_row.setObjectName("banner_top_row")
            self._banner_top_row.setLayout(top)
            root.addWidget(self._banner_top_row)

            self._banner_top_layout = top
            self._banner_root_layout = root

            for widget in (
                self.ui.lbl_twin,
                self.ui.lbl_ewin,
                self.ui.label_jump_width,
                self.ui.spin_jump_width,
                self.ui.label_spacing,
                self.ui.spin_spacing,
                self.ui.label_scale,
                self.ui.spin_scale,
                getattr(self.ui, "label_11", None),
                getattr(self.ui, "label_12", None),
                self.ui.spin_fixed_max,
                self.ui.spin_fixed_min,
                self.ui.check_labels,
                self.ui.radio_fixedscale,
                self.ui.radio_empiric,
                self.ui.radio_clip,
                self.ui.butt_render,
                getattr(self, "_line_weight_widget", None),
                getattr(self, "_psd_toolbar_holder", None),
                getattr(self.ui, "sep_info_nav", None),
                getattr(self.ui, "sep_disp_yaxis", None),
                getattr(self.ui, "sep_yaxis_render", None),
            ):
                if widget is None:
                    continue
                grid.removeWidget(widget)
                if widget not in (
                    self.ui.lbl_twin,
                    self.ui.lbl_ewin,
                    self.ui.label_jump_width,
                    self.ui.spin_jump_width,
                    self.ui.label_spacing,
                    self.ui.spin_spacing,
                    self.ui.label_scale,
                    self.ui.spin_scale,
                    self.ui.spin_fixed_max,
                    self.ui.spin_fixed_min,
                    self.ui.check_labels,
                    self.ui.radio_fixedscale,
                    self.ui.radio_empiric,
                    self.ui.radio_clip,
                    self.ui.butt_render,
                    getattr(self, "_line_weight_widget", None),
                    getattr(self, "_psd_toolbar_holder", None),
                ):
                    widget.hide()

            for name in ("toolbar_h_spacer",):
                item = getattr(self.ui, name, None)
                if item is not None:
                    try:
                        grid.removeItem(item)
                    except Exception:
                        pass

            grid.addWidget(self._banner_root, 0, 0, 4, 13)

            status_card, status_lay = self._banner_section("Window", "banner_status")
            status_lay.addWidget(self.ui.lbl_twin)
            status_lay.addWidget(self.ui.lbl_ewin)
            status_lay.addWidget(self._banner_field_row(self.ui.label_jump_width, self.ui.spin_jump_width))

            scaling_card, scaling_lay = self._banner_section("Y SCALE", "banner_scaling")
            self.ui.spin_spacing.setFixedWidth(52)
            self.ui.spin_scale.setFixedWidth(52)
            self.ui.spin_fixed_max.setFixedWidth(84)
            self.ui.spin_fixed_min.setFixedWidth(84)
            merged_grid = QGridLayout()
            merged_grid.setContentsMargins(0, 0, 0, 0)
            merged_grid.setHorizontalSpacing(3)
            merged_grid.setVerticalSpacing(2)
            max_lab = QLabel("Max", scaling_card)
            min_lab = QLabel("Min", scaling_card)
            merged_grid.addWidget(self.ui.label_spacing, 0, 0)
            merged_grid.addWidget(self.ui.spin_spacing, 0, 1)
            merged_grid.addWidget(max_lab, 0, 2)
            merged_grid.addWidget(self.ui.spin_fixed_max, 0, 3)
            merged_grid.addWidget(self.ui.label_scale, 1, 0)
            merged_grid.addWidget(self.ui.spin_scale, 1, 1)
            merged_grid.addWidget(min_lab, 1, 2)
            merged_grid.addWidget(self.ui.spin_fixed_min, 1, 3)
            scaling_lay.addLayout(merged_grid)
            toggles = QHBoxLayout()
            toggles.setContentsMargins(0, 0, 0, 0)
            toggles.setSpacing(10)
            toggles.addWidget(self.ui.check_labels)
            toggles.addWidget(self.ui.radio_clip)
            toggles.addWidget(self.ui.radio_empiric)
            toggles.addWidget(self.ui.radio_fixedscale)
            toggles.addStretch(1)
            scaling_lay.addLayout(toggles)

            action_card, action_lay = self._banner_section("TRACE VIEW", "banner_actions")
            action_lay.setSpacing(6)
            self.ui.butt_render.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Preferred)
            action_lay.addWidget(self.ui.butt_render)
            if getattr(self, "_line_weight_widget", None) is not None:
                action_lay.addWidget(self._line_weight_widget)

            # WINDOW left-anchored; stretch pushes the rest (PSD, Y SCALE, TRACE VIEW) right.
            # PSD is inserted between stretch and Y SCALE when available.
            top.addWidget(status_card)
            top.addStretch(1)
            top.addWidget(scaling_card)
            top.addWidget(action_card)

            self._banner_status_card = status_card
            self._banner_scaling_card = scaling_card
            self._banner_action_card = action_card

            # Always-visible cards keep a hard minimum so the banner never
            # collapses below a usable size.
            status_card.setMinimumWidth(210)
            status_card.setMaximumWidth(210)
            status_card.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            action_card.setMinimumWidth(100)
            action_card.setMaximumWidth(100)
            action_card.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            # Optional cards: minimum = 0 so they never prevent the window from
            # shrinking.  The threshold logic in _apply_banner_mode_inner hides
            # them before the layout would compress them, so they appear either
            # at full width or not at all.
            scaling_card.setMinimumWidth(0)
            scaling_card.setMaximumWidth(300)
            scaling_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

            host.setStyleSheet(
                """
                QFrame#ctrframe {
                    background: #2f2f2f;
                    border-top: 1px solid rgba(255,255,255,0.08);
                    border-bottom: 1px solid rgba(0,0,0,0.45);
                }
                QWidget#banner_root {
                    background: transparent;
                }
                QFrame#banner_status,
                QFrame#banner_scaling,
                QFrame#banner_actions,
                QFrame#banner_psd {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 rgba(255,255,255,0.045),
                        stop:1 rgba(0,0,0,0.08));
                    border: 1px solid rgba(255,255,255,0.10);
                    border-radius: 7px;
                }
                QLabel#banner_section_title {
                    color: #b9c6d2;
                    font-size: 9px;
                    font-weight: 600;
                    letter-spacing: 0.04em;
                    text-transform: uppercase;
                    padding: 0;
                    margin: 0;
                }
                QFrame#ctrframe QLabel {
                    font-size: 10px;
                }
                QFrame#banner_status QLabel,
                QFrame#banner_scaling QLabel,
                QFrame#banner_actions QLabel,
                QFrame#banner_psd QLabel {
                    font-size: 10px;
                }
                QFrame#banner_status QLabel#lbl_twin,
                QFrame#banner_status QLabel#lbl_ewin,
                QFrame#banner_status QLabel#label_jump_width {
                    font-size: 11px;
                    font-weight: 600;
                }
                """
            )

            self._banner_resize_filter = _BannerResizeFilter(self._apply_banner_mode, host)
            host.installEventFilter(self._banner_resize_filter)

        psd_holder = getattr(self, "_psd_toolbar_holder", None)
        top = self._banner_top_layout

        old_psd_card = getattr(self, "_banner_psd_card", None)
        if old_psd_card is not None:
            top.removeWidget(old_psd_card)
            old_psd_card.hide()
            old_psd_card.setParent(None)
            self._banner_psd_card = None

        if psd_holder is not None:
            psd_card, psd_lay = self._banner_section("PSD", "banner_psd")
            psd_card.setMinimumWidth(0)
            psd_card.setMaximumWidth(145)
            psd_card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            psd_lay.addWidget(psd_holder)
            # Insert PSD before Y SCALE (2nd from end), after the stretch spacer.
            insert_at = max(0, top.count() - 2)
            top.insertWidget(insert_at, psd_card)
            self._banner_psd_card = psd_card

        self._banner_root.updateGeometry()
        self._apply_banner_mode()

    def _move_banner_widget(self, widget, layout, index: int | None = None):
        if widget is None or layout is None:
            return
        old_parent = widget.parentWidget()
        if old_parent is not None and old_parent.layout() is not None:
            old_parent.layout().removeWidget(widget)
        if index is None:
            layout.addWidget(widget)
        else:
            layout.insertWidget(index, widget)

    def _apply_banner_mode(self):
        # Re-entrancy guard: widget.show()/setVisible() inside here triggers
        # child resize events which re-fire the event filter synchronously,
        # causing infinite recursion.  Drop any re-entrant call immediately.
        if getattr(self, "_banner_applying", False):
            return
        self._banner_applying = True
        try:
            self._apply_banner_mode_inner()
        finally:
            self._banner_applying = False

    def _apply_banner_mode_inner(self):
        host = getattr(self.ui, "ctrframe", None)
        top = getattr(self, "_banner_top_layout", None)
        if host is None or top is None:
            return

        self._banner_top_row.show()
        status  = getattr(self, "_banner_status_card",  None)
        scaling = getattr(self, "_banner_scaling_card", None)
        action  = getattr(self, "_banner_action_card",  None)
        psd     = getattr(self, "_banner_psd_card",     None)

        width = host.width()
        if width <= 0:
            # Window not yet shown/sized — start with everything visible so
            # the layout reports a sensible initial minimum size.
            for w in (status, scaling, action, psd):
                if w is not None:
                    w.show()
            return

        # Derive thresholds from the cards' own minimum widths so the logic
        # is content-aware and works across different screen resolutions.
        # OVERHEAD covers layout margins + inter-card spacing (~40 px).
        # BAND is a hysteresis dead-band that prevents rapid show/hide toggling
        # at the boundary (show if width > thresh+BAND, hide if < thresh-BAND).
        # OVERHEAD: actual inter-card spacing (6 px × 3 gaps) + banner margins (8 px) ≈ 26 px.
        # BAND: hysteresis dead-band — keeps cards from flickering at the boundary.
        OVERHEAD = 26
        BAND     = 8
        # Use maximumWidth for optional cards (their minimumWidth is 0 so the
        # layout can shrink freely; maximumWidth holds the intended display width).
        w_status  = status.minimumWidth()  if status  else 0
        w_scaling = scaling.maximumWidth() if scaling else 0
        w_psd     = psd.maximumWidth()     if psd     else 0
        w_action  = action.minimumWidth()  if action  else 0

        thresh_scaling = w_status + w_scaling + w_action + OVERHEAD
        thresh_psd     = thresh_scaling + w_psd

        was_scaling = bool(getattr(self, "_banner_scaling_visible", True))
        was_psd     = bool(getattr(self, "_banner_psd_visible",     True))

        if was_scaling:
            show_scaling = width >= thresh_scaling - BAND
        else:
            show_scaling = width >= thresh_scaling + BAND

        if was_psd:
            show_psd = width >= thresh_psd - BAND
        else:
            show_psd = width >= thresh_psd + BAND
        show_psd = show_psd and show_scaling and (psd is not None)

        self._banner_scaling_visible = show_scaling
        self._banner_psd_visible     = show_psd

        for w in (status, action):
            if w is not None:
                w.show()
        if scaling is not None:
            scaling.setVisible(show_scaling)
        if psd is not None:
            psd.setVisible(show_psd)

    def _on_line_weight_changed(self, value: float):
        self.cfg_line_weight = float(value)

        # Keep parsed cfg in sync for runtime operations that inspect cfg dict.
        if hasattr(self, "cfg") and isinstance(self.cfg, dict):
            self.cfg.setdefault("par", {})
            self.cfg["par"]["line-weight"] = str(self.cfg_line_weight)

        if hasattr(self, "_update_cols"):
            self._update_cols()

        if hasattr(self, "annot_mgr"):
            self.annot_mgr.refresh_wpx(self.cfg_line_weight)

    # --------------------------------------------------------------------------------
    #
    # on attach new EDF --> initiate segsrv_t for channel / annotation drawing 
    #
    # --------------------------------------------------------------------------------


    def _render_hypnogram(self):

        # ------------------------------------------------------------
        # initiate segsrv 
        
        self.ss = lp.segsrv( self.p )
        # reset non-render PSD segsrv so it is repopulated for the new file
        self._psd_nr_ss  = None
        self._psd_nr_chs = None

        # view 'epoch' is fixed at 30 seconds
        scope_epoch_sec = 30 

        # last time-point (secs)
        nsecs_clk = self.ss.num_seconds_clocktime_original()

        # number of scope-epochs (i.e. fixed at 0, 30s), and seconds
        self.ne = int( nsecs_clk / scope_epoch_sec )
        self.ns = nsecs_clk

        # multi-day mode: records longer than 36 hours
        self.multiday_mode = self.ns > 36 * 3600
        if hasattr(self, "_update_mode_badge"):
            self._update_mode_badge()

        # set epoch default for spectrogram/hjorth based on record type
        self._set_epoch_default(multiday=self.multiday_mode)
        if hasattr(self, "_set_actigraphy_epoch_default"):
            self._set_actigraphy_epoch_default(multiday=self.multiday_mode)

        # option defaults
        self.show_labels = True

        
        # ------------------------------------------------------------
        # set lights out/on

        res = self.p.silent_proc( 'HEADERS' )
        df = self.p.table( 'HEADERS' )

        start_date = str(df["START_DATE"].iloc[0])
        start_time = str(df["START_TIME"].iloc[0])
        stop_date = str(df["STOP_DATE"].iloc[0])
        stop_time = str(df["STOP_TIME"].iloc[0])

        # store start-of-day offset (seconds from midnight) for day-boundary math
        try:
            _stp = start_time.split(".")
            self._record_start_tod_secs = int(_stp[0]) * 3600 + int(_stp[1]) * 60 + int(_stp[2])
        except (IndexError, ValueError):
            self._record_start_tod_secs = 0
        
        start = start_date + "-" + start_time
        stop = stop_date + "-" + stop_time

        # time/date formats w/ '.' from HEADERS:
        dt_start = QtCore.QDateTime.fromString(start, "dd.MM.yy-HH.mm.ss")
        dt_stop = QtCore.QDateTime.fromString(stop, "dd.MM.yy-HH.mm.ss")

        # set widget
        self.ui.dt_lights_out.setDateTime(dt_start)
        self.ui.dt_lights_out.setDisplayFormat("dd/MM/yy-HH:mm:ss")

        self.ui.dt_lights_on.setDateTime(dt_stop)
        self.ui.dt_lights_on.setDisplayFormat("dd/MM/yy-HH:mm:ss")

        # ------------------------------------------------------------
        # hypnogram init

        h = self.ui.pgh
        pi = h.getPlotItem()
        pi.clear()

        vb = pi.getViewBox()

        h.showAxis('left', False)
        h.showAxis('bottom', False)
        h.setMenuEnabled(False)
        h.setMouseEnabled(x=False, y=False)

        pi.showAxis('left', False)
        pi.showAxis('bottom', False)
        pi.hideButtons()
        pi.setMenuEnabled(False)
        pi.layout.setContentsMargins(0, 0, 0, 0)
        pi.setContentsMargins(0, 0, 0, 0)        
        vb.setDefaultPadding(0)
        
        vb.setMouseEnabled(x=False, y=False)
        vb.wheelEvent = lambda ev: ev.accept()
        vb.doubleClickEvent = lambda ev: ev.accept()
        vb.keyPressEvent = lambda ev: ev.accept()   # swallow 'A' and everything else
        
        pi.setXRange(0, self.ns, padding=0)
        pi.setYRange(0, 1, padding=0)
        vb.setLimits(xMin=0, xMax=self.ns, yMin=0, yMax=1)  # prevent programmatic drift

        h.setXRange(0,self.ns)
        h.setYRange(0,1)

        # get full, original staging from annotations
        stgs = self._navigator_stage_query_classes()

        stgns = {'N1': 0.13333333333333333,
                 'N2': 0.06666666666666667,
                 'N3': 0.0,
                'R': 0.2,
                'SP': 0.1,
                'WP': 0.26666666666666666,
                'W': 0.26666666666666666,
                '?': 0.3333333333333333,
                'L': 0.4}

        stg_evts = self.p.fetch_annots( stgs , 30 )
        stg_evts = self._filter_navigator_stage_df(stg_evts, 'Class')
        
        if len( stg_evts ) != 0:
            starts = stg_evts[ 'Start' ].to_numpy()
            stops = stg_evts[ 'Stop' ].to_numpy()
            cols = [self.stgcols_hex.get(c, self.stgcols_hex['?']) for c in stg_evts['Class'].tolist()]
            ys = [stgns.get(c, stgns['?']) for c in stg_evts['Class'].tolist()]

            # ensure we'll see
            starts, stops = _ensure_min_px_width( vb, starts, stops, px=1)  # 1-px minimum

            # keep in seconds
            x = ( ( starts + stops ) / 2.0 ) 
            w = ( stops - starts ) 

            brushes = [QtGui.QColor(c) for c in cols]   # e.g. "#20B2DA"
            pens    = [None]*len(x)
            
            bins = defaultdict(list)
            for xi, wi, yi, ci in zip(x.tolist(), w.tolist(), ys, cols):
                bins[ci].append((xi, wi, yi ))

            for ci, items in bins.items():
                xi, wi, yi = zip(*items)
                bg = pg.BarGraphItem(
                    x=list(xi), width=list(wi), y0=[ x+0.25 for x in list(yi) ], height=[0.225]*len(xi), 
                    brush=QtGui.QColor(ci), pen=None )                
                bg.setZValue(-10)
                bg.setAcceptedMouseButtons(QtCore.Qt.NoButton)
                bg.setAcceptHoverEvents(False)
                pi.addItem(bg)

        # segment plotter
        pi.plot([0, self.ns], [0.01, 0.01], pen=pg.mkPen(0, 0, 0 ))
        
        # wire up range selector (first wiping existing one, if needed)

        if getattr(self, "sel", None) is not None:
            try:
                self.sel.dispose()
            except Exception:
                pass
            self.sel = None
        
        if self.multiday_mode:
            _click_span, _min_span, _step, _big_step = 3600.0, 60.0, 3600, 86400
        else:
            _click_span, _min_span, _step, _big_step = 30.0, 1.0, 30, 300
        self.sel = XRangeSelector(h, bounds=(0, self.ns),
                             integer=True,
                             click_span=_click_span,
                             min_span=_min_span,
                             step=_step, big_step=_big_step)
        self.sel.owner = self

        self.sel.rangeSelected.connect(self.on_window_range)  

        # clock ticks at top
        self.tb0 = TextBatch( vb, QtGui.QFont("Arial", 12), color=(180,255,255), mode='device')
        self.tb0.setZValue(10)
        tks = self.ssa.get_hour_ticks()
        if self.multiday_mode:
            if self.ns > 14 * 86400:
                stride_h = 24
            elif self.ns > 4 * 86400:
                stride_h = 12
            else:
                stride_h = 6
            stride_s = stride_h * 3600
            tks = {k: v for k, v in tks.items()
                   if (self._record_start_tod_secs + k) % stride_s == 0}
        tx = list( tks.keys() )
        tv = list( tks.values() )
        tv = [v[:-6] if v.endswith(":00:00") else v for v in tv]  # reduce to | hh
        self.tb0.setData(tx, [ 0.99 ] * len( tx ) , tv )
        self.ui.pgh.addItem(self.tb0 , ignoreBounds=True)

        # day boundary lines on navigator (multi-day mode only)
        self._day_lines_pgh = []
        if self.multiday_mode:
            for _t in self._compute_day_boundaries():
                _line = pg.InfiniteLine(
                    pos=_t,
                    angle=90,
                    pen=self._day_delimiter_pen(navigator=True),
                )
                pi.addItem(_line)
                self._day_lines_pgh.append(_line)

        # disable staging-dependent features in multi-day mode
        self.ui.butt_calc_hypnostats.setEnabled(not self.multiday_mode)
        self.ui.butt_soap.setEnabled(not self.multiday_mode)
        self.ui.butt_pops.setEnabled(not self.multiday_mode)
        if hasattr(self, "_sync_multiday_actigraphy_dock"):
            self._sync_multiday_actigraphy_dock()

        
    # --------------------------------------------------------------------------------
    #
    # called on first attaching, but also after Render: masked hypnogram + segment plot
    #
    # --------------------------------------------------------------------------------

    def _update_hypnogram(self):

        # writes on the same canvas as the hypnogram above, but only updates the
        # stuff that may change

        h = self.ui.pgh
        pi = h.getPlotItem()
        vb = pi.getViewBox()        
        
        # hypnogram vesion 2
        # get staging (in units no larger than 30 seconds)
        stgs = self._navigator_stage_query_classes()
        stg_evts = self.p.fetch_annots( stgs , 30 )
        stg_evts = self._filter_navigator_stage_df(stg_evts, 'Class')
                
        # get staging (in units no larger than 30 seconds)
        # use STAGES here so that we only get the unmasked datapoints

        mode = self._navigator_stage_mode(stg_evts['Class'].tolist()) if len(stg_evts) else 'other'
        if mode == 'sw':
            # Generic SP/WP mode: draw directly from annotations, no STAGE call needed.
            df = stg_evts[['Start', 'Stop', 'Class']].copy()
            df.rename(columns={'Start': 'START', 'Stop': 'STOP', 'Class': 'OSTAGE'}, inplace=True)
        else:
            try:
                res = self.p.silent_proc( 'EPOCH align verbose & STAGE' )
                # EPOCH align returns NE=0 when staging offsets can't align to the
                # epoch grid (e.g. after MASK+RE produces a discontinuous EDF).
                # Treat this the same as a failed STAGE call.
                df_ne = self.p.table( 'EPOCH' )
                if 'NE' not in df_ne.columns or int( df_ne.iloc[0, df_ne.columns.get_loc('NE')] ) == 0:
                    raise RuntimeError( "EPOCH align returned 0 epochs" )
            except (RuntimeError, KeyError):
                if len(stg_evts) != 0:
                    # Fallback: if STAGE fails, still render available annotations.
                    df = stg_evts[['Start', 'Stop', 'Class']].copy()
                    df.rename(columns={'Start': 'START', 'Stop': 'STOP', 'Class': 'OSTAGE'}, inplace=True)
                else:
                    QMessageBox.critical(
                        self.ui,
                        "Error running STAGE: checking for overlapping staging annotations",
                        "Problem with annotations: check for overlapping stage annotations"
                    )
                    return
            else:
                if "EPOCH: E" in res:
                    df1 = self.p.table( 'EPOCH' , 'E' )
                    df1 = df1[ ['E' , 'START' , 'STOP' ] ]
                else:
                    df1 = pd.DataFrame( columns = [ "E", "START", "STOP" ] )

                # if no valid staging, will not have any 'STAGE' output
                tbls = self.p.strata()
                has_staging = (tbls["Command"] == "STAGE").any()
                if has_staging:
                    df2 = self.p.table( 'STAGE' , 'E' )
                    df2 = df2[ ['E' , 'OSTAGE' ] ]
                else:
                    df2 = pd.DataFrame({
                        "E": df1["E"],
                        "OSTAGE": "?"
                    })
                # merge
                df = pd.merge(df1, df2, on="E", how="inner")

        df = self._filter_navigator_stage_df(df, 'OSTAGE')

        # Always draw from stg_evts (fetch_annots clocktime positions) — the same
        # source _render_hypnogram uses — so bars are never in a different coordinate
        # space.  For MASK-without-RE, EPOCH START values are in the same clocktime
        # space, so we can filter stg_evts to active epochs by matching positions.
        # For MASK+RE, EPOCH START values are restructured (0, 30, 60 …) and won't
        # match stg_evts; in that case RE has already removed inactive epochs so
        # stg_evts contains exactly the active set — use it as-is.
        if len(stg_evts) > 0:
            active = stg_evts[stg_evts['Start'].isin(df['START'])] if len(df) > 0 else stg_evts.iloc[0:0]
            if len(active) == 0:
                active = stg_evts  # post-RE: all remaining stg_evts entries are active
            df = active[['Start', 'Stop', 'Class']].copy()
            df.rename(columns={'Start': 'START', 'Stop': 'STOP', 'Class': 'OSTAGE'}, inplace=True)
            df = self._filter_navigator_stage_df(df, 'OSTAGE')

        if len( df ) != 0:
            starts = df[ 'START' ].to_numpy()
            stops = df[ 'STOP' ].to_numpy()
            cols = [self.stgcols_hex.get(c, self.stgcols_hex['?']) for c in df['OSTAGE'].tolist()]

            # ensure we'll see
            starts, stops = _ensure_min_px_width( vb, starts, stops, px=1)  # 1-px minimum
            
            # keep in seconds
            x = ( ( starts + stops ) / 2.0 ) 
            w = ( stops - starts ) 

            brushes = [QtGui.QColor(c) for c in cols]   # e.g. "#20B2DA"
            pens    = [None]*len(x)
            
            bins = defaultdict(list)
            for xi, wi, ci in zip(x.tolist(), w.tolist(), cols):
                bins[ci].append((xi, wi ))

            # clear if previously added
            if getattr(self, "updated_hypno", None) is not None:
                for it in self.updated_hypno:
                    pi.removeItem(it)
                self.updated_hypno.clear()

            self.updated_hypno = [ ] 

            # staging
            for ci, items in bins.items():
                xi, wi = zip(*items)
                bg = pg.BarGraphItem(
                    x=list(xi), width=list(wi), y0=[0.1] * len(xi), height=[0.05]*len(xi), 
                    brush=QtGui.QColor(ci), pen=None )                
                bg.setZValue(-10)
                bg.setAcceptedMouseButtons(QtCore.Qt.NoButton)
                bg.setAcceptHoverEvents(False)
                pi.addItem(bg)
                self.updated_hypno.append(bg)

            # simple segment plot
            for ci, items in bins.items():
                xi, wi = zip(*items)
                bg = pg.BarGraphItem(
                    x=list(xi), width=list(wi), y0=[0.03] * len(xi), height=[0.05]*len(xi), 
                    brush= '#FFCE1B', pen=None )                
                bg.setZValue(-10)
                bg.setAcceptedMouseButtons(QtCore.Qt.NoButton)
                bg.setAcceptHoverEvents(False)
                pi.addItem(bg)
                self.updated_hypno.append(bg)


        
    # --------------------------------------------------------------------------------
    #
    # click Render --> initiate segsrv_t for channel / annotation drawing 
    #
    # --------------------------------------------------------------------------------
    
    def _populate_segsrv(self):
        # compute on separate thread
        # --> do not touch the GUI here
        
        # segsrv options
        throttle1_sr = 100
        self._segsrv_input_throttle_sr = throttle1_sr
        self.ss.input_throttle( throttle1_sr )
        throttle2_np = 10000
        self.ss.throttle( throttle2_np )

        # special version that releases the GIL
        self.ss.segsrv.populate_lunascope( chs = self.ss_chs , anns = self.ss_anns )
        self._apply_backend_filters( throttle1_sr )
        self.ss.set_annot_format6( False ) # pyqtgraph, not plotly
        self.ss.set_clip_xaxes( False )

        # any sig-mods?
        self._render_cmaps()

    def _apply_backend_filters(self, input_throttle_sr=None):
        # Apply currently selected channel filters only after segsrv has
        # channel data loaded; applying earlier can leave stale empty buffers.
        # input_throttle_sr: if segsrv decimated channels via input_throttle(),
        # pass that SR here so filters are designed at the actual stored SR
        # rather than the original header SR.
        self.ss.clear_filters()
        for ch_label, fcode in self.fmap.items():
            if fcode == "None" or ch_label not in self.ss_chs:
                continue
            if ch_label not in self.srs:
                continue

            if fcode == "User":
                frqs = self.user_fmap_frqs.get(ch_label, [])
            else:
                frqs = self.fmap_frqs.get(fcode, [])

            if len(frqs) != 2:
                continue

            orig_sr = float(self.srs[ch_label])
            sr = min(orig_sr, float(input_throttle_sr)) if input_throttle_sr else orig_sr
            if frqs[0] < frqs[1] and frqs[1] <= sr / 2:
                order = 2
                sos = butter(order, frqs, btype='band', fs=sr, output='sos')
                self.ss.apply_filter(ch_label, sos.reshape(-1))
    
        
    def _render_signals(self):
        if getattr(self, "_pg1_probe", None) is not None:
            self._pg1_probe.clear_pinned()

        if not hasattr(self, "p"):
            QMessageBox.critical( self.ui , "Error", "No instance attached" )
            return
        
        # update hypnogram and segment plot
        self._update_hypnogram()

        # copy originally selected channels (i.e. as denominator
        # for subsequently drop in/out)
        self.ss_chs = self.ui.tbl_desc_signals.checked()
        self.ss_anns = self.ui.tbl_desc_annots.checked()

        # set palette
        self.set_palette()
        
        # for a given EDF instance, take selected channels 
        if len( self.ss_chs ) + len( self.ss_anns ) == 0:
            self._set_render_status( False , False )
            return

        # we're now going to have something to plot
        #   rendered and current
        self._set_render_status( True , True)

        # Reinitialize segsrv for each Render so that accumulated C++ state
        # (particularly sigmods registered via make_sigmod) does not grow
        # across repeated renders, which causes std::bad_alloc.
        self.ss = lp.segsrv( self.p )
        if getattr(self, '_psd_visible', False):
            self.ss.set_psd_mode(True)

        # ------------------------------------------------------------
        # do rendering on a separate thread

        # ------------------------------------------------------------
        # execute command string 'cmd' in a separate thread

        # note that we're busy
        self._busy = True

        # and do not let other jobs be run
        self._buttons( False )

        # start progress bar
        self.sb_progress.setVisible(True)
        self.sb_progress.setRange(0, 0) 
        self.sb_progress.setFormat("Running…")
        self.lock_ui()

        # set up call on different thread
        fut_ss = self._exec.submit( self._populate_segsrv )  # returns nothing
                
        def done_segsrv( _f=fut_ss ):
            try:
                exc = _f.exception()
                if exc is None:
                    # self._last_result = _f.result()  # nothing returned
                    QMetaObject.invokeMethod(self, "_segsrv_done_ok", Qt.QueuedConnection)
                else:
                    self._last_exc = exc
                    self._last_tb = f"{type(exc).__name__}: {exc}"
                    QMetaObject.invokeMethod(self, "_segsrv_done_err", Qt.QueuedConnection)
            except Exception as cb_exc:
                self._last_exc = cb_exc
                self._last_tb = f"{type(cb_exc).__name__}: {cb_exc}"
                QMetaObject.invokeMethod(self, "_segsrv_done_err", Qt.QueuedConnection)

        # add the callback
        fut_ss.add_done_callback( done_segsrv )



    @Slot()
    def _segsrv_done_ok(self):        
        try:
            self._complete_rendering()
        finally:
            self.unlock_ui()
            self._busy = False
            self._buttons( True )           
            self.sb_progress.setRange(0, 100); self.sb_progress.setValue(0)
            self.sb_progress.setVisible(False)
            
    @Slot()
    def _segsrv_done_err(self):
        try:
            # show or log the error; pick one
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self.ui, "Error rendering sample", self._last_tb)
        finally:
            self.unlock_ui()
            self._busy = False
            self._buttons( True )
            self.sb_progress.setRange(0, 100); self.sb_progress.setValue(0)
            self.sb_progress.setVisible(False)

     
    def _complete_rendering(self):

        # we can now touch the GUI
        
        # update segment plot
        self._initiate_curves()        
        
        # plot segments
#        num_epochs = self.ss.num_epochs()
#        tscale = self.ss.get_time_scale()
#        tstarts = [ tscale[idx] for idx in range(0,len(tscale),2)]
#        tstops = [ tscale[idx] for idx in range(1,len(tscale),2)]
#        times = np.concatenate((tstarts, tstops), axis=1)

        # ready to view
        self.ss.window( self.last_x1, self.last_x2)

        self._update_scaling()
        if hasattr(self, "_psd_overlay_on_new_data"):
            self._psd_overlay_on_new_data()

        
    def on_window_range(self, lo: float, hi: float):
        if not hasattr(self, "p"):
            return
        if getattr(self, "_pg1_probe", None) is not None:
            self._pg1_probe.clear_pinned()

        # time in seconds now
        if lo < 0: lo = 0
        if hi > self.ns: hi = self.ns 
        if hi < lo: hi = lo
        
        # update ss window
        t1 = ""
        t2 = ""        
        if self.rendered is True:
            if getattr(self, "ss", None) is None:
                return
            self.ss.window( lo  , hi )
            t1 = self.ss.get_window_left_hms()
            t2 = self.ss.get_window_right_hms()
        else: # annot only segsrv
            # In non-render mode with signals selected, cap window to avoid overload.
            chs = self.ui.tbl_desc_signals.checked()
            max_simple_span = 3600.0 if getattr(self, 'multiday_mode', False) else 30.0
            annotator_interval_active = self._annotator_interval_capture_active()
            if (not annotator_interval_active) and len(chs) != 0 and (hi - lo) > max_simple_span:
                prev_lo = getattr(self, "last_x1", None)
                prev_hi = getattr(self, "last_x2", None)
                prev_span = None if (prev_lo is None or prev_hi is None) else (prev_hi - prev_lo)

                if prev_span is not None and abs(prev_span - max_simple_span) < 1e-6:
                    lo, hi = prev_lo, prev_hi
                else:
                    c = 0.5 * (lo + hi)
                    lo, hi = c - 0.5 * max_simple_span, c + 0.5 * max_simple_span
                    if lo < 0:
                        lo, hi = 0.0, max_simple_span
                    if hi > self.ns:
                        hi = float(self.ns)
                        lo = max(0.0, hi - max_simple_span)

                if getattr(self, "sel", None) is not None:
                    self.sel.setRange(lo, hi, emit=False)

            if getattr(self, "ssa", None) is None:
                return
            self.ssa.window( lo  , hi )
            t1 = self.ssa.get_window_left_hms()
            t2 = self.ssa.get_window_right_hms()

        self.ui.lbl_twin.setText( f"T: {t1} - {t2}" )
        if getattr(self, 'multiday_mode', False):
            self.ui.lbl_ewin.setText(f"Day: {int(lo/86400)+1} - {int(hi/86400)+1}")
        else:
            self.ui.lbl_ewin.setText(f"E: {int(lo/30)+1} - {int(hi/30)+1}")
        self._update_pg1()
        if hasattr(self, "_annotator_on_window_range_changed"):
            self._annotator_on_window_range_changed(lo, hi)
        if hasattr(self, "_psd_overlay_on_range_changed"):
            self._psd_overlay_on_range_changed()





    # --------------------------------------------------------------------------------
    #
    # pre-Render plot set up (called on attaching the plot) 
    #
    # --------------------------------------------------------------------------------
    
    def _render_signals_simple(self):

        # update hypnogram and segment plot
        self._update_hypnogram()
        
        # get all checked channels
        self.ss_chs = self.ui.tbl_desc_signals.checked()
        self.ss_anns = self.ui.tbl_desc_annots.checked()

        # set palette
        self.set_palette()
        
        # initiate curves 
        self._initiate_curves()

        # ready view — preserve current position (defaults 0,30 on first load)
        self.ssa.window(self.last_x1, self.last_x2)
        self._update_scaling()
        self._update_pg1_simple()
        if hasattr(self, "_psd_overlay_on_new_data"):
            self._psd_overlay_on_new_data()
        




    # --------------------------------------------------------------------------------
    #
    # set up traces
    #
    # --------------------------------------------------------------------------------
    
    def _initiate_curves(self):


        #
        # get (and order) display items
        #

        self.ss_chs = self.ui.tbl_desc_signals.checked()

        self.ss_anns = self.ui.tbl_desc_annots.checked()

        # re-order channels, annots?
        if self.cmap_list:
            self.ss_chs = sorted( self.ss_chs, key=lambda x: (self.cmap_rlist.index(x) if x in self.cmap_rlist else len(self.cmap_rlist) + self.ss_chs.index(x)))
            self.ss_anns = sorted( self.ss_anns, key=lambda x: (self.cmap_list.index(x) if x in self.cmap_list else len(self.cmap_list) + self.ss_anns.index(x)))

        nchan = len( self.ss_chs )

        nann = len( self.ss_anns )

        nmods = len( self.sigmods ) 
        
        #
        # clear prior items
        #

        pi = self.ui.pg1.getPlotItem()
        pi.clear()
        if hasattr(self, "_annot_queue_on_curves_reinit"):
            self._annot_queue_on_curves_reinit()

        for curve in self.curves:
            pi.removeItem(curve)
        self.curves.clear()

        for curve in self.y0_curves:
            pi.removeItem(curve)
        self.y0_curves.clear()

        for curve in self.y_curves:
            pi.removeItem(curve)
        self.y_curves.clear()

        for curve in self.sigmod_curves:
            pi.removeItem(curve)
        self.sigmod_curves.clear()

        for curve in self.annot_curves:
            pi.removeItem(curve)
        self.annot_curves.clear()

        
        #
        # initiate channels
        #
        
        for i in range(nchan):
            pen = pg.mkPen( self.colors[i], width= self.cfg_line_weight , cosmetic=True )
            c = pg.PlotCurveItem(pen=pen, connect='finite')
            pi.addItem(c)
            self.curves.append(c)

        #
        # y=0 lines
        #

        for i in range(nchan):
            base_col = self.colors[i] if i < len(self.colors) else 'gray'
            pen = pg.mkPen(base_col, width=2, cosmetic=True)
            pen.setDashPattern([8, 4])
            c = pg.PlotCurveItem(pen=pen, connect='finite')
            pi.addItem(c)
            self.y0_curves.append(c)

        #
        # y=x lines
        #
        ycol = 'orange'
        if self.palset == 'white' or self.palset == 'muted':
            ycol = 'black'
        for i in range(self.cmap_n_ylines):
            pen = pg.mkPen( ycol , width=1, cosmetic=True )
            pen.setDashPattern([4, 8])
            c = pg.PlotCurveItem(pen=pen, connect='finite')
            pi.addItem(c)
            self.y_curves.append(c)

        #
        # initiate sigmod curves (18 per channel) [ch1 x 18, ch2 x 18 , ... ] 
        #

        self.sigmod_curve_colors = []
        for ch, mod_spec in self.sigmods.items():
            pal_label = mod_spec.get('pal', 'rwb')
            colors = self.sigmod_colors.get(pal_label, self.rwb_sigmod_colors)
            for j in range(18):
                col = colors[j % len(colors)]
                pen = pg.mkPen(col, width=self.cfg_line_weight, cosmetic=True)
                c = pg.PlotCurveItem(pen=pen, connect='finite')
                pi.addItem(c)
                self.sigmod_curves.append(c)
                self.sigmod_curve_colors.append(col)
        
        
        #
        # initiate annotations
        #

        self.annot_mgr = TrackManager( self.ui.pg1 )
        
        for i in range(nann):
            col = self.acolors[i]
            self.annot_mgr.update_track( self.ss_anns[i] , [] , [], [], [] , color = col )
            pen = pg.mkPen( col, width=1, cosmetic=True)
            c = pg.PlotCurveItem(pen=pen, connect='finite')
            pi.addItem(c)
            self.annot_curves.append(c)

        #
        # initiate gaps
        #

        self.annot_mgr.update_track( "__#gaps__"  ,
                                     [] , [], [], [] ,
                                     color = (0,25,25) ,
                                     pen = pg.mkPen((200, 200, 200), width=1) )
        
        #
        # initiate ticks
        #

        self.tb = TextBatch(pi.vb, QtGui.QFont("Arial", 12), color=(180,255,255), mode='device')
        self.tb.setZValue(10)
        self.tb.setData([ ], [ ], [ ])
        self.ui.pg1.addItem(self.tb)# , ignoreBounds=True)

        #
        # initiate labels
        #

        self.labs = TextBatch(pi.vb,
               QtGui.QFont("Arial", 10, QtGui.QFont.Normal),
               color=(255,255,255),
               mode='device',
               bg=(0,0,0,70),    # semi-transparent black (was 170)
               pad=(6,3),         # x/y padding in px
               radius=20,          # rounded corners
               outline= (255,255,255,80) )      # or (255,255,255,80)
        
        self.labs.setZValue(10)
        self.labs.setData([ ], [ ], [ ])
        self.ui.pg1.addItem(self.labs)# , ignoreBounds=True)

        # day boundary lines on main signal canvas (multi-day mode only)
        self._day_lines_pg1 = []
        if getattr(self, 'multiday_mode', False):
            for _t in self._compute_day_boundaries():
                _line = pg.InfiniteLine(
                    pos=_t,
                    angle=90,
                    pen=self._day_delimiter_pen(navigator=False),
                )
                pi.addItem(_line)
                self._day_lines_pg1.append(_line)

        self._pg1_channel_cache = []
        self._init_pg1_probe_items()
        self._init_pg1_annot_selector()
        self._update_pg1_probe_styles()
        

    # --------------------------------------------------------------------------------
    #
    # labels
    #
    # --------------------------------------------------------------------------------

    def _update_labels(self):

        # labels?
        if self.ui.check_labels.isChecked():
            self.show_labels = True
        else:
            self.show_labels = False
            
        # redraw
        self._update_pg1()

    # --------------------------------------------------------------------------------
    #
    # handle y-axis scaling
    #
    # --------------------------------------------------------------------------------

    def _update_scaling(self):

        self.pg1_header_height = 0.05

        self.pg1_footer_height = 0.025

        if len(self.ss_anns) == 0:
            self.pg1_annot_height = 0
        else:
            self.pg1_annot_height = min( 0.3 , 0.10 + len(self.ss_anns) * 0.015 ) 

        if len(self.ss_chs) == 0:
            self.pg1_annot_height = 0.8

        # if cmap values specifed, use those
        ch_set = [ ]
        for ch in self.ss_chs:
            if ch in self.cmap_fixed_min:
                self.ss.fix_physical_scale( ch , self.cmap_fixed_min[ch] , self.cmap_fixed_max[ch] )
                ch_set.append( ch )
            
        # use empirical vals (default) 
        if self.ui.radio_empiric.isChecked():
            for ch in self.ss_chs:
                if ch not in ch_set:
                    self.ss.empirical_physical_scale( ch )

            # & turn off other fixed scale , if set
            if self.ui.radio_fixedscale.isChecked() :
                with QSignalBlocker(self.ui.radio_fixedscale):
                    self.ui.radio_fixedscale.setChecked(False)

        elif self.ui.radio_fixedscale.isChecked():
            lwr = self.ui.spin_fixed_min.value()
            upr = self.ui.spin_fixed_max.value()
            if lwr >= upr:   # degenerate range: fall back to unit defaults
                lwr = -1
                upr = +1
            for ch in self.ss_chs:
                if ch not in ch_set:
                    self.ss.fix_physical_scale( ch , lwr, upr )
        else:
            for ch in self.ss_chs:
                if ch not in ch_set:
                    self.ss.free_physical_scale( ch )

        self.clip_signals = self.ui.radio_clip.isChecked()
        
        ns = len( self.ui.tbl_desc_signals.checked() )

        na = len( self.ui.tbl_desc_annots.checked() )

        yscale = 2**float( self.ui.spin_scale.value() )
        yspacing = float( self.ui.spin_spacing.value() )

        # if not annotations, take up entire screen for annots
        if ns != 0:
            yannot = self.pg1_annot_height
        else:
            yannot = 1 - self.pg1_footer_height - self.pg1_header_height 

        # update scaling (either for ss or ssa in simple rendering)

        if self.rendered is True:
            self.ss.set_scaling( ns, na,  yscale , yspacing ,
                                 self.pg1_header_height,
                                 self.pg1_footer_height ,
                                 yannot ,
                                 self.clip_signals )
        else:
            self.ssa.set_scaling( ns, na,  yscale , yspacing ,
                                  self.pg1_header_height,
                                  self.pg1_footer_height ,
                                  yannot ,
                                  self.clip_signals )


        # update main plot (passes to _update_pg1_simple() as needed)
            
        self._update_pg1()



        
    # --------------------------------------------------------------------------------
    #
    # clear main curves
    #
    # --------------------------------------------------------------------------------

    def _clear_pg1(self):

        pi = self.ui.pg1.getPlotItem()
        pi.clear() 

        for curve in self.curves:
            pi.removeItem(curve)
        self.curves.clear()

        for curve in self.y0_curves:
            pi.removeItem(curve)
        self.y0_curves.clear()

        for curve in self.y_curves:
            pi.removeItem(curve)
        self.y_curves.clear()

        for curve in self.annot_curves:
            pi.removeItem(curve)
        self.annot_curves.clear()

        self.set_palette()
        
        self._initiate_curves()
        
    
    # --------------------------------------------------------------------------------
    #
    # update main signal traces
    #
    # --------------------------------------------------------------------------------


    

    
    def _update_pg1(self):
        if getattr(self, "_pg1_probe", None) is not None:
            self._pg1_probe.clear_pinned()

        if self.rendered is not True:
            self._update_pg1_simple()
            return

        # channels
        chs = self.ui.tbl_desc_signals.checked()
        chs = [x for x in self.ss_chs if x in chs ] 

        # sigmod channels
        sigmods = {k: v for k, v in self.sigmods.items() if k in self.ss_chs}

        # annots
        anns = self.ui.tbl_desc_annots.checked()
        anns = [x for x in self.ss_anns if x in anns ]
            
        # window (sec)
        x1 = self.ss.get_window_left()
        x2 = self.ss.get_window_right()

        # store for any updates
        self.last_x1 = x1
        self.last_x2 = x2

        # get canvas
        pw = self.ui.pg1
        vb = pw.getPlotItem().getViewBox()

        # window (pixels)
        self.ss.segsrv.set_pixel_width( int( vb.width() ) )
        
        # set range (x-axis)
        vb.setRange(xRange=(x1,x2), padding=0, update=False)  # no immediate paint

        # ch-ordering? (based on index
        if self.cmap_list:
            chs = sorted( chs, key=lambda x: (self.cmap_list.index(x) if x in self.cmap_list else len(self.cmap_list) + chs.index(x)))
            anns = sorted( anns, key=lambda x: (self.cmap_list.index(x) if x in self.cmap_list else len(self.cmap_list) + anns.index(x)))

        
        # channels
        nchan = len( chs )
        idx = 0
        self._pg1_channel_cache = []
        sigmod_idx = 0 # n(ch) x 18
        tv = [ '' ] * ( len(chs) + len(anns) )
        yv = [ 0.5 ] * ( len(chs) + len(anns) )
        xv = [  x1 + ( x2 - x1 ) * 0.02 ] * ( len(chs) + len(anns) )
        for ch in chs:
            x = None
            y = None

            # y0 lines
            curve_slot = nchan - idx - 1
            curve_color = self.colors[curve_slot] if curve_slot < len(self.colors) else 'gray'
            self._set_y0_curve_pen(idx, curve_color)
                                    
            # y-lines
            if ch in self.cmap_ylines_idx:
                yidx = self.cmap_ylines_idx[ ch ]
                yval = self.cmap_ylines[ ch ]
                for i in range( len( yidx ) ):
                    yl = self.ss.get_scaled_y( ch , yval[i] )
                    self.y_curves[ yidx[i] ].setData([ x1, x2 ], [ yl , yl ])
                    
            # todo: check if we need reverse idx'ing, as per nchan
            if ch in sigmods:
                self.ss.apply_sigmod( self.sigmods[ ch ][ 'mod' ] , ch , chs.index( ch ) )
                self.curves[nchan-idx-1].setData( [ ] , [ ] )
                for b in range(18):
                    # print( 'sigmod ' , ch , b )
                    tx1 = self.ss.get_sigmod_timetrack( b )
                    ty1 = self.ss.get_sigmod_scaled_signal( b )
                    self.sigmod_curves[sigmod_idx].setData(tx1, ty1)
                    sigmod_idx = sigmod_idx + 1
            else: # regular channel
                # signals
                x = self.ss.get_timetrack( ch )
                y = self.ss.get_scaled_signal( ch , idx )
                # note: if filters set, these will have been passed to segsrv, which will
                #       take care of filtering in the above call
                # draw
                self.curves[nchan-idx-1].setData(x, y)

            ylim = self.ss.get_window_phys_range( ch )
            band0 = self.ss.get_scaled_y(ch, ylim[0])
            band1 = self.ss.get_scaled_y(ch, ylim[1])
            if self.cfg_show_zero_line:
                y0 = self.ss.get_scaled_y(ch, 0)
                band_lo = min(band0, band1)
                band_hi = max(band0, band1)
                if band_lo <= y0 <= band_hi:
                    self.y0_curves[idx].setData([x1, x2], [y0, y0])
                else:
                    self.y0_curves[idx].setData([], [])
            else:
                self.y0_curves[idx].setData([], [])
            self._cache_pg1_channel_band(
                ch=ch,
                band_lo=min(band0, band1),
                band_hi=max(band0, band1),
                phys_lo=float(ylim[0]),
                phys_hi=float(ylim[1]),
                x=x,
                y_scaled=y,
                zero_scaled=self.ss.get_scaled_y(ch, 0),
                color=curve_color,
                srv=self.ss,
            )
            if self.show_labels:
                tv[idx] = ' ' + ch + ' ' + str(round(ylim[0],3)) + ' : ' + str(round(ylim[1],3)) + ' (' + self.units[ ch ] +')'
            yv[idx] = self.ss.get_ylabel( idx )
            # next
            idx = idx + 1

        # Clear curve slots for channels that were rendered but are now deselected.
        # Without this, deselected channels leave ghost traces on screen.
        for i in range(nchan, len(self.curves)):
            self.curves[i].setData([], [])
            self.y0_curves[i].setData([], [])
        for i in range(sigmod_idx, len(self.sigmod_curves)):
            self.sigmod_curves[i].setData([], [])

        # annots
        aidx = 0
        self.ss.compile_windowed_annots( anns )
        for ann in anns:
            a0 = self.ss.get_annots_xaxes( ann )            
            if len(a0) == 0:
                self.annot_curves[ aidx ].setData( [ ] , [ ] )
                idx = idx + 1
                aidx = aidx + 1
                continue
            a1 = self.ss.get_annots_xaxes_ends( ann )            
            y0 = self.ss.get_annots_yaxes( ann )
            y1 = self.ss.get_annots_yaxes_ends( ann )
            self.annot_curves[aidx].setData( [ x1 , x2 ] , [ ( y0[0] + y1[0] ) / 2  , ( y0[0] + y1[0] ) / 2 ] )
            a0, a1 = _ensure_min_px_width( vb, a0, a1, px=1)  # 1-px minimum
            self.annot_mgr.update_track( ann , x0 = a0 , x1 = a1 , y0 = y0 , y1 = y1 , reduce = True , wpx = float(getattr(self, 'cfg_line_weight', 1.0)))
            # labels
            yv[idx] = ( y0[0] * 2 + y1[0]  ) / 3.0
            if self.show_labels: 
                if ann and str(ann).strip():
                    tv[idx] = ann
            idx = idx + 1
            aidx = aidx + 1

        xv2, yv2, tv2 = [], [], []
        for x, y, t in zip(xv, yv, tv):
            if t and str(t).strip():  # keep only non-empty labels
                xv2.append(x)
                yv2.append(y)
                tv2.append(t)

        self.labs.setData(xv2, yv2, tv2)

        # gaps (list of (start,stop) values
        gaps = self.ss.get_gaps()
        gx0 =  [ x[0] for x in gaps ]
        gx1 =  [ x[1] for x in gaps ]
        gy0 =  [ 0.01 for x in gaps ]
        gy1 =  [ 0.96 for x in gaps ]
        gaps = self.annot_mgr.update_track( "__#gaps__" ,x0 = gx0 , x1 = gx1 , y0 = gy0 , y1 = gy1 )
            
        # clock-ticks                                                                                                          
        tx1 = self.ss.get_window_left()
        tx2 = self.ss.get_window_right()
        tks = self.ss.get_clock_ticks(6, multiday=self.multiday_mode)
        tx = list( tks.keys() )
        tv = self._format_clock_tick_labels(tx, list(tks.values()), tx1, tx2)
        ty = [ 0.99 ] * len( tx )
        tv.append( self._durstr( tx1 , tx2 ) )
        tx.append( tx2 - 0.05 * ( tx2 - tx1 ) )
        ty.append( 0.03 )
        self.tb.setData(tx, ty , tv )

        self._hide_pg1_probe()

        # repaint
        vb.update()
        if hasattr(self, "_psd_overlay_on_trace_redraw"):
            self._psd_overlay_on_trace_redraw()
        if hasattr(self, "_annot_queue_overlay_on_trace_redraw"):
            self._annot_queue_overlay_on_trace_redraw()


    def _durstr( self , x , y ):
        d = y - x
        if d < 60: return str(int(d))+'s'
        d = d/60
        if d < 60: return str(int(d))+'m'
        d = d/60
        return format(d, ".1f")+'h'

    def _format_clock_tick_labels(self, tick_positions, fallback_labels, x1, x2):
        if getattr(self, 'multiday_mode', False):
            return fallback_labels

        show_seconds = (x2 - x1) <= 3600.0
        start_tod = int(getattr(self, '_record_start_tod_secs', 0))
        labels = []
        for tick, fallback in zip(tick_positions, fallback_labels):
            try:
                total = (start_tod + int(round(float(tick)))) % 86400
            except (TypeError, ValueError):
                labels.append(fallback)
                continue

            hh = total // 3600
            mm = (total % 3600) // 60
            ss = total % 60
            labels.append(f"{hh:02d}:{mm:02d}:{ss:02d}" if show_seconds else f"{hh:02d}:{mm:02d}")
        return labels

    def _compute_day_boundaries(self):
        """Return list of seconds-from-record-start for each day anchor boundary."""
        anchor_secs = getattr(self, 'cfg_day_anchor', 12) * 3600
        start_tod   = getattr(self, '_record_start_tod_secs', 0)
        secs_to_first = (anchor_secs - start_tod) % 86400
        boundaries, t = [], secs_to_first
        while t < self.ns:
            boundaries.append(t)
            t += 86400
        return boundaries

    def _day_delimiter_pen(self, navigator: bool):
        alpha = 120 if navigator else 90
        return pg.mkPen(
            (255, 255, 180, alpha),
            width=2,
            style=QtCore.Qt.DashLine,
            cosmetic=True,
        )
    
    # --------------------------------------------------------------------------------
    #
    # simple (non-segsrv) update main signal traces - called if segsrv not populated
    # restrict to single epoch plotting here
    # --------------------------------------------------------------------------------

    def _update_pg1_simple(self):
        if getattr(self, "_pg1_probe", None) is not None:
            self._pg1_probe.clear_pinned()

        # get epoch 'e' for channel 'ch =' w/ time
        # p.slice( p.e2i( 1 ) ,  chs = ['C3'] , time = True ) 
        # --> tuple x[0] header; x[1] nparray
        # --> window in timepoints:  p.e2i( 1 ) 

        # use self.ssa segsrv for annotations and mapping

        # channels
        chs = self.ui.tbl_desc_signals.checked()

        # annots
        anns = self.ui.tbl_desc_annots.checked()

        # window (sec)
        x1 = self.ssa.get_window_left()
        x2 = self.ssa.get_window_right()

        # Guard: if signals are checked and window exceeds the non-render max
        # (e.g. zoomed out to whole night while viewing annotations only, then
        # a channel is checked on), snap back to a 30s epoch at the left edge
        # before attempting to load data.
        max_simple_span = 3600.0 if getattr(self, 'multiday_mode', False) else 30.0
        if len(chs) > 0 and (x2 - x1) > max_simple_span:
            x2 = x1 + max_simple_span
            ns = getattr(self, 'ns', None)
            if ns is not None and x2 > ns:
                x2 = float(ns)
                x1 = max(0.0, x2 - max_simple_span)
            self.ssa.window(x1, x2)
            if getattr(self, "sel", None) is not None:
                self.sel.setRange(x1, x2, emit=False)

        # store for any updates
        self.last_x1 = x1
        self.last_x2 = x2

        # get canvas
        pw = self.ui.pg1
        vb = pw.getPlotItem().getViewBox()
        vb.setRange(xRange=(x1,x2), padding=0, update=False)  # no immediate paint

        # scaling
        h = 1 - self.pg1_header_height - self.pg1_footer_height - self.pg1_annot_height
        if len(chs) != 0:
            h = h / len(chs) 
        else:
            h = 0

        # re-order channels, annots?
        if self.cmap_list:
            chs = sorted( chs, key=lambda x: (self.cmap_list.index(x) if x in self.cmap_list else len(self.cmap_list) + chs.index(x)))
            anns = sorted( anns, key=lambda x: (self.cmap_list.index(x) if x in self.cmap_list else len(self.cmap_list) + anns.index(x)))
            chs.reverse()
            
        # channels
        idx = 0        
        self._pg1_channel_cache = []
        tv = [ '' ] * ( len(chs) + len(anns) )
        yv = [ 0.5 ] * ( len(chs) + len(anns) )
        xv = [ x1 + ( x2 - x1 ) * 0.02 ] * ( len(chs) + len(anns) )
        for ch in chs:
            # signals
            d = self.p.slice( self.p.s2i( [ ( x1 , x2 ) ] ) , chs = ch , time = True )[1]
            curve_color = self.colors[idx] if idx < len(self.colors) else 'gray'
            self._set_y0_curve_pen(idx, curve_color)
            # no data, e.g. in gap?
            if len(d) == 0:
                if idx < len(self.y0_curves):
                    self.y0_curves[idx].setData([], [])
                idx = idx + 1
                continue
            x = d[:,0]  # time-track
            y = d[:,1]  # unscaled signal
            # filter?
            if ch in self.fmap:
                y = self.filter_signal( y , ch, ( self.fmap[ch] , self.srs[ ch ] ) )
            y_phys_filt = y.copy()   # physical units, post-filter — used by PSD overlay
            # need to scale manually: to 0/1, either empirically, or from fixed
            if ch in self.cmap_fixed_min:
                mn = self.cmap_fixed_min[ch]
                mx = self.cmap_fixed_max[ch]
            else:
                mn, mx = min(y), max(y)
            if mx > mn: y = (y - mn) / (mx - mn)
            else: y = y - y
            # --> to grid value
            ybase = idx * h + self.pg1_footer_height
            y = ybase + y * h 
            # plot
            self.curves[idx].setData(x, y)
            if self.cfg_show_zero_line:
                if mx > mn:
                    y0 = ybase + ((0.0 - mn) / (mx - mn)) * h
                    if ybase <= y0 <= (ybase + h):
                        self.y0_curves[idx].setData([x1, x2], [y0, y0])
                    else:
                        self.y0_curves[idx].setData([], [])
                else:
                    self.y0_curves[idx].setData([], [])
            else:
                self.y0_curves[idx].setData([], [])
            self._cache_pg1_channel_band(
                ch=ch,
                band_lo=ybase,
                band_hi=ybase + h,
                phys_lo=float(mn),
                phys_hi=float(mx),
                x=x,
                y_scaled=y,
                y_phys=y_phys_filt,
                zero_scaled=(ybase + ((0.0 - mn) / (mx - mn)) * h) if mx > mn else (ybase + 0.5 * h),
                color=curve_color,
            )
            # labels
            ylim = [ mn , mx ] 
            if self.show_labels:
                tv[idx] = ' ' + ch + ' ' + str(round(ylim[0],3)) + ' : ' + str(round(ylim[1],3)) + ' (' + self.units[ ch ] +')'
            yv[idx] = ybase + 0.5 * h
            # next
            idx = idx + 1

        for i in range(idx, len(self.y0_curves)):
            self.y0_curves[i].setData([], [])

        # annots (from ssa)
        aidx = 0
        self.ssa.compile_windowed_annots( anns )
        
        for ann in anns:

            # get events
            a0 = self.ssa.get_annots_xaxes( ann )            

            # nothing to do?
            if len(a0) == 0:
                self.annot_curves[ aidx ].setData( [ ] , [ ] )
                idx = idx + 1
                aidx = aidx + 1                
                continue

            # pull
            a1 = self.ssa.get_annots_xaxes_ends( ann )
            y0 = self.ssa.get_annots_yaxes( ann )
            y1 = self.ssa.get_annots_yaxes_ends( ann )
           
            # draw
            self.annot_curves[ aidx ].setData( [ x1 , x2 ] , [ ( y0[0] + y1[0] ) / 2  , ( y0[0] + y1[0] ) / 2 ] ) 
            a0, a1 = _ensure_min_px_width( vb, a0, a1, px=1)  # 1-px minimum
            self.annot_mgr.update_track( ann , x0 = a0 , x1 = a1 , y0 = y0 , y1 = y1 , reduce = True , wpx = float(getattr(self, 'cfg_line_weight', 1.0)))

            # labels
            yv[idx] = ( y0[0] * 2 + y1[0]  ) / 3.0 
            if self.show_labels: tv[idx] = ann

            # next annot
            idx = idx + 1
            aidx = aidx + 1

            
        # add labels
        # filter out empty/blank labels before drawing
        xv2, yv2, tv2 = [], [], []
        for x, y, t in zip(xv, yv, tv):
            if t and str(t).strip():  # keep only non-empty labels
                xv2.append(x)
                yv2.append(y)
                tv2.append(t)

        self.labs.setData(xv2, yv2, tv2)

        # gaps (list of (start,stop) values
        gaps = self.ssa.get_gaps()
        x0 =  [ x[0] for x in gaps ]
        x1 =  [ x[1] for x in gaps ]
        y0 =  [ 0.01 for x in gaps ]
        y1 =  [ 0.96 for x in gaps ]
        gaps = self.annot_mgr.update_track( "__#gaps__" ,x0 = x0 , x1 = x1 , y0 = y0 , y1 = y1 )
            
        # clock-ticks
        x1 = self.ssa.get_window_left()
        x2 = self.ssa.get_window_right()
        tks = self.ssa.get_clock_ticks(6, multiday=self.multiday_mode)
        tx = list( tks.keys() )
        tv = self._format_clock_tick_labels(tx, list(tks.values()), x1, x2)
        ty = [ 0.99 ] * len( tx )
        tv.append( self._durstr( x1 , x2 ) )
        tx.append( x2 - 0.05 * ( x2 - x1 ) )
        ty.append( 0.03 )
        self.tb.setData(tx, ty , tv )

        self._hide_pg1_probe()

        # repaint
        vb.update()
        if hasattr(self, "_psd_overlay_on_trace_redraw"):
            self._psd_overlay_on_trace_redraw()
        if hasattr(self, "_annot_queue_overlay_on_trace_redraw"):
            self._annot_queue_overlay_on_trace_redraw()

    def _init_pg1_probe_items(self):
        pi = self.ui.pg1.getPlotItem()
        self._pg1_probe_line = pg.InfiniteLine(
            pos=0.0,
            angle=90,
            pen=pg.mkPen((255, 235, 120, 190), width=1, cosmetic=True),
        )
        self._pg1_probe_line.setZValue(40)
        self._pg1_probe_line.hide()
        pi.addItem(self._pg1_probe_line)

        self._pg1_probe_start_sample_line = pg.PlotCurveItem(
            pen=pg.mkPen((120, 210, 255, 230), width=2, cosmetic=True)
        )
        self._pg1_probe_start_sample_line.setZValue(41)
        self._pg1_probe_start_sample_line.hide()
        pi.addItem(self._pg1_probe_start_sample_line)

        self._pg1_probe_sample_line = pg.PlotCurveItem(
            pen=pg.mkPen((255, 235, 120, 230), width=2, cosmetic=True)
        )
        self._pg1_probe_sample_line.setZValue(41)
        self._pg1_probe_sample_line.hide()
        pi.addItem(self._pg1_probe_sample_line)

        self._pg1_probe_span_line = pg.PlotCurveItem(
            pen=pg.mkPen((180, 220, 255, 210), width=2, style=QtCore.Qt.DotLine, cosmetic=True)
        )
        self._pg1_probe_span_line.setZValue(41)
        self._pg1_probe_span_line.hide()
        pi.addItem(self._pg1_probe_span_line)

        self._pg1_probe_label = pg.TextItem(
            text="",
            color=(255, 255, 255),
            border=pg.mkPen((255, 255, 255, 80)),
            fill=pg.mkBrush(0, 0, 0, 170),
            anchor=(0, 1),
        )
        self._pg1_probe_label.setZValue(42)
        self._pg1_probe_label.hide()
        pi.addItem(self._pg1_probe_label)

        self._pg1_probe_legend = pg.TextItem(
            text="",
            color=(220, 220, 220),
            border=pg.mkPen((255, 255, 255, 50)),
            fill=pg.mkBrush(0, 0, 0, 130),
            anchor=(0.5, 1),
        )
        self._pg1_probe_legend.setZValue(42)
        self._pg1_probe_legend.hide()
        pi.addItem(self._pg1_probe_legend)

        self._pg1_probe_grid_lines = pg.PlotCurveItem(
            pen=pg.mkPen((255, 255, 255, 110), width=1, style=QtCore.Qt.DotLine, cosmetic=True),
            connect='pairs',
        )
        self._pg1_probe_grid_lines.setZValue(38)
        self._pg1_probe_grid_lines.hide()
        pi.addItem(self._pg1_probe_grid_lines)

        self._pg1_probe_zero_lines = pg.PlotCurveItem(
            pen=pg.mkPen((140, 255, 180, 170), width=1, style=QtCore.Qt.DotLine, cosmetic=True),
            connect='pairs',
        )
        self._pg1_probe_zero_lines.setZValue(40)
        self._pg1_probe_zero_lines.hide()
        pi.addItem(self._pg1_probe_zero_lines)

        self._pg1_probe_zero_baseline = pg.PlotCurveItem(
            pen=pg.mkPen((140, 255, 180, 210), width=1, cosmetic=True)
        )
        self._pg1_probe_zero_baseline.setZValue(40)
        self._pg1_probe_zero_baseline.hide()
        pi.addItem(self._pg1_probe_zero_baseline)

        self._pg1_probe_peak_max = pg.ScatterPlotItem(
            size=8,
            pen=pg.mkPen((255, 210, 120, 220)),
            brush=pg.mkBrush(255, 210, 120, 180),
            symbol='t',
        )
        self._pg1_probe_peak_max.setZValue(41)
        self._pg1_probe_peak_max.hide()
        pi.addItem(self._pg1_probe_peak_max)

        self._pg1_probe_peak_min = pg.ScatterPlotItem(
            size=8,
            pen=pg.mkPen((120, 210, 255, 220)),
            brush=pg.mkBrush(120, 210, 255, 180),
            symbol='t1',
        )
        self._pg1_probe_peak_min.setZValue(41)
        self._pg1_probe_peak_min.hide()
        pi.addItem(self._pg1_probe_peak_min)

        self._pg1_probe_band_lines = []
        for _ in range(64):
            line = pg.PlotCurveItem(
                pen=pg.mkPen((255, 255, 255, 90), width=1, style=QtCore.Qt.DotLine, cosmetic=True)
            )
            line.setZValue(39)
            line.hide()
            pi.addItem(line)
            self._pg1_probe_band_lines.append(line)

    def _pg1_probe_dark_bg(self):
        palset = getattr(self, "palset", "spectrum")
        if palset in ("white", "muted"):
            return False
        if palset == "user":
            try:
                col = pg.mkColor(getattr(self, "c1", "#101010"))
                lum = 0.2126 * col.red() + 0.7152 * col.green() + 0.0722 * col.blue()
                return lum < 160
            except Exception:
                return True
        return True

    def _update_pg1_probe_styles(self):
        dark_bg = self._pg1_probe_dark_bg()
        grid_col = (255, 255, 255, 110) if dark_bg else (0, 0, 0, 95)
        band_col = (255, 255, 255, 90) if dark_bg else (0, 0, 0, 75)
        legend_fg = (220, 220, 220) if dark_bg else (30, 30, 30)
        legend_fill = (0, 0, 0, 130) if dark_bg else (255, 255, 255, 180)
        legend_border = (255, 255, 255, 50) if dark_bg else (0, 0, 0, 60)
        label_fg = (255, 255, 255) if dark_bg else (20, 20, 20)
        label_fill = (0, 0, 0, 170) if dark_bg else (255, 255, 255, 215)
        label_border = (255, 255, 255, 80) if dark_bg else (0, 0, 0, 90)

        if self._pg1_probe_grid_lines is not None:
            self._pg1_probe_grid_lines.setPen(pg.mkPen(grid_col, width=1, style=QtCore.Qt.DotLine, cosmetic=True))
        if self._pg1_probe_zero_lines is not None:
            self._pg1_probe_zero_lines.setPen(pg.mkPen((140, 255, 180, 170) if dark_bg else (0, 120, 60, 180), width=1, style=QtCore.Qt.DotLine, cosmetic=True))
        if getattr(self, "_pg1_probe_zero_baseline", None) is not None:
            self._pg1_probe_zero_baseline.setPen(pg.mkPen((140, 255, 180, 210) if dark_bg else (0, 120, 60, 210), width=1, cosmetic=True))
        if self._pg1_probe_label is not None:
            self._pg1_probe_label.setColor(label_fg)
            self._pg1_probe_label.fill = pg.mkBrush(*label_fill)
            self._pg1_probe_label.border = pg.mkPen(label_border)
        if self._pg1_probe_legend is not None:
            self._pg1_probe_legend.setColor(legend_fg)
            self._pg1_probe_legend.fill = pg.mkBrush(*legend_fill)
            self._pg1_probe_legend.border = pg.mkPen(legend_border)
        for line in self._pg1_probe_band_lines:
            line.setPen(pg.mkPen(band_col, width=1, style=QtCore.Qt.DotLine, cosmetic=True))

    def _cache_pg1_channel_band(self, ch, band_lo, band_hi, phys_lo, phys_hi, x=None, y_scaled=None, y_phys=None, zero_scaled=None, color=None, srv=None):
        x_arr = np.asarray(x, dtype=float) if x is not None else np.empty(0, dtype=float)
        y_scaled_arr = np.asarray(y_scaled, dtype=float) if y_scaled is not None else np.empty(0, dtype=float)
        y_phys_arr = np.asarray(y_phys, dtype=float) if y_phys is not None else None

        finite = np.isfinite(x_arr) & np.isfinite(y_scaled_arr)
        if y_phys_arr is not None:
            finite &= np.isfinite(y_phys_arr)

        if np.any(finite):
            valid_idx = np.flatnonzero(finite)
            x_arr = x_arr[valid_idx]
            y_scaled_arr = y_scaled_arr[valid_idx]
            if y_phys_arr is not None:
                y_phys_arr = y_phys_arr[valid_idx]
        else:
            x_arr = np.empty(0, dtype=float)
            y_scaled_arr = np.empty(0, dtype=float)
            y_phys_arr = np.empty(0, dtype=float) if y_phys_arr is not None else None

        self._pg1_channel_cache.append({
            "ch": ch,
            "band_lo": float(min(band_lo, band_hi)),
            "band_hi": float(max(band_lo, band_hi)),
            "phys_lo": float(phys_lo),
            "phys_hi": float(phys_hi),
            "zero_scaled": None if zero_scaled is None else float(zero_scaled),
            "zero_visible": (
                zero_scaled is not None
                and float(min(band_lo, band_hi)) <= float(zero_scaled) <= float(max(band_lo, band_hi))
            ),
            "x": x_arr,
            "y_scaled": y_scaled_arr,
            "y_phys": y_phys_arr,
            "color": color,
            "_peak_cache": None,
            "srv": srv,
        })

    def _toggle_zero_lines(self):
        self.cfg_show_zero_line = not bool(getattr(self, "cfg_show_zero_line", True))
        if hasattr(self, "cfg") and isinstance(self.cfg, dict):
            self.cfg.setdefault("par", {})
            self.cfg["par"]["show-lines"] = "1" if self.cfg_show_zero_line else "0"
        self._update_pg1()

    def _set_y0_curve_pen(self, idx, color):
        if idx < 0 or idx >= len(self.y0_curves):
            return
        pen = pg.mkPen(color, width=2, cosmetic=True)
        pen.setDashPattern([8, 4])
        self.y0_curves[idx].setPen(pen)

    def _limit_probe_indices(self, idx, cap=256):
        idx = np.asarray(idx, dtype=int)
        if idx.size <= cap:
            return idx
        picks = np.linspace(0, idx.size - 1, cap, dtype=int)
        return idx[picks]

    def _envelope_prominence(self, env_vals, idx, half_win, want_max):
        n = env_vals.size
        lo = max(0, idx - half_win)
        hi = min(n, idx + half_win + 1)
        if idx <= lo or idx + 1 >= hi:
            return 0.0
        left = env_vals[lo:idx]
        right = env_vals[idx + 1:hi]
        if left.size == 0 or right.size == 0:
            return 0.0
        if want_max:
            base = max(float(np.min(left)), float(np.min(right)))
            return float(env_vals[idx] - base)
        base = min(float(np.max(left)), float(np.max(right)))
        return float(base - env_vals[idx])

    def _merge_probe_candidates(self, cand_bins, cand_src_idx, cand_scores, min_sep_bins):
        cand_bins = np.asarray(cand_bins, dtype=int)
        cand_src_idx = np.asarray(cand_src_idx, dtype=int)
        cand_scores = np.asarray(cand_scores, dtype=float)
        if cand_bins.size == 0:
            return np.empty(0, dtype=int)
        order = np.argsort(cand_bins)
        cand_bins = cand_bins[order]
        cand_src_idx = cand_src_idx[order]
        cand_scores = cand_scores[order]
        keep = []
        cluster_anchor = None
        best_src = None
        best_score = None
        for b, src_idx, score in zip(cand_bins, cand_src_idx, cand_scores):
            if cluster_anchor is None or abs(int(b) - cluster_anchor) >= min_sep_bins:
                if best_src is not None:
                    keep.append(int(best_src))
                cluster_anchor = int(b)
                best_src = int(src_idx)
                best_score = float(score)
            elif best_score is None or score > best_score:
                best_src = int(src_idx)
                best_score = float(score)
        if best_src is not None:
            keep.append(int(best_src))
        return np.asarray(keep, dtype=int)

    def _build_probe_envelope(self, tx, ys, idx_view, vx0, vx1, width_px):
        if idx_view.size == 0:
            return None
        span = max(vx1 - vx0, 1e-12)
        bins = np.floor((tx[idx_view] - vx0) * ((width_px - 1) / span)).astype(int)
        bins = np.clip(bins, 0, width_px - 1)

        ys_view = ys[idx_view].astype(float)

        # Vectorised bin statistics -- avoid Python loop over potentially large arrays.
        # Mean / count via bincount.
        cnt_y = np.bincount(bins, minlength=width_px)
        sum_y = np.bincount(bins, weights=ys_view, minlength=width_px)

        # Max per bin: sort descending, keep first occurrence of each bin.
        order_max = np.argsort(-ys_view)
        bins_max = bins[order_max]
        src_max = idx_view[order_max]
        _, first_max = np.unique(bins_max, return_index=True)
        occupied_max = bins_max[first_max]

        max_y = np.full(width_px, -np.inf, dtype=float)
        max_idx = np.full(width_px, -1, dtype=int)
        max_y[occupied_max] = ys_view[order_max[first_max]]
        max_idx[occupied_max] = src_max[first_max]

        # Min per bin: sort ascending, keep first occurrence of each bin.
        order_min = np.argsort(ys_view)
        bins_min = bins[order_min]
        src_min = idx_view[order_min]
        _, first_min = np.unique(bins_min, return_index=True)
        occupied_min = bins_min[first_min]

        min_y = np.full(width_px, np.inf, dtype=float)
        min_idx = np.full(width_px, -1, dtype=int)
        min_y[occupied_min] = ys_view[order_min[first_min]]
        min_idx[occupied_min] = src_min[first_min]

        max_valid = np.flatnonzero(max_idx >= 0)
        min_valid = np.flatnonzero(min_idx >= 0)
        ctr_valid = np.flatnonzero(cnt_y > 0)
        return {
            "max_bins": max_valid,
            "max_vals": max_y[max_valid],
            "max_src_idx": max_idx[max_valid],
            "min_bins": min_valid,
            "min_vals": min_y[min_valid],
            "min_src_idx": min_idx[min_valid],
            "ctr_bins": ctr_valid,
            "ctr_vals": sum_y[ctr_valid] / np.maximum(cnt_y[ctr_valid], 1),
        }

    def _snap_probe_candidate_to_extremum(self, tx, ys, idx_view, vx0, vx1, width_px, cand_bins, want_max, search_bins=8):
        cand_bins = np.asarray(cand_bins, dtype=int)
        if cand_bins.size == 0 or idx_view.size == 0:
            return np.empty(0, dtype=int)
        span = max(vx1 - vx0, 1e-12)
        bin_dt = span / max(float(width_px - 1), 1.0)
        tx_view = tx[idx_view]
        ys_view = ys[idx_view]
        out = []
        for b in cand_bins:
            x_target = vx0 + float(b) * bin_dt
            lo = x_target - search_bins * bin_dt
            hi = x_target + search_bins * bin_dt
            mask = (tx_view >= lo) & (tx_view <= hi)
            if not np.any(mask):
                nearest = int(np.argmin(np.abs(tx_view - x_target)))
                out.append(int(idx_view[nearest]))
                continue
            local_src = idx_view[mask]
            local_y = ys_view[mask]
            pick = int(np.argmax(local_y) if want_max else np.argmin(local_y))
            out.append(int(local_src[pick]))
        return np.asarray(out, dtype=int)

    def _smooth_probe_envelope(self, vals, win):
        vals = np.asarray(vals, dtype=float)
        if vals.size == 0 or win <= 1:
            return vals
        # Cap window to signal length so np.convolve(mode='same') never returns
        # a longer array than the input (it returns max(len(a), len(kernel))).
        win = int(max(1, min(win, vals.size)))
        if win % 2 == 0:
            win -= 1   # round down to stay ≤ vals.size and keep odd
        if win <= 1:
            return vals
        kernel = np.ones(win, dtype=float) / float(win)
        return np.convolve(vals, kernel, mode='same')

    def _plateau_aware_extrema(self, vals, want_max):
        vals = np.asarray(vals, dtype=float)
        if vals.size < 3:
            return np.empty(0, dtype=int)
        d = np.diff(vals)
        s = np.sign(d)
        if s.size == 0:
            return np.empty(0, dtype=int)

        # Fill flat runs from the left (propagates the last non-zero sign
        # forward) and from the right (propagates the next non-zero sign
        # backward).  This gives two views of each plateau:
        #   left-fill  → last bin before the descent  = plateau END
        #   right-fill → first bin after the ascent   = plateau START
        # Taking the midpoint of (start, end) centres the returned index on
        # the plateau, which for a box-smoothed spike equals the original
        # sample position.  Without this, a 3-bin smoothed spike returns
        # the right-edge bin, causing an off-by-one that makes the cursor
        # snap to a baseline sample beside the true peak.
        left = s.copy()
        for i in range(1, left.size):
            if left[i] == 0:
                left[i] = left[i - 1]
        right = s.copy()
        for i in range(right.size - 2, -1, -1):
            if right[i] == 0:
                right[i] = right[i + 1]

        if want_max:
            ends   = np.where((left[:-1]  > 0) & (left[1:]  < 0))[0] + 1
            starts = np.where((right[:-1] > 0) & (right[1:] < 0))[0] + 1
        else:
            ends   = np.where((left[:-1]  < 0) & (left[1:]  > 0))[0] + 1
            starts = np.where((right[:-1] < 0) & (right[1:] > 0))[0] + 1

        if ends.size == 0:
            return np.empty(0, dtype=int)
        # When counts agree (the normal case) return plateau centres.
        # Fall back to ends if edge effects produce a count mismatch.
        if starts.size == ends.size:
            return ((starts + ends) // 2).astype(int)
        return ends.astype(int)

    def _candidate_scores(self, positions, values, half_win, want_max):
        positions = np.asarray(positions, dtype=int)
        if positions.size == 0:
            return np.empty(0, dtype=float)
        return np.asarray(
            [self._envelope_prominence(values, int(pos), half_win, want_max) for pos in positions],
            dtype=float,
        )

    def _get_probe_peaks(self, entry):
        tx = entry["x"]
        ys = entry["y_scaled"]
        if tx.size < 3 or ys.size < 3:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)

        vb = self.ui.pg1.getPlotItem().getViewBox()
        vx0, vx1 = vb.viewRange()[0]
        width_px = max(16, int(round(float(vb.width() or 1.0))))
        cache_key = (round(vx0, 6), round(vx1, 6), width_px, tx.size)
        cache = entry.get("_peak_cache")
        if cache is not None and cache.get("key") == cache_key:
            return cache["max_idx"], cache["min_idx"]

        in_view = (tx >= vx0) & (tx <= vx1)
        if np.count_nonzero(in_view) < 3:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)

        idx_view = np.flatnonzero(in_view)
        env = self._build_probe_envelope(tx, ys, idx_view, vx0, vx1, width_px)
        if env is None:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)

        max_bins = env["max_bins"]
        min_bins = env["min_bins"]
        max_src_idx = env["max_src_idx"]
        min_src_idx = env["min_src_idx"]
        ctr_bins = env["ctr_bins"]
        ctr_vals = env["ctr_vals"]

        if ctr_vals.size < 3:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)

        # ---------------------------------------------------------------
        # Two-scale hierarchical peak detection
        #
        # COARSE scale  heavy smoothing + large prominence window
        #   Correctly scores broad/slow peaks (airflow) whose valleys sit
        #   far from the crest.  Uses only the per-bin envelope (max/min),
        #   not the mean, because for an alternating signal the per-bin
        #   mean is dragged toward zero when both crest and trough of the
        #   same cycle fall in the same bin.
        #   min_sep_coarse prevents double-marking the same slow peak.
        #
        # FINE scale    minimal smoothing (win=3) + moderate prominence window
        #   smooth_win=3 is the minimum needed so that a single-bin R-peak
        #   (zoomed-out ECG) still produces a clear local maximum; larger
        #   windows shift the smoothed maximum to land *between* closely-
        #   spaced peaks rather than on them.
        #   min_sep_fine=3 allows consecutive heart beats to coexist.
        #
        # Merge strategy:
        #   Collect coarse and fine candidates independently, then pool them
        #   and merge by pixel-bin position with min_sep_fine.  The merge
        #   selects the highest-scoring candidate within each cluster, so
        #   the coarse (better-scored for broad peaks) and fine (better-scored
        #   for narrow spikes) naturally win in their respective domains.
        # ---------------------------------------------------------------

        band_span = max(1e-12, float(entry["band_hi"] - entry["band_lo"]))
        amp_thresh = max(0.01 * band_span, 1e-4)

        span = max(vx1 - vx0, 1e-12)

        # ---------------------------------------------------------------
        # All coarse-scale parameters derive from one anchor: coarse_px,
        # the number of pixel-bins that define a "broad visible peak".
        # ~1/12 of the view is a reasonable perceptual boundary between
        # "narrow spike" and "broad wave": at 1200 px it gives ~100 px,
        # which is roughly the width of one respiratory cycle at typical
        # clinical zoom.  All constants are therefore in units of this
        # one value, not independently tuned fractions of width_px.
        # ---------------------------------------------------------------
        coarse_px = max(8, width_px // 12)          # e.g. 100 for 1200 px

        smooth_win_coarse = coarse_px | 1            # odd, ~1 coarse feature wide
        half_win_coarse   = coarse_px                # see ±1 coarse feature for prominence
        min_sep_coarse    = max(4, coarse_px // 2)   # distinct coarse peaks half-feature apart

        max_env_c = self._smooth_probe_envelope(env["max_vals"], smooth_win_coarse)
        min_env_c = self._smooth_probe_envelope(env["min_vals"], smooth_win_coarse)

        def _coarse_candidates(env_vals, env_src_idx, env_bins_arr, want_max):
            pos = self._plateau_aware_extrema(env_vals, want_max)
            if pos.size == 0 or env_src_idx.size == 0:
                return np.empty(0, int), np.empty(0, int), np.empty(0, float)
            scores = self._candidate_scores(pos, env_vals, half_win_coarse, want_max)
            mask = scores >= amp_thresh
            return env_bins_arr[pos][mask], env_src_idx[pos][mask], scores[mask]

        crs_max_b, crs_max_s, crs_max_sc = _coarse_candidates(max_env_c, max_src_idx, max_bins, True)
        crs_min_b, crs_min_s, crs_min_sc = _coarse_candidates(min_env_c, min_src_idx, min_bins, False)

        # ---- fine parameters ----
        # smooth_win=3: minimal blur so single-bin spikes survive.  A larger
        # window shifts smoothed maxima to land *between* closely-spaced
        # R-peaks rather than on them.
        # half_win_fine: ¼ of the coarse scale is enough to see the valley
        # beside a narrow spike.  min_sep_fine=3 is the perceptual pixel floor.
        smooth_win_fine = 3
        half_win_fine   = max(4, coarse_px // 4)   # ~25 for 1200 px
        min_sep_fine    = 3

        ctr_vals_f = self._smooth_probe_envelope(ctr_vals,        smooth_win_fine)
        max_env_f  = self._smooth_probe_envelope(env["max_vals"], smooth_win_fine)
        min_env_f  = self._smooth_probe_envelope(env["min_vals"], smooth_win_fine)

        def _fine_candidates(want_max):
            bl, bs, bsc = [], [], []
            # envelope signal
            env_vals  = max_env_f  if want_max else min_env_f
            env_src   = max_src_idx if want_max else min_src_idx
            env_bins2 = max_bins    if want_max else min_bins
            if env_src.size:
                pos = self._plateau_aware_extrema(env_vals, want_max)
                if pos.size:
                    sc = self._candidate_scores(pos, env_vals, half_win_fine, want_max)
                    m = sc >= amp_thresh
                    cand_bins = env_bins2[pos][m]
                    # Snap to actual sample extremum within ±(smooth_win_fine+1) bins.
                    # Corrects any residual off-by-one from plateau centre rounding.
                    snapped = self._snap_probe_candidate_to_extremum(
                        tx, ys, idx_view, vx0, vx1, width_px, cand_bins, want_max,
                        search_bins=smooth_win_fine + 1)
                    bl.append(cand_bins); bs.append(snapped); bsc.append(sc[m])
            # mean signal
            pos = self._plateau_aware_extrema(ctr_vals_f, want_max)
            if pos.size:
                snapped = self._snap_probe_candidate_to_extremum(
                    tx, ys, idx_view, vx0, vx1, width_px, ctr_bins[pos], want_max)
                sc = self._candidate_scores(pos, ctr_vals_f, half_win_fine, want_max)
                m = sc >= amp_thresh
                bl.append(ctr_bins[pos][m]); bs.append(snapped[m]); bsc.append(sc[m])
            if not bl:
                return np.empty(0, int), np.empty(0, int), np.empty(0, float)
            return np.concatenate(bl), np.concatenate(bs), np.concatenate(bsc)

        fine_max_b, fine_max_s, fine_max_sc = _fine_candidates(True)
        fine_min_b, fine_min_s, fine_min_sc = _fine_candidates(False)

        # Combine coarse and fine candidates and merge by bin position.
        # No suppression step: suppressing fine peaks near coarse ones was
        # silencing ECG R-peaks whose amplitude is modulated by respiration
        # (the coarse max-envelope shows a respiratory-frequency oscillation,
        # which looked like a valid coarse peak and blanked nearby R-peaks).
        # With the plateau-centre fix both paths snap to the same source sample
        # for the same physical peak, so the merge naturally deduplicates them.
        def _combine(cb, cs, csc, fb, fs, fsc):
            if cb.size and fb.size:
                b = np.concatenate([cb, fb])
                s = np.concatenate([cs, fs])
                sc = np.concatenate([csc, fsc])
            elif cb.size:
                b, s, sc = cb, cs, csc
            elif fb.size:
                b, s, sc = fb, fs, fsc
            else:
                return np.empty(0, int)
            return self._merge_probe_candidates(b, s, sc, min_sep_fine)

        max_idx = _combine(crs_max_b, crs_max_s, crs_max_sc, fine_max_b, fine_max_s, fine_max_sc)
        min_idx = _combine(crs_min_b, crs_min_s, crs_min_sc, fine_min_b, fine_min_s, fine_min_sc)

        max_idx = self._limit_probe_indices(max_idx, cap=256)
        min_idx = self._limit_probe_indices(min_idx, cap=256)
        entry["_peak_cache"] = {"key": cache_key, "max_idx": max_idx, "min_idx": min_idx}
        return max_idx, min_idx

    def _find_pg1_channel_band(self, y):
        for entry in self._pg1_channel_cache:
            if entry["band_lo"] <= y <= entry["band_hi"]:
                return entry
        return None

    def _probe_channel_sample(self, entry, x):
        tx = entry["x"]
        ty = entry["y_scaled"]
        if tx.size == 0 or ty.size == 0:
            return None
        idx = int(np.argmin(np.abs(tx - float(x))))
        sample_x = float(tx[idx])
        sample_y_scaled = float(ty[idx])
        if entry["y_phys"] is not None and idx < entry["y_phys"].size:
            sample_val = float(entry["y_phys"][idx])
        else:
            band_span = max(1e-12, entry["band_hi"] - entry["band_lo"])
            frac = (sample_y_scaled - entry["band_lo"]) / band_span
            sample_val = entry["phys_lo"] + frac * (entry["phys_hi"] - entry["phys_lo"])
        if not np.isfinite(sample_val):
            return None
        return {
            "sample_x": sample_x,
            "sample_y_scaled": sample_y_scaled,
            "sample_val": sample_val,
        }

    def _show_pg1_probe_band_lines(self):
        vx0, vx1 = self.ui.pg1.getPlotItem().getViewBox().viewRange()[0]
        bands = sorted(self._pg1_channel_cache, key=lambda entry: entry["band_lo"])
        boundaries = []
        for prev, curr in zip(bands, bands[1:]):
            y = 0.5 * (prev["band_hi"] + curr["band_lo"])
            boundaries.append(y)

        for idx, line in enumerate(self._pg1_probe_band_lines):
            if idx < len(boundaries):
                y = boundaries[idx]
                line.setData([vx0, vx1], [y, y])
                line.show()
            else:
                line.setData([], [])
                line.hide()

    def _pg1_probe_key_active(self, key):
        return key in self._pg1_probe_keys

    def _set_pg1_probe_key(self, key, active):
        if active:
            self._pg1_probe_keys.add(key)
        else:
            self._pg1_probe_keys.discard(key)

    def _toggle_pg1_probe_key(self, key):
        if self._pg1_probe_key_active(key):
            self._pg1_probe_keys.discard(key)
        else:
            self._pg1_probe_keys.add(key)

    def _cycle_pg1_probe_grid(self):
        self._pg1_probe_grid_idx = (self._pg1_probe_grid_idx + 1) % len(self._pg1_probe_grid_steps)

    def _current_pg1_probe_grid(self):
        return self._pg1_probe_grid_steps[self._pg1_probe_grid_idx]

    def _build_vertical_segments(self, xs, y0, y1):
        if xs is None or len(xs) == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        xs = np.asarray(xs, dtype=float)
        xdat = np.empty(xs.size * 2, dtype=float)
        ydat = np.empty(xs.size * 2, dtype=float)
        xdat[0::2] = xs
        xdat[1::2] = xs
        ydat[0::2] = y0
        ydat[1::2] = y1
        return xdat, ydat

    def _update_pg1_probe_legend(self, vb):
        if self._pg1_probe_legend is None:
            return
        vx0, vx1 = vb.viewRange()[0]
        legend = "Probe toggles: Z zero-x | P peaks | A annots | S stats | G grid | Y y=0 | Space pin"
        if bool(getattr(self, "_pg1_probe_pinned", False)):
            legend += " [pinned]"
        grid = self._current_pg1_probe_grid()
        if grid is not None:
            grid_txt = self._format_pg1_probe_dt(grid)
            legend += f" | grid={grid_txt}"
        self._pg1_probe_legend.setText(legend)
        self._pg1_probe_legend.setPos(0.5 * (vx0 + vx1), 0.955)
        self._pg1_probe_legend.show()

    def _update_pg1_probe_grid(self, vb):
        if self._pg1_probe_grid_lines is None:
            return
        step = self._current_pg1_probe_grid()
        if step is None:
            self._pg1_probe_grid_lines.setData([], [])
            self._pg1_probe_grid_lines.hide()
            return
        vx0, vx1 = vb.viewRange()[0]
        start = np.floor(vx0 / step) * step
        xs = np.arange(start, vx1 + step, step)
        xs = xs[(xs >= vx0) & (xs <= vx1)]
        xdat, ydat = self._build_vertical_segments(xs, 0.0, 1.0)
        self._pg1_probe_grid_lines.setData(xdat, ydat)
        self._pg1_probe_grid_lines.show()

    def _update_pg1_probe_zero_crossings(self, entry):
        if self._pg1_probe_zero_lines is None or self._pg1_probe_zero_baseline is None:
            return
        if not self._pg1_probe_key_active("Z"):
            self._pg1_probe_zero_lines.setData([], [])
            self._pg1_probe_zero_lines.hide()
            self._pg1_probe_zero_baseline.setData([], [])
            self._pg1_probe_zero_baseline.hide()
            return
        tx = entry["x"]
        ys = entry["y_scaled"]
        zero_scaled = entry.get("zero_scaled", None)
        zero_visible = bool(entry.get("zero_visible", False))
        if tx.size == 0 or ys.size == 0 or zero_scaled is None or not zero_visible:
            self._pg1_probe_zero_lines.setData([], [])
            self._pg1_probe_zero_lines.hide()
            self._pg1_probe_zero_baseline.setData([], [])
            self._pg1_probe_zero_baseline.hide()
            return

        self._pg1_probe_zero_baseline.setData([tx[0], tx[-1]], [zero_scaled, zero_scaled])
        self._pg1_probe_zero_baseline.show()

        vals = entry["y_phys"] if entry["y_phys"] is not None else (ys - zero_scaled)
        tx = entry["x"]
        if tx.size < 2 or vals is None or vals.size < 2:
            self._pg1_probe_zero_lines.setData([], [])
            self._pg1_probe_zero_lines.hide()
            self._pg1_probe_zero_baseline.setData([], [])
            self._pg1_probe_zero_baseline.hide()
            return
        v0 = vals[:-1]
        v1 = vals[1:]
        x0 = tx[:-1]
        x1 = tx[1:]
        mask = ((v0 == 0) | (v1 == 0) | ((v0 < 0) & (v1 > 0)) | ((v0 > 0) & (v1 < 0)))
        if not np.any(mask):
            self._pg1_probe_zero_lines.setData([], [])
            self._pg1_probe_zero_lines.hide()
            return
        x0m = x0[mask]
        x1m = x1[mask]
        v0m = v0[mask]
        v1m = v1[mask]
        denom = v1m - v0m
        frac = np.where(np.abs(denom) > 1e-12, -v0m / denom, 0.0)
        xs = x0m + frac * (x1m - x0m)
        xdat, ydat = self._build_vertical_segments(xs, entry["band_lo"], entry["band_hi"])
        self._pg1_probe_zero_lines.setData(xdat, ydat)
        self._pg1_probe_zero_lines.show()

    def _update_pg1_probe_peaks(self, entry):
        if self._pg1_probe_peak_max is None or self._pg1_probe_peak_min is None:
            return
        if not self._pg1_probe_key_active("P"):
            self._pg1_probe_peak_max.setData([], [])
            self._pg1_probe_peak_min.setData([], [])
            self._pg1_probe_peak_max.hide()
            self._pg1_probe_peak_min.hide()
            return
        vals = entry["y_phys"] if entry["y_phys"] is not None else entry["y_scaled"]
        tx = entry["x"]
        ys = entry["y_scaled"]
        if tx.size < 3 or vals is None or vals.size < 3 or ys.size < 3:
            self._pg1_probe_peak_max.setData([], [])
            self._pg1_probe_peak_min.setData([], [])
            self._pg1_probe_peak_max.hide()
            self._pg1_probe_peak_min.hide()
            return
        max_idx, min_idx = self._get_probe_peaks(entry)
        self._pg1_probe_peak_max.setData(tx[max_idx], ys[max_idx])
        self._pg1_probe_peak_min.setData(tx[min_idx], ys[min_idx])
        self._pg1_probe_peak_max.setVisible(max_idx.size > 0)
        self._pg1_probe_peak_min.setVisible(min_idx.size > 0)

    def _probe_channel_peak_sample(self, entry, x):
        tx = entry["x"]
        ys = entry["y_scaled"]
        vals = entry["y_phys"] if entry["y_phys"] is not None else ys
        if tx.size < 3 or ys.size < 3 or vals is None or vals.size < 3:
            return None
        max_idx, min_idx = self._get_probe_peaks(entry)
        peak_idx = np.sort(np.concatenate([max_idx, min_idx]))
        if peak_idx.size == 0:
            return None
        nearest = int(np.argmin(np.abs(tx[peak_idx] - float(x))))
        idx = int(peak_idx[nearest])
        return {
            "sample_x": float(tx[idx]),
            "sample_y_scaled": float(ys[idx]),
            "sample_val": float(vals[idx]),
        }

    def _probe_annotations_at_time(self, xpos):
        hits = []
        if not hasattr(self, "annot_mgr") or self.annot_mgr is None:
            return hits
        for name, track in self.annot_mgr.tracks.items():
            if name == "__#gaps__":
                continue
            x0 = np.asarray(track.get("x0", []), dtype=float)
            x1 = np.asarray(track.get("x1", []), dtype=float)
            if x0.size == 0 or x1.size == 0:
                continue
            if np.any((x0 <= xpos) & (x1 >= xpos)):
                hits.append(name)
        return hits

    def _resolve_pg1_probe_entry(self, y, locked_entry=None):
        if locked_entry is not None:
            return locked_entry
        return self._find_pg1_channel_band(y)

    def _format_pg1_probe_value(self, value):
        aval = abs(float(value))
        if aval >= 100:
            return f"{value:.1f}"
        if aval >= 10:
            return f"{value:.2f}"
        return f"{value:.3f}"

    def _format_pg1_probe_dt(self, dt_seconds):
        dt = abs(float(dt_seconds))
        if dt < 60:
            return f"{dt:.2f} s"
        if dt < 3600:
            return f"{dt / 60.0:.2f} min"
        return f"{dt / 3600.0:.2f} h"

    def _probe_channel_window_stats(self, entry, sample_a, sample_b):
        tx = entry["x"]
        if tx.size == 0:
            return None

        vals = entry["y_phys"] if entry["y_phys"] is not None else entry["y_scaled"]
        if vals is None or vals.size == 0:
            return None

        lo = min(sample_a["sample_x"], sample_b["sample_x"])
        hi = max(sample_a["sample_x"], sample_b["sample_x"])
        mask = (tx >= lo) & (tx <= hi)
        if not np.any(mask):
            return None

        win = vals[mask]
        if win.size == 0:
            return None

        vmin = float(np.min(win))
        vmax = float(np.max(win))
        return {
            "min": vmin,
            "max": vmax,
            "p2p": vmax - vmin,
            "mean": float(np.mean(win)),
        }

    def _update_pg1_probe(self, scene_pos, start_entry=None, start_sample=None):
        vb = self.ui.pg1.getPlotItem().getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos):
            self._hide_pg1_probe()
            return None, None

        x = float(vb.mapSceneToView(scene_pos).x())
        y = float(vb.mapSceneToView(scene_pos).y())

        if self._pg1_probe_line is not None:
            self._pg1_probe_line.setPos(x)
            self._pg1_probe_line.show()

        self._update_pg1_probe_grid(vb)
        self._update_pg1_probe_legend(vb)
        self._show_pg1_probe_band_lines()
        entry = self._resolve_pg1_probe_entry(y, start_entry)
        if entry is not None and self._pg1_probe_key_active("P"):
            sample = self._probe_channel_peak_sample(entry, x)
            if sample is None:
                sample = self._probe_channel_sample(entry, x)
        else:
            sample = self._probe_channel_sample(entry, x) if entry is not None else None

        if sample is None:
            if self._pg1_probe_start_sample_line is not None:
                self._pg1_probe_start_sample_line.setData([], [])
                self._pg1_probe_start_sample_line.hide()
            if self._pg1_probe_sample_line is not None:
                self._pg1_probe_sample_line.setData([], [])
                self._pg1_probe_sample_line.hide()
            if self._pg1_probe_span_line is not None:
                self._pg1_probe_span_line.setData([], [])
                self._pg1_probe_span_line.hide()
            if self._pg1_probe_zero_lines is not None:
                self._pg1_probe_zero_lines.setData([], [])
                self._pg1_probe_zero_lines.hide()
            if self._pg1_probe_peak_max is not None:
                self._pg1_probe_peak_max.setData([], [])
                self._pg1_probe_peak_max.hide()
            if self._pg1_probe_peak_min is not None:
                self._pg1_probe_peak_min.setData([], [])
                self._pg1_probe_peak_min.hide()
            if self._pg1_probe_label is not None:
                self._pg1_probe_label.hide()
            return entry, sample

        self._update_pg1_probe_zero_crossings(entry)
        self._update_pg1_probe_peaks(entry)

        if self._pg1_probe_sample_line is not None:
            self._pg1_probe_sample_line.setData(
                [sample["sample_x"], sample["sample_x"]],
                [entry["band_lo"], entry["band_hi"]],
            )
            self._pg1_probe_sample_line.show()

        units = self.units.get(entry["ch"], "")
        label = f'{entry["ch"]}: {self._format_pg1_probe_value(sample["sample_val"])}'
        if units:
            label += f" {units}"
        if (
            start_entry is not None
            and start_sample is not None
            and start_entry.get("ch") == entry["ch"]
        ):
            span = abs(sample["sample_val"] - start_sample["sample_val"])
            dt = abs(sample["sample_x"] - start_sample["sample_x"])
            stats = self._probe_channel_window_stats(entry, start_sample, sample) if self._pg1_probe_key_active("S") else None
            if self._pg1_probe_start_sample_line is not None:
                self._pg1_probe_start_sample_line.setData(
                    [start_sample["sample_x"], start_sample["sample_x"]],
                    [entry["band_lo"], entry["band_hi"]],
                )
                self._pg1_probe_start_sample_line.show()
            if self._pg1_probe_span_line is not None:
                self._pg1_probe_span_line.setData(
                    [start_sample["sample_x"], sample["sample_x"]],
                    [start_sample["sample_y_scaled"], sample["sample_y_scaled"]],
                )
                self._pg1_probe_span_line.show()
            label = (
                f'{entry["ch"]}\n'
                f'start: {self._format_pg1_probe_value(start_sample["sample_val"])}'
                + (f" {units}" if units else "")
                + f'\ncurrent: {self._format_pg1_probe_value(sample["sample_val"])}'
                + (f" {units}" if units else "")
                + f'\ndelta: {self._format_pg1_probe_value(span)}'
                + (f" {units}" if units else "")
                + f'\ndt: {self._format_pg1_probe_dt(dt)}'
            )
            if stats is not None:
                label += (
                    f'\nmin: {self._format_pg1_probe_value(stats["min"])}'
                    + (f" {units}" if units else "")
                    + f'\nmax: {self._format_pg1_probe_value(stats["max"])}'
                    + (f" {units}" if units else "")
                    + f'\np2p: {self._format_pg1_probe_value(stats["p2p"])}'
                    + (f" {units}" if units else "")
                    + f'\nmean: {self._format_pg1_probe_value(stats["mean"])}'
                    + (f" {units}" if units else "")
                )
        else:
            if self._pg1_probe_start_sample_line is not None:
                self._pg1_probe_start_sample_line.setData([], [])
                self._pg1_probe_start_sample_line.hide()
            if self._pg1_probe_span_line is not None:
                self._pg1_probe_span_line.setData([], [])
                self._pg1_probe_span_line.hide()

        if self._pg1_probe_key_active("A"):
            ann_hits = self._probe_annotations_at_time(sample["sample_x"])
            if ann_hits:
                shown = ann_hits[:6]
                suffix = " ..." if len(ann_hits) > 6 else ""
                label += "\nannots: " + ", ".join(shown) + suffix

        if self._pg1_probe_label is not None:
            vx0, vx1 = vb.viewRange()[0]
            span_active = (
                start_entry is not None
                and start_sample is not None
                and start_entry.get("ch") == entry["ch"]
            )
            ref_x = start_sample["sample_x"] if span_active else sample["sample_x"]
            other_x = sample["sample_x"] if span_active else x
            place_right = other_x < ref_x
            anchor_x = 0 if place_right else 1
            self._pg1_probe_label.setAnchor((anchor_x, 0.5))
            xoff = 0.01 * (vx1 - vx0)
            xpos = ref_x + xoff if place_right else ref_x - xoff
            ypos = min(entry["band_hi"] - 0.01, max(entry["band_lo"] + 0.01, 0.5 * (entry["band_lo"] + entry["band_hi"])))
            self._pg1_probe_label.setText(label)
            self._pg1_probe_label.setPos(xpos, ypos)
            self._pg1_probe_label.show()

        return entry, sample

    def _hide_pg1_probe(self):
        if self._pg1_probe_line is not None:
            self._pg1_probe_line.hide()
        if self._pg1_probe_start_sample_line is not None:
            self._pg1_probe_start_sample_line.setData([], [])
            self._pg1_probe_start_sample_line.hide()
        if self._pg1_probe_sample_line is not None:
            self._pg1_probe_sample_line.setData([], [])
            self._pg1_probe_sample_line.hide()
        if self._pg1_probe_span_line is not None:
            self._pg1_probe_span_line.setData([], [])
            self._pg1_probe_span_line.hide()
        if self._pg1_probe_label is not None:
            self._pg1_probe_label.hide()
        if self._pg1_probe_legend is not None:
            self._pg1_probe_legend.hide()
        if self._pg1_probe_grid_lines is not None:
            self._pg1_probe_grid_lines.setData([], [])
            self._pg1_probe_grid_lines.hide()
        if self._pg1_probe_zero_lines is not None:
            self._pg1_probe_zero_lines.setData([], [])
            self._pg1_probe_zero_lines.hide()
        if getattr(self, "_pg1_probe_zero_baseline", None) is not None:
            self._pg1_probe_zero_baseline.setData([], [])
            self._pg1_probe_zero_baseline.hide()
        if self._pg1_probe_peak_max is not None:
            self._pg1_probe_peak_max.setData([], [])
            self._pg1_probe_peak_max.hide()
        if self._pg1_probe_peak_min is not None:
            self._pg1_probe_peak_min.setData([], [])
            self._pg1_probe_peak_min.hide()
        for line in self._pg1_probe_band_lines:
            line.setData([], [])
            line.hide()
        


# ------------------------------------------------------------

    def filter_signal( self , x , ch, fs_key , order = 2):

        if fs_key[0] == 'User':
            return self.user_filter_signal( x, fs_key, ch )

        if fs_key in self.fmap_flts:
            return sosfilt( self.fmap_flts[ fs_key ] , x )
        else:
            frqs = self.fmap_frqs[ fs_key[0] ]
            sr = fs_key[1]
            # ensure below Nyquist 
            if frqs[1] <= sr / 2:
                sos = butter( order,
                              frqs , 
                              btype='band',
                              fs=sr , 
                              output='sos' )
                self.fmap_flts[ fs_key ] = sos
                return sosfilt( sos , x )
        return x
        

    def user_filter_signal( self , x , fs_key , ch, order = 2):

        if ch not in self.user_fmap_frqs:
            return x

        # edit 'User' --> specific band for this ch
        frqs = self.user_fmap_frqs[ch]
        sr = fs_key[1]
        if len(frqs) != 2:
            return x
        if not (frqs[0] < frqs[1] and frqs[0] >= 0 and frqs[1] <= sr / 2):
            return x

        # cache key for this channel's user band at this sampling rate
        key = (f"User:{frqs[0]}-{frqs[1]}", sr, ch)
        if key in self.fmap_flts:
            return sosfilt( self.fmap_flts[ key ] , x )
        else:
            sos = butter( order,
                          frqs ,
                          btype='band',
                          fs=sr ,
                          output='sos' )
            self.fmap_flts[ key ] = sos
            return sosfilt( sos , x )
        return x

# ------------------------------------------------------------

from PySide6 import QtCore, QtGui
import pyqtgraph as pg

class MainTraceProbe(QtCore.QObject):
    def __init__(self, plot, owner):
        super().__init__(plot)
        self.plot = plot
        self.pi = plot.getPlotItem() if isinstance(plot, pg.PlotWidget) else plot
        self.vb = self.pi.getViewBox()
        self.owner = owner
        self._active = False
        self._pinned = False
        self._start_entry = None
        self._start_sample = None
        self._last_scene_pos = None
        self.pi.scene().installEventFilter(self)
        try:
            self.plot.destroyed.connect(self._on_plot_destroyed)
        except Exception:
            pass

    def _on_plot_destroyed(self, *_):
        self.plot = None
        self.pi = None
        self.vb = None

    def is_active(self):
        return self._active

    def is_engaged(self):
        return self._active or self._pinned

    def is_pinned(self):
        return self._pinned

    def _sync_owner_pin_state(self):
        try:
            self.owner._pg1_probe_pinned = bool(self._pinned)
        except Exception:
            pass

    def clear_pinned(self):
        self._pinned = False
        self._sync_owner_pin_state()
        if not self._active:
            self._start_entry = None
            self._start_sample = None
            self._last_scene_pos = None

    def toggle_pinned(self):
        if self._last_scene_pos is None:
            return False
        self._pinned = not self._pinned
        self._sync_owner_pin_state()
        if not self._pinned and not self._active:
            self._start_entry = None
            self._start_sample = None
            self._last_scene_pos = None
            self.owner._hide_pg1_probe()
        else:
            self.refresh()
        return True

    def refresh(self):
        if (self._active or self._pinned) and self._last_scene_pos is not None and self.plot is not None:
            if self._start_entry is not None and self._start_sample is not None and self.owner._pg1_probe_key_active("P"):
                snapped = self.owner._probe_channel_peak_sample(self._start_entry, self._start_sample["sample_x"])
                if snapped is not None:
                    self._start_sample = snapped
            self.owner._update_pg1_probe(self._last_scene_pos, self._start_entry, self._start_sample)

    def eventFilter(self, obj, ev):
        try:
            scene = None if self.pi is None else self.pi.scene()
        except RuntimeError:
            return False
        if scene is None or obj is not scene:
            return False

        et = ev.type()
        if et == QtCore.QEvent.GraphicsSceneMousePress and ev.button() == QtCore.Qt.LeftButton:
            try:
                if self.vb is None or not self.vb.sceneBoundingRect().contains(ev.scenePos()):
                    return False
            except RuntimeError:
                return False

            mods = ev.modifiers()
            annot_mode = (hasattr(self.owner, "_annotator_mode_active") and
                          self.owner._annotator_mode_active())
            annot_interval_mode = bool(
                hasattr(self.owner, "annotator")
                and self.owner.annotator is not None
                and self.owner.annotator.current_mode() == "interval"
            )
            annot_interval_capture = bool(
                annot_interval_mode
                and hasattr(self.owner, "annotator")
                and self.owner.annotator is not None
                and self.owner.annotator.isVisible()
                and self.owner.annotator.check_enabled.isChecked()
            )

            # Interval capture must own plain left-drag on the trace.
            # Do not activate probe while annotator interval capture is active.
            if annot_interval_capture:
                return False

            if annot_mode:
                # Annotator mode: clicks on an annotation select it.
                # Clicks on blank space fall through to normal probe activation
                # so that A/S/P/Z probe keys still work.
                # In interval capture mode we prioritize drag-to-select, so plain
                # left-clicks should not be consumed by annotation selection.
                if (not annot_interval_mode) and hasattr(self.owner, "_annot_hit_test"):
                    hit = self.owner._annot_hit_test(ev.scenePos(), self.vb)
                    if hit is not None:
                        zoom = bool(mods & (QtCore.Qt.ControlModifier |
                                            QtCore.Qt.ShiftModifier))
                        self.owner._annot_select_in_dock5(*hit, zoom=zoom)
                        return True
                # no annotation hit — fall through to normal probe activation

            # Normal mode ---------------------------------------------------
            # ctrl+click: select annotation in Dock5 and zoom to it
            # shift+click: same but leave the display window unchanged
            if mods & (QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier):
                if hasattr(self.owner, "_annot_hit_test"):
                    hit = self.owner._annot_hit_test(ev.scenePos(), self.vb)
                    if hit is not None:
                        zoom = bool(mods & QtCore.Qt.ControlModifier)
                        self.owner._annot_select_in_dock5(*hit, zoom=zoom)
                        return True

            try:
                if self.plot is not None:
                    self.plot.setFocus()
            except RuntimeError:
                return False
            self._pinned = False
            self._sync_owner_pin_state()
            self._active = True
            self._last_scene_pos = ev.scenePos()
            self._start_entry, self._start_sample = self.owner._update_pg1_probe(ev.scenePos())
            return False

        if et == QtCore.QEvent.GraphicsSceneMouseMove and self._active:
            self._last_scene_pos = ev.scenePos()
            self.owner._update_pg1_probe(ev.scenePos(), self._start_entry, self._start_sample)
            return False

        if et == QtCore.QEvent.GraphicsSceneMouseRelease and ev.button() == QtCore.Qt.LeftButton and self._active:
            self._active = False
            if not self._pinned:
                self._start_entry = None
                self._start_sample = None
                self._last_scene_pos = None
                self.owner._hide_pg1_probe()
            return False

        return False

class MainTraceNavProxy(QtCore.QObject):
    def __init__(self, plot, owner):
        super().__init__(plot)
        self.plot = plot
        self.owner = owner
        self.pi = plot.getPlotItem() if isinstance(plot, pg.PlotWidget) else plot
        self.vb = self.pi.getViewBox()
        self.plot.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.plot.installEventFilter(self)
        viewport = self.plot.viewport() if hasattr(self.plot, "viewport") else None
        if viewport is not None:
            viewport.setFocusPolicy(QtCore.Qt.StrongFocus)
            viewport.installEventFilter(self)
        try:
            self.plot.destroyed.connect(self._on_plot_destroyed)
        except Exception:
            pass

    def _on_plot_destroyed(self, *_):
        self.plot = None
        self.pi = None
        self.vb = None

    def _selector(self):
        return getattr(self.owner, "sel", None)

    def _probe(self):
        return getattr(self.owner, "_pg1_probe", None)

    def _in_plot(self, global_pos):
        if self.plot is None:
            return False
        local = self.plot.mapFromGlobal(global_pos)
        return self.plot.rect().contains(local)

    def _handle_key(self, ev):
        sel = self._selector()

        if hasattr(self.owner, "_annotator_handle_maintrace_key_press"):
            if self.owner._annotator_handle_maintrace_key_press(ev):
                return True

        key = ev.key()
        mods = ev.modifiers()
        shift = bool(mods & QtCore.Qt.ShiftModifier)
        ctrl = bool(mods & QtCore.Qt.ControlModifier)

        probe = self._probe()
        if key == QtCore.Qt.Key_Space:
            if probe is not None and probe.is_engaged() and not ev.isAutoRepeat():
                if probe.toggle_pinned():
                    return True
            return False

        if key in (QtCore.Qt.Key_Z, QtCore.Qt.Key_P, QtCore.Qt.Key_A, QtCore.Qt.Key_S):
            if probe is not None and probe.is_engaged() and not ev.isAutoRepeat():
                key_map = {
                    QtCore.Qt.Key_Z: "Z",
                    QtCore.Qt.Key_P: "P",
                    QtCore.Qt.Key_A: "A",
                    QtCore.Qt.Key_S: "S",
                }
                self.owner._toggle_pg1_probe_key(key_map[key])
                probe.refresh()
                return True
            return False

        if key == QtCore.Qt.Key_Y:
            if probe is not None and probe.is_engaged() and not ev.isAutoRepeat():
                self.owner._toggle_zero_lines()
                probe.refresh()
                return True
            return False

        if key == QtCore.Qt.Key_G:
            if probe is not None and probe.is_engaged() and not ev.isAutoRepeat():
                self.owner._cycle_pg1_probe_grid()
                probe.refresh()
                return True
            return False

        if sel is None:
            return False

        if key == QtCore.Qt.Key_Left:
            dx = sel._step2() if ctrl else sel._step(shift)
            sel._nudge(-dx)
            return True
        if key == QtCore.Qt.Key_Right:
            dx = sel._step2() if ctrl else sel._step(shift)
            sel._nudge(dx)
            return True
        if key == QtCore.Qt.Key_Up:
            sel._zoom(0.8)
            return True
        if key == QtCore.Qt.Key_Down:
            sel._zoom(1.25)
            return True
        return False

    def _handle_key_release(self, ev):
        if hasattr(self.owner, "_annotator_handle_maintrace_key_release"):
            if self.owner._annotator_handle_maintrace_key_release(ev):
                return True
        return False

    def _handle_wheel(self, ev):
        sel = self._selector()
        if sel is None:
            return False
        dy = ev.angleDelta().y()
        if dy > 0:
            sel._zoom(0.8)
            return True
        if dy < 0:
            sel._zoom(1.25)
            return True
        return False

    def eventFilter(self, obj, ev):
        try:
            if self.plot is None:
                return False
            viewport = self.plot.viewport() if hasattr(self.plot, "viewport") else None
        except RuntimeError:
            return False
        if obj not in {self.plot, viewport}:
            return False
        try:
            has_focus = self.plot.hasFocus() or (viewport is not None and viewport.hasFocus())
        except RuntimeError:
            return False

        if ev.type() == QtCore.QEvent.Wheel:
            pos = ev.globalPosition().toPoint() if hasattr(ev, "globalPosition") else QtGui.QCursor.pos()
            if self._in_plot(pos) and self._handle_wheel(ev):
                ev.accept()
                return True

        if ev.type() == QtCore.QEvent.KeyPress and has_focus:
            if self._handle_key(ev):
                ev.accept()
                return True

        if ev.type() == QtCore.QEvent.KeyRelease and has_focus:
            if self._handle_key_release(ev):
                ev.accept()
                return True

        return False

class XRangeSelector(QtCore.QObject):
    """
    Background left-drag: draw/resize selection.
    Single-click: fixed `click_span` centered at click.
    Drag inside region: MOVE whole region (fixed width). If wide and near edges, LRI resizes.
    Left/Right pan. Shift+Left/Right bigger pan.
    Up/Down zoom in/out. Min span = `min_span`. Max = bounds/view.
    Emits: rangeSelected(lo: float, hi: float)
    """
    rangeSelected = QtCore.Signal(float, float)

    def __init__(self, plot, bounds=None, integer=False,
                 click_span=30.0, min_span=1.0,
                 line_width=6, step=1, big_step=10, step_px=3, big_step_px=15,
                 drag_thresh_px=6, edge_tol_px=10, thin_px=16,
                 active_predicate=None):
        super().__init__(plot)

        # resolve plot + focus widget
        self.pi  = plot.getPlotItem() if isinstance(plot, pg.PlotWidget) else plot
        self.vb  = self.pi.getViewBox()
        views = self.pi.scene().views()
        self.wid = plot if hasattr(plot, "setFocusPolicy") else (views[0] if views else None)
        if self.wid is None:
            raise RuntimeError("No focusable view for shortcuts.")
        self.wid.setFocusPolicy(QtCore.Qt.StrongFocus)

        # config
        self.integer     = bool(integer)
        self.bounds      = tuple(bounds) if bounds is not None else None
        self.click_span  = float(click_span)
        self.min_span    = max(0.0, float(min_span))
        self.step2       = 8
        self.step, self.big_step = float(step), float(big_step)
        self.step_px, self.big_step_px = int(step_px), int(big_step_px)
        self.drag_thresh_px = int(drag_thresh_px)
        self.edge_tol_px    = int(edge_tol_px)
        self.thin_px        = int(thin_px)   # width ≤ thin_px ⇒ move-only anywhere inside
        self._active_predicate = active_predicate

        # state
        self._setting_region = False
        self._region_active  = False     # LRI is handling its own drag
        self._last_emitted = None
        self._pending = None
        self._dragging_bg = False
        self._dragging_move = False
        self._moved = False
        self._press_scene = None
        self._anchor_x = None
        self._point_x = None
        self._move_width = None
        self._move_offset = 0.0
        self._disposed = False
        
        # coalesced emitter
        self._emit_timer = QtCore.QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.timeout.connect(self._flush_emit)

        # selection region
        self.region = pg.LinearRegionItem(orientation=pg.LinearRegionItem.Vertical)
        self.region.setMovable(True)
        if self.bounds is not None:
            self.region.setBounds(self.bounds)
        try:
            self.region.setBrush(pg.mkBrush(0, 170, 255, 70))
            self.region.setHoverBrush(pg.mkBrush(0, 170, 255, 110))
        except Exception:
            pass
        for ln in getattr(self.region, "lines", []):
            try:
                ln.setPen(pg.mkPen((0, 220, 255, 240), width=line_width))
                ln.setHoverPen(pg.mkPen((255, 255, 255, 255), width=line_width+4))
                ln.setCursor(QtCore.Qt.SizeHorCursor)
            except Exception:
                pass
        self.region.setZValue(220); self.region.hide()
        self.pi.addItem(self.region)

        self.point_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen((255, 220, 0, 255), width=3, style=QtCore.Qt.DashLine),
        )
        self.point_line.setZValue(230)
        self.point_line.hide()
        self.pi.addItem(self.point_line)

        # signals
        self.region.sigRegionChanged.connect(self._on_region_changed)
        if hasattr(self.region, "sigRegionChangeStarted"):
            self.region.sigRegionChangeStarted.connect(lambda: setattr(self, "_region_active", True))
        if hasattr(self.region, "sigRegionChangeFinished"):
            self.region.sigRegionChangeFinished.connect(self._on_region_finished)

        # keyboard + scene filter
        self._mk_shortcuts()
        self.pi.scene().installEventFilter(self)

    def _interaction_enabled(self):
        if not callable(self._active_predicate):
            return True
        try:
            return bool(self._active_predicate())
        except Exception:
            return False

    # ---------- shortcuts ----------
    def _mk_shortcuts(self):
        self._sc = []
        def sc(keyseq, fn):
            s = QtGui.QShortcut(QtGui.QKeySequence(keyseq), self.wid)
            # critical: restrict scope to this widget (not whole window)
            s.setContext(QtCore.Qt.WidgetWithChildrenShortcut)  # or QtCore.Qt.WidgetShortcut
            s.setAutoRepeat(True)
            s.activated.connect(fn)
            self._sc.append(s)

        sc(QtCore.Qt.Key_Left,  lambda: self._nudge(self._step(False)*-1))
        sc(QtCore.Qt.Key_Right, lambda: self._nudge(self._step(False)*+1))
        sc(QtCore.Qt.SHIFT | QtCore.Qt.Key_Left,  lambda: self._nudge(self._step(True)*-1))
        sc(QtCore.Qt.SHIFT | QtCore.Qt.Key_Right, lambda: self._nudge(self._step(True)*+1))
        sc(QtCore.Qt.CTRL | QtCore.Qt.Key_Left,  lambda: self._nudge(self._step2()*-1))
        sc(QtCore.Qt.CTRL | QtCore.Qt.Key_Right, lambda: self._nudge(self._step2()*+1))
        sc(QtCore.Qt.Key_Up,    lambda: self._zoom(0.8))
        sc(QtCore.Qt.Key_Down,  lambda: self._zoom(1.25))

        # ---- added: mouse wheel zoom ----
        def _wheel(ev):
            if not self._interaction_enabled():
                return
            dy = ev.angleDelta().y()
            if dy > 0:
                self._zoom(0.8)     # zoom in
            elif dy < 0:
                self._zoom(1.25)    # zoom out
            ev.accept()             # block default pg zoom

        self.wid.wheelEvent = _wheel
        
        
    def _step(self, big: bool):
        if self.integer:
            # 30 or 300 if standard window, but if view is smaller, scale down
            self._ensure_region_visible()
            lo, hi = self.region.getRegion()
            wd = hi - lo
            if wd < 30:
                return max(1.0, float(int(round(wd / 2.0))))
            return self.big_step if big else self.step
        px = self.big_step_px if big else self.step_px
        (xmin, xmax), w = self.vb.viewRange()[0], max(1.0, float(self.vb.width() or 1))
        return (xmax - xmin) * (float(px) / w)

    def _step2(self):
        self._ensure_region_visible()
        lo, hi = self.region.getRegion()
        wd = hi - lo
        return (hi-lo)/self.step2
    
    # ---------- helpers ----------
    def _snap(self, x): return int(round(x)) if self.integer else float(x)

    def _snap_pair(self, lo, hi):
        if not self.integer:
            return float(lo), float(hi)
        min_w = max(1, int(np.ceil(self.min_span)))
        span_i = max(min_w, int(round(hi - lo)))
        c = 0.5 * (float(lo) + float(hi))
        lo_i = int(round(c - 0.5 * span_i))
        hi_i = lo_i + span_i
        return float(lo_i), float(hi_i)

    def _in_vb(self, scene_pos): return self.vb.sceneBoundingRect().contains(scene_pos)

    def _px_to_dx(self, px):
        (xmin, xmax), w = self.vb.viewRange()[0], max(1.0, float(self.vb.width() or 1))
        return (xmax - xmin) * (float(px) / w)

    def _dx_to_px(self, dx):
        (xmin, xmax), w = self.vb.viewRange()[0], max(1.0, float(self.vb.width() or 1))
        span = max(1e-12, xmax - xmin)
        return abs(dx) * (w / span)

    def _full_bounds(self):
        # Prefer explicit bounds
        if self.bounds is not None:
            return float(self.bounds[0]), float(self.bounds[1])
        # Else use data bounds in the ViewBox
        br = self.vb.childrenBounds()  # QRectF over all child items (data coords)
        if br is not None and br.width() > 0:
            return float(br.left()), float(br.right())
        # Fallback: current view (last resort)
        xmin, xmax = self.vb.viewRange()[0]
        return float(xmin), float(xmax)

    def _inside_region_scene(self, scene_pos):
        if not self.region.isVisible():
            return False
        try:
            rect = self.region.boundingRect()
        except Exception:
            return False
        if rect is None:
            return False
        p = self.region.mapFromScene(scene_pos)
        return rect.contains(p)
    
    def _max_span(self):
        if self.bounds is not None:
            return max(0.0, float(self.bounds[1] - self.bounds[0]))
        xmin, xmax = self.vb.viewRange()[0]
        return max(0.0, float(xmax - xmin))

    def _annotator_interval_capture_active(self):
        owner = getattr(self, "owner", None)
        annotator = getattr(owner, "annotator", None)
        return bool(
            owner is not None
            and annotator is not None
            and annotator.isVisible()
            and annotator.check_enabled.isChecked()
            and annotator.current_mode() == "interval"
        )

    def _integer_zoom_ladder(self, max_w: float):
        # Fine control at short windows, then progressively coarser steps.
        base = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 45, 60, 90, 120, 180, 300]
        if max_w <= 0:
            return base
        max_i = max(1, int(np.floor(max_w)))
        ladder = [x for x in base if x <= max_i]
        if not ladder:
            ladder = [1]
        if ladder[-1] < max_i:
            # Extend smoothly to large windows in render mode;
            # avoid single-step jump from 300s to whole-night.
            curr = ladder[-1]
            while curr < max_i:
                nxt = int(np.ceil(curr * 1.5))
                if nxt <= curr:
                    nxt = curr + 1
                if nxt >= max_i:
                    ladder.append(max_i)
                    break
                ladder.append(nxt)
                curr = nxt
        return ladder

    def _clamp_pair(self, lo, hi):
        if self.bounds is None:
            return lo, hi
        b0, b1 = self.bounds
        span = hi - lo
        if span <= 0:
            x = min(max(lo, b0), b1); return x, x
        lo = max(lo, b0); hi = lo + span
        if hi > b1: hi = b1; lo = hi - span
        return lo, hi

    def _enforce_span_limits(self, lo, hi):
        span = hi - lo
        max_span = self._max_span()
        eff_min = min(self.min_span, max_span) if max_span > 0 else self.min_span
        if span < eff_min:
            c = 0.5*(lo + hi); lo, hi = c - 0.5*eff_min, c + 0.5*eff_min
        if max_span > 0 and (hi - lo) > max_span:
            c = 0.5*(lo + hi); lo, hi = c - 0.5*max_span, c + 0.5*max_span
        return self._clamp_pair(lo, hi)

    def _set_region_silent(self, lo, hi):
        self._setting_region = True
        blockers = [QtCore.QSignalBlocker(self.region)]
        for ln in getattr(self.region, "lines", []):
            try: blockers.append(QtCore.QSignalBlocker(ln))
            except Exception: pass
        self.region.setRegion((lo, hi))
        del blockers
        self._setting_region = False
        if abs(float(hi) - float(lo)) > 1e-9:
            self._point_x = None
            self._hide_point_marker()

    def _show_point_marker(self, x):
        try:
            self.point_line.setPos(float(x))
            self.point_line.show()
        except Exception:
            pass

    def _hide_point_marker(self):
        try:
            self.point_line.hide()
        except Exception:
            pass

    def clear_visuals(self):
        self._dragging_bg = False
        self._dragging_move = False
        self._moved = False
        self._press_scene = None
        self._anchor_x = None
        self._point_x = None
        self._pending = None
        self._hide_point_marker()
        try:
            self.region.hide()
        except Exception:
            pass

    def _ensure_region_visible(self, span=None):
        if self.region.isVisible():
            return
        max_span = self._max_span()
        span = min(span or self.click_span, max_span) if max_span > 0 else (span or self.click_span)
        xmin, xmax = self.vb.viewRange()[0]
        c = 0.5*(xmin + xmax)
        lo, hi = c - 0.5*span, c + 0.5*span
        lo, hi = self._enforce_span_limits(lo, hi)
        self._set_region_silent(lo, hi)
        self.region.show()
        self._hide_point_marker()
        self.wid.setFocus()
        self._schedule_emit(lo, hi)

    def _schedule_emit(self, lo, hi, snap: bool = True):
        if snap:
            lo, hi = self._snap_pair(lo, hi)
            lo, hi = self._enforce_span_limits(lo, hi)
            lo, hi = self._snap_pair(lo, hi)
        else:
            lo, hi = self._enforce_span_limits(float(lo), float(hi))
        self._pending = (float(lo), float(hi))
        if not self._emit_timer.isActive():
            self._emit_timer.start(0)

    def _flush_emit(self):
        if self._pending is None:
            return
        if self._pending != self._last_emitted:
            self._last_emitted = self._pending
            self.rangeSelected.emit(*self._pending)

    def dispose(self):
        if getattr(self, "_disposed", False):
            return
        self._disposed = True

        scene = None
        try:
            scene = self.pi.scene()
        except Exception:
            scene = None
        if scene is not None:
            try:
                scene.removeEventFilter(self)
            except Exception:
                pass

        if getattr(self, "region", None) is not None:
            try:
                self.pi.removeItem(self.region)
            except Exception:
                pass
            try:
                self.region.setParentItem(None)
            except Exception:
                pass
            try:
                self.region.deleteLater()
            except Exception:
                pass
            self.region = None

        if getattr(self, "point_line", None) is not None:
            try:
                self.pi.removeItem(self.point_line)
            except Exception:
                pass
            try:
                self.point_line.deleteLater()
            except Exception:
                pass
            self.point_line = None

        sc_list = getattr(self, "_sc", None)
        if sc_list is not None:
            for sc in sc_list:
                try:
                    sc.activated.disconnect()
                except Exception:
                    pass
                try:
                    sc.setParent(None)
                except Exception:
                    pass
                try:
                    sc.deleteLater()
                except Exception:
                    pass
            sc_list.clear()

        if getattr(self, "_emit_timer", None) is not None:
            try:
                self._emit_timer.stop()
            except Exception:
                pass

        try:
            QtCore.QObject.deleteLater(self)
        except Exception:
            pass

            
    # ---------- mouse (event filter) ----------
    def eventFilter(self, obj, ev):
        if obj is not self.pi.scene():
            return False
        if not self._interaction_enabled():
            return False
        if self._region_active:
            return False  # let LRI handle its own drags/resizes

        et = ev.type()


        if et == QtCore.QEvent.GraphicsSceneMouseDoubleClick and ev.button() == QtCore.Qt.LeftButton:
            if not self._in_vb(ev.scenePos()):
                return False

            # cancel any press/drag in progress
            self._dragging_bg = False
            self._dragging_move = False

            # check if click is inside the region (in scene coords)
            inside = False
            if self.region.isVisible():
                # scene-space hit test for robustness
                try:
                    rect = self.region.boundingRect()
                except Exception:
                    rect = None
                if rect is not None:
                    r = self.region.mapRectToScene(rect)
                    inside = r.contains(ev.scenePos())
            
            if inside:
                # shrink to one epoch centered at click
                x = self.vb.mapSceneToView(ev.scenePos()).x()
                half = 0.5 * self.click_span
                lo2, hi2 = self._enforce_span_limits(*self._clamp_pair(x - half, x + half))
            else:
                # expand to whole recording (bounds or data extent)
                lo2, hi2 = self._full_bounds()
                if self.bounds is not None:
                    lo2, hi2 = self._enforce_span_limits(lo2, hi2)

            self._set_region_silent(lo2, hi2)
            self.region.show()
            self._schedule_emit(lo2, hi2)
            return True

                  
        elif et == QtCore.QEvent.GraphicsSceneMousePress and ev.button() == QtCore.Qt.LeftButton:
            if not self._in_vb(ev.scenePos()):
                return False
            x = self.vb.mapSceneToView(ev.scenePos()).x()
            interval_capture = self._annotator_interval_capture_active()

            if self.region.isVisible():
                lo, hi = self.region.getRegion()
                w = max(hi - lo, 0.0)
                w_px = self._dx_to_px(w)
                tol_dx = self._px_to_dx(self.edge_tol_px)

                # THIN region -> move-only anywhere inside [lo, hi]
                if w_px <= self.thin_px and (lo - tol_dx) <= x <= (hi + tol_dx):
                    self._start_move_drag(x, lo, hi); return True

                # WIDE region:
                # inside core (away from edges) -> move-only
                if (lo + tol_dx) <= x <= (hi - tol_dx):
                    self._start_move_drag(x, lo, hi); return True

                # near edges -> let LRI resize
                if (lo - tol_dx) <= x <= (hi + tol_dx):
                    return False

            # outside region -> background selection drag
            self._dragging_bg = True; self._moved = False
            self._press_scene = ev.scenePos()
            self._anchor_x = self._snap(x)
            self._point_x = None
            if interval_capture:
                # Make the dragged interval visible immediately instead of
                # falling back to click-span behavior on release.
                self._set_region_silent(self._anchor_x, self._anchor_x)
                self.region.hide()
                self._show_point_marker(self._anchor_x)
            # Claim press so subsequent move/release stay with selector drag logic.
            return True

        elif et == QtCore.QEvent.GraphicsSceneMouseMove:
            if self._dragging_move:
                if not self._in_vb(ev.scenePos()):
                    return True
                x = self._snap(self.vb.mapSceneToView(ev.scenePos()).x())
                c = x - self._move_offset
                half = 0.5 * self._move_width
                lo, hi = c - half, c + half
                lo, hi = self._enforce_span_limits(lo, hi)
                self._set_region_silent(lo, hi)
                self.region.show()
                self._schedule_emit(lo, hi)
                self._moved = True
                return True

            if self._dragging_bg:
                if not self._in_vb(ev.scenePos()):
                    return True
                if (ev.scenePos() - self._press_scene).manhattanLength() >= self.drag_thresh_px:
                    self._moved = True
                if self._moved:
                    self._point_x = None
                    self._hide_point_marker()
                    x = self._snap(self.vb.mapSceneToView(ev.scenePos()).x())
                    lo, hi = sorted((self._anchor_x, x))
                    lo, hi = self._enforce_span_limits(*self._clamp_pair(lo, hi))
                    self._set_region_silent(lo, hi)
                    self.region.show()
                    self._schedule_emit(lo, hi)
                return True

        elif et == QtCore.QEvent.GraphicsSceneMouseRelease and ev.button() == QtCore.Qt.LeftButton:
            if self._dragging_move:
                self._dragging_move = False
                return True
            if self._dragging_bg:
                self._dragging_bg = False
                if not self._moved:
                    if self._annotator_interval_capture_active():
                        self._point_x = float(self._anchor_x)
                        self._set_region_silent(self._point_x, self._point_x)
                        self.region.hide()
                        self._show_point_marker(self._point_x)
                        return True
                    x = self._anchor_x
                    half = 0.5 * self.click_span
                    lo, hi = self._enforce_span_limits(*self._clamp_pair(x - half, x + half))
                    self._set_region_silent(lo, hi)
                    self.region.show()
                    self._schedule_emit(lo, hi)
                return True

        return False

    def _start_move_drag(self, x, lo, hi):
        self._dragging_move = True
        self._moved = False
        self._press_scene = None
        self._move_width = max(hi - lo, self.min_span)
        c = 0.5 * (lo + hi)
        self._move_offset = x - c

    # ---------- region + keys ----------
    def _on_region_changed(self):
        if self._setting_region:
            return
        lo, hi = self.region.getRegion()
        lo, hi = self._snap_pair(lo, hi)
        lo, hi = self._enforce_span_limits(lo, hi)
        self._set_region_silent(lo, hi)
        self._schedule_emit(lo, hi)

    def _on_region_finished(self):
        self._region_active = False
        lo, hi = self.region.getRegion()
        self._schedule_emit(*self._enforce_span_limits(lo, hi))

    def _nudge(self, dx):
        if not self._interaction_enabled():
            return
        self._ensure_region_visible()        
        lo, hi = self.region.getRegion()
        lo, hi = self._snap_pair(lo, hi)
        lo, hi = self._clamp_pair(lo + dx, hi + dx)
        lo, hi = self._enforce_span_limits(lo, hi)
        self._set_region_silent(lo, hi)
        self._schedule_emit(lo, hi)

    def _zoom(self, factor):
        if not self._interaction_enabled():
            return
        self._ensure_region_visible()
        lo, hi = self.region.getRegion()
        c = 0.5*(lo + hi)
        w = max(hi - lo, 0.0)
        max_w = self._max_span()
        min_w = min(self.min_span, max_w) if max_w > 0 else self.min_span

        if self.integer:
            ladder = self._integer_zoom_ladder(max_w)
            curr = int(max(1, round(w)))
            if factor < 1.0:
                # zoom in: next smaller rung
                smaller = [x for x in ladder if x < curr]
                new_w = float(smaller[-1] if smaller else ladder[0])
            else:
                # zoom out: next larger rung
                larger = [x for x in ladder if x > curr]
                new_w = float(larger[0] if larger else ladder[-1])
            lo2, hi2 = c - 0.5 * new_w, c + 0.5 * new_w
            lo2, hi2 = self._enforce_span_limits(lo2, hi2)
            self._set_region_silent(lo2, hi2)
            # Preserve exact midpoint while zooming through discrete rung widths.
            self._schedule_emit(lo2, hi2, snap=False)
            return

        if w <= 0:
            w = min(max_w if max_w > 0 else self.click_span, self.click_span)
        new_w = w * float(factor)
        if max_w > 0:
            new_w = min(max(new_w, min_w), max_w)
        else:
            new_w = max(new_w, min_w)
        lo2, hi2 = c - 0.5*new_w, c + 0.5*new_w
        lo2, hi2 = self._enforce_span_limits(lo2, hi2)
        self._set_region_silent(lo2, hi2)
        self._schedule_emit(lo2, hi2)

    # ---------- lifecycle ----------
    def detach(self):
        try: self.pi.scene().removeEventFilter(self)
        except Exception: pass
        for sig, slot in [
            (self.region.sigRegionChanged, self._on_region_changed),
        ]:
            try: sig.disconnect(slot)
            except TypeError: pass
        try: self.pi.removeItem(self.region)
        except Exception: pass
        for s in getattr(self, "_sc", []):
            try: s.setParent(None)
            except Exception: pass
            
    # programmatically set range
    def setRange(self, lo: float, hi: float, emit: bool = True, grab_focus: bool = False):
        lo, hi = self._enforce_span_limits(lo, hi)
        self._set_region_silent(lo, hi)
        self.region.show()
        if grab_focus:
            self.wid.setFocus()
        if emit:
            self._schedule_emit(lo, hi)


# --------------------------------------------------------------------------------
#
# text updater
#
# --------------------------------------------------------------------------------


class TextBatch(pg.GraphicsObject):
    def __init__(self, viewbox: pg.ViewBox, font: QtGui.QFont=None,
                 color=(255,255,255), mode='device',
                 bg=(0,0,0,170), pad=(6,3), radius=3, outline=None):
        super().__init__()
        self.vb = viewbox
        self.mode = mode
        self.font = font or QtGui.QFont("Sans Serif", 10)
        self.color = pg.mkColor(color)
        self.bg_brush = None if bg is None else pg.mkBrush(bg)
        self.bg_pen = QtGui.QPen(QtCore.Qt.NoPen) if not outline else pg.mkPen(outline)
        self.pad_x, self.pad_y = pad
        self.radius = radius
        self._x = np.empty(0); self._y = np.empty(0)
        self._labels = []
        self._stat = {}
        self._bbox = QtCore.QRectF()

                
    def setData(self, x, y, labels):
        x = np.asarray(x, float); y = np.asarray(y, float)
        assert len(x) == len(y) == len(labels)
        self._x, self._y = x, y
        self._labels = list(map(str, labels))
        self._stat.clear()
        self._rebuild_bbox()
        self.update()

    def setMode(self, mode):
        self.mode = mode
        self.update()

    def _rebuild_bbox(self):
        if self._x.size == 0:
            self.prepareGeometryChange()
            self._bbox = QtCore.QRectF()
            return
        xmin, xmax = np.min(self._x), np.max(self._x)
        ymin, ymax = np.min(self._y), np.max(self._y)
        self.prepareGeometryChange()
        self._bbox = QtCore.QRectF(xmin, ymin, xmax-xmin, ymax-ymin)

    def boundingRect(self):
        return self._bbox

    def _qstatic(self, s: str) -> QtGui.QStaticText:
        st = self._stat.get(s)
        if st is None:
            st = QtGui.QStaticText(s)
            st.setTextFormat(QtCore.Qt.PlainText)
            st.prepare(font=self.font)
            self._stat[s] = st
        return st

    def _draw_with_bg(self, p: QtGui.QPainter, top_left: QtCore.QPointF, st: QtGui.QStaticText):
        if self.bg_brush is not None:
            sz = st.size()  # QSizeF in current painter coord system
            r = QtCore.QRectF(top_left.x() - self.pad_x,
                              top_left.y() - self.pad_y,
                              sz.width() + 2*self.pad_x,
                              sz.height() + 2*self.pad_y)
            p.setPen(self.bg_pen)
            p.setBrush(self.bg_brush)
            if self.radius:
                p.drawRoundedRect(r, self.radius, self.radius)
            else:
                p.drawRect(r)
        # text color
        p.setPen(pg.mkPen(self.color))
        p.drawStaticText(top_left, st)

    def paint(self, p: QtGui.QPainter, opt, widget=None):
        if self._x.size == 0:
            return
        (xmin, xmax), (ymin, ymax) = self.vb.viewRange()
        m = (self._x >= xmin) & (self._x <= xmax) & (self._y >= ymin) & (self._y <= ymax)
        if not np.any(m):
            return

        p.setFont(self.font)

        if self.mode == 'data':
            # text and bg scale with view (data coords)
            for xi, yi, lab in zip(self._x[m], self._y[m], np.asarray(self._labels, object)[m]):
                st = self._qstatic(lab)
                self._draw_with_bg(p, QtCore.QPointF(xi, yi), st)
            return

        # device mode: constant pixel size (screen coords)
        p.save()
        p.resetTransform()
        mv = self.vb.mapViewToDevice
        for xi, yi, lab in zip(self._x[m], self._y[m], np.asarray(self._labels, object)[m]):
            dp = mv(QtCore.QPointF(float(xi), float(yi)))
            if dp is None:
                continue
            st = self._qstatic(lab)
            self._draw_with_bg(p, dp, st)
        p.restore()


# --------------------------------------------------------------------------------
# rect track mgr

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui, QtCore

class TrackManager:
    def __init__(self, plot):
        self.plot = plot
        self.tracks = {}  # name -> dict(item, color, pen, visible)

        # --- adaptive border controls (added) ---
        self._vb = getattr(self.plot, "getViewBox", lambda: None)()
        self._pen_thresh = 1.0  # data units per screen pixel; tune to taste
        self._pen_on = pg.mkPen((0, 0, 0, 120), width=1, cosmetic=True)  # thin, translucent
        self._pen_off = pg.mkPen(None)
        self._borders_on = None  # unknown until first check

        self._border_timer = QtCore.QTimer()
        self._border_timer.setSingleShot(True)
        self._border_timer.setInterval(50)
        self._border_timer.timeout.connect(self._update_all_pens)

        if self._vb is not None:
            self._vb.sigRangeChanged.connect(lambda *_: self._border_timer.start())
        # ----------------------------------------

    def _want_borders(self):
        if self._vb is None:
            return True
        sx, sy = self._vb.viewPixelSize()
        return max(sx, sy) < self._pen_thresh

    def _effective_pen(self, orig_pen):
        """Return the pen to apply now given zoom level."""
        want = self._want_borders()
        return (self._pen_on if orig_pen is None else pg.mkPen(orig_pen)) if want else self._pen_off

    def _update_all_pens(self):
        want = self._want_borders()
        if want == self._borders_on:
            return
        self._borders_on = want
        for t in self.tracks.values():
            # Respect original pen when borders are on; hide borders when off
            eff = self._effective_pen(t["pen"])
            t["item"].setOpts(pen=eff)

    def update_track(self, name, x0, x1, y0, y1, color=None, pen=None, reduce=False, wpx=1 ):
        """
        Replace the given track with new rectangles spanning [x0,x1] × [y0,y1].
        Arrays must be equal length.
        """
        x0 = np.asarray(x0)
        x1 = np.asarray(x1)
        y0 = np.asarray(y0)
        y1 = np.asarray(y1)
        assert x0.shape == x1.shape == y0.shape == y1.shape

        if color is None and name in self.tracks:
            color = self.tracks[name]["color"]
        if color is None:
            color = (200, 250, 240)

        if pen is None and name in self.tracks:
            pen = self.tracks[name]["pen"]  # store original user pen (may be tuple/QPen/None)
        # default black edge if never set before
        if pen is None and name not in self.tracks:
            pen = (0, 0, 0)

        # remove old
        if name in self.tracks:
            self.plot.removeItem(self.tracks[name]["item"])

        # make line,box effect
        vb = self.plot.getViewBox()
        if reduce:
            x0_all, x1_all, y0_all, y1_all = build_dual_rect_arrays(vb, x0, x1, y0, y1, hfrac=0.5, wpx=wpx)
        else:
            x0_all = x0
            x1_all = x1
            y0_all = y0
            y1_all = y1

        # Create item with adaptive pen
        eff_pen = self._effective_pen(pen)
        item = pg.BarGraphItem(x0=x0_all, x1=x1_all, y0=y0_all, y1=y1_all, brush=color, pen=eff_pen, name=name)
        self.plot.addItem(item)
        self.tracks[name] = {
            "item": item, "color": color, "pen": pen, "visible": True,
            "x0": x0, "x1": x1, "y0": y0, "y1": y1, "reduce": reduce,
        }

        # Initialize border state if first time
        if self._borders_on is None:
            self._borders_on = self._want_borders()

    def refresh_wpx(self, wpx):
        """Rebuild all reduce=True tracks with a new tick-width (wpx)."""
        vb = self.plot.getViewBox()
        for name, t in self.tracks.items():
            if not t["reduce"]:
                continue
            x0_all, x1_all, y0_all, y1_all = build_dual_rect_arrays(
                vb, t["x0"], t["x1"], t["y0"], t["y1"], hfrac=0.5, wpx=wpx)
            t["item"].setOpts(x0=x0_all, x1=x1_all, y0=y0_all, y1=y1_all)

    def toggle(self, name, on=True):
        if name in self.tracks:
            self.tracks[name]["visible"] = on
            self.tracks[name]["item"].setVisible(on)

    def clear(self, name=None):
        if name is None:
            for t in self.tracks.values():
                self.plot.removeItem(t["item"])
            self.tracks.clear()
        else:
            if name in self.tracks:
                self.plot.removeItem(self.tracks[name]["item"])
                self.tracks.pop(name)



                
# ------------------------------------------------------------

import numpy as np

def build_dual_rect_arrays(vb, x0, x1, y0, y1, hfrac=0.66, wpx=1):
    """
    Returns x0_all, x1_all, y0_all, y1_all that include:
      - a 1-px wide full-height strip at each left edge
      - a 66% height body for the remainder
    vb: pyqtgraph ViewBox (for pixel→data conversion)
    """
    x0 = np.asarray(x0); x1 = np.asarray(x1)
    y0 = np.asarray(y0); y1 = np.asarray(y1)

    # normalize heights
    ylo = np.minimum(y0, y1)
    yhi = np.maximum(y0, y1)
    h   = yhi - ylo
    ym  = 0.5 * (yhi + ylo)

    # n pixels in data units (wpx rounded to integer to avoid sub-pixel blending)
    dx, _ = vb.viewPixelSize()
    w1 = dx * max(1, round(wpx))

    # thin strip: [x0, x0+w1] at full height
    x0_thin = x0
    x1_thin = x0 + w1
    y0_thin = ylo
    y1_thin = yhi

    # body: [x0+w1, x1] at hfrac height centered
    y0_body = ym - 0.5 * h * hfrac
    y1_body = ym + 0.5 * h * hfrac
    x0_body = x0 + w1
    x1_body = x1

    # concatenate both sets
    x0_all = np.concatenate([x0_thin, x0_body])
    x1_all = np.concatenate([x1_thin, x1_body])
    y0_all = np.concatenate([y0_thin, y0_body])
    y1_all = np.concatenate([y1_thin, y1_body])

    return x0_all, x1_all, y0_all, y1_all



# ------------------------------------------------------------
        
@staticmethod
def _ensure_min_px_width(vb, x0, x1, px=1):
    x0 = np.asarray(x0, dtype=float)
    x1 = np.asarray(x1, dtype=float)

    dx, _ = vb.viewPixelSize()
    wmin = px * dx
    w = x1 - x0
    too_narrow = w < wmin
    xc = 0.5 * (x0 + x1)
    x0a = np.where(too_narrow, xc - 0.5*wmin, x0)
    x1a = np.where(too_narrow, xc + 0.5*wmin, x1)
    return x0a, x1a
