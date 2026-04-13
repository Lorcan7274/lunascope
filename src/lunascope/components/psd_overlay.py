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

import numpy as np

import pyqtgraph as pg
from scipy.signal import welch
from PySide6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout,
    QPushButton, QDoubleSpinBox, QLabel, QCheckBox,
)
from PySide6.QtCore import Qt, QTimer, QEvent, QObject, QPoint, QSize
from PySide6.QtGui import QColor, QPainter, QPen


# ---------------------------------------------------------------------------
# Default geometry
# ---------------------------------------------------------------------------

_INSET_W      = 280    # default width  (pixels)
_INSET_H      = 180    # default height (pixels)
_INSET_MARGIN = 10     # gap from bottom-right corner of pg1
_TITLE_H      = 20     # title-bar height
_GRIP_SZ      = 14     # resize grip size
_DEBOUNCE_MS  = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _psd_nperseg(n_samples: int) -> int:
    """Choose welch nperseg that keeps computation cheap.

    Frequency resolution = fs / nperseg, so we want nperseg as large as
    reasonable.  For short windows (few hundred samples) we're already
    limited.  For long windows the bottleneck is number of FFT segments,
    so we cap nperseg to bound the total work.

    n_samples   nperseg cap
    ---------   -----------
    < 512        n_samples  (no cap, short window)
    < 8 192      512
    ≥ 8 192      1 024
    """
    if n_samples < 512:
        return n_samples
    if n_samples < 8192:
        return 512
    return 1024


# ---------------------------------------------------------------------------
# Resize event filter
# ---------------------------------------------------------------------------

class _ResizeEventFilter(QObject):
    def __init__(self, callback):
        super().__init__()
        self._cb = callback

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Resize:
            self._cb()
        return False


# ---------------------------------------------------------------------------
# Title bar  (drag handle + label + close button)
# ---------------------------------------------------------------------------

class _TitleBar(QWidget):

    def __init__(self, container, close_cb):
        super().__init__(container)
        self.setFixedHeight(_TITLE_H)
        self.setCursor(Qt.SizeAllCursor)
        self.setStyleSheet("background: #2a2a3a; border-bottom: 1px solid #555;")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 2, 0)
        lay.setSpacing(0)

        lbl = QLabel("PSD", self)
        lbl.setStyleSheet("color: #cccccc; font-size: 10px; font-weight: bold;"
                          " background: transparent; border: none;")
        lay.addWidget(lbl)
        lay.addStretch()

        close_btn = QPushButton("✕", self)
        close_btn.setFixedSize(16, 16)
        close_btn.setStyleSheet(
            "QPushButton { color:#aaa; background:transparent; border:none; font-size:10px; }"
            "QPushButton:hover { color:#fff; }"
        )
        close_btn.clicked.connect(close_cb)
        lay.addWidget(close_btn)

        self._drag_active = False
        self._drag_offset = QPoint()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag_active = True
            container = self.parent()
            pw = container.parentWidget()
            origin = pw.mapToGlobal(container.pos()) if pw else container.pos()
            self._drag_offset = ev.globalPosition().toPoint() - origin

    def mouseMoveEvent(self, ev):
        if not self._drag_active:
            return
        container = self.parent()
        pw = container.parentWidget()
        new_global = ev.globalPosition().toPoint() - self._drag_offset
        new_local  = pw.mapFromGlobal(new_global) if pw else new_global
        container.move(new_local)
        container._user_positioned = True

    def mouseReleaseEvent(self, ev):
        self._drag_active = False


# ---------------------------------------------------------------------------
# Resize grip  (bottom-right corner handle)
# ---------------------------------------------------------------------------

class _ResizeGrip(QWidget):

    def __init__(self, container):
        super().__init__(container)
        self.setFixedSize(_GRIP_SZ, _GRIP_SZ)
        self.setCursor(Qt.SizeFDiagCursor)
        self._drag_active = False
        self._press_global = QPoint()
        self._press_size   = QSize()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._drag_active  = True
            self._press_global = ev.globalPosition().toPoint()
            self._press_size   = self.parent().size()

    def mouseMoveEvent(self, ev):
        if not self._drag_active:
            return
        delta = ev.globalPosition().toPoint() - self._press_global
        container = self.parent()
        new_w = max(180, self._press_size.width()  + delta.x())
        new_h = max(120, self._press_size.height() + delta.y())
        container.resize(new_w, new_h)
        container._user_positioned = True

    def mouseReleaseEvent(self, ev):
        self._drag_active = False

    def paintEvent(self, ev):
        p = QPainter(self)
        pen = QPen(QColor(130, 130, 130))
        pen.setWidth(1)
        p.setPen(pen)
        for i in (4, 7, 10):
            p.drawLine(i, _GRIP_SZ - 1, _GRIP_SZ - 1, i)
        p.end()


# ---------------------------------------------------------------------------
# Container  (title bar + PlotWidget + resize grip)
# ---------------------------------------------------------------------------

class _PSDContainer(QFrame):
    """
    Floating, draggable, resizable frame that holds the PSD PlotWidget.
    Parent must be a plain QWidget (sibling of pg1), NOT pg1 itself.
    """

    def __init__(self, parent, close_cb):
        super().__init__(parent)
        self._user_positioned = False   # True once user has dragged/resized

        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setStyleSheet(
            "QFrame { background: rgba(20,20,30,210);"
            " border: 1px solid #555; border-radius: 3px; }"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._title_bar = _TitleBar(self, close_cb)
        root.addWidget(self._title_bar)

        self.plot_widget = pg.PlotWidget(self)
        self.plot_widget.setBackground((20, 20, 30, 0))   # transparent — frame provides bg
        root.addWidget(self.plot_widget)

        self._grip = _ResizeGrip(self)
        self._reposition_grip()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._reposition_grip()

    def _reposition_grip(self):
        self._grip.move(self.width() - _GRIP_SZ, self.height() - _GRIP_SZ)
        self._grip.raise_()


# ---------------------------------------------------------------------------
# PSD overlay mixin
# ---------------------------------------------------------------------------

class PSDOverlayMixin:
    """
    On-the-fly PSD inset, draggable and resizable by the user.

    The container is a SIBLING of pg1 (child of pg1.parentWidget()), never
    a child of pg1 itself, to avoid corrupting pg1's viewbox geometry.
    """

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_psd_overlay(self):
        self._psd_visible   = False
        self._psd_use_db    = False

        self._psd_debounce = QTimer()
        self._psd_debounce.setSingleShot(True)
        self._psd_debounce.setInterval(_DEBOUNCE_MS)
        self._psd_debounce.timeout.connect(self._psd_compute_and_draw)

        # ── container is a sibling of pg1 ─────────────────────────────
        pg1        = self.ui.pg1
        pg1_parent = pg1.parentWidget() or self.ui
        self._psd_pg1_parent = pg1_parent

        self._psd_container = _PSDContainer(
            parent   = pg1_parent,
            close_cb = self._psd_close,
        )
        self._psd_container.hide()
        self._psd_curves = []

        # Convenient alias to the inner plot widget
        self._psd_inset = self._psd_container.plot_widget
        self._psd_style_inset()

        # ── re-anchor when pg1 or its parent resizes ──────────────────
        self._psd_resize_filter = _ResizeEventFilter(self._psd_anchor_inset)
        pg1.installEventFilter(self._psd_resize_filter)
        self._psd_parent_resize_filter = _ResizeEventFilter(self._psd_anchor_inset)
        pg1_parent.installEventFilter(self._psd_parent_resize_filter)

        # ── toolbar ───────────────────────────────────────────────────
        self._psd_add_toolbar_widgets()

        # ── update on channel selection changes ───────────────────────
        self._psd_connect_channel_changes()

    # ------------------------------------------------------------------

    def _psd_style_inset(self):
        pw = self._psd_inset
        pw.showAxis("left", True)
        pw.showAxis("bottom", True)
        pw.getPlotItem().hideButtons()
        pw.setMenuEnabled(False)
        vb = pw.getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.setDefaultPadding(0.02)

        for ax_name in ("left", "bottom"):
            ax = pw.getAxis(ax_name)
            ax.setTextPen(pg.mkPen((200, 200, 200)))
            ax.setPen(pg.mkPen((120, 120, 120)))
            ax.setStyle(tickLength=-4, tickTextOffset=2)
            if ax_name == "left":
                ax.setWidth(36)

        pw.getAxis("bottom").setLabel(
            "Hz", **{"color": "#aaaaaa", "font-size": "9pt"}
        )

    # ------------------------------------------------------------------

    def _psd_anchor_inset(self):
        """Move container to bottom-right of pg1 — only if user hasn't repositioned it."""
        if not hasattr(self, "_psd_container"):
            return
        if self._psd_container._user_positioned:
            return   # respect user's chosen position

        pg1        = self.ui.pg1
        pg1_parent = self._psd_pg1_parent

        br_local = QPoint(pg1.width()  - _INSET_MARGIN,
                          pg1.height() - _INSET_MARGIN)
        br = pg1.mapTo(pg1_parent, br_local)
        x  = br.x() - _INSET_W
        y  = br.y() - _INSET_H
        self._psd_container.setGeometry(x, y, _INSET_W, _INSET_H)

    # ------------------------------------------------------------------

    def _psd_add_toolbar_widgets(self):
        banner_host = getattr(self.ui, "ctrframe", None)
        parent = banner_host if banner_host is not None else self.ui.butt_render.parentWidget()
        if parent is None:
            return

        # ------------------------------------------------------------------ #
        # Full-width PSD section bar                                           #
        # ------------------------------------------------------------------ #
        bar = QWidget(parent)
        root = QVBoxLayout(bar)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(1)

        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(3)

        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(3)

        # Toggle button — compact label
        self._psd_toggle_btn = QPushButton("On", bar)
        self._psd_toggle_btn.setCheckable(True)
        self._psd_toggle_btn.setChecked(False)
        self._psd_toggle_btn.setFixedWidth(26)
        self._psd_toggle_btn.setToolTip("Toggle on-the-fly PSD inset")
        self._psd_toggle_btn.toggled.connect(self._psd_on_toggle)

        # Frequency range
        f_lbl = QLabel("f:", bar)

        self._psd_fmin_spin = QDoubleSpinBox(bar)
        self._psd_fmin_spin.setDecimals(0)
        self._psd_fmin_spin.setRange(0, 499)
        self._psd_fmin_spin.setValue(0)
        self._psd_fmin_spin.setSingleStep(1)
        self._psd_fmin_spin.setFixedWidth(34)
        self._psd_fmin_spin.setToolTip("Min PSD frequency (Hz)")
        self._psd_fmin_spin.valueChanged.connect(self._psd_on_fmin_changed)

        dash_lbl = QLabel("–", bar)

        self._psd_fmax_spin = QDoubleSpinBox(bar)
        self._psd_fmax_spin.setDecimals(0)
        self._psd_fmax_spin.setRange(1, 500)
        self._psd_fmax_spin.setValue(30)
        self._psd_fmax_spin.setSingleStep(5)
        self._psd_fmax_spin.setFixedWidth(34)
        self._psd_fmax_spin.setToolTip("Max PSD frequency (Hz)")
        self._psd_fmax_spin.valueChanged.connect(self._psd_on_fmax_changed)

        hz_lbl = QLabel("Hz", bar)

        self._psd_db_chk = QCheckBox("dB", bar)
        self._psd_db_chk.setChecked(False)
        self._psd_db_chk.setToolTip("Show PSD in decibels (10·log₁₀)")
        self._psd_db_chk.toggled.connect(self._psd_on_db_toggled)

        row1.addWidget(self._psd_toggle_btn)
        row1.addWidget(self._psd_db_chk)
        row1.addStretch(1)

        row2.addWidget(f_lbl)
        row2.addWidget(self._psd_fmin_spin)
        row2.addWidget(dash_lbl)
        row2.addWidget(self._psd_fmax_spin)
        row2.addWidget(hz_lbl)
        row2.addStretch(1)

        root.addLayout(row1)
        root.addLayout(row2)

        self._psd_toolbar_holder = bar
        if hasattr(self, "_style_control_banner"):
            self._style_control_banner()

    # ------------------------------------------------------------------

    def _psd_connect_channel_changes(self):
        try:
            mdl = self.ui.tbl_desc_signals.model()
            src = mdl.sourceModel() if hasattr(mdl, "sourceModel") else mdl
            src.dataChanged.connect(self._psd_on_channel_selection_changed)
        except Exception:
            pass

    def _psd_on_channel_selection_changed(self, *_):
        if self._psd_visible:
            self._psd_schedule_update()

    # ------------------------------------------------------------------
    # Toggle / close / settings
    # ------------------------------------------------------------------

    def _psd_on_toggle(self, checked: bool):
        self._psd_visible = checked
        # Render mode uses segsrv-backed PSD. Non-render mode computes PSD
        # directly from the cached visible traces and does not need backend prep.
        if getattr(self, "rendered", False):
            try:
                self.ss.set_psd_mode(checked)
            except Exception as e:
                print(f"[PSD] set_psd_mode({checked}): {e}")
        if checked:
            self._psd_anchor_inset()
            self._psd_container.show()
            self._psd_container.raise_()
            self._psd_schedule_update()
        else:
            self._psd_container.hide()

    def _psd_close(self):
        """Called by the ✕ button inside the container."""
        self._psd_toggle_btn.setChecked(False)   # triggers _psd_on_toggle(False)

    def _psd_on_db_toggled(self, checked: bool):
        self._psd_use_db = checked
        self._psd_schedule_update()

    def _psd_on_fmin_changed(self, value: float):
        # keep fmin < fmax
        if value >= self._psd_fmax_spin.value():
            self._psd_fmax_spin.blockSignals(True)
            self._psd_fmax_spin.setValue(value + 1)
            self._psd_fmax_spin.blockSignals(False)
        self._psd_schedule_update()

    def _psd_on_fmax_changed(self, value: float):
        # keep fmax > fmin
        if value <= self._psd_fmin_spin.value():
            self._psd_fmin_spin.blockSignals(True)
            self._psd_fmin_spin.setValue(value - 1)
            self._psd_fmin_spin.blockSignals(False)
        self._psd_schedule_update()

    # ------------------------------------------------------------------
    # Scheduling hooks
    # ------------------------------------------------------------------

    def _psd_adaptive_debounce_ms(self) -> int:
        """Return debounce delay scaled to current window duration.

        Shorter windows → cheaper computation → we can afford a tighter
        debounce without risking UI stutter.

        Duration      Debounce
        --------      --------
        < 30 s        50 ms
        30 – 120 s    100 ms
        > 120 s       200 ms
        """
        try:
            vr       = self.ui.pg1.getViewBox().viewRange()
            duration = vr[0][1] - vr[0][0]   # seconds
        except Exception:
            return _DEBOUNCE_MS
        if duration < 30:
            return 50
        if duration < 120:
            return 100
        return 200

    def _psd_schedule_update(self, *_):
        if not self._psd_visible:
            return
        self._psd_debounce.start(self._psd_adaptive_debounce_ms())

    def _psd_overlay_on_range_changed(self):
        """Call from on_window_range, AFTER _update_pg1 has populated cache."""
        self._psd_schedule_update()

    def _psd_overlay_on_new_data(self):
        """Call after render / simple-render completes."""
        if self._psd_visible:
            self._psd_schedule_update()

    def _psd_overlay_on_trace_redraw(self):
        """Call after the visible trace cache has been rebuilt."""
        if self._psd_visible:
            self._psd_schedule_update()

    # ------------------------------------------------------------------
    # Compute & draw
    # ------------------------------------------------------------------

    def _psd_compute_and_draw(self):
        if not self._psd_visible:
            return
        channels, freqs_list, psd_list, colors = self._psd_gather_data()
        if not channels:
            self._psd_clear_curves()
            return
        fmax = float(self._psd_fmax_spin.value())
        self._psd_draw(channels, freqs_list, psd_list, colors, fmax)

    # ------------------------------------------------------------------

    def _psd_gather_data(self):
        cache = getattr(self, "_pg1_channel_cache", [])
        if not cache:
            return [], [], [], []

        channels, freqs_list, psd_list, colors = [], [], [], []

        render_entries    = [(i, e) for i, e in enumerate(cache) if e.get("srv") is not None]
        nonrender_entries = [(i, e) for i, e in enumerate(cache) if e.get("srv") is None]

        # ── render mode: segsrv already has PSD mode on and window set ──────
        for i, entry in render_entries:
            ch    = entry.get("ch")
            srv   = entry.get("srv")
            color = entry.get("color") or "gray"
            try:
                srv.get_scaled_signal(ch, i)
                f   = np.asarray(srv.get_psd_freqs(ch))
                pxx = np.asarray(srv.get_psd_power(ch))
                if len(f) < 4:
                    continue
                channels.append(ch)
                freqs_list.append(f)
                psd_list.append(pxx)
                colors.append(color)
            except Exception as e:
                print(f"[PSD] render ch={ch!r}: {e}")

        # ── non-render mode: compute directly from the cached visible traces ─
        if nonrender_entries:
            for i, entry in nonrender_entries:
                ch = entry.get("ch")
                color = entry.get("color") or "gray"
                try:
                    f, pxx = self._psd_from_cached_trace(entry)
                    if len(f) < 4:
                        continue
                    channels.append(ch)
                    freqs_list.append(f)
                    psd_list.append(pxx)
                    colors.append(color)
                except Exception as e:
                    print(f"[PSD] non-render ch={ch!r}: {e}")

        return channels, freqs_list, psd_list, colors

    def _psd_from_cached_trace(self, entry):
        x = np.asarray(entry.get("x"), dtype=float)
        y = np.asarray(entry.get("y_phys"), dtype=float)
        if x.size < 8 or y.size < 8 or x.size != y.size:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)

        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        x = x[finite]
        y = y[finite]
        if x.size < 8:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)

        dx = np.diff(x)
        dx = dx[np.isfinite(dx) & (dx > 0)]
        if dx.size == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)
        dt = float(np.median(dx))
        if not np.isfinite(dt) or dt <= 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)

        fs = 1.0 / dt
        nperseg = _psd_nperseg(int(y.size))
        if nperseg < 8:
            return np.empty(0, dtype=float), np.empty(0, dtype=float)

        noverlap = min(nperseg // 2, nperseg - 1)
        freqs, pxx = welch(
            y,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend="constant",
            scaling="density",
        )
        return np.asarray(freqs, dtype=float), np.asarray(pxx, dtype=float)

    # ------------------------------------------------------------------

    def _psd_dummy_data(self):
        checked = []
        if hasattr(self, "ui"):
            try:
                checked = list(self.ui.tbl_desc_signals.checked())
            except Exception:
                pass
        if not checked:
            checked = ["CH1", "CH2"]

        rng = np.random.default_rng(42)
        channels, freqs_list, psd_list, colors = [], [], [], []
        f = np.linspace(0.5, 30, 200)
        for i, ch in enumerate(checked[:8]):
            pxx  = 10.0 / (f + 0.5) + 0.5 * np.exp(-0.5 * ((f - 12) / 2) ** 2)
            pxx += rng.uniform(0, 0.05, size=len(f))
            color = (self.colors[i]
                     if hasattr(self, "colors") and i < len(self.colors)
                     else "gray")
            channels.append(ch)
            freqs_list.append(f.copy())
            psd_list.append(pxx)
            colors.append(color)

        return channels, freqs_list, psd_list, colors

    # ------------------------------------------------------------------

    def _psd_clear_curves(self):
        pi = self._psd_inset.getPlotItem()
        for item in self._psd_curves:
            try:
                pi.removeItem(item)
            except Exception:
                pass
        self._psd_curves.clear()

    # ------------------------------------------------------------------

    def _psd_draw(self, channels, freqs_list, psd_list, colors, fmax: float):
        self._psd_clear_curves()

        fmin = float(self._psd_fmin_spin.value())

        # ── pass 1: pre-process all channels so we know the global y range ──
        processed = []   # list of (f_trim, pxx_trim, color)
        for ch, f, pxx, color in zip(channels, freqs_list, psd_list, colors):
            mask = (f >= fmin) & (f <= fmax)
            if not np.any(mask):
                continue
            f_trim   = f[mask]
            pxx_trim = pxx[mask].copy()
            if self._psd_use_db:
                pxx_trim = 10.0 * np.log10(np.maximum(pxx_trim, 1e-30))
            processed.append((f_trim, pxx_trim, color))

        if not processed:
            return

        all_pxx = np.concatenate([pd for _, pd, _ in processed])
        all_pxx = all_pxx[np.isfinite(all_pxx)]
        if len(all_pxx) == 0:
            return

        all_f  = np.concatenate([fd for fd, _, _ in processed])
        fmin_v = float(np.min(all_f[np.isfinite(all_f)])) if np.any(np.isfinite(all_f)) else fmin
        ymin_v = float(np.min(all_pxx))
        ymax_v = float(np.max(all_pxx))
        pad    = (ymax_v - ymin_v) * 0.05 if ymax_v > ymin_v else 1.0

        # In linear mode, fill should stop at the physical floor of 0.0.
        # In dB mode, keep the shared lower pad so curves fill to the plot base.
        if self._psd_use_db:
            fill_base = ymin_v - pad
            y_floor = fill_base
        else:
            fill_base = 0.0
            y_floor = 0.0

        # ── pass 2: add curves using the shared fill_base ─────────────────
        for f_trim, pxx_trim, color in processed:
            qc         = QColor(pg.mkColor(color))
            fill_color = (qc.red(), qc.green(), qc.blue(), 55)
            pen        = pg.mkPen(color, width=1.5, cosmetic=True)
            brush      = pg.mkBrush(*fill_color)

            curve = pg.PlotDataItem(
                f_trim, pxx_trim,
                pen=pen, fillLevel=fill_base, brush=brush,
            )
            self._psd_inset.getPlotItem().addItem(curve)
            self._psd_curves.append(curve)

        self._psd_inset.setXRange(fmin_v, fmax, padding=0)
        self._psd_inset.setYRange(y_floor, ymax_v + pad, padding=0)
