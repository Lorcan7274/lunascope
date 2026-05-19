
#  --------------------------------------------------------------------
#
#  This file is part of Luna.
#
#  LUNA is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  LUNA is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Luna. If not, see <http://www.gnu.org/licenses/>.
#
#  Please see LICENSE.txt for more details.
#
#  --------------------------------------------------------------------

"""Explorer: Topo tab — EEG topographic maps.

Two sub-tabs:
  Results  — render any scalar-per-channel result table as a topo map;
             supports row filtering and epoch/band stepping.
  Live     — animated topo + scrolling EEG traces; plays through the
             current EDF at variable speed with selectable window size.
"""

import math
import time
import traceback
from concurrent.futures import Future

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, QMetaObject, QTimer
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFrame,
    QHBoxLayout, QLabel, QPushButton, QRadioButton, QSizePolicy,
    QSlider, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .explorer_base import BG, FG, GRID, _ExplorerTab
from .topo_clocs import get_positions, load_clocs_file
from .topo_core import create_topo_axes, draw_topo, TopoRenderer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANDS: dict[str, tuple[float, float]] = {
    "delta":  (0.5,  4.0),
    "theta":  (4.0,  8.0),
    "alpha":  (8.0, 12.0),
    "sigma": (11.0, 16.0),
    "beta":  (16.0, 30.0),
    "gamma": (30.0, 50.0),
}

WINDOW_SIZES = [1, 2, 4, 8, 16, 30]   # seconds

SPEED_OPTIONS = [("0.1×", 0.1), ("0.25×", 0.25), ("0.5×", 0.5),
                 ("1×", 1.0), ("2×", 2.0), ("5×", 5.0), ("10×", 10.0)]

CMAP_OPTIONS = ["RdBu_r", "coolwarm", "viridis", "plasma", "hot", "RdYlBu_r"]

_TRACE_PALETTE = [
    "#4ec9b0", "#ce9178", "#9cdcfe", "#dcdcaa", "#c586c0",
    "#569cd6", "#f44747", "#b5cea8", "#6a9955", "#d7ba7d",
]

_TIMER_MS       = 50    # ~20 fps target
_MIN_INTERP_DEF = 8
_MIN_TOPO_CHANS = 16
_MAX_SCALE_WINDOWS = 1500
_WELCH_SEG_SEC  = 2.0
_DYN_SCALE_RELEASE_SEC = 3.0


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color: {GRID};")
    return f


def _label(text: str, color: str = FG, bold: bool = False) -> QLabel:
    lb = QLabel(text)
    style = f"color: {color};"
    if bold:
        style += " font-weight: bold;"
    lb.setStyleSheet(style)
    return lb


def _combo(items: list[str]) -> QComboBox:
    cb = QComboBox()
    cb.addItems(items)
    cb.setStyleSheet(
        f"QComboBox {{ background: #161b22; color: {FG}; border: 1px solid {GRID}; }}"
        f"QComboBox QAbstractItemView {{ background: #161b22; color: {FG}; }}"
    )
    return cb


def _button(text: str, fixed_w: int | None = None) -> QPushButton:
    btn = QPushButton(text)
    btn.setStyleSheet(
        f"QPushButton {{ background: #21262d; color: {FG}; border: 1px solid {GRID}; "
        f"padding: 2px 8px; }} "
        f"QPushButton:hover {{ background: #30363d; }} "
        f"QPushButton:disabled {{ color: #555; }}"
    )
    if fixed_w is not None:
        btn.setFixedWidth(fixed_w)
    return btn


def _row(*widgets, stretch_idx: int | None = None) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    for i, w in enumerate(widgets):
        if isinstance(w, QWidget):
            lay.addWidget(w)
        elif isinstance(w, int):       # raw stretch value
            lay.addStretch(w)
    if stretch_idx is None:
        lay.addStretch(1)
    return lay


def _fmt_time(sec: float) -> str:
    sec = max(0.0, sec)
    h   = int(sec // 3600)
    m   = int((sec % 3600) // 60)
    s   = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


# ---------------------------------------------------------------------------
# Band-power / amplitude computation helpers
# ---------------------------------------------------------------------------

def _band_power(sig: np.ndarray, sr: float, flo: float, fhi: float) -> float:
    """Return log10 band power (Welch) for a 1-D signal."""
    from scipy.signal import welch
    n = len(sig)
    if n < 4:
        return np.nan
    nperseg = min(n, max(4, int(_WELCH_SEG_SEC * sr)))  # 2-second segments if possible
    freqs, psd = welch(sig.astype(float), fs=sr, nperseg=nperseg)
    idx = (freqs >= flo) & (freqs <= fhi)
    if not idx.any():
        return np.nan
    power = float(np.trapz(psd[idx], freqs[idx]))
    return float(np.log10(max(power, 1e-30)))


def _rms(sig: np.ndarray) -> float:
    if len(sig) == 0:
        return np.nan
    return float(np.sqrt(np.mean(sig.astype(float) ** 2)))


# ---------------------------------------------------------------------------
# MplCanvas — thin wrapper used in both sub-tabs
# ---------------------------------------------------------------------------

class _TopoCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(facecolor=BG)
        super().__init__(fig)
        if parent:
            self.setParent(parent)
        self._ax, self._cax = create_topo_axes(fig)
        self._ax.set_facecolor(BG)
        self._ax.set_axis_off()
        fig.patch.set_facecolor(BG)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(280, 280)


# ---------------------------------------------------------------------------
# Results panel
# ---------------------------------------------------------------------------

class _ResultsPanel(QWidget):
    """Static topo from a loaded Luna output table."""

    def __init__(self, ctrl, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        self._user_clocs: dict | None = None
        self._step_values: list = []
        self._step_idx: int = 0
        self._build()

    # ------------------------------------------------------------------

    def _build(self):
        self.setStyleSheet(f"background: {BG}; color: {FG};")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        # --- Table ---
        self._combo_table = _combo([])
        self._combo_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_refresh = _button("↻", fixed_w=28)
        btn_refresh.clicked.connect(self.refresh_tables)
        outer.addLayout(_row(_label("Table:"), self._combo_table, btn_refresh))

        # --- Filter row ---
        self._combo_filter_col = _combo([])
        self._combo_filter_col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._edit_filter_val  = QtWidgets.QLineEdit()
        self._edit_filter_val.setPlaceholderText("value (optional)")
        self._edit_filter_val.setStyleSheet(
            f"background: #161b22; color: {FG}; border: 1px solid {GRID}; padding: 2px;")
        self._edit_filter_val.setFixedWidth(100)
        outer.addLayout(_row(_label("Filter:"), self._combo_filter_col, _label("="),
                             self._edit_filter_val))

        # --- Step row (for multi-row tables: step through a column) ---
        self._combo_step_col = _combo([])
        self._combo_step_col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._btn_step_back = _button("◀", fixed_w=28)
        self._btn_step_fwd  = _button("▶", fixed_w=28)
        self._lbl_step      = _label("—", color="#888")
        self._lbl_step.setMinimumWidth(120)
        self._lbl_step.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._btn_step_back.clicked.connect(lambda: self._step(-1))
        self._btn_step_fwd.clicked.connect(lambda:  self._step(+1))
        outer.addLayout(_row(_label("Step:"), self._combo_step_col,
                             self._btn_step_back, self._btn_step_fwd, self._lbl_step))

        # --- Value column ---
        self._combo_val_col = _combo([])
        self._combo_val_col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        outer.addLayout(_row(_label("Values:"), self._combo_val_col))

        outer.addWidget(_sep())

        # --- Display options ---
        self._radio_dots  = QRadioButton("Dots")
        self._radio_interp = QRadioButton("Interp")
        self._radio_both  = QRadioButton("Both")
        self._radio_both.setChecked(True)
        for rb in (self._radio_dots, self._radio_interp, self._radio_both):
            rb.setStyleSheet(f"color: {FG};")
        grp = QButtonGroup(self)
        for rb in (self._radio_dots, self._radio_interp, self._radio_both):
            grp.addButton(rb)
        self._combo_cmap   = _combo(CMAP_OPTIONS)
        self._chk_labels   = QCheckBox("Labels")
        self._chk_labels.setChecked(True)
        self._chk_labels.setStyleSheet(f"color: {FG};")
        self._chk_labels.toggled.connect(lambda *_: self._plot())
        self._spin_dot_size = QSpinBox()
        self._spin_dot_size.setRange(8, 160)
        self._spin_dot_size.setValue(35)
        self._spin_dot_size.setFixedWidth(64)
        self._spin_dot_size.setStyleSheet(
            f"background: #161b22; color: {FG}; border: 1px solid {GRID};")
        self._spin_dot_size.valueChanged.connect(lambda *_: self._plot())
        outer.addLayout(_row(_label("Mode:"), self._radio_dots, self._radio_interp,
                             self._radio_both, _label("  Cmap:"), self._combo_cmap,
                             self._chk_labels, _label("Dot size:"), self._spin_dot_size))

        # --- Clocs override ---
        btn_clocs = _button("Load coords…")
        self._lbl_clocs = _label("(default)", color="#888")
        btn_clocs.clicked.connect(self._load_clocs)
        outer.addLayout(_row(btn_clocs, self._lbl_clocs))

        outer.addWidget(_sep())

        # --- Plot button ---
        btn_plot = _button("Plot ▶")
        btn_plot.setFixedHeight(28)
        btn_plot.clicked.connect(self._plot)
        outer.addLayout(_row(btn_plot))

        # --- Canvas ---
        self._canvas = _TopoCanvas(self)
        outer.addWidget(self._canvas, stretch=1)

        # wire table change
        self._combo_table.currentIndexChanged.connect(self._on_table_changed)
        self._combo_step_col.currentIndexChanged.connect(self._on_step_col_changed)

    # ------------------------------------------------------------------

    def refresh_tables(self):
        results = getattr(self.ctrl, "results", None) or {}
        cur = self._combo_table.currentData()
        self._combo_table.blockSignals(True)
        self._combo_table.clear()
        for key in sorted(results.keys()):
            display = " : ".join(str(x) for x in key) if isinstance(key, tuple) else str(key)
            self._combo_table.addItem(display, key)
        idx = self._combo_table.findData(cur)
        if idx >= 0:
            self._combo_table.setCurrentIndex(idx)
        self._combo_table.blockSignals(False)
        self._on_table_changed()

    def _on_table_changed(self, *_):
        key     = self._combo_table.currentData()
        results = getattr(self.ctrl, "results", None) or {}
        df      = results.get(key) if key else None
        if df is None:
            for cb in (self._combo_filter_col, self._combo_step_col, self._combo_val_col):
                cb.clear()
            return
        cols = list(df.columns)
        # filter column (categorical / non-numeric preferred)
        non_num = [c for c in cols if df[c].dtype == object or str(df[c].dtype).startswith("cat")]
        for cb, src in ((self._combo_filter_col, non_num or cols),
                        (self._combo_step_col,   non_num or cols)):
            cb.blockSignals(True)
            cb.clear()
            cb.addItem("(none)", None)
            for c in src:
                cb.addItem(str(c), c)
            cb.blockSignals(False)
        # value column (numeric)
        num_cols = [c for c in cols if np.issubdtype(df[c].dtype, np.number)]
        self._combo_val_col.blockSignals(True)
        self._combo_val_col.clear()
        for c in (num_cols or cols):
            self._combo_val_col.addItem(str(c), c)
        self._combo_val_col.blockSignals(False)
        self._on_step_col_changed()

    def _on_step_col_changed(self, *_):
        df = self._current_df()
        col = self._combo_step_col.currentData()
        if df is None or col is None:
            self._step_values = []
            self._lbl_step.setText("—")
            return
        self._step_values = sorted(df[col].dropna().unique().tolist())
        self._step_idx    = 0
        self._update_step_label()

    def _update_step_label(self):
        n = len(self._step_values)
        if n == 0:
            self._lbl_step.setText("—")
        else:
            v = self._step_values[self._step_idx % n]
            self._lbl_step.setText(f"{v}  ({self._step_idx+1}/{n})")

    def _step(self, direction: int):
        n = len(self._step_values)
        if n == 0:
            return
        self._step_idx = (self._step_idx + direction) % n
        self._update_step_label()
        self._plot()

    def _current_df(self):
        key     = self._combo_table.currentData()
        results = getattr(self.ctrl, "results", None) or {}
        return results.get(key) if key else None

    def _load_clocs(self):
        from ..file_dialogs import open_file_name
        path, _ = open_file_name(self, "Channel coordinates (LABEL X Y Z)", "",
                                 "Text files (*.txt *.tsv *.csv);;All files (*)")
        if path:
            try:
                self._user_clocs = load_clocs_file(path)
                import os
                self._lbl_clocs.setText(os.path.basename(path))
            except Exception as exc:
                self._lbl_clocs.setText(f"Error: {exc}")

    # ------------------------------------------------------------------

    def _plot(self):
        df = self._current_df()
        if df is None:
            return

        try:
            # apply filter
            filt_col = self._combo_filter_col.currentData()
            filt_val = self._edit_filter_val.text().strip()
            if filt_col and filt_val:
                try:
                    fv = type(df[filt_col].iloc[0])(filt_val)
                except Exception:
                    fv = filt_val
                df = df[df[filt_col] == fv]

            # apply step filter
            step_col = self._combo_step_col.currentData()
            if step_col and self._step_values:
                sv = self._step_values[self._step_idx % len(self._step_values)]
                df = df[df[step_col] == sv]

            # require CH column
            ch_col = None
            for c in df.columns:
                if str(c).upper() == "CH":
                    ch_col = c
                    break
            if ch_col is None:
                self._show_msg("Table has no CH column")
                return

            val_col = self._combo_val_col.currentData()
            if val_col is None:
                return

            # Value columns must be numeric. When a table has no numeric
            # columns, the combo can still contain strings as a fallback.
            coerced = np.asarray(pd.to_numeric(df[val_col], errors="coerce"), dtype=float)
            if not np.isfinite(coerced).any():
                self._show_msg(f"Value column '{val_col}' is not numeric")
                return

            # build values dict
            sub = df[[ch_col, val_col]].copy()
            sub[val_col] = coerced
            sub = sub.dropna()
            values = {
                str(row[ch_col]).upper(): row[val_col]
                for _, row in sub.iterrows()
            }
            if not values:
                self._show_msg("No data after filtering")
                return

            positions = get_positions(list(values.keys()), self._user_clocs)
            if not positions:
                self._show_msg("No channel positions matched")
                return
            if len(positions) < _MIN_TOPO_CHANS:
                self._show_msg(
                    f"Need at least {_MIN_TOPO_CHANS} mappable channels "
                    f"(found {len(positions)})"
                )
                return

            # render
            mode = ("dots"  if self._radio_dots.isChecked()  else
                    "interp" if self._radio_interp.isChecked() else "both")
            fig  = self._canvas.figure
            ax, cax = create_topo_axes(fig)
            draw_topo(ax, values, positions,
                      cax=cax,
                      mode=mode,
                      cmap=self._combo_cmap.currentText(),
                      dot_size=float(self._spin_dot_size.value()),
                      show_labels=self._chk_labels.isChecked(),
                      bg=BG, fg=FG)
            self._canvas.draw_idle()

        except Exception:
            traceback.print_exc()

    def _show_msg(self, msg: str):
        fig = self._canvas.figure
        ax, cax = create_topo_axes(fig)
        cax.set_visible(False)
        ax.set_facecolor(BG)
        ax.set_axis_off()
        ax.text(0.5, 0.5, msg, color=FG, ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        self._canvas.draw_idle()


# ---------------------------------------------------------------------------
# Live panel
# ---------------------------------------------------------------------------

class _LiveTopoPanel(QWidget):
    """Animated topo + scrolling traces for the current EDF."""

    def __init__(self, ctrl, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl

        # loaded data state
        self._raw:       dict  = {}   # {ch: {'vals': np.float32, 'sr': float, 't_start': float, 'n': int}}
        self._gaps:      list  = []   # list of (start_sec, stop_sec)
        self._total_sec: float = 0.0
        self._topo_renderer: TopoRenderer | None = None
        self._user_clocs: dict | None = None
        self._pending_load_payload = None
        self._pending_load_error: str | None = None
        self._scale_samples_cache: dict[tuple, np.ndarray] = {}
        self._scale_job_key = None
        self._pending_scale_key = None
        self._pending_scale_samples = None
        self._pending_scale_error: str | None = None
        self._dynamic_scale_vmin: float | None = None
        self._dynamic_scale_vmax: float | None = None
        self._dynamic_scale_wall: float | None = None

        # playback state
        self._cursor_sec: float = 0.0
        self._playing:    bool  = False
        self._last_tick_wall: float | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(_TIMER_MS)
        self._timer.timeout.connect(self._tick)

        # trace curves (set up after load)
        self._trace_curves:  list = []
        self._trace_offsets: dict = {}   # {ch: float y-offset}
        self._trace_scales:  dict = {}   # {ch: (lo, hi) physical range}
        self._loaded_chs:    list = []

        self._build()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build(self):
        self.setStyleSheet(f"background: {BG}; color: {FG};")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        # --- Load row ---
        self._btn_load   = _button("Load ▶")
        self._btn_load.setToolTip("Fetch all signal data for the current record")
        self._btn_load.clicked.connect(self._start_load)
        self._lbl_status = _label("No data loaded", color="#888")
        btn_clocs = _button("Coords…")
        btn_clocs.clicked.connect(self._load_clocs)
        outer.addLayout(_row(self._btn_load, self._lbl_status, btn_clocs))

        # --- Transport row ---
        self._btn_play  = _button("▶ Play", fixed_w=70)
        self._btn_stop  = _button("■", fixed_w=30)
        self._btn_back  = _button("◀◀", fixed_w=36)
        self._btn_fwd   = _button("▶▶", fixed_w=36)
        for b in (self._btn_play, self._btn_stop, self._btn_back, self._btn_fwd):
            b.setEnabled(False)
        self._btn_play.clicked.connect(self._toggle_play)
        self._btn_stop.clicked.connect(self._stop)
        self._btn_back.clicked.connect(lambda: self._seek_step(-1))
        self._btn_fwd.clicked.connect(lambda:  self._seek_step(+1))

        self._combo_speed = _combo([s for s, _ in SPEED_OPTIONS])
        self._combo_speed.setCurrentIndex(3)   # 1× default
        self._combo_speed.setFixedWidth(60)

        self._combo_win = _combo([f"{w}s" for w in WINDOW_SIZES])
        self._combo_win.setCurrentIndex(2)   # 4s default
        self._combo_win.setFixedWidth(60)
        self._combo_win.currentIndexChanged.connect(self._on_window_changed)

        outer.addLayout(_row(self._btn_back, self._btn_play, self._btn_stop,
                             self._btn_fwd, _label("  Speed:"), self._combo_speed,
                             _label("Window:"), self._combo_win))

        # --- Mode row ---
        self._radio_amp  = QRadioButton("Amplitude (RMS)")
        self._radio_band = QRadioButton("Band power")
        self._radio_band.setChecked(True)
        for rb in (self._radio_amp, self._radio_band):
            rb.setStyleSheet(f"color: {FG};")
        bg2 = QButtonGroup(self)
        bg2.addButton(self._radio_amp)
        bg2.addButton(self._radio_band)
        bg2.buttonClicked.connect(self._on_metric_controls_changed)
        self._combo_band = _combo(list(BANDS.keys()))
        self._combo_band.setCurrentText("sigma")
        self._combo_band.setFixedWidth(80)
        self._combo_band.currentIndexChanged.connect(self._on_metric_controls_changed)
        self._chk_custom_band = QCheckBox("Custom Hz:")
        self._chk_custom_band.setStyleSheet(f"color: {FG};")
        self._chk_custom_band.toggled.connect(self._on_metric_controls_changed)
        self._spin_flo = QDoubleSpinBox(); self._spin_flo.setRange(0.1, 200); self._spin_flo.setValue(11.0); self._spin_flo.setFixedWidth(60)
        self._spin_fhi = QDoubleSpinBox(); self._spin_fhi.setRange(0.1, 200); self._spin_fhi.setValue(16.0); self._spin_fhi.setFixedWidth(60)
        for sp in (self._spin_flo, self._spin_fhi):
            sp.setStyleSheet(f"background: #161b22; color: {FG}; border: 1px solid {GRID};")
        self._spin_flo.valueChanged.connect(self._on_metric_controls_changed)
        self._spin_fhi.valueChanged.connect(self._on_metric_controls_changed)
        outer.addLayout(_row(self._radio_amp, self._radio_band, self._combo_band,
                             self._chk_custom_band, self._spin_flo, _label("–"), self._spin_fhi))

        # --- Display options row ---
        self._radio_dots2  = QRadioButton("Dots")
        self._radio_interp2 = QRadioButton("Interp")
        self._radio_both2   = QRadioButton("Both")
        self._radio_both2.setChecked(True)
        for rb in (self._radio_dots2, self._radio_interp2, self._radio_both2):
            rb.setStyleSheet(f"color: {FG};")
        bg3 = QButtonGroup(self)
        for rb in (self._radio_dots2, self._radio_interp2, self._radio_both2):
            bg3.addButton(rb)
        bg3.buttonClicked.connect(self._on_mode_changed)
        self._combo_cmap = _combo(CMAP_OPTIONS)
        self._combo_cmap.currentIndexChanged.connect(self._on_cmap_changed)
        self._chk_labels = QCheckBox("Labels")
        self._chk_labels.setChecked(True)
        self._chk_labels.setStyleSheet(f"color: {FG};")
        self._chk_labels.toggled.connect(self._on_labels_toggled)
        self._spin_dot_size = QSpinBox()
        self._spin_dot_size.setRange(8, 160)
        self._spin_dot_size.setValue(35)
        self._spin_dot_size.setFixedWidth(64)
        self._spin_dot_size.setStyleSheet(
            f"background: #161b22; color: {FG}; border: 1px solid {GRID};")
        self._spin_dot_size.valueChanged.connect(self._on_dot_size_changed)
        outer.addLayout(_row(_label("Mode:"), self._radio_dots2, self._radio_interp2,
                             self._radio_both2, _label("  Cmap:"), self._combo_cmap,
                             self._chk_labels, _label("Dot size:"), self._spin_dot_size))

        # --- Color scale row ---
        self._combo_scale = _combo(["Dynamic", "Whole record"])
        self._combo_scale.setFixedWidth(120)
        self._combo_scale.currentIndexChanged.connect(self._on_scale_controls_changed)
        self._spin_clip = QSpinBox()
        self._spin_clip.setRange(0, 25)
        self._spin_clip.setValue(0)
        self._spin_clip.setSuffix("%")
        self._spin_clip.setFixedWidth(70)
        self._spin_clip.setStyleSheet(
            f"background: #161b22; color: {FG}; border: 1px solid {GRID};")
        self._spin_clip.valueChanged.connect(self._on_scale_controls_changed)
        self._lbl_scale_status = _label("", color="#888")
        self._lbl_scale_status.setMinimumWidth(110)
        self._chk_show_traces = QCheckBox("Show traces")
        self._chk_show_traces.setChecked(True)
        self._chk_show_traces.setStyleSheet(f"color: {FG};")
        self._chk_show_traces.toggled.connect(self._on_show_traces_toggled)
        outer.addLayout(_row(_label("Scale:"), self._combo_scale,
                             _label("Clip tails:"), self._spin_clip,
                             self._lbl_scale_status, self._chk_show_traces))

        outer.addWidget(_sep())

        # --- Main display: topo (left) + traces (right) ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: #30363d; }")

        self._topo_canvas = _TopoCanvas()
        splitter.addWidget(self._topo_canvas)

        self._pg_widget = self._build_trace_widget()
        splitter.addWidget(self._pg_widget)
        self._trace_splitter = splitter
        splitter.setSizes([380, 620])

        outer.addWidget(splitter, stretch=1)

        # --- Scrubber ---
        scrub_row = QHBoxLayout()
        scrub_row.setSpacing(6)
        self._lbl_cursor = _label("00:00:00.00", color=FG)
        self._lbl_cursor.setFixedWidth(90)
        self._scrubber   = QSlider(Qt.Horizontal)
        self._scrubber.setRange(0, 10000)
        self._scrubber.setEnabled(False)
        self._scrubber.setStyleSheet(
            f"QSlider::groove:horizontal {{ background: {GRID}; height: 6px; }}"
            f"QSlider::handle:horizontal {{ background: #58a6ff; width: 12px; margin: -4px 0; border-radius: 6px; }}"
        )
        self._scrubber.sliderMoved.connect(self._on_scrubber_moved)
        self._lbl_total = _label("/ 00:00:00", color="#888")
        scrub_row.addWidget(self._lbl_cursor)
        scrub_row.addWidget(self._scrubber, stretch=1)
        scrub_row.addWidget(self._lbl_total)
        outer.addLayout(scrub_row)

        # initial placeholder topo
        self._show_topo_msg("Load a file and click  Load ▶")

    def _build_trace_widget(self):
        import pyqtgraph as pg
        pg.setConfigOptions(antialias=True)
        pw = pg.PlotWidget(background=BG)
        pw.hideAxis("left")
        pw.hideAxis("bottom")
        pw.setMouseEnabled(x=False, y=False)
        pw.setMenuEnabled(False)
        self._pg = pw
        return pw

    # ------------------------------------------------------------------
    # Property helpers
    # ------------------------------------------------------------------

    def _window_sec(self) -> float:
        return float(WINDOW_SIZES[self._combo_win.currentIndex()])

    def _speed(self) -> float:
        return SPEED_OPTIONS[self._combo_speed.currentIndex()][1]

    def _topo_mode(self) -> str:
        if self._radio_dots2.isChecked():   return "dots"
        if self._radio_interp2.isChecked(): return "interp"
        return "both"

    def _get_band(self) -> tuple[float, float]:
        if self._chk_custom_band.isChecked():
            return float(self._spin_flo.value()), float(self._spin_fhi.value())
        return BANDS[self._combo_band.currentText()]

    def _scale_mode(self) -> str:
        return "record" if self._combo_scale.currentIndex() == 1 else "dynamic"

    def _clip_tail_pct(self) -> int:
        return int(self._spin_clip.value())

    def _metric_cache_key(self) -> tuple:
        win = self._window_sec()
        if self._radio_band.isChecked():
            flo, fhi = self._get_band()
            return ("band", round(flo, 4), round(fhi, 4), round(win, 4))
        return ("amp", round(win, 4))

    def _show_traces(self) -> bool:
        return self._chk_show_traces.isChecked()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    def _load_clocs(self):
        from ..file_dialogs import open_file_name
        path, _ = open_file_name(self, "Channel coordinates (LABEL X Y Z)", "",
                                 "Text files (*.txt *.tsv *.csv);;All files (*)")
        if path:
            try:
                self._user_clocs = load_clocs_file(path)
            except Exception as exc:
                self._lbl_status.setText(f"Coords error: {exc}")

    def _start_load(self):
        ctrl = self.ctrl
        if not getattr(ctrl, "p", None):
            self._lbl_status.setText("No EDF loaded")
            return
        if getattr(ctrl, "_busy", False):
            return

        # channels: rendered ones, or all EDF channels
        chs = list(getattr(ctrl, "ss_chs", None) or [])
        if not chs:
            try:
                chs = ctrl.p.channels()["Channels"].tolist()
            except Exception:
                chs = []
        if not chs:
            self._lbl_status.setText("No channels available")
            return

        # only load channels with known positions
        positions = get_positions(chs, self._user_clocs)
        topo_chs  = [ch for ch in chs if ch in positions or ch.upper() in
                     {k.upper() for k in positions}]
        # keep all channels for traces, but only topo channels for topo map
        # read total_sec now on main thread (ctrl.ns is set after render)
        total_sec_hint = float(getattr(ctrl, "ns", 0.0) or 0.0)

        self._lbl_status.setText(f"Loading {len(chs)} channels…")
        self._btn_load.setEnabled(False)
        ctrl._busy = True
        ctrl.sb_progress.setVisible(True)
        ctrl.sb_progress.setRange(0, 0)
        ctrl.lock_ui("Loading topo data…")
        self._scale_samples_cache.clear()
        self._scale_job_key = None
        self._pending_scale_key = None
        self._pending_scale_samples = None
        self._pending_scale_error = None
        self._dynamic_scale_vmin = None
        self._dynamic_scale_vmax = None
        self._dynamic_scale_wall = None
        self._lbl_scale_status.setText("")

        fut: Future = ctrl._exec.submit(
            self._fetch_data, ctrl.p, chs, total_sec_hint
        )

        def _done(f: Future):
            err = f.exception()
            if err:
                self._pending_load_error = str(err)
                QMetaObject.invokeMethod(
                    self, "_deliver_load_error",
                    Qt.QueuedConnection,
                )
            else:
                self._pending_load_payload = f.result()
                QMetaObject.invokeMethod(
                    self, "_deliver_load_ok",
                    Qt.QueuedConnection,
                )

        fut.add_done_callback(_done)

    @staticmethod
    def _fetch_data(p, chs: list[str], total_sec_hint: float = 0.0):
        """Background: fetch all signal data per channel."""
        import lunapi as lp

        # total duration — prefer the hint from main thread; fall back to segsrv
        total_sec = total_sec_hint
        if total_sec <= 0:
            try:
                total_sec = float(lp.segsrv(p).num_seconds_clocktime_original())
            except Exception:
                total_sec = 86400.0   # 24-hour ceiling; slice will stop at EDF end

        # gaps from segments table
        gaps: list[tuple[float, float]] = []
        try:
            seg_df = p.segments()
            if seg_df is not None and len(seg_df) > 1:
                # rows are contiguous segments; gaps are between them
                starts = seg_df.iloc[:, 0].values   # START column
                stops  = seg_df.iloc[:, 1].values   # STOP column
                for i in range(len(stops) - 1):
                    gaps.append((float(stops[i]), float(starts[i + 1])))
        except Exception:
            pass

        raw: dict = {}
        for ch in chs:
            try:
                ivals = p.s2i([(0.0, total_sec)])
                d = p.slice(ivals, ch, time=True)
                if d is None or len(d) < 2:
                    continue
                mat = d[1]
                if mat is None or len(mat) == 0:
                    continue
                times = mat[:, 0]
                vals  = mat[:, 1].astype(np.float32)
                # estimate sample rate from time column
                dt_arr = np.diff(times[:min(1000, len(times))])
                dt = float(np.median(dt_arr)) if len(dt_arr) > 0 else 1.0
                sr = 1.0 / dt if dt > 0 else 1.0
                t_start = float(times[0])
                raw[ch] = {
                    "vals":    vals,
                    "sr":      sr,
                    "t_start": t_start,
                    "n":       len(vals),
                }
            except Exception:
                continue

        if total_sec == 0.0 and raw:
            ch0 = next(iter(raw))
            total_sec = raw[ch0]["t_start"] + raw[ch0]["n"] / raw[ch0]["sr"]

        return raw, gaps, total_sec

    @QtCore.Slot()
    def _deliver_load_error(self):
        msg = self._pending_load_error or "Unknown load error"
        self._pending_load_error = None
        self.ctrl._busy = False
        self.ctrl.sb_progress.setRange(0, 100)
        self.ctrl.sb_progress.setVisible(False)
        self.ctrl.unlock_ui()
        self._btn_load.setEnabled(True)
        self._lbl_status.setText(f"Error: {msg}")

    @QtCore.Slot()
    def _deliver_load_ok(self):
        payload = self._pending_load_payload
        self._pending_load_payload = None
        if payload is None:
            self._deliver_load_error()
            return
        raw, gaps, total_sec = payload
        self.ctrl._busy = False
        self.ctrl.sb_progress.setRange(0, 100)
        self.ctrl.sb_progress.setVisible(False)
        self.ctrl.unlock_ui()
        self._btn_load.setEnabled(True)

        if not raw:
            self._lbl_status.setText("No data returned")
            return

        self._raw       = raw
        self._gaps      = gaps
        self._total_sec = total_sec
        self._cursor_sec = 0.0
        self._loaded_chs = list(raw.keys())

        # build positions for topo-capable channels
        positions = get_positions(self._loaded_chs, self._user_clocs)
        topo_chs  = [ch for ch in self._loaded_chs if ch in positions]

        n_ch   = len(self._loaded_chs)
        n_topo = len(topo_chs)
        self._lbl_status.setText(
            f"{n_ch} ch loaded  ({n_topo} with topo coords) | "
            f"{_fmt_time(total_sec)}"
        )
        self._lbl_total.setText(f"/ {_fmt_time(total_sec)}")

        # set up scrubber
        self._scrubber.setEnabled(True)
        self._scrubber.setValue(0)

        # enable transport buttons
        for b in (self._btn_play, self._btn_stop, self._btn_back, self._btn_fwd):
            b.setEnabled(True)

        # per-channel scale (5th–95th percentile for trace display)
        self._trace_scales = {}
        for ch, r in raw.items():
            v    = r["vals"]
            lo   = float(np.percentile(v[::max(1, len(v)//5000)], 5))
            hi   = float(np.percentile(v[::max(1, len(v)//5000)], 95))
            if lo == hi:
                lo -= 1.0
                hi += 1.0
            self._trace_scales[ch] = (lo, hi)

        # set up trace widget
        self._setup_traces()

        # set up topo renderer (pre-compute interpolation weights)
        if topo_chs:
            if len(topo_chs) < _MIN_TOPO_CHANS:
                self._topo_renderer = None
                self._show_topo_msg(
                    f"Need at least {_MIN_TOPO_CHANS} mappable channels "
                    f"for topo view (found {len(topo_chs)})"
                )
            else:
                self._topo_renderer = TopoRenderer(
                    positions, min_interp=_MIN_INTERP_DEF,
                    bg=BG, fg=FG,
                )
                fig = self._topo_canvas.figure
                ax, cax = create_topo_axes(fig)
                self._topo_renderer.setup(
                    ax, fig, cax=cax,
                    cmap=self._combo_cmap.currentText(),
                    dot_size=float(self._spin_dot_size.value()),
                    show_labels=self._chk_labels.isChecked(),
                    mode=self._topo_mode(),
                )
                self._topo_canvas.draw_idle()
        else:
            self._topo_renderer = None
            self._show_topo_msg("No channels with known coordinates")

        # render first frame
        self._render_frame(0.0)
        self._ensure_record_scale()

    # ------------------------------------------------------------------
    # Trace widget setup
    # ------------------------------------------------------------------

    def _setup_traces(self):
        import pyqtgraph as pg
        self._pg.clear()
        self._trace_curves = []
        self._trace_offsets = {}

        chs = self._loaded_chs
        n   = len(chs)
        for i, ch in enumerate(chs):
            color = _TRACE_PALETTE[i % len(_TRACE_PALETTE)]
            curve = self._pg.plot(pen=pg.mkPen(color=color, width=1))
            self._trace_curves.append(curve)
            offset = (n - i - 1) * 1.2   # top channel at highest y
            self._trace_offsets[ch] = offset
            label = pg.TextItem(ch, color=color, anchor=(1.0, 0.5))
            self._pg.addItem(label)
            label.setPos(0.0, offset)

        # y range
        self._pg.setYRange(-0.2, n * 1.2 + 0.2, padding=0)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _toggle_play(self):
        if self._playing:
            self._pause()
        else:
            self._play()

    def pause_playback(self):
        if self._playing:
            self._pause()

    def _play(self):
        if not self._raw:
            return
        self._playing = True
        self._last_tick_wall = time.perf_counter()
        self._btn_play.setText("⏸ Pause")
        self._timer.start()

    def _pause(self):
        self._playing = False
        self._last_tick_wall = None
        self._btn_play.setText("▶ Play")
        self._timer.stop()

    def _stop(self):
        self._pause()
        self._seek(0.0)

    def _seek(self, sec: float):
        self._cursor_sec = max(0.0, min(sec, self._total_sec))
        if self._playing:
            self._last_tick_wall = time.perf_counter()
        self._update_scrubber()
        self._render_frame(self._cursor_sec)

    def _seek_step(self, direction: int):
        """Jump one window forward/backward."""
        self._seek(self._cursor_sec + direction * self._window_sec())

    def _on_scrubber_moved(self, val: int):
        if self._total_sec <= 0:
            return
        sec = (val / 10000.0) * self._total_sec
        self._cursor_sec = sec
        if self._playing:
            self._last_tick_wall = time.perf_counter()
        self._render_frame(sec)

    def _update_scrubber(self):
        if self._total_sec > 0:
            pos = int(self._cursor_sec / self._total_sec * 10000)
            self._scrubber.blockSignals(True)
            self._scrubber.setValue(pos)
            self._scrubber.blockSignals(False)
        self._lbl_cursor.setText(_fmt_time(self._cursor_sec))

    def _tick(self):
        try:
            now = time.perf_counter()
            prev = self._last_tick_wall
            self._last_tick_wall = now
            elapsed = (_TIMER_MS / 1000.0) if prev is None else max(0.0, now - prev)
            # Avoid a huge jump if the app was stalled or backgrounded.
            elapsed = min(elapsed, 0.25)

            self._cursor_sec += elapsed * self._speed()
            if self._cursor_sec >= self._total_sec:
                self._cursor_sec = self._total_sec
                self._pause()
            self._update_scrubber()
            self._render_frame(self._cursor_sec)
        except Exception as exc:
            self._pause()
            self._lbl_status.setText(f"Playback error: {exc}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    def _render_frame(self, cursor: float):
        if not self._raw:
            return
        win = self._window_sec()
        t0, t1 = cursor, cursor + win

        # check if window is in a gap
        in_gap = any(g0 < t1 and g1 > t0 for g0, g1 in self._gaps)

        # compute per-channel values for topo
        topo_values: dict[str, float] = {}
        use_band = self._radio_band.isChecked()
        flo, fhi = self._get_band()

        for ch, r in self._raw.items():
            seg_t, seg_v = self._slice_raw(r, t0, t1)
            if seg_v is None or len(seg_v) < 4:
                topo_values[ch] = np.nan
                continue
            if use_band:
                topo_values[ch] = _band_power(seg_v, r["sr"], flo, fhi)
            else:
                topo_values[ch] = _rms(seg_v)

        # update topo
        if self._topo_renderer is not None:
            ax = self._topo_canvas.figure.axes[0] if self._topo_canvas.figure.axes else None
            if ax is not None:
                if in_gap:
                    ax.set_title("— gap —", color="#888", fontsize=8, pad=2)
                else:
                    ax.set_title("", color=FG, fontsize=8, pad=2)
                vmin, vmax = self._topo_scale_limits(topo_values)
                self._topo_renderer.update(topo_values, vmin=vmin, vmax=vmax)
                self._topo_canvas.draw_idle()

        # update traces
        if self._show_traces():
            self._update_traces(t0, t1)

    def _slice_raw(self, r: dict, t0: float, t1: float):
        """Return (times, vals) numpy slice for window [t0, t1] from channel record r."""
        i0 = max(0, int((t0 - r["t_start"]) * r["sr"]))
        i1 = min(r["n"], int((t1 - r["t_start"]) * r["sr"]) + 1)
        if i0 >= i1:
            return None, None
        times = r["t_start"] + np.arange(i0, i1, dtype=np.float32) / r["sr"]
        vals  = r["vals"][i0:i1]
        return times, vals

    def _update_traces(self, t0: float, t1: float):
        for i, ch in enumerate(self._loaded_chs):
            if i >= len(self._trace_curves):
                break
            r = self._raw.get(ch)
            if r is None:
                self._trace_curves[i].setData([], [])
                continue
            times, vals = self._slice_raw(r, t0, t1)
            if vals is None or len(vals) == 0:
                self._trace_curves[i].setData([], [])
                continue
            # normalise to [0, 1] then offset
            lo, hi = self._trace_scales.get(ch, (float(vals.min()), float(vals.max())))
            if hi > lo:
                y_norm = (vals.astype(float) - lo) / (hi - lo)
            else:
                y_norm = np.zeros_like(vals, dtype=float)
            offset = self._trace_offsets.get(ch, 0.0)
            self._trace_curves[i].setData(
                times.astype(float), y_norm + offset
            )

        # update x range
        self._pg.setXRange(t0, t1, padding=0.01)

    # ------------------------------------------------------------------
    # React to display option changes
    # ------------------------------------------------------------------

    def _on_mode_changed(self, *_):
        """Rebuild renderer when dots/interp/both changes."""
        if self._topo_renderer is None or not self._raw:
            return
        self._rebuild_renderer()

    def _on_cmap_changed(self, *_):
        if self._topo_renderer is None or not self._raw:
            return
        self._rebuild_renderer()

    def _on_labels_toggled(self, *_):
        if self._topo_renderer is None or not self._raw:
            return
        self._rebuild_renderer()

    def _on_dot_size_changed(self, *_):
        if self._topo_renderer is None or not self._raw:
            return
        self._topo_renderer.set_dot_size(float(self._spin_dot_size.value()))
        self._topo_canvas.draw_idle()

    def _on_window_changed(self, *_):
        self._invalidate_metric_cache()
        self._ensure_record_scale()
        if self._raw:
            self._render_frame(self._cursor_sec)

    def _on_scale_controls_changed(self, *_):
        self._dynamic_scale_vmin = None
        self._dynamic_scale_vmax = None
        self._dynamic_scale_wall = None
        self._ensure_record_scale()
        if self._raw:
            self._render_frame(self._cursor_sec)

    def _on_show_traces_toggled(self, checked: bool):
        self._pg_widget.setVisible(checked)
        if hasattr(self, "_trace_splitter"):
            if checked:
                self._trace_splitter.setSizes([380, 620])
            else:
                self._trace_splitter.setSizes([1, 0])
        if checked and self._raw:
            t0 = self._cursor_sec
            t1 = t0 + self._window_sec()
            self._update_traces(t0, t1)

    def _on_metric_controls_changed(self, *_):
        self._invalidate_metric_cache()
        self._ensure_record_scale()
        if self._raw:
            self._render_frame(self._cursor_sec)

    def _invalidate_metric_cache(self):
        self._scale_job_key = None
        self._pending_scale_key = None
        self._pending_scale_samples = None
        self._pending_scale_error = None
        self._lbl_scale_status.setText("")

    def _rebuild_renderer(self):
        positions = get_positions(self._loaded_chs, self._user_clocs)
        if not positions:
            self._topo_renderer = None
            self._show_topo_msg("No channels with known coordinates")
            return
        if len(positions) < _MIN_TOPO_CHANS:
            self._topo_renderer = None
            self._show_topo_msg(
                f"Need at least {_MIN_TOPO_CHANS} mappable channels "
                f"for topo view (found {len(positions)})"
            )
            return
        self._topo_renderer = TopoRenderer(
            positions, min_interp=_MIN_INTERP_DEF, bg=BG, fg=FG,
        )
        fig = self._topo_canvas.figure
        ax, cax = create_topo_axes(fig)
        self._topo_renderer.setup(
            ax, fig, cax=cax,
            cmap=self._combo_cmap.currentText(),
            dot_size=float(self._spin_dot_size.value()),
            show_labels=self._chk_labels.isChecked(),
            mode=self._topo_mode(),
        )
        self._topo_canvas.draw_idle()
        self._render_frame(self._cursor_sec)
        self._ensure_record_scale()

    def _show_topo_msg(self, msg: str):
        fig = self._topo_canvas.figure
        ax, cax = create_topo_axes(fig)
        cax.set_visible(False)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        ax.set_axis_off()
        ax.text(0.5, 0.5, msg, color=FG, ha="center", va="center",
                transform=ax.transAxes, fontsize=9, wrap=True,
                multialignment="center")
        self._topo_canvas.draw_idle()

    def _topo_scale_limits(self, topo_values: dict[str, float]) -> tuple[float | None, float | None]:
        frame_vals = np.asarray(list(topo_values.values()), dtype=float)
        frame_vals = frame_vals[np.isfinite(frame_vals)]
        if frame_vals.size == 0:
            return None, None

        if self._scale_mode() == "record":
            samples = self._scale_samples_cache.get(self._metric_cache_key())
            if samples is not None and len(samples):
                vals = samples
            else:
                vals = frame_vals
                self._ensure_record_scale()
            lo, hi = self._apply_clip(vals)
        else:
            lo, hi = self._smoothed_dynamic_limits(frame_vals)
        if lo == hi:
            pad = 1.0 if lo == 0 else max(abs(lo) * 0.05, 1e-6)
            lo -= pad
            hi += pad
        return float(lo), float(hi)

    def _smoothed_dynamic_limits(self, frame_vals: np.ndarray) -> tuple[float, float]:
        lo_now, hi_now = self._apply_clip(frame_vals)
        now = time.perf_counter()

        if self._dynamic_scale_vmin is None or self._dynamic_scale_vmax is None:
            self._dynamic_scale_vmin = lo_now
            self._dynamic_scale_vmax = hi_now
            self._dynamic_scale_wall = now
            return lo_now, hi_now

        prev_lo = self._dynamic_scale_vmin
        prev_hi = self._dynamic_scale_vmax
        prev_wall = self._dynamic_scale_wall
        dt = 0.0 if prev_wall is None else max(0.0, now - prev_wall)
        alpha = math.exp(-dt / _DYN_SCALE_RELEASE_SEC) if _DYN_SCALE_RELEASE_SEC > 0 else 0.0

        # Expand immediately to include new extremes, but contract more slowly
        # so the color scale has some visual memory and doesn't breathe per-frame.
        if lo_now < prev_lo:
            lo = lo_now
        else:
            lo = alpha * prev_lo + (1.0 - alpha) * lo_now

        if hi_now > prev_hi:
            hi = hi_now
        else:
            hi = alpha * prev_hi + (1.0 - alpha) * hi_now

        self._dynamic_scale_vmin = float(lo)
        self._dynamic_scale_vmax = float(hi)
        self._dynamic_scale_wall = now
        return self._dynamic_scale_vmin, self._dynamic_scale_vmax

    def _apply_clip(self, vals: np.ndarray) -> tuple[float, float]:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0.0, 1.0
        pct = self._clip_tail_pct()
        if pct > 0:
            lo = float(np.percentile(vals, pct))
            hi = float(np.percentile(vals, 100 - pct))
        else:
            lo = float(np.min(vals))
            hi = float(np.max(vals))
        return lo, hi

    def _ensure_record_scale(self):
        if self._scale_mode() != "record":
            self._lbl_scale_status.setText("")
            return
        if not self._raw:
            self._lbl_scale_status.setText("")
            return

        key = self._metric_cache_key()
        if key in self._scale_samples_cache:
            n = len(self._scale_samples_cache[key])
            self._lbl_scale_status.setText(f"fixed ({n} pts)")
            return
        if self._scale_job_key == key:
            self._lbl_scale_status.setText("computing…")
            return

        self._scale_job_key = key
        self._lbl_scale_status.setText("computing…")
        use_band = self._radio_band.isChecked()
        flo, fhi = self._get_band()
        win = self._window_sec()
        fut: Future = self.ctrl._exec.submit(
            self._compute_record_scale_samples,
            self._raw, self._total_sec, use_band, flo, fhi, win
        )

        def _done(f: Future, cache_key=key):
            err = f.exception()
            if err:
                self._pending_scale_error = str(err)
                self._pending_scale_key = cache_key
                QMetaObject.invokeMethod(
                    self, "_deliver_scale_error",
                    Qt.QueuedConnection,
                )
            else:
                self._pending_scale_key = cache_key
                self._pending_scale_samples = f.result()
                QMetaObject.invokeMethod(
                    self, "_deliver_scale_ok",
                    Qt.QueuedConnection,
                )

        fut.add_done_callback(_done)

    @staticmethod
    def _compute_record_scale_samples(
        raw: dict,
        total_sec: float,
        use_band: bool,
        flo: float,
        fhi: float,
        win: float,
    ) -> np.ndarray:
        if not raw or total_sec <= 0 or win <= 0:
            return np.array([], dtype=float)

        max_start = max(total_sec - win, 0.0)
        if max_start == 0.0:
            starts = [0.0]
        else:
            n_windows = int(min(_MAX_SCALE_WINDOWS, max(1, np.ceil(max_start / win) + 1)))
            starts = np.linspace(0.0, max_start, n_windows)

        samples: list[float] = []
        for t0 in starts:
            t1 = t0 + win
            for r in raw.values():
                i0 = max(0, int((t0 - r["t_start"]) * r["sr"]))
                i1 = min(r["n"], int((t1 - r["t_start"]) * r["sr"]) + 1)
                if i0 >= i1:
                    continue
                seg_v = r["vals"][i0:i1]
                if seg_v is None or len(seg_v) < 4:
                    continue
                if use_band:
                    val = _band_power(seg_v, r["sr"], flo, fhi)
                else:
                    val = _rms(seg_v)
                if np.isfinite(val):
                    samples.append(float(val))

        return np.asarray(samples, dtype=float)

    @QtCore.Slot()
    def _deliver_scale_ok(self):
        key = self._pending_scale_key
        samples = self._pending_scale_samples
        self._pending_scale_key = None
        self._pending_scale_samples = None
        self._scale_job_key = None
        if key is None or samples is None:
            return
        self._scale_samples_cache[key] = samples
        if self._scale_mode() == "record" and key == self._metric_cache_key():
            n = len(samples)
            self._lbl_scale_status.setText(f"fixed ({n} pts)")
            if self._raw:
                self._render_frame(self._cursor_sec)

    @QtCore.Slot()
    def _deliver_scale_error(self):
        self._scale_job_key = None
        self._pending_scale_key = None
        self._pending_scale_samples = None
        msg = self._pending_scale_error or "scale error"
        self._pending_scale_error = None
        self._lbl_scale_status.setText(msg)


# ---------------------------------------------------------------------------
# Outer TopoTab
# ---------------------------------------------------------------------------

class TopoTab(_ExplorerTab):
    """Explorer tab: EEG topographic maps (Results + Live sub-tabs)."""

    def __init__(self, ctrl, parent=None):
        super().__init__(ctrl, parent)
        self._build_widget()

    def _build_widget(self):
        root = QWidget()
        root.setStyleSheet(f"background: {BG}; color: {FG};")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)

        self._subtabs = QTabWidget()
        self._subtabs.setTabPosition(QTabWidget.North)
        self._subtabs.setDocumentMode(True)
        self._subtabs.setStyleSheet(
            f"QTabWidget::pane {{ border: 1px solid {GRID}; }}"
            f"QTabBar::tab {{ background: {BG}; color: {FG}; padding: 4px 12px; }}"
            f"QTabBar::tab:selected {{ background: #21262d; }}"
        )

        self._results = _ResultsPanel(self.ctrl)
        self._live    = _LiveTopoPanel(self.ctrl)

        self._subtabs.addTab(self._results, "Results")
        self._subtabs.addTab(self._live,    "Live")
        self._subtabs.currentChanged.connect(self._on_subtab_changed)

        layout.addWidget(self._subtabs)
        self._root = root

    # ------------------------------------------------------------------
    # Public interface used by explorer_dock.py
    # ------------------------------------------------------------------

    def refresh_tables(self):
        """Called when ctrl.sig_results_changed fires."""
        self._results.refresh_tables()

    def pause_live_playback(self):
        self._live.pause_playback()

    def _on_subtab_changed(self, idx):
        if idx != 1:
            self.pause_live_playback()
