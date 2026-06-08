from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt, correlate

import pyqtgraph as pg
from PySide6 import QtCore
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from .explorer_base import BG, FG, GRID, SEP, _ExplorerTab


# ---------------------------------------------------------------------------
# Timing / buffer constants
# ---------------------------------------------------------------------------

_ANALYSIS_SR_CAP      = 128.0   # max analysis sample rate (Hz)
_METRIC_WIN_S         = 8.0     # rolling correlation window (s)
_METRIC_STEP_S        = 0.5     # step between metric centres (s)
_METRIC_HISTORY_S     = 120.0   # seconds of history in metric panels
_METRIC_LOOKAHEAD_S   = 60.0    # seconds precomputed ahead of cursor
_METRIC_EXTEND_MARGIN = 20.0    # start re-fetch this many s before buffer edge
_LAG_VALUES = np.linspace(-2.0, 2.0, 81, dtype=float)
_PLAYBACK_INTERVAL_MS = 33      # ~30 fps


# ---------------------------------------------------------------------------
# Frozen dataclasses (kept for external callers / presets)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandDefinition:
    label: str
    low_hz: float
    high_hz: float
    color_role: str = ""


@dataclass(frozen=True)
class BandPreset:
    name: str
    intended_signal_types: tuple[str, ...]
    bands: tuple[BandDefinition, ...]


@dataclass(frozen=True)
class ChannelInfo:
    name: str
    sample_rate: float = float("nan")
    units: str = ""
    signal_type: str = "Other"


@dataclass
class _MetricResult:
    """All data produced by one worker run. Fields are mutable (corr_smooth recomputed on persist change)."""
    # Native-SR display data
    a_times:  np.ndarray
    a_values: np.ndarray
    b_times:  np.ndarray
    b_values: np.ndarray
    info_a:   ChannelInfo
    info_b:   ChannelInfo
    # Analysis info
    target_sr: float
    # Metric time-series
    centers:     np.ndarray   # (n,)  time of each window centre
    corr_raw:    np.ndarray   # (n,)  Pearson r
    corr_smooth: np.ndarray   # (n,)  EWMA-smoothed (recomputed when persist changes)
    best_lag:    np.ndarray   # (n,)  dominant lag (s)
    lag_image:   np.ndarray   # (n_lags, n)
    # Coverage window
    t0: float
    t1: float


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _format_sr(v: float) -> str:
    if not np.isfinite(v) or v <= 0: return "n/a"
    return f"{v:.0f} Hz" if v >= 100 else (f"{v:.1f} Hz" if v >= 10 else f"{v:.2f} Hz")


def _format_secs(v: float) -> str:
    if not np.isfinite(v) or v < 0: return "n/a"
    if v >= 3600:
        h, r = divmod(int(v), 3600); m, s = divmod(r, 60)
        return f"{h}:{m:02d}:{s:02d}"
    if v >= 60:
        m, s = divmod(int(v), 60); return f"{m}:{s:02d}"
    return f"{v:.1f} s" if v >= 10 else f"{v:.2f} s"


# ---------------------------------------------------------------------------
# Signal-type inference
# ---------------------------------------------------------------------------

def _infer_signal_type(name: str, units: str) -> str:
    text = f"{name} {units}".lower()
    if any(t in text for t in ("ecg", "ekg", "heart", "hr")):        return "ECG"
    if any(t in text for t in ("resp", "airflow", "thor", "abd", "nasal")): return "Resp"
    if any(t in text for t in ("emg", "chin", "leg")):               return "EMG"
    eeg = ("eeg","fp","fz","cz","pz","oz","f3","f4","c3","c4",
           "o1","o2","t3","t4","t5","t6","a1","a2","m1","m2")
    if any(t in text for t in eeg) or units.strip().lower() in {"uv","µv"}:
        return "EEG"
    return "Other"


# ---------------------------------------------------------------------------
# Band presets
# ---------------------------------------------------------------------------

def _band_presets() -> tuple[BandPreset, ...]:
    return (
        BandPreset("Generic", ("EEG","ECG","Resp","EMG","Other"), ()),
        BandPreset("EEG Classic", ("EEG",), (
            BandDefinition("delta", 0.5,  4.0),
            BandDefinition("theta", 4.0,  8.0),
            BandDefinition("alpha", 8.0, 12.0),
            BandDefinition("sigma",11.0, 16.0),
            BandDefinition("beta", 16.0, 30.0),
            BandDefinition("low gamma", 30.0, 55.0),
        )),
        BandPreset("Sleep EEG", ("EEG",), (
            BandDefinition("SO",    0.3,  1.25),
            BandDefinition("delta", 1.0,  4.0),
            BandDefinition("theta", 4.0,  8.0),
            BandDefinition("sigma",11.0, 16.0),
            BandDefinition("beta", 16.0, 30.0),
        )),
        BandPreset("Generic Physiology", ("ECG","Resp","Other"), (
            BandDefinition("ultra-slow",       0.003, 0.03),
            BandDefinition("infra-slow",       0.01,  0.1),
            BandDefinition("respiration-ish",  0.1,   0.5),
            BandDefinition("cardiac-ish",      0.8,   2.5),
        )),
    )


# ---------------------------------------------------------------------------
# Signal math helpers
# ---------------------------------------------------------------------------

def _robust_zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0: return arr
    finite = np.isfinite(arr)
    if not np.any(finite): return np.zeros_like(arr)
    work = arr[finite]
    med = float(np.median(work))
    mad = float(np.median(np.abs(work - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 1e-12:
        std = float(np.std(work))
        scale = std if np.isfinite(std) and std > 1e-12 else 1.0
    out = np.zeros_like(arr)
    out[finite] = (work - med) / scale
    return out


def _interp_to_grid(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    times  = np.asarray(times,  dtype=float)
    values = np.asarray(values, dtype=float)
    if times.size < 2 or values.size < 2 or grid.size == 0:
        return np.zeros(grid.shape, dtype=float)
    ok = np.isfinite(times) & np.isfinite(values)
    times, values = times[ok], values[ok]
    if times.size < 2: return np.zeros(grid.shape, dtype=float)
    uniq_t, uniq_idx = np.unique(times, return_index=True)
    if uniq_t.size < 2: return np.zeros(grid.shape, dtype=float)
    return np.interp(grid, uniq_t, values[uniq_idx]).astype(float)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 8 or a.size != b.size: return float("nan")
    ok = np.isfinite(a) & np.isfinite(b)
    if np.sum(ok) < 8: return float("nan")
    aa, bb = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    da, db = float(np.std(aa)), float(np.std(bb))
    if da <= 1e-12 or db <= 1e-12: return float("nan")
    return float(np.mean((aa / da) * (bb / db)))


def _ewma_array(values: np.ndarray, alpha: float) -> np.ndarray:
    out  = np.empty_like(values, dtype=float)
    prev = float("nan")
    for i, v in enumerate(values):
        fv = float(v)
        if np.isfinite(fv):
            prev = fv if not np.isfinite(prev) else alpha * fv + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _fft_xcorr_profile(
    a: np.ndarray, b: np.ndarray, sr: float, lag_values: np.ndarray
) -> tuple[np.ndarray, float]:
    out = np.full(lag_values.shape, np.nan, dtype=float)
    if a.size < 16 or a.size != b.size or not np.isfinite(sr) or sr <= 0:
        return out, float("nan")
    aa = a - a.mean(); bb = b - b.mean()
    da, db = float(np.std(aa)), float(np.std(bb))
    if da <= 1e-12 or db <= 1e-12: return out, float("nan")
    aa /= da; bb /= db
    n = len(aa)
    xcorr = correlate(aa, bb, mode="full", method="fft") / n
    lags_s = np.arange(-(n - 1), n, dtype=float) / sr
    in_range = (lag_values >= lags_s[0]) & (lag_values <= lags_s[-1])
    if np.any(in_range):
        out[in_range] = np.interp(lag_values[in_range], lags_s, xcorr)
    finite = np.isfinite(out)
    if not np.any(finite): return out, float("nan")
    best = float(lag_values[finite][np.argmax(np.abs(out[finite]))])
    return out, best


# ---------------------------------------------------------------------------
# Cancellation token (shared between tab and worker)
# ---------------------------------------------------------------------------

class _CancelToken:
    __slots__ = ("_ev",)
    def __init__(self):         self._ev = threading.Event()
    def cancel(self):           self._ev.set()
    @property
    def cancelled(self) -> bool: return self._ev.is_set()


# ---------------------------------------------------------------------------
# Background metric worker — pure numpy, no Qt calls except Signal emission
# ---------------------------------------------------------------------------

class _MetricWorker(QtCore.QThread):
    result_ready: Signal = Signal(object)
    status_msg:   Signal = Signal(str)

    # Class-level lock: Luna's p.slice() may not be thread-safe.
    # This serialises EDF reads while allowing concurrent numpy work.
    _slice_lock = threading.Lock()

    def __init__(self, session, a_name, b_name, cursor, ns, target_sr, token, parent=None):
        super().__init__(parent)
        self._session   = session
        self._a_name    = a_name
        self._b_name    = b_name
        self._cursor    = float(cursor)
        self._ns        = float(ns)
        self._target_sr = float(target_sr)
        self._token     = token

    def run(self):
        try:
            result = self._compute()
        except Exception as e:
            self.status_msg.emit(f"Worker error: {e}")
            return
        if result is not None and not self._token.cancelled:
            self.result_ready.emit(result)

    def _compute(self) -> _MetricResult | None:
        t0 = max(0.0, self._cursor - _METRIC_HISTORY_S)
        t1 = min(self._ns, self._cursor + _METRIC_LOOKAHEAD_S)
        if t1 - t0 < 4.0:
            return None

        # Serialise EDF access
        with self._slice_lock:
            if self._token.cancelled: return None
            info_a = self._session.channel_info(self._a_name)
            info_b = self._session.channel_info(self._b_name)
            raw_a  = self._session.extract_channel_slice(self._a_name, t0, t1)
            if self._token.cancelled: return None
            raw_b  = self._session.extract_channel_slice(self._b_name, t0, t1)

        if raw_a is None or raw_b is None or self._token.cancelled:
            return None

        # Build common analysis grid (target_sr, clipped to real overlap)
        sr = self._target_sr
        n  = max(2, int(math.floor((t1 - t0) * sr)) + 1)
        grid  = np.linspace(t0, t1, n, dtype=float)
        a_aln = _interp_to_grid(raw_a["times"], raw_a["values"], grid)
        b_aln = _interp_to_grid(raw_b["times"], raw_b["values"], grid)

        if self._token.cancelled: return None

        # Rolling metrics
        win_n  = int(_METRIC_WIN_S  * sr)
        step_n = max(1, int(_METRIC_STEP_S * sr))
        starts = np.arange(0, max(0, len(grid) - win_n), step_n)
        if starts.size == 0: return None

        centers   = grid[starts + win_n // 2]
        corr_raw  = np.full(len(centers), np.nan, dtype=float)
        best_lag  = np.full(len(centers), np.nan, dtype=float)
        lag_image = np.full((len(_LAG_VALUES), len(centers)), np.nan, dtype=float)

        # Check cancel every 20 windows (~10 ms intervals at 128 Hz)
        for i, s in enumerate(starts):
            if i % 20 == 0 and self._token.cancelled:
                return None
            a_seg = a_aln[s : s + win_n]
            b_seg = b_aln[s : s + win_n]
            corr_raw[i] = _safe_corr(a_seg, b_seg)
            lp, bl = _fft_xcorr_profile(a_seg, b_seg, sr, _LAG_VALUES)
            lag_image[:, i] = lp
            best_lag[i]     = bl

        if self._token.cancelled: return None

        return _MetricResult(
            a_times=raw_a["times"], a_values=raw_a["values"],
            b_times=raw_b["times"], b_values=raw_b["values"],
            info_a=info_a, info_b=info_b,
            target_sr=sr,
            centers=centers, corr_raw=corr_raw,
            corr_smooth=_ewma_array(corr_raw, 0.18),  # default; recomputed on persist change
            best_lag=best_lag, lag_image=lag_image,
            t0=float(t0), t1=float(t1),
        )


# ---------------------------------------------------------------------------
# Channel catalog + raw extraction (main-thread object, EDF calls serialised via worker lock)
# ---------------------------------------------------------------------------

class TwoChannelAnalysisSession(QtCore.QObject):
    def __init__(self, ctrl, parent=None):
        super().__init__(parent or ctrl)
        self.ctrl = ctrl
        self._catalog:     list[ChannelInfo] = []
        self._catalog_key: tuple | None      = None

    def _record_key(self) -> tuple:
        return (id(getattr(self.ctrl, "p", None)),
                float(getattr(self.ctrl, "ns", 0.0) or 0.0))

    def invalidate(self):
        self._catalog     = []
        self._catalog_key = None

    def current_window(self) -> tuple[float, float]:
        lo = float(getattr(self.ctrl, "last_x1", 0.0) or 0.0)
        hi = float(getattr(self.ctrl, "last_x2", 0.0) or 0.0)
        return lo, hi

    def channel_catalog(self) -> list[ChannelInfo]:
        key = self._record_key()
        if self._catalog and self._catalog_key == key:
            return list(self._catalog)
        p = getattr(self.ctrl, "p", None)
        catalog: list[ChannelInfo] = []
        if p is not None:
            try:
                hdr = p.table("HEADERS", "CH")
                if hdr is not None and not hdr.empty and "CH" in hdr.columns:
                    for _, row in hdr.iterrows():
                        name = str(row.get("CH", "")).strip()
                        if not name: continue
                        try:   sr = float(row.get("SR", np.nan))
                        except Exception: sr = float("nan")
                        units = str(row.get("PDIM", "") or "").strip()
                        catalog.append(ChannelInfo(
                            name=name, sample_rate=sr, units=units,
                            signal_type=_infer_signal_type(name, units),
                        ))
            except Exception:
                pass
        self._catalog     = sorted(catalog, key=lambda i: i.name.lower())
        self._catalog_key = key
        return list(self._catalog)

    def channel_info(self, channel: str) -> ChannelInfo:
        for info in self.channel_catalog():
            if info.name == channel: return info
        return ChannelInfo(name=str(channel or ""))

    def extract_channel_slice(self, channel: str, t0: float, t1: float) -> dict[str, Any] | None:
        p = getattr(self.ctrl, "p", None)
        if p is None or not channel: return None
        t0, t1 = max(0.0, float(t0)), max(float(t0), float(t1))
        try:
            raw = p.slice(p.s2i([(t0, t1)]), chs=channel, time=True)
        except Exception:
            return None
        arr = None if raw is None else raw[1]
        if arr is None or len(arr) == 0: return None
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2: return None
        info = self.channel_info(channel)
        return {
            "channel": channel,
            "times":       arr[:, 0].copy(),
            "values":      arr[:, 1].copy(),
            "sample_rate": float(info.sample_rate),
            "units":       info.units,
            "signal_type": info.signal_type,
        }


# ---------------------------------------------------------------------------
# pyqtgraph curve factory
# ---------------------------------------------------------------------------

def _make_curve(plot_item: pg.PlotItem, pen, antialias: bool = True) -> pg.PlotDataItem:
    c = pg.PlotDataItem(antialias=antialias)
    c.setPen(pen)
    c.setClipToView(True)
    c.setDownsampling(auto=True, method="peak")
    plot_item.addItem(c)
    return c


# ---------------------------------------------------------------------------
# Main tab widget
# ---------------------------------------------------------------------------

class TwoChannelDynamicsTab(_ExplorerTab):
    _SPEEDS      = (("0.2×", 0.2), ("0.5×", 0.5), ("1×", 1.0), ("2×", 2.0), ("5×", 5.0))
    _WINDOWS     = (("5 s", 5.0), ("10 s", 10.0), ("30 s", 30.0), ("60 s", 60.0), ("5 min", 300.0))
    _PERSISTENCE = (("Fast", 0.45), ("Medium", 0.18), ("Slow", 0.07))

    def __init__(self, ctrl, parent=None):
        super().__init__(ctrl, parent)
        self._session = TwoChannelAnalysisSession(ctrl, self)
        self._presets = _band_presets()
        self._building_controls = False
        self._last_record_key: tuple | None = None

        # Playback state
        self._cursor_time:      float        = 0.0
        self._view_span:        float        = 10.0
        self._playing:          bool         = False
        self._playback_speed:   float        = 1.0
        self._playback_last_ts: float | None = None
        self._slider_lock:      bool         = False
        self._metric_tick:      int          = 0
        self._label_ts:         float | None = None

        # Worker state
        self._result:       _MetricResult | None = None
        self._active_token: _CancelToken  | None = None
        self._worker:       _MetricWorker | None = None

        self._playback_timer = QtCore.QTimer(self)
        self._playback_timer.setInterval(_PLAYBACK_INTERVAL_MS)
        self._playback_timer.timeout.connect(self._on_playback_tick)

        self._build_widget()
        if hasattr(self.ctrl, "sig_window_range_changed"):
            self.ctrl.sig_window_range_changed.connect(
                self._on_main_window_changed, Qt.QueuedConnection)

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def _build_widget(self):
        root  = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(4)

        # ── Selection ──────────────────────────────────────────────────
        sel_box  = QGroupBox("Selection")
        sel_grid = QGridLayout(sel_box)
        sel_grid.setContentsMargins(8, 10, 8, 8); sel_grid.setSpacing(6)
        combo_a    = QComboBox(); combo_b    = QComboBox()
        combo_mode = QComboBox(); combo_norm = QComboBox()
        btn_refresh = QPushButton("Refresh")
        combo_mode.addItems([p.name for p in self._presets])
        combo_norm.addItem("Original Units",   "native")
        combo_norm.addItem("Standardized (z)", "zscore")
        sel_grid.addWidget(QLabel("Signal A"), 0, 0); sel_grid.addWidget(combo_a, 0, 1)
        sel_grid.addWidget(QLabel("Signal B"), 0, 2); sel_grid.addWidget(combo_b, 0, 3)
        sel_grid.addWidget(QLabel("Preset"),   1, 0); sel_grid.addWidget(combo_mode, 1, 1)
        sel_grid.addWidget(QLabel("Scale"),    1, 2); sel_grid.addWidget(combo_norm, 1, 3)
        sel_grid.addWidget(btn_refresh, 0, 4, 2, 1)

        # ── Playback controls ──────────────────────────────────────────
        pb_box  = QGroupBox("Playback")
        pb_grid = QGridLayout(pb_box)
        pb_grid.setContentsMargins(8, 10, 8, 8); pb_grid.setSpacing(6)
        btn_play  = QPushButton("▶ Play")
        btn_pause = QPushButton("⏸ Pause")
        btn_stop  = QPushButton("⏹ Stop")
        btn_adopt = QPushButton("Adopt Window")
        chk_follow    = QCheckBox("Follow Main")
        combo_speed   = QComboBox(); combo_window = QComboBox(); combo_persist = QComboBox()
        slider        = QSlider(Qt.Horizontal); slider.setRange(0, 10000)
        lbl_cursor    = QLabel("Cursor: —")
        lbl_status    = QLabel("Select two channels.")
        lbl_status.setWordWrap(True)
        for label, val in self._SPEEDS:       combo_speed.addItem(label, val)
        for label, val in self._WINDOWS:      combo_window.addItem(label, val)
        for label, val in self._PERSISTENCE:  combo_persist.addItem(label, val)
        combo_speed.setCurrentIndex(2); combo_window.setCurrentIndex(1); combo_persist.setCurrentIndex(1)
        pb_grid.addWidget(btn_play,  0, 0); pb_grid.addWidget(btn_pause, 0, 1)
        pb_grid.addWidget(btn_stop,  0, 2); pb_grid.addWidget(btn_adopt, 0, 3)
        pb_grid.addWidget(chk_follow, 0, 4)
        pb_grid.addWidget(QLabel("Speed"),   1, 0); pb_grid.addWidget(combo_speed,   1, 1)
        pb_grid.addWidget(QLabel("Window"),  1, 2); pb_grid.addWidget(combo_window,  1, 3)
        pb_grid.addWidget(QLabel("Persist"), 1, 4); pb_grid.addWidget(combo_persist, 1, 5)
        pb_grid.addWidget(lbl_cursor, 2, 0, 1, 3); pb_grid.addWidget(lbl_status, 2, 3, 1, 3)
        pb_grid.addWidget(slider, 3, 0, 1, 6)

        # ── Channel info ───────────────────────────────────────────────
        meta_box  = QGroupBox("Channel info")
        meta_form = QFormLayout(meta_box)
        meta_form.setContentsMargins(8, 10, 8, 8); meta_form.setSpacing(3)
        lbl_a_meta  = QLabel("—"); lbl_b_meta  = QLabel("—")
        lbl_overlap = QLabel("—"); lbl_align   = QLabel("—")
        meta_form.addRow("A",       lbl_a_meta)
        meta_form.addRow("B",       lbl_b_meta)
        meta_form.addRow("Buffer",  lbl_overlap)
        meta_form.addRow("Grid",    lbl_align)

        # ── Plots ──────────────────────────────────────────────────────
        gfx = pg.GraphicsLayoutWidget()
        gfx.setBackground(BG)
        gfx.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        plot_a    = gfx.addPlot(row=0, col=0)
        plot_b    = gfx.addPlot(row=1, col=0)
        plot_corr = gfx.addPlot(row=2, col=0)
        plot_lag  = gfx.addPlot(row=3, col=0)
        plot_b.setXLink(plot_a)

        _ap = pg.mkPen(FG); _gp = pg.mkPen(GRID)
        for p in (plot_a, plot_b, plot_corr, plot_lag):
            p.showGrid(x=True, y=True, alpha=0.15)
            p.getAxis("left").setTextPen(_ap);   p.getAxis("bottom").setTextPen(_ap)
            p.getAxis("left").setPen(_gp);       p.getAxis("bottom").setPen(_gp)
            p.hideButtons(); p.setMenuEnabled(False)

        plot_a.setTitle("Signal A",                color=FG, size="10pt")
        plot_b.setTitle("Signal B",                color=FG, size="10pt")
        plot_corr.setTitle("Rolling Correlation",  color=FG, size="10pt")
        plot_lag.setTitle("Cross-Correlation Lag", color=FG, size="10pt")
        for p in (plot_a, plot_b, plot_corr):
            p.getAxis("bottom").setStyle(showValues=False)
        plot_lag.getAxis("bottom").setLabel("time (s)", color=FG)
        plot_corr.setYRange(-1.05, 1.05, padding=0.0)
        plot_corr.getAxis("left").setLabel("r",      color=FG)
        plot_lag.getAxis("left").setLabel("lag (s)", color=FG)
        plot_lag.setYRange(float(_LAG_VALUES[0]), float(_LAG_VALUES[-1]), padding=0.0)

        A_COL = (80,  200, 255)
        B_COL = (255, 130,  90)

        curve_a_glow = _make_curve(plot_a, pg.mkPen((*A_COL, 40),  width=8))
        curve_a      = _make_curve(plot_a, pg.mkPen((*A_COL, 230), width=1.8))
        curve_b_glow = _make_curve(plot_b, pg.mkPen((*B_COL, 40),  width=8))
        curve_b      = _make_curve(plot_b, pg.mkPen((*B_COL, 230), width=1.8))

        cursor_pen = pg.mkPen("#ffe66d", width=1.2, style=Qt.DashLine)
        cursor_a   = pg.InfiniteLine(pos=0.0, angle=90, pen=cursor_pen)
        cursor_b   = pg.InfiniteLine(pos=0.0, angle=90, pen=cursor_pen)
        plot_a.addItem(cursor_a); plot_b.addItem(cursor_b)

        corr_raw_curve    = _make_curve(plot_corr, pg.mkPen((100, 140, 220, 130), width=1.0))
        corr_smooth_curve = _make_curve(plot_corr, pg.mkPen((246, 211, 101, 240), width=2.2))
        zero_line = pg.InfiniteLine(pos=0.0, angle=0,  pen=pg.mkPen((180, 180, 180, 60), width=1))
        corr_now  = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen((255, 230, 100, 140), width=1))
        plot_corr.addItem(zero_line); plot_corr.addItem(corr_now)

        lag_img        = pg.ImageItem()
        best_lag_curve = _make_curve(plot_lag, pg.mkPen((255, 255, 255, 200), width=1.5))
        lag_now  = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen((255, 230, 100, 140), width=1))
        lag_zero = pg.InfiniteLine(pos=0.0, angle=0,  pen=pg.mkPen((200, 200, 200,  70), width=1))
        plot_lag.addItem(lag_img)
        plot_lag.addItem(lag_now); plot_lag.addItem(lag_zero)

        try:
            cmap = pg.colormap.get("CET-D1")
            lag_img.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
        except Exception:
            pass

        outer.addWidget(sel_box); outer.addWidget(pb_box)
        outer.addWidget(meta_box); outer.addWidget(gfx, 1)

        root.setStyleSheet(f"""
            QWidget    {{ color: {FG}; }}
            QGroupBox  {{ color: {FG}; border: 1px solid {SEP}; margin-top: 8px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
            QComboBox, QPushButton, QSlider {{
                background: #161b22; color: {FG};
                border: 1px solid {SEP}; padding: 3px 6px; border-radius: 3px;
            }}
            QPushButton:hover {{ background: #21262d; }}
            QCheckBox  {{ color: {FG}; }}
        """)

        # Wire signals
        btn_refresh.clicked.connect(self.refresh_controls)
        combo_a.currentTextChanged.connect(self._on_selection_changed)
        combo_b.currentTextChanged.connect(self._on_selection_changed)
        combo_mode.currentTextChanged.connect(self._on_selection_changed)
        combo_norm.currentIndexChanged.connect(self._on_scale_changed)
        combo_speed.currentIndexChanged.connect(self._on_speed_changed)
        combo_window.currentIndexChanged.connect(self._on_window_combo_changed)
        combo_persist.currentIndexChanged.connect(self._on_persist_changed)
        btn_play.clicked.connect(self._play)
        btn_pause.clicked.connect(self._pause)
        btn_stop.clicked.connect(self._stop)
        btn_adopt.clicked.connect(self._adopt_main_window)
        slider.valueChanged.connect(self._on_slider_changed)

        # Store widget refs
        self._root            = root
        self._combo_a         = combo_a
        self._combo_b         = combo_b
        self._combo_mode      = combo_mode
        self._combo_norm      = combo_norm
        self._combo_speed     = combo_speed
        self._combo_window    = combo_window
        self._combo_persist   = combo_persist
        self._chk_follow      = chk_follow
        self._slider          = slider
        self._lbl_cursor      = lbl_cursor
        self._lbl_status      = lbl_status
        self._lbl_a_meta      = lbl_a_meta
        self._lbl_b_meta      = lbl_b_meta
        self._lbl_overlap     = lbl_overlap
        self._lbl_align       = lbl_align
        self._plot_a          = plot_a
        self._plot_b          = plot_b
        self._plot_corr       = plot_corr
        self._plot_lag        = plot_lag
        self._curve_a         = curve_a
        self._curve_b         = curve_b
        self._curve_a_glow    = curve_a_glow
        self._curve_b_glow    = curve_b_glow
        self._cursor_a        = cursor_a
        self._cursor_b        = cursor_b
        self._corr_raw_curve    = corr_raw_curve
        self._corr_smooth_curve = corr_smooth_curve
        self._corr_now        = corr_now
        self._lag_img         = lag_img
        self._best_lag_curve  = best_lag_curve
        self._lag_now         = lag_now

        self.refresh_controls()

    # ------------------------------------------------------------------
    # Controls lifecycle
    # ------------------------------------------------------------------

    def refresh_controls(self):
        p   = getattr(self.ctrl, "p", None)
        key = (id(p), float(getattr(self.ctrl, "ns", 0.0) or 0.0))
        if key != self._last_record_key:
            self._session.invalidate()
            self._last_record_key = key
            self._cancel_worker()
            self._result = None

        catalog = self._session.channel_catalog()
        prev_a  = self._combo_a.currentText()
        prev_b  = self._combo_b.currentText()

        self._building_controls = True
        try:
            for combo in (self._combo_a, self._combo_b):
                combo.blockSignals(True); combo.clear(); combo.addItem("")
            for info in catalog:
                self._combo_a.addItem(info.name)
                self._combo_b.addItem(info.name)
            for combo, prev in ((self._combo_a, prev_a), (self._combo_b, prev_b)):
                idx = combo.findText(prev)
                if idx >= 0: combo.setCurrentIndex(idx)
            if self._combo_a.currentIndex() <= 0 and len(catalog) >= 1:
                self._combo_a.setCurrentIndex(1)
            if self._combo_b.currentIndex() <= 0 and len(catalog) >= 2:
                self._combo_b.setCurrentIndex(2)
            elif self._combo_b.currentIndex() <= 0 and len(catalog) == 1:
                self._combo_b.setCurrentIndex(1)
        finally:
            for combo in (self._combo_a, self._combo_b):
                combo.blockSignals(False)
            self._building_controls = False

        # Seed cursor at centre of current main-window view
        lo, hi = self._session.current_window()
        if hi > lo:
            self._view_span = max(1.0, hi - lo)
            self._set_cursor(0.5 * (lo + hi))

        self._start_worker()

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def _start_worker(self):
        a_name = self._combo_a.currentText().strip()
        b_name = self._combo_b.currentText().strip()
        if not a_name or not b_name:
            self._clear_plots()
            self._lbl_status.setText("Select two channels.")
            return

        ns = self._record_duration()
        if ns <= 0:
            self._lbl_status.setText("No recording loaded.")
            return

        # Cancel any in-flight worker; its result will be dropped via stale token.
        if self._active_token:
            self._active_token.cancel()

        info_a    = self._session.channel_info(a_name)
        info_b    = self._session.channel_info(b_name)
        valid_srs = [s for s in (info_a.sample_rate, info_b.sample_rate)
                     if np.isfinite(s) and s > 0]
        target_sr = min(_ANALYSIS_SR_CAP, max(8.0, min(valid_srs) if valid_srs else 64.0))

        token = _CancelToken()
        self._active_token = token

        worker = _MetricWorker(
            self._session, a_name, b_name,
            self._cursor_time, ns, target_sr, token,
            parent=self,
        )
        worker.result_ready.connect(
            lambda r, t=token: self._on_result_ready(r, t), Qt.QueuedConnection)
        worker.status_msg.connect(self._lbl_status.setText, Qt.QueuedConnection)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self._worker = worker
        self._lbl_status.setText("Computing…")

    def _cancel_worker(self):
        if self._active_token:
            self._active_token.cancel()
            self._active_token = None
        self._worker = None

    def _on_result_ready(self, result: _MetricResult, token: _CancelToken):
        # Drop stale results from superseded workers
        if token is not self._active_token or token.cancelled:
            return

        # Apply user's current persistence alpha to the fresh data
        result.corr_smooth = _ewma_array(result.corr_raw, self._current_persistence_alpha())

        self._result = result
        self._apply_result_to_plots(result)
        self._update_meta_labels(result)
        self._scroll_viewport()
        self._lbl_status.setText(
            f"Ready — {_format_secs(result.t1 - result.t0)} buffered"
            f"  |  {_format_sr(result.target_sr)} analysis SR"
            f"  |  {len(result.centers)} metric pts")

    # ------------------------------------------------------------------
    # Plot population — called once per new result, never during playback
    # ------------------------------------------------------------------

    def _apply_result_to_plots(self, result: _MetricResult):
        mode = str(self._combo_norm.currentData() or "native")
        a_v  = _robust_zscore(result.a_values) if mode == "zscore" else result.a_values
        b_v  = _robust_zscore(result.b_values) if mode == "zscore" else result.b_values

        self._curve_a_glow.setData(result.a_times, a_v)
        self._curve_a.setData(result.a_times, a_v)
        self._curve_b_glow.setData(result.b_times, b_v)
        self._curve_b.setData(result.b_times, b_v)

        self._set_stable_y(self._plot_a, a_v)
        self._set_stable_y(self._plot_b, b_v)

        units_a = "z" if mode == "zscore" else (result.info_a.units or "")
        units_b = "z" if mode == "zscore" else (result.info_b.units or "")
        self._plot_a.getAxis("left").setLabel(result.info_a.name, units=units_a)
        self._plot_b.getAxis("left").setLabel(result.info_b.name, units=units_b)

        self._corr_raw_curve.setData(result.centers, result.corr_raw)
        self._corr_smooth_curve.setData(result.centers, result.corr_smooth)

        valid = np.isfinite(result.best_lag)
        if np.any(valid):
            self._best_lag_curve.setData(result.centers[valid], result.best_lag[valid])
        else:
            self._best_lag_curve.setData([], [])

        img = result.lag_image
        if img.size > 0 and np.any(np.isfinite(img)):
            img_disp = np.where(np.isfinite(img), img, 0.0)
            self._lag_img.setImage(img_disp, autoLevels=False)
            self._lag_img.setLevels((-1.0, 1.0))
            x0 = float(result.centers[0])
            x1 = float(result.centers[-1]) if len(result.centers) > 1 else x0 + 1.0
            self._lag_img.setRect(QtCore.QRectF(
                x0, float(_LAG_VALUES[0]),
                max(1e-6, x1 - x0), float(_LAG_VALUES[-1] - _LAG_VALUES[0]),
            ))
        else:
            self._lag_img.clear()

    def _set_stable_y(self, plot: pg.PlotItem, values: np.ndarray):
        finite = values[np.isfinite(values)]
        if finite.size < 4: return
        lo = float(np.percentile(finite, 1.0))
        hi = float(np.percentile(finite, 99.0))
        if hi <= lo: hi = lo + 1.0
        pad = 0.1 * (hi - lo)
        plot.setYRange(lo - pad, hi + pad, padding=0.0)

    def _update_meta_labels(self, result: _MetricResult):
        self._lbl_a_meta.setText(
            f"{result.info_a.signal_type}  |  {_format_sr(result.info_a.sample_rate)}"
            f"  |  {result.info_a.units or 'unitless'}")
        self._lbl_b_meta.setText(
            f"{result.info_b.signal_type}  |  {_format_sr(result.info_b.sample_rate)}"
            f"  |  {result.info_b.units or 'unitless'}")
        span = max(0.0, result.t1 - result.t0)
        self._lbl_overlap.setText(
            f"{_format_secs(span)}  ({result.t0:.1f}–{result.t1:.1f} s)")
        native_a = result.info_a.sample_rate <= _ANALYSIS_SR_CAP
        native_b = result.info_b.sample_rate <= _ANALYSIS_SR_CAP
        self._lbl_align.setText(
            f"{_format_sr(result.target_sr)}"
            f"  |  A: {'native' if native_a else 'interp'}"
            f"  B: {'native' if native_b else 'interp'}")

    def _clear_plots(self):
        for c in (self._curve_a, self._curve_b, self._curve_a_glow, self._curve_b_glow,
                  self._corr_raw_curve, self._corr_smooth_curve, self._best_lag_curve):
            c.setData([], [])
        self._lag_img.clear()

    # ------------------------------------------------------------------
    # Viewport scroll — the ONLY work done on every playback frame
    # ------------------------------------------------------------------

    def _scroll_viewport(self, full: bool = True):
        lo, hi = self._visible_bounds()
        t = self._cursor_time

        # Waveform plots + cursors: every frame
        self._cursor_a.setPos(t)
        self._cursor_b.setPos(t)
        self._plot_a.setXRange(lo, hi, padding=0.0)
        self._plot_b.setXRange(lo, hi, padding=0.0)

        # Metric panels: throttled (every 3rd frame ≈ 10 fps)
        if full:
            h_lo = t - _METRIC_HISTORY_S
            self._plot_corr.setXRange(h_lo, t + 10.0, padding=0.0)
            self._plot_lag.setXRange(h_lo, t + 10.0, padding=0.0)
            self._corr_now.setPos(t)
            self._lag_now.setPos(t)

        self._sync_slider_to_cursor()
        now = time.monotonic()
        if self._label_ts is None or (now - self._label_ts) >= 0.1:
            self._lbl_cursor.setText(
                f"Cursor: {_format_secs(t)}  |  window {_format_secs(hi - lo)}")
            self._label_ts = now

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _play(self):
        if self._playing: return
        if self._result is None:
            self._start_worker()
        self._playing          = True
        self._playback_last_ts = time.monotonic()
        self._playback_timer.start()

    def _pause(self):
        self._playing          = False
        self._playback_last_ts = None
        self._playback_timer.stop()

    def _stop(self):
        self._pause()
        lo, hi = self._session.current_window()
        self._set_cursor(0.5 * (lo + hi) if hi > lo else 0.0)
        self._scroll_viewport()

    def _on_playback_tick(self):
        if not self._playing: return

        now = time.monotonic()
        dt  = 0.033 if self._playback_last_ts is None else max(0.0, now - self._playback_last_ts)
        self._playback_last_ts = now
        self._set_cursor(self._cursor_time + dt * self._playback_speed)

        ns = self._record_duration()
        if ns > 0 and self._cursor_time >= ns:
            self._pause()
            return

        # Trigger lookahead re-fetch when buffer edge is approaching.
        # Only if no worker is currently running (avoids re-triggering before previous finishes).
        if (self._result is not None and
                self._cursor_time > self._result.t1 - _METRIC_EXTEND_MARGIN and
                (self._active_token is None or self._active_token.cancelled) and
                (self._worker is None or not self._worker.isRunning())):
            self._start_worker()

        # Throttle expensive metric-panel updates to ~10 fps
        self._metric_tick = (self._metric_tick + 1) % 3
        self._scroll_viewport(full=(self._metric_tick == 0))

    # ------------------------------------------------------------------
    # Cursor / slider
    # ------------------------------------------------------------------

    def _record_duration(self) -> float:
        return float(getattr(self.ctrl, "ns", 0.0) or 0.0)

    def _visible_bounds(self) -> tuple[float, float]:
        ns   = self._record_duration()
        half = 0.5 * self._view_span
        lo   = max(0.0, self._cursor_time - half)
        hi   = lo + self._view_span
        if ns > 0 and hi > ns:
            hi = ns; lo = max(0.0, hi - self._view_span)
        return float(lo), float(hi)

    def _set_cursor(self, value: float, sync: bool = True):
        ns      = self._record_duration()
        clamped = min(max(float(value), 0.0), ns) if ns > 0 else max(float(value), 0.0)
        self._cursor_time = clamped
        if sync: self._sync_slider_to_cursor()

    def _sync_slider_to_cursor(self):
        ns  = self._record_duration()
        pos = 0 if ns <= 0 else int(round(10000.0 * self._cursor_time / ns))
        self._slider_lock = True
        try:     self._slider.setValue(max(0, min(10000, pos)))
        finally: self._slider_lock = False

    # ------------------------------------------------------------------
    # Control callbacks
    # ------------------------------------------------------------------

    def _on_selection_changed(self):
        if self._building_controls: return
        self._result = None
        self._clear_plots()
        self._start_worker()

    def _on_scale_changed(self):
        if self._result is not None:
            self._apply_result_to_plots(self._result)
        self._scroll_viewport()

    def _on_persist_changed(self):
        if self._result is not None:
            self._result.corr_smooth = _ewma_array(
                self._result.corr_raw, self._current_persistence_alpha())
            self._corr_smooth_curve.setData(
                self._result.centers, self._result.corr_smooth)
        self._scroll_viewport()

    def _on_speed_changed(self):
        self._playback_speed = float(self._combo_speed.currentData() or 1.0)

    def _on_window_combo_changed(self):
        self._view_span = float(self._combo_window.currentData() or 10.0)
        self._scroll_viewport()

    def _on_main_window_changed(self, lo: float, hi: float):
        if not self._chk_follow.isChecked(): return
        self._pause()
        if hi > lo:
            self._view_span = max(1.0, hi - lo)
            self._set_cursor(0.5 * (lo + hi))
        self._start_worker()

    def _on_slider_changed(self, value: int):
        if self._slider_lock: return
        ns = self._record_duration()
        if ns <= 0: return
        self._pause()
        self._set_cursor((float(value) / 10000.0) * ns, sync=False)
        if (self._result is not None and
                self._result.t0 <= self._cursor_time <= self._result.t1):
            self._scroll_viewport()       # cursor inside buffer: just pan, no re-fetch
        else:
            self._start_worker()          # cursor jumped outside buffer: re-fetch

    def _adopt_main_window(self):
        lo, hi = self._session.current_window()
        if hi > lo:
            self._view_span = max(1.0, hi - lo)
            self._set_cursor(0.5 * (lo + hi))
        self._start_worker()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _selected_preset(self) -> BandPreset:
        name = self._combo_mode.currentText().strip()
        for p in self._presets:
            if p.name == name: return p
        return self._presets[0]

    def _current_persistence_alpha(self) -> float:
        return float(min(max(self._combo_persist.currentData() or 0.18, 0.01), 0.95))


__all__ = [
    "BandDefinition",
    "BandPreset",
    "ChannelInfo",
    "TwoChannelAnalysisSession",
    "TwoChannelDynamicsTab",
]
