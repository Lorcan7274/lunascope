"""Explorer tab for FIR and CWT design previews."""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .explorer_base import BG, FG, GRID, SEP, _ExplorerTab
from ..file_dialogs import open_file_name


@dataclass
class _ModeResult:
    mode: str
    command: str
    stdout: str
    tables: dict[str, pd.DataFrame]


_MAX_PLOT_PTS = 4096


def _downsample(arr: np.ndarray, max_pts: int = _MAX_PLOT_PTS) -> np.ndarray:
    if len(arr) <= max_pts:
        return arr
    idx = np.round(np.linspace(0, len(arr) - 1, max_pts)).astype(int)
    return arr[idx]


def _fmt_num(value: float) -> str:
    try:
        f = float(value)
    except Exception:
        return str(value)
    if math.isfinite(f) and abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return f"{f:g}"


class FilterDesignTab(_ExplorerTab):
    """Explorer tab: FIR and CWT design previews via isolated Luna runs."""

    _sig_ok = QtCore.Signal(object)
    _sig_err = QtCore.Signal(str)

    def __init__(self, ctrl, parent=None):
        super().__init__(ctrl, parent)
        self._last_result: _ModeResult | None = None
        self._range_controls: dict[str, tuple[QCheckBox, QLineEdit, QLineEdit]] = {}
        self._build_widget()
        self._sig_ok.connect(self._on_ok, Qt.QueuedConnection)
        self._sig_err.connect(self._on_err, Qt.QueuedConnection)

    def refresh_controls(self):
        self._update_fir_visibility()
        self._update_cwt_visibility()

    @staticmethod
    def build_fir_command(cfg: dict) -> str:
        fs = float(cfg["fs"])
        if fs <= 0:
            raise ValueError("Sampling rate must be positive.")
        nyq = fs / 2.0

        fir_type = cfg["fir_type"]
        design_mode = cfg["design_mode"]

        if fir_type != "file":
            f1 = float(cfg["f1"])
            if f1 <= 0:
                raise ValueError("Cutoff frequency must be positive.")
            if f1 >= nyq:
                raise ValueError(
                    f"Cutoff frequency ({_fmt_num(f1)} Hz) must be less than Nyquist ({_fmt_num(nyq)} Hz)."
                )
            if fir_type in {"bandpass", "bandstop"}:
                f2 = float(cfg["f2"])
                if f2 <= 0:
                    raise ValueError("Upper cutoff frequency must be positive.")
                if f2 >= nyq:
                    raise ValueError(
                        f"Upper cutoff frequency ({_fmt_num(f2)} Hz) must be less than Nyquist ({_fmt_num(nyq)} Hz)."
                    )
                if f1 >= f2:
                    raise ValueError("Lower cutoff must be less than upper cutoff.")

            fix_nyquist = float(cfg.get("fix_nyquist", 0.0) or 0.0)
            if fix_nyquist < 0:
                raise ValueError("Fix-Nyquist must be non-negative.")

            if design_mode == "kaiser":
                split_bp = bool(cfg.get("split_bandpass")) and fir_type == "bandpass"
                if split_bp:
                    for lbl, key in [("HP ripple", "ripple_hp"), ("LP ripple", "ripple_lp")]:
                        v = float(cfg[key])
                        if v <= 0 or v >= 1:
                            raise ValueError(f"{lbl} must be between 0 and 1 (exclusive).")
                    for lbl, key in [("HP transition width", "tw_hp"), ("LP transition width", "tw_lp")]:
                        if float(cfg[key]) <= 0:
                            raise ValueError(f"{lbl} must be positive.")
                else:
                    ripple = float(cfg["ripple"])
                    if ripple <= 0 or ripple >= 1:
                        raise ValueError("Ripple must be between 0 and 1 (exclusive).")
                    tw = float(cfg["tw"])
                    if tw <= 0:
                        raise ValueError("Transition width must be positive.")
            elif design_mode == "fixed":
                order = int(cfg["order"])
                if order < 1:
                    raise ValueError("Filter order must be at least 1.")
            elif design_mode != "file":
                raise ValueError(f"Unknown FIR design mode: {design_mode}")

        parts = ["FILTER-DESIGN", f"fs={_fmt_num(fs)}"]

        if fir_type == "file":
            file_path = str(cfg.get("file_path", "")).strip()
            if not file_path:
                raise ValueError("Select a coefficient file to summarize.")
            parts.append(f"file={file_path}")
            return " ".join(parts)

        if fir_type == "lowpass":
            parts.append(f"lowpass={_fmt_num(cfg['f1'])}")
        elif fir_type == "highpass":
            parts.append(f"highpass={_fmt_num(cfg['f1'])}")
        elif fir_type == "bandpass":
            parts.append(f"bandpass={_fmt_num(cfg['f1'])},{_fmt_num(cfg['f2'])}")
        elif fir_type == "bandstop":
            parts.append(f"bandstop={_fmt_num(cfg['f1'])},{_fmt_num(cfg['f2'])}")
        else:
            raise ValueError(f"Unknown FIR type: {fir_type}")

        fix_nyquist = float(cfg.get("fix_nyquist", 0.0) or 0.0)
        if fix_nyquist > 0:
            parts.append(f"fix-nyquist={_fmt_num(fix_nyquist)}")

        if design_mode == "kaiser":
            if fir_type == "bandpass" and cfg.get("split_bandpass"):
                parts.append(
                    "ripple=" + ",".join([_fmt_num(cfg["ripple_hp"]), _fmt_num(cfg["ripple_lp"])])
                )
                parts.append(
                    "tw=" + ",".join([_fmt_num(cfg["tw_hp"]), _fmt_num(cfg["tw_lp"])])
                )
            else:
                parts.append(f"ripple={_fmt_num(cfg['ripple'])}")
                parts.append(f"tw={_fmt_num(cfg['tw'])}")
        elif design_mode == "fixed":
            parts.append(f"order={int(cfg['order'])}")
            window = cfg.get("window", "")
            if window and window != "default":
                parts.append(window)

        return " ".join(parts)

    @staticmethod
    def build_cwt_command(cfg: dict) -> str:
        fs = float(cfg["fs"])
        if fs <= 0:
            raise ValueError("Sampling rate must be positive.")
        fc = float(cfg["fc"])
        if fc <= 0:
            raise ValueError("Center frequency must be positive.")
        if fc >= fs / 2.0:
            raise ValueError(
                f"Center frequency ({_fmt_num(fc)} Hz) must be less than Nyquist ({_fmt_num(fs / 2.0)} Hz)."
            )
        mode = cfg["mode"]
        if mode == "cycles":
            cycles = int(cfg["cycles"])
            if cycles < 1:
                raise ValueError("Cycles must be at least 1.")
        elif mode == "fwhm":
            fwhm = float(cfg["fwhm"])
            if fwhm <= 0:
                raise ValueError("FWHM must be positive.")
            length = float(cfg["length"])
            if length <= 0:
                raise ValueError("Length must be positive.")
        else:
            raise ValueError(f"Unknown CWT mode: {mode}")

        parts = [
            "CWT-DESIGN",
            f"fs={_fmt_num(fs)}",
            f"fc={_fmt_num(fc)}",
        ]
        if mode == "cycles":
            parts.append(f"cycles={int(cfg['cycles'])}")
        elif mode == "fwhm":
            parts.append(f"fwhm={_fmt_num(cfg['fwhm'])}")
            parts.append(f"len={_fmt_num(cfg['length'])}")
        return " ".join(parts)

    @staticmethod
    def build_ngaus_command(cfg: dict) -> str:
        fs = float(cfg["fs"])
        if fs <= 0:
            raise ValueError("Sampling rate must be positive.")
        fc = float(cfg["fc"])
        if fc <= 0:
            raise ValueError("Center frequency must be positive.")
        if fc >= fs / 2.0:
            raise ValueError(
                f"Center frequency ({_fmt_num(fc)} Hz) must be less than Nyquist ({_fmt_num(fs / 2.0)} Hz)."
            )
        fwhm = float(cfg["fwhm"])
        if fwhm <= 0:
            raise ValueError("FWHM must be positive.")
        n_str = str(cfg.get("n", "")).strip()
        if n_str:
            try:
                n_val = int(float(n_str))
            except ValueError:
                raise ValueError(f"Invalid value for N: {n_str!r}")
            if n_val < 1:
                raise ValueError("N must be at least 1.")

        parts = [
            "FILTER-DESIGN",
            f"fs={_fmt_num(fs)}",
            f"ngaus={_fmt_num(fc)},{_fmt_num(fwhm)}",
        ]
        if n_str:
            parts.append(f"n={n_val}")
        return " ".join(parts)

    @staticmethod
    def run_design_worker(mode: str, command: str) -> _ModeResult:
        import lunapi as lp

        proj = lp.proj(verbose=False)
        inst = proj.empty_inst("__design__", 1, 1)
        stdout = inst.eval_lunascope(command) or ""
        tables: dict[str, pd.DataFrame] = {}
        tbls = inst.strata()
        if tbls is not None:
            for row in tbls.itertuples(index=False):
                key = f"{row.Command}_{row.Strata}"
                tables[key] = inst.table(row.Command, row.Strata)
        return _ModeResult(mode=mode, command=command, stdout=stdout, tables=tables)

    @staticmethod
    def summarize_fir(result: _ModeResult) -> str:
        summary = result.tables.get("FILTER_DESIGN_FIR")
        freq = result.tables.get("FILTER_DESIGN_F_FIR")
        time = result.tables.get("FILTER_DESIGN_FIR_SEC")
        taps = result.tables.get("FILTER_DESIGN_FIR_TAP")
        if summary is None or freq is None or time is None:
            return "No FIR output tables were returned."

        label = str(summary.iloc[0].get("FIR", ""))
        fs = float(summary.iloc[0].get("FS", np.nan))
        ntaps = int(float(summary.iloc[0].get("NTAPS", np.nan)))
        mag = pd.to_numeric(freq.get("MAG"), errors="coerce")
        mag_db = pd.to_numeric(freq.get("MAG_DB"), errors="coerce")
        phase = pd.to_numeric(freq.get("PHASE"), errors="coerce")
        f = pd.to_numeric(freq.get("F"), errors="coerce")
        ir = pd.to_numeric(time.get("IR"), errors="coerce")
        sr = pd.to_numeric(time.get("SR"), errors="coerce")
        tau = None
        if len(f) > 1:
            try:
                phi_u = np.unwrap(phase.to_numpy(dtype=float))
                omega = 2.0 * np.pi * f.to_numpy(dtype=float)
                tau = -np.gradient(phi_u, omega)
            except Exception:
                tau = None

        lines = [
            f"Filter: {label}",
            "-----",
            f"Sampling rate: {_fmt_num(fs)} Hz",
            f"Taps: {ntaps}",
            "",
            "Frequency response",
            f"Peak magnitude: {_fmt_num(mag.max())}",
            f"Min magnitude: {_fmt_num(mag.min())}",
            f"Peak gain: {_fmt_num(mag_db.max())} dB",
            f"Minimum gain: {_fmt_num(mag_db.min())} dB",
            f"DC gain: {_fmt_num(mag.iloc[(f - 0).abs().idxmin()])}",
            f"Nyquist gain: {_fmt_num(mag.iloc[(f - (fs / 2.0)).abs().idxmin()])}",
            "",
            "Time-domain response",
            f"Impulse energy: {_fmt_num(np.nansum(ir ** 2))}",
            f"Step final value: {_fmt_num(sr.dropna().iloc[-1])}",
        ]
        if tau is not None and len(tau):
            mask = mag.to_numpy(dtype=float) >= 0.05
            vals = tau[mask] if np.any(mask) else tau
            lines.insert(11, f"Median group delay: {_fmt_num(np.nanmedian(vals))} s")
            lines.insert(12, f"Median group delay (samples): {_fmt_num(np.nanmedian(vals) * fs)}")
        if taps is not None:
            w = pd.to_numeric(taps.get("W"), errors="coerce").dropna().to_numpy()
            if len(w):
                sym = np.allclose(w, w[::-1], atol=1e-8, rtol=1e-6)
                lines.extend(["", "Coefficients", f"Tap symmetry: {'yes' if sym else 'no'}"])
        return "\n".join(lines)

    @staticmethod
    def summarize_cwt(result: _ModeResult) -> str:
        summary = result.tables.get("CWT_DESIGN_PARAM")
        freq = result.tables.get("CWT_DESIGN_F_PARAM")
        coeff = result.tables.get("CWT_DESIGN_PARAM_SEC")
        if summary is None or freq is None or coeff is None:
            return "No CWT output tables were returned."

        row = summary.iloc[0]
        lines = [
            f"Wavelet: {row.get('PARAM', '')}",
            "-----",
            "Frequency support",
            f"FWHM_F: {_fmt_num(row.get('FWHM_F', np.nan))} Hz",
            f"Lower half-max: {_fmt_num(row.get('FWHM_LWR', np.nan))} Hz",
            f"Upper half-max: {_fmt_num(row.get('FWHM_UPR', np.nan))} Hz",
        ]
        if "FWHM" in summary.columns and pd.notna(row.get("FWHM")):
            lines.append(f"Time-domain FWHM: {_fmt_num(row.get('FWHM'))} s")

        re = pd.to_numeric(coeff.get("REAL"), errors="coerce")
        im = pd.to_numeric(coeff.get("IMAG"), errors="coerce")
        t = pd.to_numeric(coeff.get("SEC"), errors="coerce")
        mag = np.sqrt(re ** 2 + im ** 2)
        if len(mag):
            peak_idx = int(np.nanargmax(mag.to_numpy()))
            half = float(np.nanmax(mag.to_numpy())) * 0.5
            above = np.flatnonzero(mag.to_numpy() >= half)
            if len(above):
                support = float(t.iloc[above[-1]] - t.iloc[above[0]])
                lines.extend(["", "Time support", f"50% magnitude support: {_fmt_num(support)} s"])
            else:
                lines.extend(["", "Time support"])
            lines.append(f"Peak coefficient time: {_fmt_num(t.iloc[peak_idx])} s")

        resp_mag = pd.to_numeric(freq.get("MAG"), errors="coerce")
        resp_f = pd.to_numeric(freq.get("F"), errors="coerce")
        if len(resp_mag):
            peak_idx = int(np.nanargmax(resp_mag.to_numpy()))
            lines.append(f"Peak response frequency: {_fmt_num(resp_f.iloc[peak_idx])} Hz")
        return "\n".join(lines)

    @staticmethod
    def summarize_ngaus(result: _ModeResult) -> str:
        summary = result.tables.get("FILTER_DESIGN_FIR")
        freq    = result.tables.get("FILTER_DESIGN_F_FIR")
        time    = result.tables.get("FILTER_DESIGN_FIR_SEC")
        if summary is None or freq is None:
            return "No NGaus output tables were returned."

        row  = summary.iloc[0]
        fc   = float(row.get("FC",   np.nan))
        fs   = float(row.get("FS",   np.nan))
        fwhm = float(row.get("FWHM", np.nan))
        n    = int(float(row.get("N", 0)))

        f   = pd.to_numeric(freq.get("F"),   errors="coerce")
        mag = pd.to_numeric(freq.get("MAG"), errors="coerce")

        lines = [
            "Filter: Narrow Gaussian (NGaus)",
            "-----",
            f"Center frequency: {_fmt_num(fc)} Hz",
            f"FWHM (frequency): {_fmt_num(fwhm)} Hz",
            f"Sampling rate: {_fmt_num(fs)} Hz",
            f"N (kernel samples): {n}",
            f"Kernel duration: {_fmt_num(n / fs)} s",
            "",
            "Frequency response",
        ]

        mag_arr = mag.to_numpy(dtype=float)
        f_arr   = f.to_numpy(dtype=float)
        if len(mag_arr):
            peak_idx = int(np.nanargmax(mag_arr))
            lines.append(f"Peak response: {_fmt_num(f_arr[peak_idx])} Hz")
            peak_mag = float(np.nanmax(mag_arr))
            half = peak_mag * 0.5
            above = np.flatnonzero(mag_arr >= half)
            if len(above) >= 2:
                bw = float(f_arr[above[-1]] - f_arr[above[0]])
                lines.append(f"Half-max bandwidth: {_fmt_num(bw)} Hz")
            lines.append(f"Peak magnitude: {_fmt_num(peak_mag)}")

        if time is not None:
            sec = pd.to_numeric(time.get("SEC"), errors="coerce")
            ir  = pd.to_numeric(time.get("IR"),  errors="coerce")
            env = ir.abs().to_numpy(dtype=float)
            sec_arr = sec.to_numpy(dtype=float)
            lines.append("")
            lines.append("Time-domain response")
            if len(env):
                peak_env = float(np.nanmax(env))
                peak_t   = float(sec_arr[int(np.nanargmax(env))])
                lines.append(f"Peak envelope: {_fmt_num(peak_env)} (at {_fmt_num(peak_t)} s)")
                half_e = peak_env * 0.5
                above_t = np.flatnonzero(env >= half_e)
                if len(above_t) >= 2:
                    fwhm_t = float(sec_arr[above_t[-1]] - sec_arr[above_t[0]])
                    lines.append(f"FWHM (envelope): {_fmt_num(fwhm_t)} s")
                # Warn if kernel looks truncated: edges still >2% of peak.
                # For a Gaussian envelope exp(-t²/2σ²), N_needed scales as
                # N * sqrt(ln(0.02) / ln(edge_frac)).
                edge_frac = max(float(env[0]), float(env[-1])) / peak_env if peak_env > 0 else 0.0
                if 0.0 < edge_frac and edge_frac > 0.02:
                    try:
                        ratio = math.sqrt(math.log(0.02) / math.log(edge_frac))
                        n_needed = int(math.ceil(n * ratio))
                        if n_needed % 2 == 0:
                            n_needed += 1
                        lines.append(
                            f"WARNING: kernel may be truncated "
                            f"(edge = {edge_frac * 100:.1f}% of peak). "
                            f"Try N >= {n_needed}."
                        )
                    except (ValueError, ZeroDivisionError):
                        pass

        lines.extend(["", "Phase: zero-phase (frequency-domain Gaussian)"])
        return "\n".join(lines)

    def _build_widget(self):
        root = QWidget()
        root.setStyleSheet(
            f"""
            QWidget {{ color: {FG}; }}
            QGroupBox {{ border: 1px solid {GRID}; margin-top: 8px; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
            QPlainTextEdit, QLineEdit, QComboBox {{
                background: #161b22; color: {FG}; border: 1px solid {GRID}; padding: 4px;
            }}
            QPushButton {{ background: #21262d; border: 1px solid {SEP}; padding: 6px 10px; }}
            QPushButton:hover {{ background: #30363d; }}
            """
        )

        outer = QVBoxLayout(root)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        top_split = QtWidgets.QSplitter(Qt.Horizontal)
        top_split.setChildrenCollapsible(False)

        self._subtabs = QTabWidget()
        self._subtabs.addTab(self._build_fir_tab(),   "FIR")
        self._subtabs.addTab(self._build_ngaus_tab(), "NGaus")
        self._subtabs.addTab(self._build_cwt_tab(),   "CWT")
        self._subtabs.currentChanged.connect(self._on_mode_changed)

        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFrameShape(QFrame.NoFrame)
        panel_scroll.viewport().installEventFilter(self)

        panel = QWidget()
        panel_lay = QVBoxLayout(panel)
        panel_lay.setContentsMargins(0, 0, 0, 0)
        panel_lay.setSpacing(10)

        top_split.addWidget(self._subtabs)

        self._txt_stats = QPlainTextEdit()
        self._txt_stats.setReadOnly(True)
        self._txt_stats.setPlaceholderText("Summary statistics")
        self._txt_stats.setStyleSheet(
            f"QPlainTextEdit {{ font-size: 15px; line-height: 1.35; padding: 12px; }}"
        )
        top_split.addWidget(self._txt_stats)
        top_split.setSizes([760, 380])
        top_split.setStretchFactor(0, 2)
        top_split.setStretchFactor(1, 1)
        panel_lay.addWidget(top_split, 0)

        run_row = QHBoxLayout()
        self._btn_run = QPushButton("Run Design")
        self._btn_run.clicked.connect(self._run_current)
        self._lbl_cmd = QLineEdit()
        self._lbl_cmd.setReadOnly(True)
        self._lbl_cmd.setPlaceholderText("Built Luna command")
        run_row.addWidget(self._btn_run, 0)
        run_row.addWidget(self._lbl_cmd, 1)
        panel_lay.addLayout(run_row)

        canvas_host = QFrame()
        canvas_host.setFrameShape(QFrame.NoFrame)
        canvas_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        canvas_host.setMinimumHeight(900)
        panel_lay.addWidget(canvas_host, 1)

        panel_scroll.setWidget(panel)
        self._canvas_host = canvas_host
        self._canvas_scroll = panel_scroll

        outer.addWidget(panel_scroll, 1)
        self._root = root
        self._clear_view()
        self.refresh_controls()

    def _build_fir_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        w = QWidget()
        g = QGridLayout(w)
        g.setContentsMargins(10, 10, 10, 10)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(6)

        self._fir_type = QComboBox()
        self._fir_type.addItem("Low-pass", "lowpass")
        self._fir_type.addItem("High-pass", "highpass")
        self._fir_type.addItem("Band-pass", "bandpass")
        self._fir_type.addItem("Band-stop", "bandstop")
        self._fir_type.addItem("Coefficient file", "file")
        self._fir_type.currentIndexChanged.connect(self._update_fir_visibility)
        self._fir_type.currentIndexChanged.connect(self._update_fir_defaults)

        self._fir_design_mode = QComboBox()
        self._fir_design_mode.addItem("Kaiser", "kaiser")
        self._fir_design_mode.addItem("Fixed order", "fixed")
        self._fir_design_mode.currentIndexChanged.connect(self._update_fir_visibility)

        self._fir_fs          = QLineEdit("200")
        self._fir_f1          = QLineEdit("35")
        self._fir_f2          = QLineEdit("35")
        self._fir_ripple      = QLineEdit("0.02")
        self._fir_tw          = QLineEdit("1")

        self._fir_split_bp = QCheckBox("Split HP/LP ripple + TW")
        self._fir_split_bp.toggled.connect(self._update_fir_visibility)

        self._fir_ripple_hp   = QLineEdit("0.02")
        self._fir_ripple_lp   = QLineEdit("0.02")
        self._fir_tw_hp       = QLineEdit("1")
        self._fir_tw_lp       = QLineEdit("1")
        self._fir_order       = QLineEdit("100")

        self._fir_window = QComboBox()
        self._fir_window.addItem("Default (Hamming)", "default")
        self._fir_window.addItem("Rectangular", "rectangular")
        self._fir_window.addItem("Bartlett", "bartlett")
        self._fir_window.addItem("Hann", "hann")
        self._fir_window.addItem("Blackman", "blackman")

        self._fir_fix_nyquist = QLineEdit("0.5")
        self._fir_fix_nyquist.setToolTip(
            "Clamp upper transition-band edge this many Hz below Nyquist.\n"
            "Relevant for high-pass or band-stop filters whose upper cutoff\n"
            "is close to Nyquist (fs/2). Set to 0 to disable."
        )

        self._fir_file = QLineEdit()
        self._fir_file.setPlaceholderText("Path to coefficient file")
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_fir_file)
        file_row = QWidget()
        file_lay = QHBoxLayout(file_row)
        file_lay.setContentsMargins(0, 0, 0, 0)
        file_lay.setSpacing(6)
        file_lay.addWidget(self._fir_file, 1)
        file_lay.addWidget(btn_browse, 0)

        self._fir_rows: dict[str, tuple[QLabel, QWidget]] = {}

        def add_row(row: int, left_label: str, left_widget: QWidget, left_key: str,
                    right_label: str | None = None, right_widget: QWidget | None = None,
                    right_key: str | None = None):
            llbl = QLabel(left_label)
            g.addWidget(llbl, row, 0)
            g.addWidget(left_widget, row, 1)
            self._fir_rows[left_key] = (llbl, left_widget)
            if right_label is not None and right_widget is not None and right_key is not None:
                rlbl = QLabel(right_label)
                g.addWidget(rlbl, row, 2)
                g.addWidget(right_widget, row, 3)
                self._fir_rows[right_key] = (rlbl, right_widget)

        add_row(0, "Type:", self._fir_type, "type", "Design mode:", self._fir_design_mode, "design_mode")
        add_row(1, "Sampling rate:", self._fir_fs, "fs", "Fix Nyquist:", self._fir_fix_nyquist, "fix")
        add_row(2, "Cutoff / lower:", self._fir_f1, "f1", "Upper cutoff:", self._fir_f2, "f2")
        add_row(3, "Ripple:", self._fir_ripple, "ripple", "Transition width:", self._fir_tw, "tw")
        add_row(4, "Band-pass split:", self._fir_split_bp, "split", "Order:", self._fir_order, "order")
        add_row(5, "HP ripple / LP ripple:", self._pair_widget(self._fir_ripple_hp, self._fir_ripple_lp), "bp_ripple",
                "HP TW / LP TW:", self._pair_widget(self._fir_tw_hp, self._fir_tw_lp), "bp_tw")
        add_row(6, "Window:", self._fir_window, "window")
        add_row(7, "Coefficient file:", file_row, "file")
        add_row(8, "Freq X-axis:", self._make_range_widget("fir_freq"), "range_freq")
        add_row(9, "Time X-axis:", self._make_range_widget("fir_time"), "range_time")
        add_row(10, "Tap X-axis:", self._make_range_widget("fir_tap"), "range_tap")
        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        outer.addWidget(w, 0, Qt.AlignTop)
        outer.addStretch(1)
        return page

    def _build_cwt_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        w = QWidget()
        g = QGridLayout(w)
        g.setContentsMargins(10, 10, 10, 10)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(6)

        self._cwt_mode = QComboBox()
        self._cwt_mode.addItem("Cycles", "cycles")
        self._cwt_mode.addItem("FWHM (time)", "fwhm")
        self._cwt_mode.currentIndexChanged.connect(self._update_cwt_visibility)

        self._cwt_fs     = QLineEdit("200")
        self._cwt_fc     = QLineEdit("15")
        self._cwt_cycles = QLineEdit("7")
        self._cwt_fwhm   = QLineEdit("2")
        self._cwt_len    = QLineEdit("20")

        self._cwt_rows: dict[str, tuple[QLabel, QWidget]] = {}

        def add_row(row: int, left_label: str, left_widget: QWidget, left_key: str,
                    right_label: str | None = None, right_widget: QWidget | None = None,
                    right_key: str | None = None):
            llbl = QLabel(left_label)
            g.addWidget(llbl, row, 0)
            g.addWidget(left_widget, row, 1)
            self._cwt_rows[left_key] = (llbl, left_widget)
            if right_label is not None and right_widget is not None and right_key is not None:
                rlbl = QLabel(right_label)
                g.addWidget(rlbl, row, 2)
                g.addWidget(right_widget, row, 3)
                self._cwt_rows[right_key] = (rlbl, right_widget)

        add_row(0, "Mode:", self._cwt_mode, "mode", "Sampling rate:", self._cwt_fs, "fs")
        add_row(1, "Center frequency:", self._cwt_fc, "fc", "Cycles:", self._cwt_cycles, "cycles")
        add_row(2, "FWHM:", self._cwt_fwhm, "fwhm", "Length (sec):", self._cwt_len, "length")
        add_row(3, "Freq X-axis:", self._make_range_widget("cwt_freq"), "range_freq")
        add_row(4, "Time X-axis:", self._make_range_widget("cwt_time"), "range_time")
        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        outer.addWidget(w, 0, Qt.AlignTop)
        outer.addStretch(1)
        return page

    def _build_ngaus_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        w = QWidget()
        g = QGridLayout(w)
        g.setContentsMargins(10, 10, 10, 10)
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(6)

        self._ngaus_fs   = QLineEdit("256")
        self._ngaus_fc   = QLineEdit("10")
        self._ngaus_fwhm = QLineEdit("2")
        self._ngaus_n    = QLineEdit()
        self._ngaus_n.setPlaceholderText("auto")

        def add_row(row, left_label, left_widget,
                    right_label=None, right_widget=None):
            llbl = QLabel(left_label)
            g.addWidget(llbl, row, 0)
            g.addWidget(left_widget, row, 1)
            if right_label is not None and right_widget is not None:
                rlbl = QLabel(right_label)
                g.addWidget(rlbl, row, 2)
                g.addWidget(right_widget, row, 3)

        add_row(0, "Sampling rate:", self._ngaus_fs,
                   "Center frequency:", self._ngaus_fc)
        add_row(1, "FWHM (Hz):", self._ngaus_fwhm,
                   "N (samples):", self._ngaus_n)
        add_row(2, "Freq X-axis:", self._make_range_widget("ngaus_freq"))
        add_row(3, "Time X-axis:", self._make_range_widget("ngaus_time"))
        g.setColumnStretch(1, 1)
        g.setColumnStretch(3, 1)
        outer.addWidget(w, 0, Qt.AlignTop)
        outer.addStretch(1)
        return page

    def _pair_widget(self, left: QWidget, right: QWidget) -> QWidget:
        host = QWidget()
        lay = QHBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(left, 1)
        lay.addWidget(right, 1)
        return host

    def _make_range_widget(self, key: str) -> QWidget:
        auto = QCheckBox("Auto")
        auto.setChecked(True)
        xmin = QLineEdit()
        xmax = QLineEdit()
        xmin.setPlaceholderText("min")
        xmax.setPlaceholderText("max")
        auto.toggled.connect(lambda checked, a=auto, lo=xmin, hi=xmax: self._on_range_auto_toggled(a, lo, hi, checked))

        def _on_user_edit(a=auto, lo=xmin, hi=xmax):
            if a.isChecked():
                a.setChecked(False)  # triggers _on_range_auto_toggled → redraw
            else:
                try:
                    valid = float(hi.text().strip()) > float(lo.text().strip())
                except ValueError:
                    valid = False
                if valid:
                    for w in (lo, hi):
                        w.setStyleSheet("")
                        w.setToolTip("")
                self._redraw_last_result()

        xmin.editingFinished.connect(_on_user_edit)
        xmax.editingFinished.connect(_on_user_edit)

        host = QWidget()
        lay = QHBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(auto, 0)
        lay.addWidget(QLabel("min"), 0)
        lay.addWidget(xmin, 1)
        lay.addWidget(QLabel("max"), 0)
        lay.addWidget(xmax, 1)
        self._range_controls[key] = (auto, xmin, xmax)
        return host

    def _on_range_auto_toggled(self, auto: QCheckBox, xmin: QLineEdit, xmax: QLineEdit, checked: bool):
        if checked:
            for w in (xmin, xmax):
                w.setStyleSheet("")
                w.setToolTip("")
        self._redraw_last_result()

    def _browse_fir_file(self):
        fn, _ = open_file_name(self._root, "Select coefficient file", "", "All files (*)")
        if fn:
            self._fir_file.setText(fn)

    def _update_fir_visibility(self):
        fir_type = self._fir_type.currentData()
        design_mode = self._fir_design_mode.currentData()
        is_file = fir_type == "file"
        is_band = fir_type in {"bandpass", "bandstop"}
        is_bandpass = fir_type == "bandpass"
        use_kaiser = design_mode == "kaiser" and not is_file
        use_fixed = design_mode == "fixed" and not is_file
        split_bp = is_bandpass and use_kaiser and self._fir_split_bp.isChecked()

        self._set_row_visible(self._fir_rows["design_mode"], not is_file)
        self._set_row_visible(self._fir_rows["f1"], not is_file)
        self._set_row_visible(self._fir_rows["f2"], is_band and not is_file)
        self._set_row_visible(self._fir_rows["ripple"], use_kaiser and not split_bp)
        self._set_row_visible(self._fir_rows["tw"], use_kaiser and not split_bp)
        self._set_row_visible(self._fir_rows["split"], is_bandpass and use_kaiser)
        self._set_row_visible(self._fir_rows["bp_ripple"], split_bp)
        self._set_row_visible(self._fir_rows["bp_tw"], split_bp)
        self._set_row_visible(self._fir_rows["order"], use_fixed)
        self._set_row_visible(self._fir_rows["window"], use_fixed)
        self._set_row_visible(self._fir_rows["file"], is_file)
        self._set_row_visible(self._fir_rows["fix"], not is_file)

    def _update_cwt_visibility(self):
        mode = self._cwt_mode.currentData()
        self._set_row_visible(self._cwt_rows["cycles"], mode == "cycles")
        self._set_row_visible(self._cwt_rows["fwhm"], mode == "fwhm")
        self._set_row_visible(self._cwt_rows["length"], mode == "fwhm")

    def _update_fir_defaults(self):
        defaults = {
            "lowpass":  (35.0,  None),
            "highpass": (0.3,   None),
            "bandpass": (0.3,   35.0),
            "bandstop": (48.0,  52.0),
        }
        fir_type = self._fir_type.currentData()
        f1, f2 = defaults.get(fir_type, (0.3, 35.0))
        self._fir_f1.setText(_fmt_num(f1))
        if f2 is not None:
            self._fir_f2.setText(_fmt_num(f2))

    def _set_row_visible(self, row: tuple[QLabel, QWidget], visible: bool):
        label, widget = row
        label.setVisible(visible)
        widget.setVisible(visible)

    def _current_mode(self) -> str:
        idx = self._subtabs.currentIndex()
        if idx == 0:
            return "fir"
        if idx == 1:
            return "ngaus"
        return "cwt"

    def _on_mode_changed(self, *_):
        self._clear_view()
        self.refresh_controls()

    def _clear_view(self):
        self._last_result = None
        self._lbl_cmd.clear()
        self._txt_stats.clear()
        self._render_empty("Run a FIR, CWT, or NGaus design to preview the response.")

    def _redraw_last_result(self):
        if self._last_result is None:
            return
        if self._last_result.mode == "fir":
            self._plot_fir(self._last_result)
        elif self._last_result.mode == "cwt":
            self._plot_cwt(self._last_result)
        else:
            self._plot_ngaus(self._last_result)

    def _configure_range_control(self, key: str, lo: float, hi: float):
        if key not in self._range_controls:
            return
        auto, xmin, xmax = self._range_controls[key]
        lo = float(lo)
        hi = float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi):
            return
        if hi < lo:
            lo, hi = hi, lo
        if abs(hi - lo) < 1e-12:
            pad = max(1.0, abs(lo) * 0.1)
            lo -= pad
            hi += pad
        if auto.isChecked():
            xmin.setText(_fmt_num(lo))
            xmax.setText(_fmt_num(hi))

    def _apply_xrange(self, ax, key: str, lo: float, hi: float):
        self._configure_range_control(key, lo, hi)
        auto, xmin, xmax = self._range_controls[key]
        if auto.isChecked():
            ax.set_xlim(lo, hi)
            return
        try:
            user_lo = float(xmin.text().strip())
            user_hi = float(xmax.text().strip())
        except ValueError:
            ax.set_xlim(lo, hi)
            return
        if user_hi > user_lo:
            for w in (xmin, xmax):
                w.setStyleSheet("")
                w.setToolTip("")
            ax.set_xlim(user_lo, user_hi)
        else:
            for w in (xmin, xmax):
                w.setStyleSheet("border: 1px solid #f85149;")
                w.setToolTip("min must be less than max — auto range applied")
            ax.set_xlim(lo, hi)

    @staticmethod
    def _fval(widget: QLineEdit, name: str) -> float:
        try:
            return float(widget.text().strip())
        except ValueError:
            raise ValueError(f"Invalid value for {name}: {widget.text()!r}")

    def _collect_current_command(self) -> tuple[str, str]:
        mode = self._current_mode()
        fv = self._fval
        if mode == "fir":
            cfg = {
                "fir_type": self._fir_type.currentData(),
                "design_mode": self._fir_design_mode.currentData(),
                "fs":         fv(self._fir_fs,          "sampling rate"),
                "f1":         fv(self._fir_f1,          "cutoff / lower"),
                "f2":         fv(self._fir_f2,          "upper cutoff"),
                "ripple":     fv(self._fir_ripple,      "ripple"),
                "tw":         fv(self._fir_tw,          "transition width"),
                "split_bandpass": self._fir_split_bp.isChecked(),
                "ripple_hp":  fv(self._fir_ripple_hp,   "HP ripple"),
                "ripple_lp":  fv(self._fir_ripple_lp,   "LP ripple"),
                "tw_hp":      fv(self._fir_tw_hp,       "HP transition width"),
                "tw_lp":      fv(self._fir_tw_lp,       "LP transition width"),
                "order":      int(fv(self._fir_order,   "order")),
                "window":     self._fir_window.currentData(),
                "fix_nyquist": fv(self._fir_fix_nyquist, "fix-nyquist"),
                "file_path":  self._fir_file.text().strip(),
            }
            return mode, self.build_fir_command(cfg)

        if mode == "cwt":
            cfg = {
                "mode":   self._cwt_mode.currentData(),
                "fs":     fv(self._cwt_fs,     "sampling rate"),
                "fc":     fv(self._cwt_fc,     "center frequency"),
                "cycles": int(fv(self._cwt_cycles, "cycles")),
                "fwhm":   fv(self._cwt_fwhm,   "FWHM"),
                "length": fv(self._cwt_len,    "length"),
            }
            return mode, self.build_cwt_command(cfg)

        # ngaus
        cfg = {
            "fs":   fv(self._ngaus_fs,   "sampling rate"),
            "fc":   fv(self._ngaus_fc,   "center frequency"),
            "fwhm": fv(self._ngaus_fwhm, "FWHM"),
            "n":    self._ngaus_n.text().strip(),
        }
        return mode, self.build_ngaus_command(cfg)

    def _start_work(self, msg: str = "Designing…") -> bool:
        if getattr(self.ctrl, "_busy", False):
            return False
        self.ctrl._busy = True
        self.ctrl.sb_progress.setVisible(True)
        self.ctrl.sb_progress.setRange(0, 0)
        self.ctrl.sb_progress.setFormat(msg)
        return True

    def _end_work(self):
        self.ctrl._busy = False
        self.ctrl.sb_progress.setRange(0, 100)
        self.ctrl.sb_progress.setValue(0)
        self.ctrl.sb_progress.setVisible(False)

    def _run_current(self):
        try:
            mode, command = self._collect_current_command()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self._root, "Filter Design", str(e))
            return

        self._lbl_cmd.setText(command)
        if not self._start_work("Designing…"):
            return

        try:
            fut = self.ctrl._exec.submit(self.run_design_worker, mode, command)
        except Exception as e:
            self._end_work()
            QtWidgets.QMessageBox.critical(self._root, "Filter Design", f"Could not start worker:\n{e}")
            return

        def _done(_f=fut):
            try:
                self._sig_ok.emit(_f.result())
            except Exception:
                self._sig_err.emit(traceback.format_exc())

        fut.add_done_callback(_done)

    @QtCore.Slot(object)
    def _on_ok(self, result: _ModeResult):
        try:
            self._last_result = result
            self._lbl_cmd.setText(result.command)
            if result.mode == "fir":
                self._txt_stats.setPlainText(self.summarize_fir(result))
                self._plot_fir(result)
            elif result.mode == "cwt":
                self._txt_stats.setPlainText(self.summarize_cwt(result))
                self._plot_cwt(result)
            else:
                self._txt_stats.setPlainText(self.summarize_ngaus(result))
                self._plot_ngaus(result)
        finally:
            self._end_work()

    @QtCore.Slot(str)
    def _on_err(self, tb: str):
        try:
            QtWidgets.QMessageBox.critical(self._root, "Filter Design error", tb[:1200])
        finally:
            self._end_work()

    def _plot_fir(self, result: _ModeResult):
        freq = result.tables.get("FILTER_DESIGN_F_FIR")
        time = result.tables.get("FILTER_DESIGN_FIR_SEC")
        # taps = result.tables.get("FILTER_DESIGN_FIR_TAP")  # reserved for FIR coefficients plot
        if freq is None or time is None:
            self._render_empty("FIR plot data unavailable")
            return

        canvas = self._ensure_canvas()
        if canvas is None:
            return
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(BG)
        axes = fig.subplots(2, 2)
        fig.set_size_inches(12, 8, forward=True)

        f      = pd.to_numeric(freq["F"],      errors="coerce").to_numpy(dtype=float)
        mag    = pd.to_numeric(freq["MAG"],    errors="coerce").to_numpy(dtype=float)
        mag_db = pd.to_numeric(freq["MAG_DB"], errors="coerce").to_numpy(dtype=float)
        sec_raw = pd.to_numeric(time["SEC"], errors="coerce").to_numpy(dtype=float)
        ir_raw  = pd.to_numeric(time["IR"],  errors="coerce").to_numpy(dtype=float)
        sr_raw  = pd.to_numeric(time["SR"],  errors="coerce").to_numpy(dtype=float)

        # Downsample time-domain arrays: huge filters (e.g. 53k taps) would
        # block the main thread for seconds while matplotlib renders them.
        sec = _downsample(sec_raw)
        ir  = _downsample(ir_raw)
        sr  = _downsample(sr_raw)

        ax = axes[0, 0]
        ax.plot(f, mag, color="#58a6ff", lw=1.4)
        self._style_ax(ax, "Magnitude", "Hz", "MAG")
        self._apply_xrange(ax, "fir_freq", float(f.min()), float(f.max()))

        ax = axes[0, 1]
        ax.plot(f, mag_db, color="#f2cc60", lw=1.4)
        self._style_ax(ax, "Magnitude (dB)", "Hz", "dB")
        self._apply_xrange(ax, "fir_freq", float(f.min()), float(f.max()))

        ax = axes[1, 0]
        ax.plot(sec, ir, color="#ff7b72", lw=1.2)
        self._style_ax(ax, "Impulse Response", "sec", "value")
        self._apply_xrange(ax, "fir_time", float(sec_raw.min()), float(sec_raw.max()))

        ax = axes[1, 1]
        ax.plot(sec, sr, color="#a371f7", lw=1.2)
        self._style_ax(ax, "Step Response", "sec", "value")
        self._apply_xrange(ax, "fir_time", float(sec_raw.min()), float(sec_raw.max()))

        # FIR coefficients plot removed for now
        # ax = axes[2, 1]
        # ...

        fig.tight_layout(pad=1.2)
        self._finalize_canvas_draw()

    def _plot_ngaus(self, result: _ModeResult):
        freq = result.tables.get("FILTER_DESIGN_F_FIR")
        time = result.tables.get("FILTER_DESIGN_FIR_SEC")
        if freq is None or time is None:
            self._render_empty("NGaus plot data unavailable")
            return

        canvas = self._ensure_canvas()
        if canvas is None:
            return
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(BG)
        axes = fig.subplots(2, 2)
        fig.set_size_inches(12, 8, forward=True)

        f      = pd.to_numeric(freq["F"],      errors="coerce")
        mag    = pd.to_numeric(freq["MAG"],    errors="coerce")
        mag_db = pd.to_numeric(freq["MAG_DB"], errors="coerce")
        sec    = pd.to_numeric(time["SEC"],    errors="coerce")
        ir     = pd.to_numeric(time["IR"],     errors="coerce")
        env    = ir.abs()

        ax = axes[0, 0]
        ax.plot(f, mag, color="#58a6ff", lw=1.4)
        self._style_ax(ax, "Magnitude", "Hz", "MAG")
        self._apply_xrange(ax, "ngaus_freq", float(f.min()), float(f.max()))

        ax = axes[0, 1]
        ax.plot(f, mag_db, color="#f2cc60", lw=1.4)
        self._style_ax(ax, "Magnitude (dB)", "Hz", "dB")
        self._apply_xrange(ax, "ngaus_freq", float(f.min()), float(f.max()))

        ax = axes[1, 0]
        ax.plot(sec, ir, color="#7ee787", lw=1.2)
        self._style_ax(ax, "Impulse Response", "sec", "value")
        self._apply_xrange(ax, "ngaus_time", float(sec.min()), float(sec.max()))

        ax = axes[1, 1]
        ax.plot(sec, env, color="#a371f7", lw=1.2)
        self._style_ax(ax, "Impulse Envelope |IR|", "sec", "|value|")
        self._apply_xrange(ax, "ngaus_time", float(sec.min()), float(sec.max()))

        fig.tight_layout(pad=1.2)
        self._finalize_canvas_draw()

    def _plot_cwt(self, result: _ModeResult):
        freq = result.tables.get("CWT_DESIGN_F_PARAM")
        coeff = result.tables.get("CWT_DESIGN_PARAM_SEC")
        summary = result.tables.get("CWT_DESIGN_PARAM")
        if freq is None or coeff is None or summary is None:
            self._render_empty("CWT plot data unavailable")
            return

        canvas = self._ensure_canvas()
        if canvas is None:
            return
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(BG)
        axes = fig.subplots(3, 1)
        fig.set_size_inches(11, 10, forward=True)

        f = pd.to_numeric(freq["F"], errors="coerce")
        mag = pd.to_numeric(freq["MAG"], errors="coerce")
        sec = pd.to_numeric(coeff["SEC"], errors="coerce")
        re = pd.to_numeric(coeff["REAL"], errors="coerce")
        im = pd.to_numeric(coeff["IMAG"], errors="coerce")
        env = np.sqrt(re ** 2 + im ** 2)

        ax = axes[0]
        ax.plot(f, mag, color="#58a6ff", lw=1.4)
        row = summary.iloc[0]
        lwr = float(row.get("FWHM_LWR", np.nan))
        upr = float(row.get("FWHM_UPR", np.nan))
        if np.isfinite(lwr):
            ax.axvline(lwr, color="#8b949e", ls="--", lw=0.8)
        if np.isfinite(upr):
            ax.axvline(upr, color="#8b949e", ls="--", lw=0.8)
        self._style_ax(ax, "Frequency Response", "Hz", "MAG")
        self._apply_xrange(ax, "cwt_freq", float(f.min()), float(f.max()))

        ax = axes[1]
        ax.plot(sec, re, color="#7ee787", lw=1.1, label="REAL")
        ax.plot(sec, im, color="#ff7b72", lw=1.1, label="IMAG")
        self._style_ax(ax, "Wavelet Coefficients", "sec", "value")
        self._apply_xrange(ax, "cwt_time", float(sec.min()), float(sec.max()))
        leg = ax.legend(frameon=False, fontsize=8, loc="upper right")
        for txt in leg.get_texts():
            txt.set_color(FG)

        ax = axes[2]
        ax.plot(sec, env, color="#f2cc60", lw=1.2)
        self._style_ax(ax, "Wavelet Envelope |w|", "sec", "|w|")
        self._apply_xrange(ax, "cwt_time", float(sec.min()), float(sec.max()))

        fig.tight_layout(pad=1.2)
        self._finalize_canvas_draw()
