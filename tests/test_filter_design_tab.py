from __future__ import annotations

import pytest

pytestmark = pytest.mark.qt


def test_build_fir_command_bandpass_split_kaiser():
    from lunascope.components.explorer_filter_design import FilterDesignTab

    cmd = FilterDesignTab.build_fir_command(
        {
            "fir_type": "bandpass",
            "design_mode": "kaiser",
            "fs": 128,
            "f1": 0.3,
            "f2": 35,
            "split_bandpass": True,
            "ripple_hp": 0.02,
            "ripple_lp": 0.03,
            "tw_hp": 0.5,
            "tw_lp": 1.5,
            "fix_nyquist": 0.5,
        }
    )

    assert cmd == (
        "FILTER-DESIGN fs=128 bandpass=0.3,35 fix-nyquist=0.5 "
        "ripple=0.02,0.03 tw=0.5,1.5"
    )


def test_build_fir_command_fixed_order_default_window():
    from lunascope.components.explorer_filter_design import FilterDesignTab

    cmd = FilterDesignTab.build_fir_command(
        {
            "fir_type": "lowpass",
            "design_mode": "fixed",
            "fs": 200,
            "f1": 20,
            "f2": 0,
            "order": 40,
            "window": "default",
            "fix_nyquist": 0,
        }
    )

    assert cmd == "FILTER-DESIGN fs=200 lowpass=20 order=40"


def test_build_cwt_command_fwhm_mode():
    from lunascope.components.explorer_filter_design import FilterDesignTab

    cmd = FilterDesignTab.build_cwt_command(
        {"mode": "fwhm", "fs": 100, "fc": 12, "fwhm": 2, "length": 20}
    )
    assert cmd == "CWT-DESIGN fs=100 fc=12 fwhm=2 len=20"


def test_fir_design_worker_returns_step_response_table():
    from lunascope.components.explorer_filter_design import FilterDesignTab

    result = FilterDesignTab.run_design_worker(
        "fir", "FILTER-DESIGN fs=100 lowpass=20 order=20 hann"
    )

    assert "FILTER_DESIGN_FIR_SEC" in result.tables
    df = result.tables["FILTER_DESIGN_FIR_SEC"]
    assert "IR" in df.columns
    assert "SR" in df.columns
    assert len(df) > 0


def test_cwt_design_worker_returns_summary_tables():
    from lunascope.components.explorer_filter_design import FilterDesignTab

    result = FilterDesignTab.run_design_worker(
        "cwt", "CWT-DESIGN fs=100 fc=12 cycles=7"
    )

    assert "CWT_DESIGN_PARAM" in result.tables
    assert "CWT_DESIGN_F_PARAM" in result.tables
    assert "CWT_DESIGN_PARAM_SEC" in result.tables
    summary = result.tables["CWT_DESIGN_PARAM"]
    assert "FWHM_F" in summary.columns


def test_filter_design_tab_constructs(qapp):
    from PySide6.QtCore import QObject, Signal
    from concurrent.futures import ThreadPoolExecutor

    from lunascope.components.explorer_filter_design import FilterDesignTab

    class _Ctrl(QObject):
        sig_results_changed = Signal()

        def __init__(self):
            super().__init__()
            self._exec = ThreadPoolExecutor(max_workers=1)

    ctrl = _Ctrl()
    tab = FilterDesignTab(ctrl)
    try:
        assert tab.widget() is not None
        assert tab._subtabs.count() == 2
        assert tab._subtabs.tabText(0) == "FIR"
        assert tab._subtabs.tabText(1) == "CWT"
    finally:
        ctrl._exec.shutdown(wait=False, cancel_futures=True)
