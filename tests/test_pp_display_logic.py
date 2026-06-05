"""Logic tests for posterior-probability display behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from lunascope.components.cmaps import CMapsMixin, parse_cmap
from lunascope.components.signals import SignalsMixin


class _DummyText:
    def __init__(self, text: str = ""):
        self._text = text

    def toPlainText(self):
        return self._text

    def setPlainText(self, text):
        self._text = text

    def setText(self, text):
        self._text = text


class _DummyUI:
    def __init__(self):
        self.txt_cmap = _DummyText("")
        self.txt_pops_path = _DummyText("")
        self.txt_pops_model = _DummyText("")


class _DummyCMaps(CMapsMixin):
    def __init__(self):
        self.ui = _DummyUI()
        self.stgcols_hex = {
            "N1": "#20B2DA",
            "N2": "#0000FF",
            "N3": "#000080",
            "R": "#FF0000",
            "SP": "#800080",
            "WP": "#008000",
            "W": "#008000",
            "?": "#808080",
            "L": "#FFFF00",
        }
        self.cmap = {}
        self.cfg_pp_style = True


class _DummySignals(SignalsMixin):
    def __init__(self):
        self.cfg_pp_style = True
        self.cmap_fixed_min = {}
        self.cmap_fixed_max = {}


def test_parse_cmap_reads_pp_style_flag():
    cfg = parse_cmap("[par]\npp-style = N\n")
    assert cfg["par"]["pp-style"] == "N"


def test_init_cmaps_defaults_pp_style_on():
    obj = _DummyCMaps()
    obj._init_cmaps()
    assert obj.cfg_pp_style is True


def test_pp_channel_detection_is_exact_case_sensitive_prefix():
    obj = _DummySignals()
    assert obj._is_pp_channel("PP_POST")
    assert not obj._is_pp_channel("pp_POST")
    assert not obj._is_pp_channel("XPP_POST")


def test_resolve_channel_phys_range_uses_pp_defaults():
    obj = _DummySignals()
    lo, hi = obj._resolve_channel_phys_range("PP_POST", np.array([3.0, 4.0]))
    assert (lo, hi) == (0.0, 1.0)


def test_resolve_channel_phys_range_explicit_ylim_beats_pp_defaults():
    obj = _DummySignals()
    obj.cmap_fixed_min["PP_POST"] = -2.0
    obj.cmap_fixed_max["PP_POST"] = 2.0
    lo, hi = obj._resolve_channel_phys_range("PP_POST", np.array([0.2, 0.8]))
    assert (lo, hi) == (-2.0, 2.0)


def test_resolve_channel_phys_range_non_pp_uses_empirical_data():
    obj = _DummySignals()
    lo, hi = obj._resolve_channel_phys_range("EEG_C3", np.array([-1.5, 0.5, 3.5]))
    assert (lo, hi) == (-1.5, 3.5)


def test_pp_fill_enabled_only_for_pp_channels():
    obj = _DummySignals()
    assert obj._channel_uses_pp_fill("PP_N2")
    assert not obj._channel_uses_pp_fill("EEG_C3")


def test_pp_fill_disabled_for_sigmod_channels_in_render_mode():
    obj = _DummySignals()
    assert not obj._channel_uses_pp_fill("PP_N2", {"PP_N2": {"mod": "x", "pal": "y"}})
    assert obj._channel_uses_pp_fill("PP_W", {"PP_N2": {"mod": "x", "pal": "y"}})


def test_pp_fill_curve_does_not_peak_downsample(qapp):
    obj = _DummySignals()
    curve = obj._pp_fill_curve("cyan")
    assert curve.opts["clipToView"] is True
    assert curve.opts["autoDownsample"] is False
    assert curve.opts["downsampleMethod"] == "peak"
    assert curve.opts["fillLevel"] == 0.0


def test_pp_fill_data_collapses_duplicate_x_to_upper_envelope():
    obj = _DummySignals()
    x = np.array([2.0, 1.0, 1.0, 2.0, 3.0, 3.0])
    y = np.array([0.4, 0.1, 0.9, 0.7, -0.2, 0.3])
    fill_x, fill_y = obj._prepare_pp_fill_data(x, y, band_lo=0.0)
    np.testing.assert_allclose(fill_x, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(fill_y, [0.9, 0.7, 0.3])


def test_resolved_signal_color_uses_pp_stage_color_independent_of_slot_order():
    obj = _DummySignals()
    obj.cmap = {}
    obj.stgcols_hex = {
        "N1": "#20B2DA",
        "N2": "#0000FF",
        "N3": "#000080",
        "R": "#FF0000",
        "W": "#008000",
    }
    assert obj._resolved_signal_color("PP_N1", "#123456") == "#20B2DA"
    assert obj._resolved_signal_color("PP_R", "#123456") == "#FF0000"
    assert obj._resolved_signal_color("PP_coda_N1", "#123456") == "#20B2DA"
    assert obj._resolved_signal_color("PP_INVALID", "#123456") == "#FFFFFF"
    assert obj._resolved_signal_color("PP_coda_X", "#123456") == "#123456"
    assert obj._resolved_signal_color("EEG_C3", "#123456") == "#123456"


def test_pp_stage_parser_uses_final_token_and_falls_back_cleanly():
    obj = _DummyCMaps()
    assert obj._pp_stage_from_channel("PP_N1") == "N1"
    assert obj._pp_stage_from_channel("PP_coda_N1") == "N1"
    assert obj._pp_stage_from_channel("PP_coda_X") is None
    assert obj._pp_stage_from_channel("PP_X") is None


def test_pp_signal_colors_default_to_stage_colors():
    obj = _DummyCMaps()
    out = obj._update_pp_signal_cols(["#111111", "#222222"], ["PP_N1", "PP_W"])
    assert out == ["#20B2DA", "#008000"]


def test_pp_signal_colors_use_white_for_pp_invalid():
    obj = _DummyCMaps()
    out = obj._update_pp_signal_cols(["#111111", "#222222"], ["PP_INVALID", "PP_family_INVALID"])
    assert out == ["#FFFFFF", "#FFFFFF"]


def test_pp_signal_colors_respect_explicit_config_override():
    obj = _DummyCMaps()
    obj.cmap["PP_N1"] = "#abcdef"
    out = obj._update_pp_signal_cols(["#111111"], ["PP_N1"])
    assert out == ["#abcdef"]


def test_pp_signal_colors_disable_with_pp_style_false():
    obj = _DummyCMaps()
    obj.cfg_pp_style = False
    out = obj._update_pp_signal_cols(["#111111"], ["PP_N1"])
    assert out == ["#111111"]


def test_pp_signal_colors_follow_provided_channel_order():
    obj = _DummyCMaps()
    out = obj._update_pp_signal_cols(
        ["#1", "#2", "#3", "#4", "#5", "#6"],
        ["PP_W", "PP_R", "PP_N3", "PP_coda_N2", "PP_N1", "PP_coda_X"],
    )
    assert out == ["#008000", "#FF0000", "#000080", "#0000FF", "#20B2DA", "#6"]


def test_pp_channels_group_by_stage_when_no_explicit_palette_order():
    obj = _DummyCMaps()
    ordered = obj._order_pp_channels(["EEG_C3", "PP_W", "PP_coda_N1", "PP_N2", "PP_N1", "PP_coda_W", "PP_misc_X"])
    assert ordered == ["PP_N1", "PP_coda_N1", "PP_N2", "PP_W", "PP_coda_W", "PP_misc_X", "EEG_C3"]


def test_pp_channel_grouping_disables_with_explicit_style_flag_false():
    obj = _DummyCMaps()
    obj.cfg_pp_style = False
    ordered = obj._order_pp_channels(["EEG_C3", "PP_W", "PP_N1"])
    assert ordered == ["EEG_C3", "PP_W", "PP_N1"]


def test_hypno_density_tolerates_missing_pp_columns():
    from lunascope.components.plts import hypno_density

    probs = pd.DataFrame(
        {
            "PP_N2": [0.6, 0.2],
            "PP_R": [0.1, 0.7],
            "PRED": ["N2", "R"],
        }
    )
    fig, ax = plt.subplots()
    try:
        out = hypno_density(probs, ax=ax)
        assert out is ax
        assert ax.get_xlim()[1] >= 2
    finally:
        plt.close(fig)
