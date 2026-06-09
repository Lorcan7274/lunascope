from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from lunascope.components.plts import plot_hjorth, plot_spec, plot_tf_heatmap
from lunascope.components.spectrogram import SpecMixin


class _FakeP:
    def silent_proc_lunascope(self, _cmd):
        return {
            "SIGSTATS: CH_E": pd.DataFrame(
                {
                    "E": [1, 2, 3],
                    "H1": [1.0, 2.0, 3.0],
                    "H2": [0.2, 0.5, 0.8],
                    "H3": [0.3, 0.6, 0.9],
                }
            ),
            "EPOCH: E": pd.DataFrame(
                {
                    "E": [1, 2, 3],
                    "START": [0.0, 30.0, 60.0],
                }
            ),
        }


def _gui():
    return SimpleNamespace(spin_win=SimpleNamespace(value=lambda: 0.0))


def test_plot_spec_adds_axes_labels_and_colorbar_when_legend_enabled():
    fig, ax = plt.subplots()
    try:
        xi = np.array([0.0, 30.0, 60.0])
        yi = np.array([0.5, 1.0, 1.5])
        zi = np.array([[1.0, 2.0], [3.0, 4.0]])

        plot_spec(xi, yi, zi, "C3", 0.5, 1.5, ax=ax, gui=_gui(), show_legend=True)

        assert ax.get_xlabel() == "Time (s)"
        assert ax.get_ylabel() == "Frequency (Hz)"
        assert len(fig.axes) == 2
        assert fig.axes[1].get_ylabel() == "PSD (dB)"
    finally:
        plt.close(fig)


def test_plot_spec_removes_colorbar_when_legend_disabled_on_redraw():
    fig, ax = plt.subplots()
    try:
        xi = np.array([0.0, 30.0, 60.0])
        yi = np.array([0.5, 1.0, 1.5])
        zi = np.array([[1.0, 2.0], [3.0, 4.0]])

        plot_spec(xi, yi, zi, "C3", 0.5, 1.5, ax=ax, gui=_gui(), show_legend=True)
        plot_spec(xi, yi, zi, "C3", 0.5, 1.5, ax=ax, gui=_gui(), show_legend=False)

        assert len(fig.axes) == 1
        assert all(axis.get_ylabel() != "PSD (dB)" for axis in fig.axes)
    finally:
        plt.close(fig)


def test_plot_hjorth_adds_axes_labels_and_colorbar_when_legend_enabled():
    fig, ax = plt.subplots()
    try:
        plot_hjorth("C3", ax=ax, p=_FakeP(), gui=_gui(), epoch_dur=30, show_legend=True)

        assert ax.get_xlabel() == "Time (s)"
        assert ax.get_ylabel() == "Hjorth activity"
        assert len(fig.axes) == 2
        assert fig.axes[1].get_ylabel() == "Normalized mobility / complexity"
    finally:
        plt.close(fig)


def test_plot_hjorth_keeps_compact_axis_hidden_when_legend_disabled():
    fig, ax = plt.subplots()
    try:
        plot_hjorth("C3", ax=ax, p=_FakeP(), gui=_gui(), epoch_dur=30, show_legend=False)

        assert len(fig.axes) == 1
        assert not ax.axison
    finally:
        plt.close(fig)


def test_heatmap_background_black_after_hjorth_compact_redraw():
    fig, ax = plt.subplots()
    try:
        plot_hjorth("C3", ax=ax, p=_FakeP(), gui=_gui(), epoch_dur=30, show_legend=False)
        zi = np.ma.array([[1.0, np.nan]], mask=[[False, True]])
        plot_tf_heatmap(
            np.array([0.0, 30.0, 60.0]),
            np.array([0.5, 1.5]),
            zi,
            "MTM",
            ax,
            show_legend=False,
        )

        assert ax.get_facecolor()[:3] == (0.0, 0.0, 0.0)
        assert fig.get_facecolor()[:3] == (0.0, 0.0, 0.0)
        np.testing.assert_allclose(
            ax.collections[0].cmap.get_bad()[:3],
            (0.0, 0.0, 0.0),
        )
    finally:
        plt.close(fig)


def test_dock6_legend_checkbox_exists_and_defaults_unchecked():
    tree = ET.parse("src/lunascope/ui/main.ui")
    root = tree.getroot()
    checkbox = root.find(".//widget[@class='QCheckBox'][@name='check_spec_legend']")

    assert checkbox is not None
    assert checkbox.find("./property[@name='checked']") is None


def test_timefreq_grid_preserves_elapsed_time_gaps():
    spec = SpecMixin()
    xi, yi, zi = spec._grid_elapsed_points(
        [0.0, 30.0, 120.0],
        [1.0, 1.0, 1.0],
        [10.0, 20.0, 50.0],
        0.5,
        1.5,
        0.0,
        total_epochs=5,
        total_seconds=150.0,
        timeline_starts=[0.0, 30.0, 120.0],
    )

    assert xi.tolist() == [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    assert yi.tolist() == [0.5, 1.5]
    assert zi.shape == (1, 5)
    assert zi[0, 0] == 10.0
    assert zi[0, 1] == 20.0
    assert np.ma.is_masked(zi[0, 2])
    assert np.ma.is_masked(zi[0, 3])
    assert zi[0, 4] == 50.0


def test_timefreq_cache_key_changes_when_data_cache_invalidated():
    spec = SpecMixin()
    spec._spec_cache = {}
    spec._spec_data_version = 0
    params = {"ch": "C3", "minf": 0.5, "maxf": 20.0, "winsor": 0.0}

    key_before = spec._spec_cache_key("irasa", params)
    spec._invalidate_spec_data_cache()
    key_after = spec._spec_cache_key("irasa", params)

    assert key_before != key_after
    assert key_before[1:] == key_after[1:]


def test_mtm_zoom_is_not_cached():
    spec = SpecMixin()

    assert not spec._should_cache_timefreq("mtm", {"mtm_mode": "zoom"})
    assert spec._should_cache_timefreq("mtm", {"mtm_mode": "whole"})
    assert spec._should_cache_timefreq("hjorth", {})
    assert spec._should_cache_timefreq("welch", {})
    assert spec._should_cache_timefreq("irasa", {})


def test_hjorth_cache_key_tracks_data_and_winsor():
    spec = SpecMixin()
    spec._spec_data_version = 0
    params = {"ch": "C3", "minf": None, "maxf": None, "winsor": 0.0, "epoch_dur": 30}

    key_before = spec._spec_cache_key("hjorth", params)
    key_winsor = spec._spec_cache_key("hjorth", {**params, "winsor": 0.1})
    spec._spec_data_version = 1
    key_after = spec._spec_cache_key("hjorth", params)

    assert key_before != key_winsor
    assert key_before != key_after


def test_mtm_zoom_center_edges_do_not_smear_first_segment():
    spec = SpecMixin()

    xi, yi, zi = spec._grid_from_points(
        [10.0, 10.1, 10.2],
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0],
        default_x_step=0.1,
    )

    assert np.allclose(xi, [9.95, 10.05, 10.15, 10.25])
    assert yi.tolist() == [0.5, 1.5]
    assert zi.shape == (1, 3)
