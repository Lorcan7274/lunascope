"""Tests for :mod:`lunascope.components.topo_core`.

The renderer is exercised against an Agg matplotlib backend (set in
``conftest.py``) so it works fully offline.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

scipy = pytest.importorskip("scipy")  # interpolation requires scipy

from lunascope.components import topo_clocs, topo_core


@pytest.fixture
def positions_10_chan():
    return topo_clocs.get_positions(
        ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    )


@pytest.fixture
def values_10_chan():
    return {
        "Fp1": 1.0,
        "Fp2": 0.5,
        "F3": -0.2,
        "F4": -0.1,
        "C3": 0.0,
        "C4": 0.3,
        "P3": -0.5,
        "P4": -0.7,
        "O1": 0.1,
        "O2": 0.2,
    }


def test_draw_topo_basic(positions_10_chan, values_10_chan):
    fig, ax = plt.subplots()
    try:
        topo_core.draw_topo(
            ax,
            values_10_chan,
            positions_10_chan,
            mode="both",
            show_labels=True,
        )
        # Expect at least one collection (scatter), one patch (head circle),
        # and a colorbar axes attached to the figure.
        assert ax.collections, "scatter / pcolormesh should add collections"
        assert any(p.__class__.__name__ == "Circle" for p in ax.patches)
        assert len(fig.axes) >= 2  # main + colorbar
    finally:
        plt.close(fig)


def test_draw_topo_no_matching_channels(positions_10_chan):
    fig, ax = plt.subplots()
    try:
        topo_core.draw_topo(ax, {"BOGUS": 1.0}, positions_10_chan)
        # Should have rendered the "no matching channels" message rather
        # than crashing.
        texts = [t.get_text() for t in ax.texts]
        assert any("No matching" in t for t in texts)
    finally:
        plt.close(fig)


def test_draw_topo_dots_only_with_few_channels():
    """Below the interpolation threshold, only dots are drawn."""
    pos = topo_clocs.get_positions(["Fp1", "Fp2", "Cz"])
    vals = {"Fp1": 1.0, "Fp2": -1.0, "Cz": 0.0}
    fig, ax = plt.subplots()
    try:
        topo_core.draw_topo(ax, vals, pos, mode="both")
        # No QuadMesh from pcolormesh
        from matplotlib.collections import QuadMesh

        assert not any(isinstance(c, QuadMesh) for c in ax.collections)
    finally:
        plt.close(fig)


def test_topo_renderer_setup_and_update(positions_10_chan, values_10_chan):
    renderer = topo_core.TopoRenderer(positions_10_chan, grid_res=24)
    fig, ax = plt.subplots()
    try:
        renderer.setup(ax, fig, cmap="viridis")
        # Update twice to verify the artist data actually changes
        renderer.update(values_10_chan)
        first = renderer._scatter.get_array().copy()
        flipped = {k: -v for k, v in values_10_chan.items()}
        renderer.update(flipped)
        second = renderer._scatter.get_array().copy()
        np.testing.assert_array_equal(np.asarray(second), -np.asarray(first))
    finally:
        plt.close(fig)


def test_topo_renderer_update_handles_all_nan(positions_10_chan):
    renderer = topo_core.TopoRenderer(positions_10_chan, grid_res=16)
    fig, ax = plt.subplots()
    try:
        renderer.setup(ax, fig)
        # All-NaN input must not raise
        renderer.update({k: np.nan for k in positions_10_chan})
    finally:
        plt.close(fig)
