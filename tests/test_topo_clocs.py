"""Tests for :mod:`lunascope.components.topo_clocs`.

The module ports EEG channel coordinates from luna-base; we verify the
coordinate table, the azimuthal-equidistant projection, and the small
loader for user override files.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from lunascope.components import topo_clocs as clocs


# ---------------------------------------------------------------------------
# cart_to_plot2d
# ---------------------------------------------------------------------------


def test_vertex_projects_to_origin():
    """Cz (vertex) sits straight above origin; projection is the origin."""
    px, py = clocs.cart_to_plot2d(0.0, 0.0, 1.0)
    assert px == pytest.approx(0.0, abs=1e-9)
    assert py == pytest.approx(0.0, abs=1e-9)


def test_anterior_point_projects_above_origin():
    """An anterior unit vector (+x) goes to the top of the head."""
    px, py = clocs.cart_to_plot2d(1.0, 0.0, 0.0)
    assert px == pytest.approx(0.0, abs=1e-9)
    assert py > 0


def test_right_hemisphere_projects_to_right_screen_side():
    """Negative y (right hemisphere in luna's convention) maps to +x_plot."""
    px, _py = clocs.cart_to_plot2d(0.0, -1.0, 0.0)
    assert px > 0


def test_zero_vector_returns_origin():
    px, py = clocs.cart_to_plot2d(0.0, 0.0, 0.0)
    assert (px, py) == (0.0, 0.0)


def test_equatorial_radius_is_half():
    """Any point on the equator (z=0) projects to radius 0.5."""
    px, py = clocs.cart_to_plot2d(1.0, 0.0, 0.0)
    r = math.hypot(px, py)
    assert r == pytest.approx(0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# get_positions
# ---------------------------------------------------------------------------


def test_get_positions_returns_known_channels_only():
    out = clocs.get_positions(["Fp1", "Cz", "BOGUS_CHAN"])
    assert "Fp1" in out
    assert "Cz" in out
    assert "BOGUS_CHAN" not in out


def test_get_positions_preserves_input_label_case():
    out = clocs.get_positions(["fp1"])
    assert "fp1" in out  # original case preserved as key


def test_get_positions_user_overrides_take_precedence():
    overrides = {"Cz": (10.0, 0.0, 0.0)}
    out = clocs.get_positions(["Cz"], user_overrides=overrides)
    px, py = out["Cz"]
    # Overridden coord is anterior unit vector → should project to (0, 0.5)
    assert px == pytest.approx(0.0, abs=1e-9)
    assert py == pytest.approx(0.5, rel=1e-6)


def test_alias_channels_map_to_same_coordinate():
    """T3 and T7 are documented aliases — must resolve identically."""
    pos = clocs.get_positions(["T3", "T7"])
    assert pos["T3"] == pos["T7"]
    assert clocs.get_positions(["T5"])["T5"] == clocs.get_positions(["P7"])["P7"]


# ---------------------------------------------------------------------------
# load_clocs_file
# ---------------------------------------------------------------------------


def test_load_clocs_file_round_trip(tmp_path: Path):
    f = tmp_path / "clocs.txt"
    f.write_text(
        "# comment line\n"
        "% another comment\n"
        "Fp1 1.0 2.0 3.0\n"
        "Cz, 0.0, 0.0, 1.0\n"
        "bad line should be skipped\n"
        "Oz NOT NUMERIC 0 0\n"
        "\n"
    )
    coords = clocs.load_clocs_file(str(f))
    assert coords["FP1"] == (1.0, 2.0, 3.0)
    assert coords["CZ"] == (0.0, 0.0, 1.0)
    assert "OZ" not in coords  # malformed → skipped


def test_load_clocs_file_handles_empty(tmp_path: Path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    assert clocs.load_clocs_file(str(f)) == {}


# ---------------------------------------------------------------------------
# all_known_labels
# ---------------------------------------------------------------------------


def test_all_known_labels_includes_core_montage():
    labels = set(clocs.all_known_labels())
    for ch in ("FP1", "FP2", "CZ", "OZ", "T7", "T8", "P7", "P8"):
        assert ch in labels


def test_all_known_labels_is_sorted():
    labels = clocs.all_known_labels()
    assert labels == sorted(labels)
