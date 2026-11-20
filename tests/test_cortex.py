"""Meaningful boundaries for cortical atlases, scalar maps and bounded playback."""
import json
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pandas as pd
import pytest

from lunascope.components.atlas_model import Atlas, pick_triangle
from lunascope.components.cortical_data import (ChunkCache, PlaybackClock, fetch_chunk,
    project_resources, recording_info, result_values, signal_unit, SignalChunk, summarize)


@pytest.fixture
def atlas_directory(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"geometry": {"frame": "RAS"}}))
    (tmp_path / "parcels.tsv").write_text(
        "parcel_id\tname\themisphere\tluna_signal\n10\tAlpha\tlh\tRegion_A\n20\tBeta\trh\tRegion_B\n")
    vertices = np.array([[-2, 0, 0], [-1, 1, 0], [-1, 0, 1],
                         [2, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=np.float32)
    np.savez(tmp_path / "cortex_mesh.npz", pial_vertices=vertices,
             inflated_vertices=vertices * 2, faces=[[0, 1, 2], [3, 4, 5], [0, 3, 2]],
             hemisphere=[0, 0, 0, 1, 1, 1])
    np.save(tmp_path / "vertex_parcels.npy", [10, 10, -1, 20, 20, 20])
    return tmp_path


@pytest.fixture
def atlas(atlas_directory):
    return Atlas.load(atlas_directory)


def test_names_are_exact_case_insensitive_and_order_independent(atlas):
    assert atlas.match(["Other", "REGION_b", "region_a", "DK_fake"]) == ["region_a", "REGION_b"]
    assert atlas.match(["DK_fake"]) == [None, None]
    assert atlas.match(["new_Region_A"], prefix="new_") == ["new_Region_A", None]
    assert atlas.match(["A"], mapping={10: "a"}) == ["A", None]
    with pytest.raises(ValueError, match="Duplicate"):
        atlas.match(["Region_A", "region_a"])
    with pytest.raises(ValueError, match="multiple parcels"):
        atlas.match([], mapping={10: "same", 20: "SAME"})


def test_edf_labels_match_loaded_model_without_mapping(atlas):
    atlas.parcels = (replace(atlas.parcels[0], signal="Area_LEFT_caudalanterior"),
                     replace(atlas.parcels[1], signal="Area_RIGHT_caudalanterior"))
    labels = [p.signal[:16].swapcase() for p in atlas.parcels]
    assert atlas.match([labels[1], "DK_fake", labels[0]]) == labels
    # Other short prefixes must not become aliases.
    assert atlas.match([p.signal[:15] for p in atlas.parcels]) == [None, None]
    # Prefer full labels if both representations are present.
    assert atlas.match(labels + [p.signal for p in atlas.parcels]) == [p.signal for p in atlas.parcels]
    frame = pd.DataFrame({"CH": labels[::-1], "DENS": [2., 1.]})
    values, _ = result_values(atlas, frame, "DENS")
    np.testing.assert_allclose(values, [1., 2.])


@pytest.mark.parametrize("full_available", [False, True])
def test_edf_alias_collisions_require_explicit_mapping(atlas, full_available):
    stem = "CommonSignalName"  # exactly 16 characters
    atlas.parcels = (replace(atlas.parcels[0], signal=stem + "_left"),
                     replace(atlas.parcels[1], signal=stem + "_right"))
    names = [stem]
    if full_available:
        names.append(atlas.parcels[0].signal)
    with pytest.raises(ValueError, match="Ambiguous EDF label"):
        atlas.match(names)
    # Explicitly renamed signals disambiguate the actual exported recording.
    assert atlas.match([stem, "RightShort"], mapping={10: stem, 20: "RightShort"}) == [stem, "RightShort"]


def test_edf_alias_does_not_steal_another_parcels_exact_name(atlas):
    stem = "CommonSignalName"
    atlas.parcels = (replace(atlas.parcels[0], signal=stem),
                     replace(atlas.parcels[1], signal=stem + "_long"))
    with pytest.raises(ValueError, match="Ambiguous EDF label"):
        atlas.match([stem])


def test_explicit_names_override_model_and_prefixes_get_edf_representation(atlas):
    atlas.parcels = tuple(replace(p, signal=p.signal + "_longer_signal") for p in atlas.parcels)
    assert atlas.match([p.signal[:16] for p in atlas.parcels], mapping={10: "Renamed"})[0] is None
    prefixed = [("new_" + p.signal)[:16] for p in atlas.parcels]
    assert atlas.match(prefixed[::-1], prefix="new_") == prefixed


def test_topology_majority_ties_and_hemisphere_identity(atlas, atlas_directory):
    assert atlas.face_parcels().tolist() == [0, 1, -1]
    assert [p.hemisphere for p in atlas.parcels] == ["left", "right"]
    np.save(atlas_directory / "vertex_parcels.npy", [20, 10, -1, 20, 20, 20])
    with pytest.raises(ValueError, match="Hemisphere mismatch"):
        Atlas.load(atlas_directory)


def test_invalid_geometry_rejected(atlas_directory):
    metadata = atlas_directory / "metadata.json"
    metadata.write_text('{"geometry":{"frame":"unknown"}}')
    with pytest.raises(ValueError, match="RAS"):
        Atlas.load(atlas_directory)


def test_results_missing_regions_and_supplied_units(atlas):
    df = pd.DataFrame({"ID": ["s1"], "CH": ["REGION_B"], "DENS": [3.]})
    df.attrs["units"] = {"DENS": "events/min"}
    values, units = result_values(atlas, df, "DENS")
    assert np.isnan(values[0]) and values[1] == 3
    assert units == "events/min"


def test_results_duplicates_require_explicit_aggregation(atlas):
    df = pd.DataFrame({"CH": ["Region_A"] * 2, "DENS": [2., 4.]})
    with pytest.raises(ValueError, match="Multiple rows"):
        result_values(atlas, df, "DENS")
    values, _ = result_values(atlas, df, "DENS", aggregation="mean")
    assert values[0] == 3


@pytest.mark.parametrize("column", ["ID", "STAGE", "E", "BAND"])
def test_results_do_not_pool_dimensions_even_with_aggregation(atlas, column):
    df = pd.DataFrame({"CH": ["Region_A"] * 2, "DENS": [2., 4.], column: ["a", "b"]})
    with pytest.raises(ValueError, match=f"step {column}"):
        result_values(atlas, df, "DENS", aggregation="mean")
    values, _ = result_values(atlas, df, "DENS", filters={column: "b"})
    assert values[0] == 4


def test_pairwise_tables_rejected(atlas):
    with pytest.raises(ValueError, match="pairwise"):
        result_values(atlas, pd.DataFrame({"CH1": [], "CH2": [], "R": []}), "R")


def test_flat_colors_one_parcel_at_a_time(atlas):
    from lunascope.components.cortical_view import parcel_colors
    for active in range(2):
        values = np.full(2, np.nan)
        values[active] = 1
        colors = parcel_colors(values, atlas.face_parcels(), "viridis", (0, 1))
        np.testing.assert_allclose(colors[2], [.32, .34, .37, 1])
        np.testing.assert_allclose(colors[1 - active], colors[2])
        assert not np.allclose(colors[active], colors[2])


def test_picking_nearest_two_sided_triangle_and_miss():
    triangles = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                          [[0, 0, -2], [1, 0, -2], [0, 1, -2]]], dtype=float)
    assert pick_triangle(np.array([.2, .2, 2]), np.array([0., 0, -1]), triangles) == 0
    assert pick_triangle(np.array([.2, .2, -3]), np.array([0., 0, 1]), triangles) == 1
    assert pick_triangle(np.array([2., 2, 2]), np.array([0., 0, -1]), triangles) is None


def test_playback_uses_monotonic_elapsed_time():
    now = [100.]
    clock = PlaybackClock(lambda: now[0])
    clock.play()
    now[0] += 2.3
    assert clock.position() == pytest.approx(2.3)
    clock.pause()
    now[0] += 10
    assert clock.position() == pytest.approx(2.3)
    clock.seek(40)
    clock.play(2)
    now[0] += .5
    assert clock.position() == pytest.approx(41)


def test_rms_and_band_power_have_expected_magnitude():
    times = np.arange(0, 4, .002)
    values = 2 * np.sin(2 * np.pi * 10 * times)
    rms = summarize(times, values, np.array([2.]), "RMS amplitude", 1., (8, 12))
    power = summarize(times, values, np.array([2.]), "Band power", 1., (8, 12))
    assert rms[0] == pytest.approx(np.sqrt(2), rel=.005)
    assert power[0] == pytest.approx(2, rel=.005)
    assert np.isnan(summarize(times, values, np.array([0.]), "RMS amplitude", 1., (8, 12))[0])


def test_metrics_do_not_bridge_gaps_or_bad_samples():
    times = np.r_[np.arange(0, 2, .01), np.arange(4, 6, .01)]
    samples = np.ones(len(times))
    cursors = np.array([1., 2., 3., 4., 5.])
    result = summarize(times, samples, cursors, "RMS amplitude", 1., (.5, 4))
    np.testing.assert_allclose(result[[0, 4]], 1.)
    assert np.isnan(result[1:4]).all()
    instantaneous = summarize(times, samples, np.array([3.]), "Signed amplitude", 1, (0, 1))
    assert np.isnan(instantaneous).all()
    samples[100] = np.nan
    assert np.isnan(summarize(times, samples, np.array([1.]), "RMS amplitude", 1, (0, 1))).all()


class Recording:
    def __init__(self):
        self.requests = []

    def headers(self):
        return pd.DataFrame({"CH": ["REGION_B", "Region_A"], "PDIM": ["nAm", "nAm"]})

    def segments(self):
        return pd.DataFrame({"ID": ["S1", "S1"], "SEG": [1, 2], "DUR": [12, 35980],
                             "START": [0., 20.], "STOP": [12., 36000.]})

    def s2i(self, intervals):
        return intervals

    def slice(self, intervals, chs, time):
        self.requests.append((intervals, chs))
        times = np.concatenate([np.arange(a, b, .002) for a, b in intervals])
        return [None, np.column_stack([times, np.ones(len(times)) * 3])]


def test_chunk_reads_bounded_intervals_and_recorded_units(atlas):
    recording = Recording()
    names, units, segments = recording_info(recording, atlas)
    assert names == ["Region_A", "REGION_B"] and units == ["nAm", "nAm"]
    block = fetch_chunk(recording, names, segments, 1, "RMS amplitude", 1., (.5, 4), Event())
    assert block.values.shape == (250, 2)
    assert np.isnan(block.values[(block.cursors >= 12), :]).all()
    assert all(sum(b - a for a, b in intervals) <= 11 for intervals, _ in recording.requests)
    assert all(len(trace[0]) <= 2000 for trace in block.traces.values())
    np.testing.assert_allclose(block.values[0], 3.)


def test_cache_bound_and_cancelled_reads(atlas):
    cache = ChunkCache()
    for i in range(100):
        cache.put(i, np.zeros((250, 2)))
    assert list(cache.blocks) == [97, 98, 99]
    cancelled = Event()
    cancelled.set()
    recording = Recording()
    assert fetch_chunk(recording, ["Region_A"], np.array([[0, 100]]), 0,
                       "RMS amplitude", 1., (.5, 4), cancelled) is None
    assert not recording.requests


def test_project_structured_mapping_and_stored_units(tmp_path):
    directory, mapping, units = project_resources({
        "PROJECT_FILE": pd.DataFrame({"FILE": ["metadata"], "PATH": [str(tmp_path / "metadata.json")]}),
        "PROJECT_OUTPUT": pd.DataFrame({"PARCEL_ID": [10, 20], "LABEL": ["ShortL", "ShortR"],
                                         "STORED_UNIT": ["nAm", "nAm"], "NATIVE_UNIT": ["A m", "A m"]})})
    assert directory == tmp_path and mapping == {10: "ShortL", 20: "ShortR"}
    assert units == {"shortl": "nAm", "shortr": "nAm"}
    assert signal_unit("nAm", "A m") == "nAm"
    assert signal_unit("n/a", units["shortl"]) == "nAm"
    assert signal_unit(None) == "unknown"


def test_ambiguous_project_mapping_is_rejected():
    cohort = pd.DataFrame({"ID": ["s1", "s2"], "PARCEL_ID": [10, 10], "LABEL": ["L", "Different"]})
    with pytest.raises(ValueError, match="Cohort"):
        project_resources({"PROJECT_OUTPUT": cohort})


def test_real_lunapi_segment_schema_and_bounded_slice(atlas):
    import lunapi as lp
    project = lp.proj()
    recording = project.empty_inst("cortex-test", 30, 1)
    signal = np.sin(2 * np.pi * 10 * np.arange(3000) / 100)
    recording.insert_signal("REGION_B", signal, 100)
    recording.insert_signal("Region_A", signal, 100)
    names, units, segments = recording_info(recording, atlas)
    assert names == ["Region_A", "REGION_B"]
    assert len(set(units)) == 1
    np.testing.assert_allclose(segments, [[0, 30]])
    chunk = fetch_chunk(recording, names, segments, 1, "RMS amplitude", 1., (8, 12), Event())
    np.testing.assert_allclose(chunk.values, np.sqrt(.5), atol=1e-4)


@pytest.mark.parametrize("mode", ["RMS amplitude", "Band power"])
def test_full_night_shape_keeps_only_short_blocks(mode):
    recording, cache = Recording(), ChunkCache()
    names = [f"Region_{i}" for i in range(68)]
    start = time.monotonic()
    for index in (0, 1800, 3599, 2000):
        block = fetch_chunk(recording, names, np.array([[0., 36000.]]), index,
                            mode, 1., (.5, 4), Event())
        cache.put(index, block)
    stored_bytes = sum(block.values.nbytes + block.cursors.nbytes
                       + sum(t.nbytes + s.nbytes for t, s in block.traces.values())
                       for block in cache.blocks.values())
    assert len(cache.blocks) == 3 and stored_bytes < 6_000_000
    assert all(sum(b-a for a, b in intervals) <= 11 for intervals, _ in recording.requests)
    print(f"{mode}: 4 × 68-channel chunks from a synthetic ten-hour timeline in "
          f"{time.monotonic()-start:.3f}s, cache {stored_bytes/1e6:.2f} MB")


@pytest.mark.qt
def test_qt_camera_orientation_and_unprojection_picking(qapp, atlas):
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QVector3D
    from lunascope.components.cortical_view import CorticalView
    view = CorticalView()
    view.set_atlas(atlas)
    for hemisphere, axis, sign in (("Left", 0, -1), ("Right", 0, 1)):
        view.set_display(hemisphere, "pial")
        view.preset("Lateral")
        inverse, valid = view._matrices()[0].inverted()
        assert valid
        origin = inverse.map(QVector3D(0, 0, -1)).toTuple()
        assert origin[axis] * sign > 0
    vertices = atlas.surfaces["pial"]
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    vertices = (vertices - center) / np.linalg.norm(vertices - center, axis=1).max()
    view._triangles = {"pial": vertices[atlas.faces]}
    view._geometry_dirty = False
    view.set_display("Left", "pial")
    view.preset("Lateral")
    centroid = view._triangles["pial"][0].mean(axis=0)
    projected = view._matrices()[0].map(QVector3D(*map(float, centroid)))
    point = QPointF((projected.x()+1)*view.width()/2, (1-projected.y())*view.height()/2)
    picked = []
    view.identified.connect(lambda i, clicked: picked.append(i))
    view._pick(point, True)
    assert picked == [0]
    view.deleteLater()


@pytest.mark.qt
@pytest.mark.skipif(os.environ.get("LUNASCOPE_TEST_OPENGL") != "1",
                    reason="Requires a native desktop; enable LUNASCOPE_TEST_OPENGL=1")
def test_native_composited_cortex_after_lazy_creation(atlas_directory, tmp_path):
    """Check the actual window compositor, not only grabFramebuffer().

    Run in a fresh process so the shared format is set before QApplication.
    Missing/mirrored overlays were invisible to the original standalone check.
    """
    script = textwrap.dedent('''
        import sys
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor
        from PySide6.QtCore import QObject, QPoint, QTimer
        from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget
        from lunascope.qt_opengl import configure_opengl
        from lunascope.components.explorer_cortex import CortexTab
        configure_opengl()
        app = QApplication([])
        class Controller(QObject):
            pass
        ctrl = Controller()
        ctrl.results = {}
        ctrl._exec = ThreadPoolExecutor(max_workers=1)
        window = QMainWindow()
        tabs = QTabWidget()
        tabs.addTab(QWidget(), "Other")
        cortex = CortexTab(ctrl)
        tabs.addTab(cortex.widget(), "Cortex")
        window.setCentralWidget(tabs)
        window.resize(1100, 900)
        window.show()
        def load():
            cortex.activate()
            cortex._set_atlas(sys.argv[1], manual=True)
            tabs.setCurrentIndex(1)
            cortex._show_values(np.array([.2, .8]), "nAm")
            cortex.view.set_values(np.array([.2, .8]), "viridis", (0, 1), "nAm")
        def check():
            try:
                assert cortex.view.isValid(), "No native OpenGL context"
                image = window.screen().grabWindow(window.winId()).toImage()
                image.save(sys.argv[2])
                assert not image.isNull(), "No composited window capture"
                ratio = image.width() / window.width()
                def pixel(x, y):
                    p = cortex.view.mapTo(window, QPoint(x, y))
                    return np.array(image.pixelColor(round(p.x()*ratio), round(p.y()*ratio)).getRgb()[:3])
                # The cortex must draw into its own FBO. The compositor must
                # retain both the dark background and correctly placed legend.
                np.testing.assert_allclose(pixel(2, cortex.view.height()//2), [13,17,23], atol=2)
                np.testing.assert_allclose(pixel(cortex.view.width()//2, cortex.view.height()-36),
                                           [33,145,140], atol=8)
                print("Native embedded Cortex composition PASS", flush=True)
                app.exit(0)
            except BaseException:
                import traceback
                traceback.print_exc()
                app.exit(1)
        QTimer.singleShot(200, load)
        QTimer.singleShot(1200, check)
        result = app.exec()
        ctrl._exec.shutdown()
        sys.exit(result)
    ''')
    environment = dict(os.environ)
    if environment.get("QT_QPA_PLATFORM") in (None, "offscreen", "minimal"):
        environment["QT_QPA_PLATFORM"] = "cocoa" if sys.platform == "darwin" else "windows" if sys.platform == "win32" else "xcb"
    result = subprocess.run([sys.executable, "-c", script, str(atlas_directory), str(tmp_path / "composition.png")],
                            env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.qt
def test_lazy_ui_lifecycle_and_stale_delivery(qapp, atlas_directory):
    from PySide6.QtCore import QObject
    from lunascope.components.explorer_cortex import CortexTab
    class Controller(QObject):
        pass
    controller = Controller()
    controller.results = {"SPINDLES_CH": pd.DataFrame({"CH": ["Region_A"], "DENS": [2.]})}
    with ThreadPoolExecutor(max_workers=1) as executor:
        controller._exec = executor
        panel = CortexTab(controller)
        assert panel.view is None
        panel.refresh_tables()
        assert panel.view is None
        panel.activate()
        panel._set_atlas(atlas_directory)
        assert panel.view.values[0] == 2 and np.isnan(panel.view.values[1])
        panel.clock.play()
        panel.panels.setCurrentWidget(panel.live_panel)
        assert panel.clock.started is None
        old_generation = panel._generation
        panel.recording_changed()
        panel._receive((old_generation, "info", None, ("stale",), None))
        assert panel._info is None
        # Replacing an atlas with another parcel count while Live is selected
        # must discard the previous ordered vector before recoloring.
        (atlas_directory / "parcels.tsv").write_text(
            "parcel_id\tname\themisphere\tluna_signal\n10\tAlpha\tlh\tRegion_A\n")
        np.save(atlas_directory / "vertex_parcels.npy", [10, 10, -1, -1, -1, -1])
        panel._set_atlas(atlas_directory)
        panel._recolor()
        assert len(panel.view.values) == 1 and np.isnan(panel.view.values).all()
        controller.results = {"PROJECT_OUTPUT": pd.DataFrame({"PARCEL_ID": [10], "LABEL": ["ShortL"]})}
        panel.results_changed()
        assert panel._project_snapshot[1] == {10: "ShortL"}
        panel.recording_changed(clear_resources=True)
        panel.refresh_tables()
        assert panel._project_snapshot == (None, {}, {})
        panel._root.deleteLater()


@pytest.mark.qt
def test_live_coverage_distinguishes_matches_from_incomplete_windows(qapp, atlas_directory):
    from PySide6.QtCore import QObject
    from lunascope.components.explorer_cortex import CortexTab
    controller = QObject()
    controller.results = {}
    controller.p = object()
    panel = CortexTab(controller)
    panel.activate()
    panel._set_atlas(atlas_directory)
    panel.panels.setCurrentWidget(panel.live_panel)
    panel._recording = controller.p
    panel._info = (["Region_A", "REGION_B"], ["nAm", "nAm"], np.array([[0., 30.]]))
    panel._duration = 30.
    times = np.arange(0, 10, .04)
    values = np.tile([1., 2.], (len(times), 1))
    values[:13] = np.nan
    panel.cache.put(0, SignalChunk(times, values, {}))
    panel._pending = ("chunk", 1)
    panel._tick()
    assert "2/2 signals matched" in panel.status.text()
    assert "0/2 valid regions" in panel.status.text()
    assert "Window starts before recording" in panel.status.text()
    panel.clock.seek(1.)
    panel._tick()
    assert "2/2 valid regions" in panel.status.text()
    np.testing.assert_allclose(panel.view.values, [1., 2.])
    panel.recording_changed()
    panel._root.deleteLater()
