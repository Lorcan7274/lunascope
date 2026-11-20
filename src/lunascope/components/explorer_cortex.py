"""Explorer Cortex: scalar Results and bounded, recorded Live playback."""
from __future__ import annotations

import json
from pathlib import Path
from threading import Event

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider, QTabWidget,
    QVBoxLayout, QWidget)

from .atlas_model import Atlas
from .cortical_data import (ChunkCache, PlaybackClock, fetch_chunk, project_resources,
                            recording_info, result_values, signal_unit)
from .explorer_base import _ExplorerTab


def combo(items):
    widget = QComboBox()
    widget.addItems(items)
    return widget


def spin(value, low, high, decimals=2):
    widget = QDoubleSpinBox()
    widget.setDecimals(decimals)
    widget.setRange(low, high)
    widget.setValue(value)
    return widget


def row(layout, *widgets):
    line = QHBoxLayout()
    for widget in widgets:
        line.addWidget(QLabel(widget) if isinstance(widget, str) else widget)
    layout.addLayout(line)


class CortexTab(_ExplorerTab):
    delivered = Signal(object)

    def __init__(self, ctrl, parent=None):
        super().__init__(ctrl, parent)
        self._root = QWidget()
        self.layout = QVBoxLayout(self._root)
        self.atlas = None
        self.view = None
        self.mapping = {}
        self.project_units = {}
        self._manual_atlas = False
        self._manual_mapping = False
        self._project_directory = None
        self._project_snapshot = (None, {}, {})
        self.clock = PlaybackClock()
        self.cache = ChunkCache()
        self._generation = 0
        self._cancelled = Event()
        self._future = None
        self._pending = None
        self._info = None
        self._recording = None
        self._selected = -1
        self._hover = -1
        self._scale_ready = False
        self._ready = False
        self._palette_initialized = False
        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._tick)
        self.delivered.connect(self._receive)

    def activate(self):
        if not self._ready:
            self._build()
        self.refresh_tables()

    def _build(self):
        from .cortical_view import CorticalView
        self._ready = True
        load = QPushButton("Atlas directory…")
        load.clicked.connect(self._choose_atlas)
        mapping = QPushButton("Signal mapping JSON…")
        mapping.clicked.connect(self._choose_mapping)
        self.prefix = QLineEdit()
        self.prefix.setPlaceholderText("Explicit signal prefix (optional)")
        self.prefix.editingFinished.connect(self._mapping_changed)
        row(self.layout, load, mapping, self.prefix)
        self.atlas_label = QLabel("Select the extracted atlas resource directory")
        self.layout.addWidget(self.atlas_label)
        self.panels = QTabWidget()
        self.results_panel, self.live_panel = QWidget(), QWidget()
        self.panels.addTab(self.results_panel, "Results")
        self.panels.addTab(self.live_panel, "Live")
        self.layout.addWidget(self.panels)
        self._build_results()
        self._build_live()
        self.panels.currentChanged.connect(self._panel_changed)
        self.hemi = combo(["Both", "Left", "Right"])
        self.surface = combo(["pial", "inflated"])
        preset = combo(["Lateral", "Medial", "Superior", "Inferior"])
        preset.setCurrentText("Superior")
        reset = QPushButton("Reset")
        row(self.layout, "Hemisphere:", self.hemi, "Surface:", self.surface, preset, reset)
        self.cmap = combo(["viridis", "plasma", "inferno", "RdBu_r", "coolwarm"])
        self.low, self.high = spin(0, -1e15, 1e15, 12), spin(1, -1e15, 1e15, 12)
        for widget in (self.low, self.high):
            widget.setMaximumWidth(150)
        self.dynamic = QCheckBox("Dynamic limits")
        auto = QPushButton("Robust auto limits")
        row(self.layout, self.cmap, "Limits:", self.low, self.high, auto, self.dynamic)
        self.view = CorticalView()
        self.layout.addWidget(self.view, 1)
        self.detail = QLabel("Click a region to select it; drag to rotate; wheel to zoom")
        self.layout.addWidget(self.detail)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.layout.addWidget(self.status)
        self.view.identified.connect(self._identify)
        self.view.failed.connect(self.status.setText)
        self.hemi.currentTextChanged.connect(self._display_changed)
        self.surface.currentTextChanged.connect(self._display_changed)
        preset.textActivated.connect(self.view.preset)
        reset.clicked.connect(self.view.reset)
        self.cmap.currentTextChanged.connect(self._recolor)
        self.low.valueChanged.connect(self._manual_limits)
        self.high.valueChanged.connect(self._manual_limits)
        auto.clicked.connect(self._auto_limits)
        self.dynamic.toggled.connect(self._recolor)
        # Traces are constructed only when Cortex is first opened.
        import pyqtgraph as pg
        self.trace = pg.PlotWidget()
        self.trace.setMaximumHeight(150)
        self.trace.showGrid(x=True, y=True, alpha=.2)
        self.trace.hide()
        self.layout.addWidget(self.trace)

    def _build_results(self):
        layout = QVBoxLayout(self.results_panel)
        self.table, self.variable = combo([]), combo([])
        self.aggregate = combo(["Require unique rows", "mean", "median", "min", "max"])
        self.metric_units = QLineEdit()
        self.metric_units.setPlaceholderText("Metric units (if absent from table metadata)")
        row(layout, "Table:", self.table, "Variable:", self.variable, self.aggregate)
        self.filters = QFormLayout()
        layout.addLayout(self.filters)
        self.step_column, self.step_value = combo([]), combo([])
        back, forward, plot = QPushButton("◀"), QPushButton("▶"), QPushButton("Plot")
        row(layout, "Step:", self.step_column, back, self.step_value, forward, plot, self.metric_units)
        self.table.currentIndexChanged.connect(self._table_changed)
        self.step_column.currentIndexChanged.connect(self._step_changed)
        self.step_value.currentIndexChanged.connect(self._plot_results)
        self.variable.currentIndexChanged.connect(self._result_metric_changed)
        self.aggregate.currentIndexChanged.connect(self._plot_results)
        self.metric_units.editingFinished.connect(self._plot_results)
        plot.clicked.connect(self._plot_results)
        back.clicked.connect(lambda: self._step(-1))
        forward.clicked.connect(lambda: self._step(1))
        self.filter_widgets = {}

    def _build_live(self):
        layout = QVBoxLayout(self.live_panel)
        load = QPushButton("Load recording channels")
        play, pause, stop = QPushButton("Play"), QPushButton("Pause"), QPushButton("Stop")
        self.speed = combo(["0.25", "0.5", "1", "2", "4"])
        self.speed.setCurrentText("1")
        self.mode = combo(["RMS amplitude", "Band power", "Signed amplitude"])
        self.window = spin(1., .1, 10.)
        self.band_low, self.band_high = spin(.5, 0., 1000.), spin(4., .01, 1000.)
        self.log_power = QCheckBox("log10 power")
        self.show_trace = QCheckBox("Selected region trace")
        row(layout, load, play, pause, stop, "Speed:", self.speed, self.mode)
        row(layout, "Window (s):", self.window, "Band (Hz):", self.band_low,
            self.band_high, self.log_power, self.show_trace)
        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setRange(0, 100000)
        self.time_label = QLabel("0.00 s")
        row(layout, self.scrubber, self.time_label)
        load.clicked.connect(self._load_recording)
        play.clicked.connect(self._play)
        pause.clicked.connect(self.pause_live_playback)
        stop.clicked.connect(self._stop)
        self.speed.currentTextChanged.connect(self._speed_changed)
        self.scrubber.sliderPressed.connect(self.pause_live_playback)
        self.scrubber.valueChanged.connect(self._scrub)
        self.mode.currentTextChanged.connect(self._metric_changed)
        for widget in (self.window, self.band_low, self.band_high):
            widget.valueChanged.connect(self._metric_changed)
        self.log_power.toggled.connect(self._metric_changed)
        self.show_trace.toggled.connect(self._trace_changed)

    def _choose_atlas(self):
        directory = QFileDialog.getExistingDirectory(self._root, "Atlas resource directory")
        if directory:
            self._set_atlas(directory, manual=True)

    def _set_atlas(self, directory, manual=False):
        try:
            atlas = Atlas.load(directory)
        except Exception as error:
            self.status.setText(f"Cannot load atlas: {error}")
            return False
        self.recording_changed()
        self.atlas = atlas
        self._manual_atlas = manual
        if manual:
            self.mapping, self.project_units = {}, {}
            self._manual_mapping = False
            directory, mapping, units = self._project_snapshot
            if self._project_matches(directory):
                self.mapping, self.project_units = mapping, units
        self.view.set_atlas(atlas)
        self._values = np.full(len(atlas.parcels), np.nan)
        self._units = ""
        self._selected = self._hover = -1
        self.surface.setCurrentText("pial")
        self.surface.model().item(1).setEnabled("inflated" in atlas.surfaces)
        self._scale_ready = False
        self.atlas_label.setText(f"{atlas.directory.name} · {len(atlas.parcels)} regions · "
                                 f"{atlas.metadata.get('geometry', {}).get('frame', '')}")
        self._neutral("Atlas loaded; load recording channels for Live playback")
        self._plot_results()
        return True

    def _project_matches(self, directory):
        if self.atlas is None or directory is None:
            return False
        try:
            return json.loads((directory / "metadata.json").read_text()) == self.atlas.metadata
        except (OSError, ValueError):
            return False

    def _choose_mapping(self):
        path, _ = QFileDialog.getOpenFileName(self._root, "Parcel ID → signal name JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            mapping = json.loads(Path(path).read_text())
            if not isinstance(mapping, dict) or not all(isinstance(v, str) for v in mapping.values()):
                raise ValueError("Expected a JSON object mapping parcel IDs to signal names")
            if self.atlas:
                if not set(map(int, mapping)) <= {p.id for p in self.atlas.parcels}:
                    raise ValueError("Mapping contains unknown parcel IDs")
                self.atlas.match([], mapping=mapping)
            self.mapping = mapping
            self._manual_mapping = True
            self._mapping_changed()
        except Exception as error:
            self.status.setText(f"Cannot load mapping: {error}")

    def _mapping_changed(self):
        if not self._manual_mapping:
            directory, mapping, units = self._project_snapshot
            if not self.prefix.text() and self._project_matches(directory):
                self.mapping, self.project_units = mapping, units
            else:
                self.mapping, self.project_units = {}, {}
        self.recording_changed()
        self._scale_ready = False
        self._plot_results()

    def results_changed(self):
        # Commands may replace samples while retaining the same lunapi instance.
        self.recording_changed()
        self.refresh_tables(capture_project=True)

    def refresh_tables(self, *, capture_project=False):
        results = getattr(self.ctrl, "results", None) or {}
        resource_errors = []
        try:
            # Only new command output supplies recording provenance. Merely
            # reopening Cortex must not revive a previous recording's mapping.
            if capture_project:
                snapshot = project_resources(results)
                if snapshot[0] or snapshot[1]:
                    self._project_snapshot = snapshot
        except Exception as error:
            self._project_snapshot = (None, {}, {})
            if not self._manual_mapping:
                self.mapping, self.project_units = {}, {}
            resource_errors.append(str(error))
        if not self._ready:
            return
        current = self.table.currentData()
        self.table.blockSignals(True)
        self.table.clear()
        for key, frame in results.items():
            if frame is not None and "CH" in frame and not any(c in frame for c in ("CH1", "CH2")):
                self.table.addItem(str(key), key)
        self.table.setCurrentIndex(max(0, self.table.findData(current)))
        self.table.blockSignals(False)
        try:
            directory, mapping, units = self._project_snapshot
            if not self._manual_atlas and directory and directory != self._project_directory:
                if not self._set_atlas(directory):
                    raise ValueError(f"Cannot load PROJECT atlas at {directory}; select its resource directory manually")
                if self.atlas and self.atlas.directory == directory:
                    self._project_directory = directory
            if (not self._manual_mapping and mapping and not self.prefix.text()
                    and self._project_matches(directory)):
                if mapping != self.mapping:
                    self.recording_changed()
                self.mapping = mapping
                self.project_units = units
        except Exception as error:
            resource_errors.append(str(error))
        self._table_changed()
        if resource_errors:
            self.status.setText("; ".join(resource_errors))

    def _frame(self):
        return (getattr(self.ctrl, "results", None) or {}).get(self.table.currentData())

    def _table_changed(self, *_):
        frame = self._frame()
        self.variable.blockSignals(True)
        old_variable = self.variable.currentText()
        self.variable.clear()
        self.step_column.blockSignals(True)
        self.step_column.clear()
        self.step_column.addItem("(none)", None)
        while self.filters.rowCount():
            self.filters.removeRow(0)
        self.filter_widgets = {}
        if frame is not None:
            for col in frame:
                if col != "CH" and pd.api.types.is_numeric_dtype(frame[col]):
                    self.variable.addItem(str(col), col)
                if col != "CH" and (not pd.api.types.is_numeric_dtype(frame[col])
                        or col in ("ID", "STAGE", "SS", "E", "EPOCH", "BAND")):
                    values = frame[col].drop_duplicates().tolist()
                    self.step_column.addItem(str(col), col)
                    widget = combo([])
                    widget.addItem("(all — select or step to filter)", None)
                    for index, value in enumerate(values):
                        widget.addItem(str(value), index)
                    widget.currentIndexChanged.connect(self._plot_results)
                    self.filters.addRow(str(col), widget)
                    self.filter_widgets[col] = (widget, values)
        self.variable.setCurrentIndex(max(0, self.variable.findText(old_variable)))
        self.variable.blockSignals(False)
        self.step_column.blockSignals(False)
        self._scale_ready = False
        variable = self.variable.currentData()
        if frame is not None and variable is not None and (not self._palette_initialized or old_variable != str(variable)):
            self.cmap.setCurrentText("RdBu_r" if (frame[variable] < 0).any() else "viridis")
            self._palette_initialized = True
        self._step_changed()

    def _step_changed(self, *_):
        self.step_value.blockSignals(True)
        self.step_value.clear()
        column = self.step_column.currentData()
        if column in self.filter_widgets:
            for index, value in enumerate(self.filter_widgets[column][1]):
                self.step_value.addItem(str(value), index)
        self.step_value.blockSignals(False)
        self._plot_results()

    def _step(self, direction):
        if self.step_value.count():
            self.step_value.setCurrentIndex((self.step_value.currentIndex() + direction) % self.step_value.count())

    def _result_metric_changed(self, *_):
        self._scale_ready = False
        frame, variable = self._frame(), self.variable.currentData()
        if frame is not None and variable is not None:
            self.cmap.setCurrentText("RdBu_r" if (frame[variable] < 0).any() else "viridis")
        self._plot_results()

    def _plot_results(self, *_):
        if self.atlas is None or self.view is None or self.panels.currentWidget() is not self.results_panel:
            return
        frame, variable = self._frame(), self.variable.currentData()
        if frame is None or variable is None:
            self._neutral("Select a scalar-per-region result table")
            return
        filters = {col: values[widget.currentData()] for col, (widget, values) in self.filter_widgets.items()
                   if widget.currentData() is not None}
        step = self.step_column.currentData()
        if step is not None and self.step_value.currentData() is not None:
            filters[step] = self.filter_widgets[step][1][self.step_value.currentData()]
        try:
            values, units = result_values(self.atlas, frame, variable, filters=filters,
                aggregation=None if self.aggregate.currentIndex() == 0 else self.aggregate.currentText(),
                prefix=self.prefix.text(), mapping=self.mapping)
            self._show_values(values, units or self.metric_units.text())
            self.status.setText(f"{np.isfinite(values).sum()}/{len(values)} regions available · {variable}")
        except Exception as error:
            self._neutral(str(error))

    def _invalidate(self):
        self.pause_live_playback()
        self._generation += 1
        self._cancelled.set()
        if self._future is not None:
            self._future.cancel()
        self._cancelled = Event()
        self._pending = None
        self.cache = ChunkCache()

    def recording_changed(self, clear_resources=False):
        self._invalidate()
        # Before the controller detaches/reinitializes Luna, finish the current
        # bounded read. Cancellation is checked before every subsequent channel.
        if self._future is not None:
            try:
                self._future.result()
            except Exception:
                pass
        self._future = None
        self._info = self._recording = None
        if clear_resources:
            self._project_snapshot = (None, {}, {})
            self._project_directory = None
            self.project_units = {}
            if not self._manual_mapping:
                self.mapping = {}
        self.clock.seek(0.)
        self._scale_ready = False
        if self.atlas is not None and self.view is not None:
            self._neutral("Recording changed; load recording channels for Live playback")

    def _submit(self, kind, key, function, *args):
        generation = self._generation
        self._pending = (kind, key)
        self._future = self.ctrl._exec.submit(function, *args)
        def done(future):
            try:
                payload, error = future.result(), None
            except Exception as exc:
                payload, error = None, str(exc)
            self.delivered.emit((generation, kind, key, payload, error))
        self._future.add_done_callback(done)

    def _load_recording(self):
        self.recording_changed()
        if self.atlas is None or getattr(self.ctrl, "p", None) is None:
            self.status.setText("Load an atlas and a recording first")
            return
        if getattr(self.ctrl, "_busy", False):
            self.status.setText("Wait for the current analysis to finish")
            return
        self._recording = self.ctrl.p
        self.status.setText("Matching regional signals…")
        self._submit("info", None, recording_info, self._recording, self.atlas, self.prefix.text(), dict(self.mapping))

    def _receive(self, message):
        generation, kind, key, payload, error = message
        if generation != self._generation:
            return
        self._pending = None
        if error:
            self.pause_live_playback()
            self._neutral(error)
            return
        if kind == "info":
            self._info = payload
            names, units, segments = payload
            if not any(names):
                self._info = None
                self._neutral("No recording signals match the atlas luna_signal names or explicit mapping")
                return
            self._duration = float(segments[:, 1].max())
            self._tick()
        elif payload is not None:
            self.cache.put(key, payload)
            self._tick()

    def _metric_changed(self, *_):
        self._invalidate()
        self._scale_ready = False
        signed = self.mode.currentText() == "Signed amplitude"
        self.cmap.setCurrentText("RdBu_r" if signed else "viridis")
        self._tick()

    def _play(self):
        if self._info is None:
            self._load_recording()
            return
        self.clock.play(float(self.speed.currentText()))
        self.timer.start()
        self._tick()

    def pause_live_playback(self):
        self.clock.pause()
        self.timer.stop()

    def _stop(self):
        self._invalidate()
        self.clock.seek(0.)
        self._tick()

    def _speed_changed(self, *_):
        if self.clock.started is not None:
            self.clock.play(float(self.speed.currentText()))

    def _scrub(self, value):
        if self._info:
            self._invalidate()
            self.clock.seek(value / 100000 * self._duration)
            self._tick()

    def _tick(self):
        if self._info is None or self.panels.currentWidget() is not self.live_panel:
            return
        if self._recording is not getattr(self.ctrl, "p", None):
            self.recording_changed()
            return
        cursor = min(self.clock.position(), self._duration)
        if cursor >= self._duration:
            self.pause_live_playback()
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(int(cursor / max(self._duration, 1e-9) * 100000))
        self.scrubber.blockSignals(False)
        half = 0 if self.mode.currentText() == "Signed amplitude" else self.window.value() / 2
        self.time_label.setText(f"{cursor:.2f} / {self._duration:.2f} s · window {cursor-half:.2f}–{cursor+half:.2f} s")
        index = int(cursor // 10)
        block = self.cache.get(index)
        if block is not None and len(block.cursors):
            loc = min(len(block.cursors) - 1, max(0, int(round((cursor - index * 10) * 25))))
            summarized = float(block.cursors[loc])
            self.time_label.setText(f"{cursor:.2f} / {self._duration:.2f} s · "
                                    f"window {summarized-half:.2f}–{summarized+half:.2f} s")
            values = block.values[loc].astype(float)
            names, units, _ = self._info
            # Header units have priority; PROJECT stored units fill absent headers.
            effective = [signal_unit(u, self.project_units.get(n.casefold(), ""))
                         for n, u in zip(names, units) if n]
            unique = set(effective)
            if len(unique) > 1:
                self.pause_live_playback()
                self._neutral("Regional signals have different units; convert them before playback")
                return
            unit = next(iter(unique), "unknown")
            if self.mode.currentText() == "Band power":
                unit = f"({unit})²"
                if self.log_power.isChecked():
                    values = np.log10(np.where(values > 0, values, np.nan))
                    unit = f"log10[{unit}]"
            self._show_values(values, unit)
            count = np.isfinite(values).sum()
            matched = sum(name is not None for name in names)
            reason = ""
            if not count:
                if summarized - half < self._info[2][0, 0]:
                    reason = " · Window starts before recording; advance the timeline"
                else:
                    reason = " · Gap or insufficient analysis window"
            self.status.setText(f"{matched}/{len(values)} signals matched · "
                                f"{count}/{len(values)} valid regions · {self.mode.currentText()}" + reason +
                                (" · DYNAMIC LIMITS" if self.dynamic.isChecked() else " · fixed limits"))
            self._update_trace(block)
        else:
            self._neutral("Buffering regional signals…" if cursor < self._duration else "End of recording")
        if self._pending is None and cursor < self._duration and not getattr(self.ctrl, "_busy", False):
            request = index if block is None else index + 1
            if request * 10 < self._duration and self.cache.get(request) is None:
                names, _, segments = self._info
                self._submit("chunk", request, fetch_chunk, self._recording, names, segments,
                    request, self.mode.currentText(), self.window.value(),
                    (self.band_low.value(), self.band_high.value()), self._cancelled)

    def _panel_changed(self, *_):
        self.pause_live_playback()
        self._scale_ready = False
        self.trace.hide()
        if self.panels.currentWidget() is self.results_panel:
            self._plot_results()
        else:
            self.cmap.setCurrentText("RdBu_r" if self.mode.currentText() == "Signed amplitude" else "viridis")
            if self._info:
                self._tick()
            elif self.atlas:
                self._neutral("Load recording channels to begin recorded-data playback")

    def _display_changed(self, *_):
        if self.atlas and self.surface.currentText() in self.atlas.surfaces:
            self.view.set_display(self.hemi.currentText(), self.surface.currentText())

    def _show_values(self, values, units):
        self._values, self._units = values, units
        if self.dynamic.isChecked() or not self._scale_ready:
            self._auto_limits()
        self._recolor()

    def _neutral(self, message):
        if self.atlas:
            self._values = np.full(len(self.atlas.parcels), np.nan)
            self._units = getattr(self, "_units", "")
            self._recolor()
        self.status.setText(message)
        if hasattr(self, "trace"):
            self.trace.clear()

    def _auto_limits(self, *_):
        if not hasattr(self, "_values"):
            return
        finite = self._values[np.isfinite(self._values)]
        if not len(finite):
            return
        lo, hi = np.percentile(finite, [2, 98])
        signed = self.cmap.currentText() in ("RdBu_r", "coolwarm")
        if signed:
            hi = max(abs(lo), abs(hi), 1e-12)
            lo = -hi
        else:
            lo = min(0., lo)
            hi = max(hi, lo + 1e-12)
        for widget, value in ((self.low, lo), (self.high, hi)):
            widget.blockSignals(True)
            widget.setValue(float(value))
            widget.blockSignals(False)
        self._scale_ready = True
        self._recolor()

    def _manual_limits(self, *_):
        self._scale_ready = True
        self._recolor()

    def _recolor(self, *_):
        if self.atlas is None or not hasattr(self, "_values"):
            return
        if self.high.value() <= self.low.value():
            self.status.setText("Upper color limit must exceed lower limit")
            return
        self.view.set_values(self._values, self.cmap.currentText(), (self.low.value(), self.high.value()), self._units)
        self._update_detail()

    def _identify(self, index, clicked):
        if clicked:
            self._selected = index
        self._hover = index
        self._update_detail()
        if clicked and self._info:
            block = self.cache.get(int(self.clock.position() // 10))
            if block:
                self._update_trace(block)

    def _update_detail(self):
        index = self._hover if self._hover >= 0 else self._selected
        if index < 0 or self.atlas is None:
            self.detail.setText("Neutral / medial wall · click a region to select it")
            return
        parcel = self.atlas.parcels[index]
        value = self.view.values[index]
        label = self.mapping.get(parcel.id, self.mapping.get(str(parcel.id), self.prefix.text() + parcel.signal))
        if self.panels.currentWidget() is self.live_panel and self._info:
            label = self._info[0][index] or label
        self.detail.setText(f"{'Selected' if index == self._selected else 'Hover'}: {parcel.name} · {parcel.hemisphere} · "
                            f"{label} · {value:.5g} {self._units}" if np.isfinite(value) else
                            f"{parcel.name} · {parcel.hemisphere} · {label} · unavailable")

    def _trace_changed(self, *_):
        self.trace.setVisible(self.show_trace.isChecked() and self.panels.currentWidget() is self.live_panel)
        self._tick()

    def _update_trace(self, block):
        show = self.show_trace.isChecked() and self.panels.currentWidget() is self.live_panel
        self.trace.setVisible(show)
        if not show:
            return
        self.trace.clear()
        if self._selected in block.traces:
            times, samples = block.traces[self._selected]
            self.trace.plot(times, samples, pen="#58a6ff", connect="finite")
            name = self._info[0][self._selected]
            units = signal_unit(self._info[1][self._selected], self.project_units.get(name.casefold(), ""))
            self.trace.setLabel("left", self.atlas.parcels[self._selected].name, units=units)
            self.trace.setLabel("bottom", "Recording time", units="s")
