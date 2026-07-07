
#  --------------------------------------------------------------------
#
#  This file is part of Luna.
#
#  LUNA is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Luna is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Luna. If not, see <http:#www.gnu.org/licenses/>.
#
#  Please see LICENSE.txt for more details.
#
#  --------------------------------------------------------------------

"""Snapshot dock — record/replay named views of the current subject.

Each recorded row is a lightweight "recipe" (subject ID, time window,
shown channels, per-channel display filters, scaling settings, shown
annotation classes, free-text note) that can be replayed later by
re-attaching the subject and re-applying that view state. A cached PNG
grab of the live plot is stored alongside each row as a guaranteed-
faithful fallback preview, since the recipe alone cannot reproduce
signals that were transformed via ad hoc Luna commands elsewhere in the
app (there is no command log to replay).
"""

import json
import shutil
import time
import uuid
from pathlib import Path

import pandas as pd

from PySide6.QtCore import QEvent, QModelIndex, QObject, QSortFilterProxyModel, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QPixmap, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView, QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSizePolicy, QSplitter, QTableView, QVBoxLayout,
    QWidget,
)

from ..helpers import AuxiliaryWindow, is_dark_palette, screen_clamp
from ..runtime_paths import app_cache_root
from ..file_dialogs import open_file_name, save_file_name
from .tbl_funcs import attach_comma_filter


def _snapshot_cache_root() -> Path:
    # A single, persistent, shared cache — not scoped to an app launch or a
    # cohort. It survives restarts; "Source" is a separate, purely
    # user-facing tag (see below) and is never used to partition this
    # directory. "Clear ALL Cached Images…" is the deliberate escape hatch
    # for reclaiming disk space, rather than an automatic per-session purge.
    return app_cache_root() / "snapshots"


def _snapshot_autosave_path() -> Path:
    # A *sibling* of _snapshot_cache_root(), not inside it — "Clear ALL
    # Cached Images" recursively wipes that whole folder, and the autosaved
    # table (the actual recipes) must never be collateral damage of a
    # "just clear the preview images" action.
    return app_cache_root() / "snapshots_autosave.tsv"


class _SnapshotPreviewLabel(QLabel):
    """Preview label that rescales its cached pixmap on resize (so dragging
    the splitter or resizing the dock actually gives a bigger image) and
    opens the full-resolution PNG in the system viewer on double-click."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._raw_pixmap = None
        self._image_path = ""
        self.setCursor(Qt.PointingHandCursor)

    def set_preview(self, pixmap: QPixmap, image_path: str = ""):
        self._raw_pixmap = pixmap
        self._image_path = image_path or ""
        self.setCursor(Qt.PointingHandCursor)
        self._rescale()

    def clear_preview(self, text: str = "(no preview)"):
        self._raw_pixmap = None
        self._image_path = ""
        self.setCursor(Qt.ArrowCursor)
        self.setPixmap(QPixmap())
        self.setText(text)

    def _rescale(self):
        if self._raw_pixmap is None or self._raw_pixmap.isNull():
            return
        self.setPixmap(
            self._raw_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def mouseDoubleClickEvent(self, event):
        if self._image_path and Path(self._image_path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._image_path))
        super().mouseDoubleClickEvent(event)


class _SnapshotHypnoCaptureFilter(QObject):
    """Mirrors the trace canvas's own `C` handling (MainTraceNavProxy in
    signals.py) but installed on the hypnogram/navigator widget (`pgh`)
    instead — so Capture also works with focus there, matching the two
    widgets a user actually browses with. Narrowly scoped: only events
    delivered to this specific widget (and its viewport) are ever seen
    here, unlike an application-wide filter."""

    def __init__(self, plot, owner):
        super().__init__(plot)
        self.plot = plot
        self.owner = owner
        self.plot.setFocusPolicy(Qt.StrongFocus)
        self.plot.installEventFilter(self)
        viewport = self.plot.viewport() if hasattr(self.plot, "viewport") else None
        if viewport is not None:
            viewport.setFocusPolicy(Qt.StrongFocus)
            viewport.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() != QEvent.KeyPress:
            return False
        if event.key() != Qt.Key_C or event.isAutoRepeat():
            return False
        if event.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
            return False
        if not self.owner._snapshot_mode_active():
            return False
        if self.owner._snapshot_record_current_view():
            return True
        return False


class SnapshotMixin:
    """Adds the Snapshot floating dock/mode to the Lunascope controller."""

    _SNAPSHOT_FLOAT_SIZE = (760, 560)

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_snapshot(self):
        self._snapshot_mode_on = False
        self._snapshot_rows: list[dict] = []
        self._snapshot_nav_guard = False  # re-entrancy guard while replaying
        self._snapshot_editing_row_idx = None  # row the note/source fields are editing, if any

        # C also works with focus on the hypnogram/navigator (pgh) — the
        # trace canvas (pg1) already handles it via MainTraceNavProxy in
        # signals.py.
        pgh = getattr(self.ui, "pgh", None)
        if pgh is not None:
            self._snapshot_hypno_capture = _SnapshotHypnoCaptureFilter(pgh, self)

        dock = AuxiliaryWindow("Snapshots", self.ui)
        dock.setObjectName("dock_snapshot")
        dock.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        dock.setWindowFlag(Qt.WindowMaximizeButtonHint, True)

        root = QWidget(dock)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # ---- note + source + record row ------------------------------
        note_frame = QFrame(root)
        nl = QHBoxLayout(note_frame)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.setSpacing(6)
        note_lbl = QLabel("Note:")
        note_edit = QLineEdit()
        note_edit.setObjectName("snapshot_note_edit")
        note_edit.setPlaceholderText("e.g. spindle, artifact…")
        note_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        source_lbl = QLabel("Source:")
        source_edit = QLineEdit()
        source_edit.setObjectName("snapshot_source_edit")
        source_edit.setPlaceholderText("e.g. cohort/sample-list name (your own label)")
        source_edit.setToolTip(
            "A free-text tag for grouping/filtering your own snapshots — e.g.\n"
            "which cohort or sample list this came from. Defaults to the\n"
            "currently loaded sample list's label when left blank, but it's\n"
            "just a label: it is never used to look up or load a sample list."
        )
        source_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        record_btn = QPushButton("Capture  (C)")
        record_btn.setObjectName("snapshot_record_btn")
        record_btn.setToolTip(
            "Capture the current view (subject, window, channels, filters,\n"
            "scaling, shown annotations) as a new snapshot row."
        )
        nl.addWidget(source_lbl)
        nl.addWidget(source_edit)
        nl.addWidget(note_lbl)
        nl.addWidget(note_edit)
        nl.addWidget(record_btn)
        outer.addWidget(note_frame)

        # ---- status label -------------------------------------------
        status_lbl = QLabel("")
        status_lbl.setObjectName("snapshot_status_lbl")
        status_lbl.setWordWrap(True)
        status_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        outer.addWidget(status_lbl)

        # ---- filter row ------------------------------------------------
        filter_frame = QFrame(root)
        fl = QHBoxLayout(filter_frame)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(6)
        filter_lbl = QLabel("Filter:")
        filter_edit = QLineEdit()
        filter_edit.setObjectName("snapshot_filter_edit")
        filter_edit.setPlaceholderText("Filter rows by any column (comma = OR)…")
        filter_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        fl.addWidget(filter_lbl)
        fl.addWidget(filter_edit)
        outer.addWidget(filter_frame)

        # ---- table ------------------------------------------------------
        model = QStandardItemModel(0, 6, root)
        model.setHorizontalHeaderLabels(
            ["Source", "ID", "Start (s)", "Dur (s)", "Channels", "Note"]
        )
        table = QTableView(root)
        table.setObjectName("snapshot_table")
        table.setModel(model)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # ID column (1) auto-sizes to fit whatever ID strings are actually
        # present, so long subject IDs don't get truncated.
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        # Wraps `model` in a QSortFilterProxyModel (matches any column,
        # case-insensitive, comma-separated OR terms) — same helper already
        # used for the signals/annotations tables. table.model() is the
        # proxy from here on; `model` (the source) is what _snapshot_rows
        # indices always refer to.
        attach_comma_filter(table, filter_edit)

        # ---- preview pane -------------------------------------------------
        preview_lbl = _SnapshotPreviewLabel()
        preview_lbl.setObjectName("snapshot_preview_lbl")
        preview_lbl.setAlignment(Qt.AlignCenter)
        preview_lbl.setMinimumHeight(120)
        preview_lbl.setFrameShape(QFrame.StyledPanel)
        preview_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_lbl.setToolTip("Double-click to open the full-resolution image")
        preview_lbl.setText("(no preview)")

        # Splitter so the user can drag to give the image more (or less)
        # room — resizing the dock alone previously did nothing for it.
        splitter = QSplitter(Qt.Vertical, root)
        splitter.setObjectName("snapshot_splitter")
        splitter.addWidget(table)
        splitter.addWidget(preview_lbl)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, 1)

        # ---- buttons row --------------------------------------------------
        btn_frame = QFrame(root)
        bl = QHBoxLayout(btn_frame)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(6)
        jump_btn = QPushButton("Jump to Live Data")
        jump_btn.setObjectName("snapshot_jump_btn")
        jump_btn.setToolTip(
            "Load the selected row's actual view (subject, window, channels,\n"
            "filters, scaling) — selecting a row only shows its cached image;\n"
            "this is the explicit step that switches the live viewer."
        )
        delete_btn = QPushButton("Delete Row")
        delete_btn.setObjectName("snapshot_delete_btn")
        delete_btn.setToolTip("Delete the selected row and its cached preview image.")
        delete_all_btn = QPushButton("Delete All…")
        delete_all_btn.setObjectName("snapshot_delete_all_btn")
        delete_all_btn.setToolTip(
            "Delete every row in this table and every cached preview image\n"
            "in the shared cache folder. Asks for confirmation first."
        )
        save_btn = QPushButton("Save Table…")
        save_btn.setObjectName("snapshot_save_btn")
        load_btn = QPushButton("Load Table…")
        load_btn.setObjectName("snapshot_load_btn")
        for b in (jump_btn, delete_btn, delete_all_btn, save_btn, load_btn):
            bl.addWidget(b)
        outer.addWidget(btn_frame)

        # ---- attach dock ----------------------------------------------
        dock.setWidget(root)
        _w, _h = screen_clamp(*self._SNAPSHOT_FLOAT_SIZE)
        dock.resize(_w, _h)
        parent_geo = self.ui.frameGeometry()
        dock.move(
            parent_geo.x() + (parent_geo.width() - _w) // 2,
            parent_geo.y() + 80,
        )
        dock.hide()

        # ---- store refs -----------------------------------------------
        self.ui.dock_snapshot = dock
        self.ui.snapshot_note_edit = note_edit
        self.ui.snapshot_source_edit = source_edit
        self.ui.snapshot_filter_edit = filter_edit
        self.ui.snapshot_record_btn = record_btn
        self.ui.snapshot_status_lbl = status_lbl
        self.ui.snapshot_table = table
        self.ui.snapshot_preview_lbl = preview_lbl
        self.ui.snapshot_jump_btn = jump_btn
        self.ui.snapshot_delete_btn = delete_btn
        self.ui.snapshot_delete_all_btn = delete_all_btn
        self.ui.snapshot_save_btn = save_btn
        self.ui.snapshot_load_btn = load_btn
        self.ui.snapshot_splitter = splitter
        self._snapshot_model = model

        if is_dark_palette():
            root.setStyleSheet("QLabel { color: #d7e3f4; }")

        # ---- wire signals -------------------------------------------------
        record_btn.clicked.connect(self._snapshot_record_current_view)
        delete_btn.clicked.connect(self._snapshot_delete_selected_row)
        delete_all_btn.clicked.connect(self._snapshot_delete_all)
        save_btn.clicked.connect(self._snapshot_save_table)
        load_btn.clicked.connect(self._snapshot_load_table)
        jump_btn.clicked.connect(self._snapshot_jump_to_live_selected)
        table.selectionModel().currentRowChanged.connect(self._snapshot_row_changed)
        note_edit.editingFinished.connect(self._snapshot_note_edited)
        source_edit.editingFinished.connect(self._snapshot_source_edited)
        # visibilityChanged (not the toggle action's `toggled`) — it fires
        # unconditionally on both show and hide, including a native close
        # via the window's own close button (AuxiliaryWindow.hideEvent
        # blocks the toggle action's own `toggled` signal while unchecking
        # it, so that one can't be relied on here).
        dock.visibilityChanged.connect(self._snapshot_toggle_mode)

        # Auto-restore whatever was there last time — the table should
        # survive an app restart without the user having to remember to
        # Save/Load manually.
        autosave_path = _snapshot_autosave_path()
        if autosave_path.exists():
            self._snapshot_read_tsv(autosave_path)
            self._snapshot_set_status(
                f"Restored {len(self._snapshot_rows)} snapshot(s) from last session."
            )

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    def _snapshot_mode_active(self) -> bool:
        return bool(getattr(self, "_snapshot_mode_on", False) and hasattr(self, "p"))

    def _snapshot_toggle_mode(self, visible: bool):
        # Driven by the dock's own visibilityChanged — show/raise/hide is
        # already handled by AuxiliaryWindow itself (including when closed
        # via its native close button), so this only tracks mode state.
        self._snapshot_mode_on = bool(visible)
        if self._snapshot_mode_on:
            self._snapshot_set_status(
                "Snapshot mode ON — press C over the trace view or "
                "hypnogram to capture."
            )
        else:
            self._snapshot_set_status("Snapshot mode off.")

    def _snapshot_set_status(self, text: str):
        lbl = getattr(self.ui, "snapshot_status_lbl", None)
        if lbl is not None:
            lbl.setText(text)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _snapshot_record_current_view(self) -> bool:
        if not self._snapshot_mode_active():
            self._snapshot_set_status("Snapshot mode is off.")
            return False
        if not hasattr(self, "p"):
            self._snapshot_set_status("Nothing to record — no subject attached.")
            return False

        try:
            id_str = self.p.id()
        except Exception:
            self._snapshot_set_status("Could not read current subject ID.")
            return False

        lo, hi = self._annotator_current_range()

        channels = list(self.ui.tbl_desc_signals.checked())
        if not channels:
            self._snapshot_set_status("No channels are currently shown — nothing recorded.")
            return False
        annots = (
            list(self.ui.tbl_desc_annots.checked())
            if hasattr(self.ui, "tbl_desc_annots") else []
        )

        fmap = getattr(self, "fmap", {}) or {}
        user_fmap_frqs = getattr(self, "user_fmap_frqs", {}) or {}
        filters = {ch: fmap[ch] for ch in channels if ch in fmap}
        user_frqs = {ch: list(user_fmap_frqs[ch]) for ch in filters if ch in user_fmap_frqs}

        scaling = self._snapshot_read_scaling()
        note = self.ui.snapshot_note_edit.text().strip()

        # Source is a loose, user-owned tag (never a path/identity key — see
        # plan). Seed it from the currently-loaded sample list's own label
        # only when the box is blank, and write that suggestion back into
        # the box so it's visible/editable and stays sticky for next time.
        source = self.ui.snapshot_source_edit.text().strip()
        if not source and hasattr(self, "_get_slist_label_full_text"):
            try:
                suggested = self._get_slist_label_full_text()
            except Exception:
                suggested = ""
            if suggested and suggested != "(none)":
                source = suggested
                self.ui.snapshot_source_edit.setText(source)

        row_idx = len(self._snapshot_rows)
        image_path = self._snapshot_capture_image()

        row = {
            "id": id_str,
            "source": source,
            "t1": float(lo),
            "t2": float(hi),
            "channels": channels,
            "annots": annots,
            "filters": filters,
            "user_frqs": user_frqs,
            "scaling": scaling,
            "note": note,
            "image": str(image_path) if image_path else "",
            # Remember whether this view was a full "Render" or the
            # lightweight preview — replay should restore the same mode,
            # not silently downgrade a rendered snapshot to a preview.
            "rendered": bool(getattr(self, "rendered", False)),
        }
        self._snapshot_rows.append(row)
        self._snapshot_append_table_row(row)
        self._snapshot_set_status(f"Recorded snapshot #{row_idx + 1} for {id_str}.")
        self._snapshot_autosave()

        # The note/source boxes are no longer "editing" any previously-
        # selected row — otherwise focus later bouncing back out (e.g. to
        # click the trace canvas) would silently overwrite that older row's
        # note/source with whatever's now typed for the *next* capture.
        self._snapshot_editing_row_idx = None

        # Recording must not steal keyboard focus from the trace canvas —
        # otherwise a second `C` press in a row silently goes nowhere.
        pw = getattr(self.ui, "pg1", None)
        if pw is not None:
            pw.setFocus()
        return True

    def _snapshot_read_scaling(self) -> dict:
        ui = self.ui
        mode = "default"
        if getattr(ui, "radio_fixedscale", None) is not None and ui.radio_fixedscale.isChecked():
            mode = "fixed"
        elif getattr(ui, "radio_empiric", None) is not None and ui.radio_empiric.isChecked():
            mode = "empiric"
        elif getattr(ui, "radio_clip", None) is not None and ui.radio_clip.isChecked():
            mode = "clip"
        scaling = {"mode": mode}
        for key, attr in (
            ("spacing", "spin_spacing"),
            ("scale", "spin_scale"),
            ("fixed_min", "spin_fixed_min"),
            ("fixed_max", "spin_fixed_max"),
        ):
            w = getattr(ui, attr, None)
            if w is not None:
                try:
                    scaling[key] = w.value()
                except Exception:
                    pass
        return scaling

    def _snapshot_apply_scaling(self, scaling: dict):
        ui = self.ui
        mode = (scaling or {}).get("mode")
        if mode == "fixed" and getattr(ui, "radio_fixedscale", None) is not None:
            ui.radio_fixedscale.setChecked(True)
        elif mode == "empiric" and getattr(ui, "radio_empiric", None) is not None:
            ui.radio_empiric.setChecked(True)
        elif mode == "clip" and getattr(ui, "radio_clip", None) is not None:
            ui.radio_clip.setChecked(True)
        for key, attr in (
            ("spacing", "spin_spacing"),
            ("scale", "spin_scale"),
            ("fixed_min", "spin_fixed_min"),
            ("fixed_max", "spin_fixed_max"),
        ):
            if key not in (scaling or {}):
                continue
            w = getattr(ui, attr, None)
            if w is not None:
                try:
                    w.setValue(scaling[key])
                except Exception:
                    pass
        if hasattr(self, "_update_scaling"):
            self._update_scaling()

    def _snapshot_capture_image(self):
        pw = getattr(self.ui, "pg1", None)
        if pw is None:
            return None
        try:
            pixmap = pw.grab()
            if pixmap.isNull():
                return None
        except Exception:
            return None
        cache_dir = _snapshot_cache_root()
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Globally unique — this directory is now shared across every
            # session/table forever, so a simple per-session row index is
            # no longer safe as a filename.
            fname = f"snap_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.png"
            path = cache_dir / fname
            pixmap.save(str(path), "PNG")
            return path
        except Exception:
            return None

    def _snapshot_append_table_row(self, row: dict):
        model = self._snapshot_model
        items = [
            QStandardItem(row.get("source", "")),
            QStandardItem(row["id"]),
            QStandardItem(f"{row['t1']:.1f}"),
            QStandardItem(f"{row['t2'] - row['t1']:.1f}"),
            QStandardItem(", ".join(row["channels"])),
            QStandardItem(row["note"]),
        ]
        for it in items:
            it.setEditable(False)
        model.appendRow(items)
        n = model.rowCount()
        table = self.ui.snapshot_table

        # table.model() is a filter proxy (attach_comma_filter) — map the
        # newly-appended *source* row into proxy space before selecting it.
        # If an active filter hides the new row, there's nothing valid to
        # select; the preview below still shows regardless.
        view_model = table.model()
        src_index = model.index(n - 1, 0)
        target_index = (
            view_model.mapFromSource(src_index)
            if isinstance(view_model, QSortFilterProxyModel)
            else src_index
        )
        if target_index.isValid():
            # Auto-selecting the row we just recorded must not re-trigger a
            # "jump to live" replay of the view we're already looking at —
            # block the selection signal for this one selectRow() call.
            sel_model = table.selectionModel()
            blocked = sel_model.blockSignals(True) if sel_model is not None else False
            try:
                table.selectRow(target_index.row())
            finally:
                if sel_model is not None:
                    sel_model.blockSignals(blocked)
        self._snapshot_show_preview(row)

    # ------------------------------------------------------------------
    # Preview / navigation
    # ------------------------------------------------------------------

    def _snapshot_source_row(self, index: QModelIndex) -> int:
        """Map a QModelIndex from the table (which sits behind a filter
        proxy) to a row index into self._snapshot_rows / self._snapshot_model.
        Returns -1 if invalid (e.g. filtered out, or no selection)."""
        if not index.isValid():
            return -1
        model = index.model()
        if isinstance(model, QSortFilterProxyModel):
            index = model.mapToSource(index)
            if not index.isValid():
                return -1
        return index.row()

    def _snapshot_show_preview(self, row: dict):
        lbl = getattr(self.ui, "snapshot_preview_lbl", None)
        if lbl is None:
            return
        img = (row or {}).get("image") or ""
        if img and Path(img).exists():
            pixmap = QPixmap(img)
            if not pixmap.isNull():
                lbl.set_preview(pixmap, img)
                return
        lbl.clear_preview()

    def _snapshot_row_changed(self, current: QModelIndex, _prev: QModelIndex):
        row_idx = self._snapshot_source_row(current)
        if row_idx < 0 or row_idx >= len(self._snapshot_rows):
            return
        row = self._snapshot_rows[row_idx]
        # Selecting a row only shows its cached image — it does NOT jump to
        # the live view. A live jump (subject switch + possible re-render)
        # can be slow, and doing it on every row you scan through makes
        # browsing the table sluggish. Click "Jump to Live Data" to load
        # the actual view for whatever row you land on.
        self._snapshot_show_preview(row)
        # Selecting a row also puts the note/source boxes into "edit this
        # row" mode — see _snapshot_note_edited()/_snapshot_source_edited().
        self._snapshot_editing_row_idx = row_idx
        self.ui.snapshot_note_edit.setText(row.get("note", ""))
        self.ui.snapshot_source_edit.setText(row.get("source", ""))
        self._snapshot_set_status(
            f"Showing cached image for {row['id']} "
            "(click “Jump to Live Data” to load the live view)."
        )

    def _snapshot_note_edited(self):
        idx = self._snapshot_editing_row_idx
        if idx is None or not (0 <= idx < len(self._snapshot_rows)):
            return
        text = self.ui.snapshot_note_edit.text().strip()
        row = self._snapshot_rows[idx]
        if row.get("note", "") == text:
            return
        row["note"] = text
        item = self._snapshot_model.item(idx, 5)
        if item is not None:
            item.setText(text)
        self._snapshot_autosave()

    def _snapshot_source_edited(self):
        idx = self._snapshot_editing_row_idx
        if idx is None or not (0 <= idx < len(self._snapshot_rows)):
            return
        text = self.ui.snapshot_source_edit.text().strip()
        row = self._snapshot_rows[idx]
        if row.get("source", "") == text:
            return
        row["source"] = text
        item = self._snapshot_model.item(idx, 0)
        if item is not None:
            item.setText(text)
        self._snapshot_autosave()

    def _snapshot_jump_to_live_selected(self):
        row_idx = self._snapshot_source_row(self.ui.snapshot_table.currentIndex())
        if 0 <= row_idx < len(self._snapshot_rows):
            self._snapshot_jump_to_live(self._snapshot_rows[row_idx])

    def _snapshot_jump_to_live(self, row: dict):
        if self._snapshot_nav_guard or not hasattr(self, "proj"):
            return
        self._snapshot_nav_guard = True
        try:
            id_str = row["id"]
            current_id = None
            if hasattr(self, "p"):
                try:
                    current_id = self.p.id()
                except Exception:
                    current_id = None

            if current_id != id_str:
                ok = False
                try:
                    ok = bool(self._restore_attached_subject(id_str))
                except Exception:
                    ok = False
                if not ok:
                    self._snapshot_set_status(
                        f"Subject '{id_str}' not found in the current cohort — "
                        "showing cached image only."
                    )
                    return

            wanted = set(row.get("channels", []))
            existing = self._snapshot_available_channel_labels()
            present = wanted & existing
            missing = wanted - existing

            if not present:
                self._snapshot_set_status(
                    "None of this snapshot's channels are present now "
                    "(likely derived/transformed in the original session) — "
                    "showing cached image only."
                )
                return

            self.ui.tbl_desc_signals.set_checked_by_labels(present)

            if not hasattr(self, "fmap"):
                self.fmap = {}
            if not hasattr(self, "user_fmap_frqs"):
                self.user_fmap_frqs = {}
            for ch, code in row.get("filters", {}).items():
                if ch in present:
                    self.fmap[ch] = code
            for ch, frqs in row.get("user_frqs", {}).items():
                if ch in present:
                    self.user_fmap_frqs[ch] = list(frqs)

            if hasattr(self.ui, "tbl_desc_annots"):
                wanted_annots = set(row.get("annots", []))
                existing_annots = self._snapshot_available_annot_labels()
                self.ui.tbl_desc_annots.set_checked_by_labels(wanted_annots & existing_annots)

            # `set_checked_by_labels` only *schedules* its curve-rebuild via a
            # debounced 0ms QTimer (tbl_funcs.py) — too unreliable to wait on.
            # `_clear_pg1()` is what that debounce eventually calls anyway
            # (via `_redraw_signal_plot_if_ready`); calling it here directly
            # rebuilds self.curves synchronously from whatever's now checked,
            # so the scaling/window-range calls below never see a stale list.
            if hasattr(self, "_clear_pg1"):
                self._clear_pg1()

            ns = float(getattr(self, "ns", 0.0) or 0.0)
            lo, hi = float(row["t1"]), float(row["t2"])
            clipped = False
            if ns > 0 and hi > ns:
                hi = ns
                clipped = True
            if lo < 0:
                lo = 0.0
                clipped = True
            if hi < lo:
                lo, hi = hi, lo

            self._snapshot_apply_scaling(row.get("scaling") or {})

            # A real (re-)render is needed unless we're *already* rendered
            # with every requested channel loaded — otherwise on_window_range()
            # alone would either silently clamp a wide window (simple mode
            # caps at max_simple_span) or silently skip drawing a channel
            # that isn't in self.ss_chs (signals.py:1921-1923), or — the
            # case this fixes — silently leave a snapshot that was
            # originally a full Render looking like a plain preview after
            # a subject switch, since _attach_inst() always resets
            # self.rendered to False. We track whether the snapshot being
            # replayed was itself a Render (row["rendered"]) precisely so
            # replay restores the same mode it was captured in, rather than
            # guessing from window width/channel overlap alone.
            max_simple_span = 3600.0 if getattr(self, "multiday_mode", False) else 30.0
            rendered_chs = set(map(str, getattr(self, "_rendered_chs", []) or getattr(self, "ss_chs", [])))
            already_matches = getattr(self, "rendered", False) and not bool(present - rendered_chs)
            window_needs_render = (hi - lo) > max_simple_span
            was_rendered = bool(row.get("rendered", False))
            needs_full_render = (
                hasattr(self, "_render_signals")
                and not already_matches
                and (window_needs_render or was_rendered)
            )

            if needs_full_render:
                self.sel.setRange(lo, hi, emit=False)
                self._render_signals()
                msg = f"Rendering full view for snapshot ({id_str})… this may take a moment."
                if missing:
                    msg += f" ({len(missing)} channel(s) not found and skipped.)"
                self._snapshot_set_status(msg)
            else:
                self.sel.setRange(lo, hi)
                self.on_window_range(lo, hi)

                if missing:
                    self._snapshot_set_status(
                        f"{len(missing)} channel(s) not found and skipped: "
                        f"{', '.join(sorted(missing))}"
                    )
                elif clipped:
                    self._snapshot_set_status(
                        "Recorded window extended past the current recording — clipped."
                    )
                else:
                    self._snapshot_set_status(f"Jumped to snapshot for {id_str}.")
        finally:
            self._snapshot_nav_guard = False

    def _snapshot_labels_from_view(self, view) -> set:
        try:
            model = view.model()
            if isinstance(model, QSortFilterProxyModel):
                model = model.sourceModel()
            labels = set()
            for r in range(model.rowCount()):
                labels.add(str(model.data(model.index(r, 1))))
            return labels
        except Exception:
            return set()

    def _snapshot_available_channel_labels(self) -> set:
        return self._snapshot_labels_from_view(self.ui.tbl_desc_signals)

    def _snapshot_available_annot_labels(self) -> set:
        if not hasattr(self.ui, "tbl_desc_annots"):
            return set()
        return self._snapshot_labels_from_view(self.ui.tbl_desc_annots)

    # ------------------------------------------------------------------
    # Row management
    # ------------------------------------------------------------------

    def _snapshot_delete_selected_row(self):
        row_idx = self._snapshot_source_row(self.ui.snapshot_table.currentIndex())
        if not (0 <= row_idx < len(self._snapshot_rows)):
            return
        img = self._snapshot_rows[row_idx].get("image") or ""
        if img:
            try:
                Path(img).unlink(missing_ok=True)
            except Exception:
                pass
        self._snapshot_rows.pop(row_idx)
        self._snapshot_model.removeRow(row_idx)
        self._snapshot_editing_row_idx = None
        self._snapshot_show_preview({})
        self._snapshot_autosave()

    def _snapshot_delete_all(self):
        reply = QMessageBox.question(
            self.ui,
            "Delete all snapshots?",
            f"This deletes all {len(self._snapshot_rows)} row(s) in this table and\n"
            "every cached preview image in the shared cache folder — not just\n"
            "this table's images, but any left over from other saved tables\n"
            "too. This cannot be undone.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._snapshot_rows = []
        self._snapshot_model.removeRows(0, self._snapshot_model.rowCount())
        self._snapshot_editing_row_idx = None
        self._snapshot_show_preview({})

        cache_dir = _snapshot_cache_root()
        n_removed = 0
        if cache_dir.exists():
            # Recurse, not just top-level files: earlier versions of this
            # feature wrote into per-launch session_<timestamp> subfolders,
            # so anyone upgrading may have legacy nested folders here too.
            n_removed = sum(1 for f in cache_dir.rglob("*") if f.is_file())
            try:
                shutil.rmtree(cache_dir)
            except Exception:
                pass
            cache_dir.mkdir(parents=True, exist_ok=True)

        self._snapshot_set_status(
            f"Deleted all snapshot rows and {n_removed} cached image file(s)."
        )
        self._snapshot_autosave()

    # ------------------------------------------------------------------
    # Save / load table
    # ------------------------------------------------------------------

    def _snapshot_write_tsv(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("id\tsource\tstart\tstop\tchannels\tfilters\tscaling\tannots\tnote\timage\trendered\n")
            for row in self._snapshot_rows:
                fh.write(
                    "\t".join([
                        row["id"],
                        row.get("source", "").replace("\t", " "),
                        f"{row['t1']:.3f}",
                        f"{row['t2']:.3f}",
                        "|".join(row.get("channels", [])),
                        json.dumps(row.get("filters", {})),
                        json.dumps(row.get("scaling", {})),
                        "|".join(row.get("annots", [])),
                        row.get("note", "").replace("\t", " "),
                        row.get("image", ""),
                        "1" if row.get("rendered", False) else "0",
                    ]) + "\n"
                )

    def _snapshot_read_tsv(self, path) -> bool:
        """Replace the in-memory table with the contents of `path`. Returns
        False (leaving the table untouched) if the file can't be read."""
        try:
            df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception:
            return False

        self._snapshot_rows = []
        self._snapshot_model.removeRows(0, self._snapshot_model.rowCount())

        for _, r in df.iterrows():
            try:
                row = {
                    "id": str(r.get("id", "")),
                    "source": str(r.get("source", "")),
                    "t1": float(r.get("start", 0.0) or 0.0),
                    "t2": float(r.get("stop", 0.0) or 0.0),
                    "channels": [c for c in str(r.get("channels", "")).split("|") if c],
                    "filters": json.loads(r.get("filters") or "{}"),
                    "scaling": json.loads(r.get("scaling") or "{}"),
                    "annots": [a for a in str(r.get("annots", "")).split("|") if a],
                    "note": str(r.get("note", "")),
                    "image": str(r.get("image", "")),
                    "rendered": str(r.get("rendered", "0")).strip() == "1",
                }
            except Exception:
                continue
            self._snapshot_rows.append(row)
            self._snapshot_append_table_row(row)

        if self._snapshot_rows:
            self.ui.snapshot_note_edit.setText(self._snapshot_rows[-1]["note"])
            self.ui.snapshot_source_edit.setText(self._snapshot_rows[-1].get("source", ""))
        return True

    def _snapshot_autosave(self):
        """Persist the working table to a fixed location automatically, so
        it survives an app restart without the user having to remember to
        click Save. Deliberately a sibling of (not inside) the image cache
        folder, so "Clear ALL Cached Images" can never delete it."""
        try:
            _snapshot_cache_root().parent.mkdir(parents=True, exist_ok=True)
            self._snapshot_write_tsv(_snapshot_autosave_path())
        except Exception:
            pass

    def _snapshot_save_table(self):
        path, _ = save_file_name(
            self.ui, "Save Snapshot Table", "snapshots.tsv",
            "Snapshot table (*.tsv);;All files (*)",
        )
        if not path:
            return
        try:
            self._snapshot_write_tsv(path)
        except Exception as e:
            QMessageBox.critical(self.ui, "Error", f"Could not save snapshot table:\n{e}")
            return
        self._snapshot_set_status(f"Saved {len(self._snapshot_rows)} snapshot(s) to {path}")

    def _snapshot_load_table(self):
        path, _ = open_file_name(
            self.ui, "Load Snapshot Table", "",
            "Snapshot table (*.tsv);;All files (*)",
        )
        if not path:
            return
        if not self._snapshot_read_tsv(path):
            QMessageBox.critical(self.ui, "Error", f"Could not load snapshot table:\n{path}")
            return
        self._snapshot_set_status(f"Loaded {len(self._snapshot_rows)} snapshot(s) from {path}")
        self._snapshot_autosave()
