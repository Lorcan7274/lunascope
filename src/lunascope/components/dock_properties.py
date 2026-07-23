"""Properties dialogs and context menus for the signal/annotation docks."""

from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QBrush, QIcon, QKeySequence, QPixmap, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QMessageBox,
)


_FILTER_NAMES = (
    "None", "0.3-35Hz", "Slow", "Delta", "Theta", "Alpha",
    "Sigma", "Beta", "Gamma", "User",
)
_UNCHANGED = object()
_COLOR_ROLE = Qt.UserRole + 41


def _is_checked_state(state):
    value = getattr(state, "value", state)
    checked = getattr(Qt.Checked, "value", Qt.Checked)
    partial = getattr(Qt.PartiallyChecked, "value", Qt.PartiallyChecked)
    return value in (checked, partial)


class _ColorCheckDelegate(QStyledItemDelegate):
    """Paint compact, clickable selection boxes using the item's color."""

    def paint(self, painter, option, index):
        if index.column() != 0:
            return super().paint(painter, option, index)
        painter.save()
        selected = bool(option.state & QStyle.State_Selected)
        bg = option.palette.highlight() if selected else option.palette.base()
        painter.fillRect(option.rect, bg)

        size = min(14, max(11, option.rect.height() - 4))
        box = option.rect.adjusted(
            (option.rect.width() - size) // 2,
            (option.rect.height() - size) // 2,
            -(option.rect.width() - size) // 2,
            -(option.rect.height() - size) // 2,
        )
        checked = _is_checked_state(index.data(Qt.CheckStateRole))
        value = index.data(_COLOR_ROLE)
        color = QColor(value) if value is not None else QColor("#1A1D21")
        painter.setBrush(color if checked and color.isValid() else QColor("#1A1D21"))
        painter.setPen(QPen(QColor("#FFFFFF") if checked else QColor("#8B949E"), 1))
        painter.drawRect(box)
        if checked:
            painter.setPen(QPen(QColor("#FFFFFF"), 1.4))
            x, y, w, h = box.x(), box.y(), box.width(), box.height()
            painter.drawLine(x + 3, y + h // 2, x + w // 2 - 1, y + h - 3)
            painter.drawLine(x + w // 2 - 1, y + h - 3, x + w - 2, y + 3)
        painter.restore()


def _source_model(view):
    model = view.model()
    if model is None:
        return None
    return model.sourceModel() if hasattr(model, "sourceModel") else model


def _header_column(model, names, fallback=1):
    if model is None:
        return fallback
    wanted = {str(x).upper() for x in names}
    for col in range(model.columnCount()):
        header = str(model.headerData(col, Qt.Horizontal) or "").strip().upper()
        if header in wanted:
            return col
    return fallback if model.columnCount() > fallback else 0


def _source_row_for_index(view, index):
    model = view.model()
    if model is None or not index.isValid():
        return None
    source_index = model.mapToSource(index) if hasattr(model, "mapToSource") else index
    return source_index.row() if source_index.isValid() else None


def _color_icon(value):
    if isinstance(value, QColor):
        color = QColor(value)
    elif isinstance(value, (tuple, list)) and len(value) in (3, 4):
        color = QColor(*[int(x) for x in value])
    else:
        color = QColor(str(value))
    if not color.isValid():
        return QIcon()
    pix = QPixmap(11, 11)
    pix.fill(color)
    painter = QPainter(pix)
    painter.setPen(QPen(QColor("#FFFFFF"), 1))
    painter.drawRect(0, 0, 10, 10)
    painter.end()
    return QIcon(pix)


class _TargetList(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(5)
        self.setHeaderLabels(("Target", "Color", "Filter", "Y range", "Reference lines"))
        self.setRootIsDecorated(False)
        self.setUniformRowHeights(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setDropIndicatorShown(True)
        # Allow a selected block (or non-contiguous multi-selection) to move
        # as one group during an InternalMove drag.
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def set_targets(self, labels, selected, values=None):
        self.clear()
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragEnabled(True)
        selected = set(selected)
        values = values or {}
        for label in labels:
            record = values.get(label, {})
            item = QTreeWidgetItem(self, [str(label), "", "", "", ""])
            item.setFlags(
                (item.flags() & ~Qt.ItemIsUserCheckable & ~Qt.ItemIsDropEnabled)
                | Qt.ItemIsSelectable
                | Qt.ItemIsEnabled
                | Qt.ItemIsDragEnabled
            )
            item.setData(0, Qt.UserRole, str(label))
            item.setSelected(label in selected)
            self.set_item_summary(item, record)

    def set_item_summary(self, item, record):
        color = record.get("col")
        item.setText(1, str(color or ""))
        bounds = record.get("f")
        item.setText(2, "–".join(str(float(x)) for x in bounds) + " Hz" if bounds else "")
        ylim = record.get("ylim")
        item.setText(3, "–".join(str(float(x)) for x in ylim) if ylim else "")
        lines = record.get("y")
        item.setText(4, ", ".join(str(float(x)) for x in lines) if lines else "")

    def selected_labels(self):
        return [
            self.topLevelItem(i).data(0, Qt.UserRole)
            for i in range(self.topLevelItemCount())
            if self.topLevelItem(i).isSelected()
        ]

    checked_labels = selected_labels

    def ordered_labels(self):
        return [
            self.topLevelItem(i).data(0, Qt.UserRole)
            for i in range(self.topLevelItemCount())
        ]

    def set_reordering_enabled(self, enabled):
        enabled = bool(enabled)
        self.setDragEnabled(enabled)
        self.setAcceptDrops(enabled)
        self.setDragDropMode(
            QAbstractItemView.InternalMove if enabled else QAbstractItemView.NoDragDrop
        )

    def dropEvent(self, event):
        """Move selected top-level rows before/after the row under the cursor."""
        if self.dragDropMode() != QAbstractItemView.InternalMove:
            return super().dropEvent(event)
        point = event.position().toPoint()
        target = self.itemAt(point)
        if target is None:
            insert_at = self.topLevelItemCount()
        else:
            insert_at = self.indexOfTopLevelItem(target)
            if point.y() >= self.visualItemRect(target).center().y():
                insert_at += 1

        moving = sorted(
            (item for item in self.selectedItems() if self.indexOfTopLevelItem(item) >= 0),
            key=self.indexOfTopLevelItem,
        )
        if not moving:
            return

        # Rebuild the complete top-level sequence. Incremental take/insert
        # arithmetic is vulnerable to Qt changing the drop target selection;
        # rebuilding from the captured item objects guarantees no row can be
        # swallowed or become a child of another row.
        all_items = [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
        moving_ids = {id(item) for item in moving}
        remaining = [item for item in all_items if id(item) not in moving_ids]
        moving_indices = [all_items.index(item) for item in moving]
        insert_at = max(
            0,
            min(insert_at - sum(index < insert_at for index in moving_indices), len(remaining)),
        )
        desired = remaining[:insert_at] + moving + remaining[insert_at:]
        for index in range(self.topLevelItemCount() - 1, -1, -1):
            self.takeTopLevelItem(index)
        for item in desired:
            self.addTopLevelItem(item)
        for item in moving:
            item.setSelected(True)
        self.setCurrentItem(moving[0])
        self._reload_field_values_if_available()
        # Report IgnoreAction (not acceptProposedAction()/MoveAction): this
        # method already performs the entire move by hand (take + reinsert).
        # If QDrag::exec() is left to report back MoveAction, Qt's own
        # QAbstractItemView::startDrag() runs its default post-drag cleanup
        # (clearOrRemove()) afterward, which deletes whatever is *currently
        # selected* — i.e. the rows we just moved and reselected above,
        # since InternalMove's clearOrRemove() call isn't itself conditioned
        # on drag mode. That silently removes the just-dropped row(s) right
        # after this method returns.
        event.setDropAction(Qt.IgnoreAction)
        event.accept()

    def _reload_field_values_if_available(self):
        callback = getattr(self, "_reload_callback", None)
        if callback is not None:
            callback()


class _PropertiesDialog(QDialog):
    def __init__(self, parent, kind, seed, labels, values, initial_targets=None,
                 fmap_frqs=None, on_trace_view_change=None):
        super().__init__(parent)
        self.kind = kind
        self.values = values
        self.fmap_frqs = dict(fmap_frqs or {})
        self._trace_view_callback = on_trace_view_change
        self.setWindowTitle(f"{kind.title()} properties")
        self.setMinimumSize(650, 600)
        self.resize(760, 600)

        root = QVBoxLayout(self)
        left = QVBoxLayout()
        root.addLayout(left, 1)

        initial_targets = list(initial_targets or [seed])
        left.addWidget(QLabel("Targets:"))

        select_row = QHBoxLayout()
        select_all = QPushButton("Select all", self)
        select_none = QPushButton("Select none", self)
        select_row.addWidget(select_all)
        select_row.addWidget(select_none)
        select_row.addStretch(1)
        left.addLayout(select_row)
        select_all.clicked.connect(lambda: self._set_visible(Qt.Checked))
        select_none.clicked.connect(lambda: self._set_visible(Qt.Unchecked))

        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Filter targets…")
        left.addWidget(self.search)

        self.targets = _TargetList(self)
        self.targets.set_targets(labels, initial_targets, values)
        header = self.targets.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, self.targets.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        if kind != "signal":
            for column in (2, 3, 4):
                self.targets.setColumnHidden(column, True)
        self._initial_target_order = self.targets.ordered_labels()
        left.addWidget(self.targets, 1)
        self.targets.itemSelectionChanged.connect(self._reload_field_values)
        self.targets._reload_callback = self._reload_field_values
        self.targets.model().rowsMoved.connect(lambda *_: self._reload_field_values())

        self.search.textChanged.connect(self._filter_targets)

        form = QFormLayout()
        root.addLayout(form)

        self.use_color = QCheckBox("Use explicit color", self)
        self.use_color.setTristate(True)
        self.use_color.stateChanged.connect(lambda *_: self._mark_field_dirty("col"))
        self.color_button = QPushButton(self)
        self.color_button.setFixedWidth(100)
        self.color_button.clicked.connect(self._choose_color)
        color_row = QHBoxLayout()
        color_row.addWidget(self.use_color)
        color_row.addWidget(self.color_button)
        color_row.addStretch(1)
        form.addRow("Color", color_row)

        self.use_ylim = None
        self.y_min = None
        self.y_max = None
        self.y_lines = None
        self.filter_combo = None
        self.trace_spacing = None
        self.trace_scale = None
        self.trace_labels = None
        self.trace_reset = None
        self._trace_view_initial = {}
        if kind == "signal":
            self.filter_combo = QComboBox(self)
            self.filter_combo.setMaximumWidth(260)
            self.filter_combo.addItems(("Mixed",) + _FILTER_NAMES)
            mixed_item = self.filter_combo.model().item(0)
            if mixed_item is not None:
                mixed_item.setEnabled(False)
            self.filter_combo.currentTextChanged.connect(
                lambda *_: self._mark_field_dirty("f")
            )
            form.addRow("Filter", self.filter_combo)

            self.use_ylim = QCheckBox("Use fixed physical range", self)
            self.y_min = QDoubleSpinBox(self)
            self.y_max = QDoubleSpinBox(self)
            for spin in (self.y_min, self.y_max):
                spin.setRange(-1e12, 1e12)
                spin.setDecimals(6)
                spin.setSingleStep(0.1)
                spin.setMaximumWidth(115)
                spin.valueChanged.connect(lambda *_: self._mark_field_dirty("ylim"))
            self.use_ylim.stateChanged.connect(lambda *_: self._mark_field_dirty("ylim"))
            range_row = QHBoxLayout()
            range_row.addWidget(self.use_ylim)
            range_row.addWidget(QLabel("min"))
            range_row.addWidget(self.y_min)
            range_row.addWidget(QLabel("max"))
            range_row.addWidget(self.y_max)
            form.addRow("Y range", range_row)

            self.y_lines = QLineEdit(self)
            self.y_lines.setMaximumWidth(320)
            self.y_lines.setPlaceholderText("e.g. 0, 10, 100")
            self.y_lines.textChanged.connect(lambda *_: self._mark_field_dirty("y"))
            form.addRow("Y reference lines", self.y_lines)

            self.trace_labels = QCheckBox("Labels", self)
            self.trace_labels.setChecked(bool(getattr(parent, "ui", parent).check_labels.isChecked()))
            self.trace_labels.stateChanged.connect(lambda *_: self._trace_view_changed("labels"))

            self.trace_spacing = QDoubleSpinBox(self)
            self.trace_spacing.setRange(0.0, 1.0)
            self.trace_spacing.setDecimals(3)
            self.trace_spacing.setSingleStep(0.05)
            self.trace_spacing.setFixedWidth(64)
            self.trace_spacing.setValue(float(getattr(parent, "ui", parent).spin_spacing.value()))
            self.trace_spacing.valueChanged.connect(lambda *_: self._trace_view_changed("spacing"))
            self.trace_scale = QDoubleSpinBox(self)
            self.trace_scale.setRange(-2.0, 2.0)
            self.trace_scale.setDecimals(3)
            self.trace_scale.setSingleStep(0.1)
            self.trace_scale.setFixedWidth(64)
            self.trace_scale.setValue(float(getattr(parent, "ui", parent).spin_scale.value()))
            self.trace_scale.valueChanged.connect(lambda *_: self._trace_view_changed("scale"))
            self.trace_reset = QPushButton("Reset", self)
            self.trace_reset.setFixedWidth(54)
            self.trace_reset.setToolTip("Reset trace spacing and scale to their defaults")
            self.trace_reset.clicked.connect(self._reset_trace_view_scaling)
            trace_row = QHBoxLayout()
            trace_row.setSpacing(6)
            trace_row.addWidget(QLabel("Spacing"))
            trace_row.addWidget(self.trace_spacing)
            trace_row.addWidget(QLabel("Scale"))
            trace_row.addWidget(self.trace_scale)
            trace_row.addWidget(self.trace_reset)
            trace_row.addWidget(self.trace_labels)
            trace_row.addStretch(1)
            form.addRow("Trace view", trace_row)

            self._trace_view_initial = {
                "spacing": self.trace_spacing.value(),
                "scale": self.trace_scale.value(),
                "labels": self.trace_labels.isChecked(),
            }

        self._dirty_fields = set()
        self._mixed_fields = set()
        self._loading_values = False
        self._reload_field_values()

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("Cancel", self)
        apply = QPushButton("Update config", self)
        apply.setDefault(True)
        apply.setMinimumWidth(90)
        apply.setStyleSheet(
            "QPushButton { background: #2F6DB2; color: #FFFFFF; border: 1px solid #4C8DD2; }"
            "QPushButton:hover { background: #3D82C5; }"
            "QPushButton:pressed { background: #25598F; }"
        )
        buttons.addWidget(cancel)
        buttons.addSpacing(12)
        buttons.addWidget(apply)
        cancel.clicked.connect(self._restore_trace_view)
        cancel.clicked.connect(self.reject)
        apply.clicked.connect(self.accept)
        root.addLayout(buttons)

    def _trace_view_value(self, field):
        if field == "spacing":
            return self.trace_spacing.value()
        if field == "scale":
            return self.trace_scale.value()
        return self.trace_labels.isChecked()

    def _trace_view_changed(self, field):
        self._mark_field_dirty(field)
        callback = self._trace_view_callback
        if callback is not None:
            callback(field, self._trace_view_value(field))

    def _reset_trace_view_scaling(self):
        self.trace_spacing.setValue(1.0)
        self.trace_scale.setValue(0.0)

    def _restore_trace_view(self):
        callback = self._trace_view_callback
        if callback is None:
            return
        for field, value in self._trace_view_initial.items():
            callback(field, value)

    def _set_visible(self, state):
        for i in range(self.targets.topLevelItemCount()):
            item = self.targets.topLevelItem(i)
            if not item.isHidden():
                item.setSelected(state == Qt.Checked)

    def _filter_targets(self, text):
        needle = str(text).casefold().strip()
        for i in range(self.targets.topLevelItemCount()):
            item = self.targets.topLevelItem(i)
            item.setHidden(bool(needle) and needle not in item.text(0).casefold())
        self.targets.set_reordering_enabled(not bool(needle))

    def _selected_records(self):
        labels = self.targets.selected_labels()
        return labels, [self.values.get(label, {}) for label in labels]

    def _refresh_target_summary(self):
        selected = set(self.targets.selected_labels())
        for i in range(self.targets.topLevelItemCount()):
            item = self.targets.topLevelItem(i)
            label = item.data(0, Qt.UserRole)
            record = dict(self.values.get(label, {}))
            if label in selected:
                if "col" in self._dirty_fields:
                    record["col"] = self._color if self.use_color.isChecked() else None
                if self.kind == "signal":
                    if "f" in self._dirty_fields:
                        try:
                            record["f"] = self._filter_bounds(self.filter_combo.currentText())
                        except ValueError:
                            record["f"] = None
                    if "ylim" in self._dirty_fields:
                        record["ylim"] = (
                            (self.y_min.value(), self.y_max.value())
                            if self.use_ylim.isChecked() and self.y_min.value() < self.y_max.value()
                            else None
                        )
                    if "y" in self._dirty_fields:
                        try:
                            record["y"] = [
                                float(x.strip()) for x in self.y_lines.text().split(",") if x.strip()
                            ] or None
                        except ValueError:
                            record["y"] = None
                # Keep pending edits in the dialog's local snapshot so
                # changing the row selection does not visually revert them.
                self.values[label] = dict(record)
            self.targets.set_item_summary(item, record)

    def _common_value(self, records, key):
        values = [record.get(key) for record in records]
        if not values or any(value != values[0] for value in values[1:]):
            self._mixed_fields.add(key)
            return _UNCHANGED
        return values[0]

    def _reload_field_values(self):
        if self._loading_values:
            return
        labels, records = self._selected_records()
        if not records:
            return
        self._loading_values = True
        self._mixed_fields.clear()
        try:
            color = self._common_value(records, "col")
            colors = [record.get("col") for record in records]
            color_seed = next((value for value in colors if value), "#4FC3F7")
            self._color_seed = color_seed
            if color is _UNCHANGED:
                self.use_color.setCheckState(Qt.PartiallyChecked)
                self._set_color_button("Mixed")
            else:
                self.use_color.setCheckState(Qt.Checked if color else Qt.Unchecked)
                self._set_color_button(color or "#4FC3F7")

            if self.kind != "signal":
                return

            filt = self._common_value(records, "f")
            self._current_filter_value = filt
            if filt is _UNCHANGED:
                self.filter_combo.setCurrentText("Mixed")
            else:
                self.filter_combo.setCurrentText(self._filter_name(filt))

            ylim = self._common_value(records, "ylim")
            self.use_ylim.setCheckState(
                Qt.PartiallyChecked if ylim is _UNCHANGED
                else (Qt.Checked if ylim is not None else Qt.Unchecked)
            )
            if ylim is _UNCHANGED:
                ylim = next((record.get("ylim") for record in records if record.get("ylim")), None)
            self.y_min.setValue(float(ylim[0]) if ylim else -1.0)
            self.y_max.setValue(float(ylim[1]) if ylim else 1.0)

            y_lines = self._common_value(records, "y")
            self.y_lines.setPlaceholderText("Mixed" if y_lines is _UNCHANGED else "e.g. 0, 10, 100")
            self.y_lines.setText(
                "" if y_lines is _UNCHANGED else ", ".join(str(x) for x in (y_lines or []))
            )
        finally:
            self._loading_values = False
            # Trace spacing/scale are dialog-global rather than target fields;
            # keep their pending edits when the user changes the selected
            # signal row.  Target-specific edits retain the existing reload
            # behavior.
            self._dirty_fields.intersection_update({"spacing", "scale", "labels"})

    def _filter_name(self, value):
        if value is None:
            return "None"
        if isinstance(value, (tuple, list)) and len(value) == 2:
            fmap = self.fmap_frqs
            for name, bounds in fmap.items():
                if tuple(bounds) == tuple(value):
                    return str(name)
            return "User"
        return "None"

    def _filter_bounds(self, name):
        if name in ("None", "Mixed"):
            return None
        fmap = self.fmap_frqs
        bounds = fmap.get(name)
        if bounds is None:
            raise ValueError(f"Unknown signal filter: {name}")
        if name == "User":
            value = getattr(self, "_current_filter_value", None)
            if isinstance(value, (tuple, list)) and len(value) == 2:
                return tuple(float(x) for x in value)
            raise ValueError("User filter requires an existing custom frequency range.")
        return tuple(float(x) for x in bounds)

    def _mark_field_dirty(self, field):
        if not self._loading_values:
            self._dirty_fields.add(field)
            self._refresh_target_summary()

    def _set_color_button(self, value):
        self._color = str(value)
        self.color_button.setText(self._color)
        self.color_button.setStyleSheet(
            f"background-color: {self._color}; color: {'black' if QColor(self._color).lightness() > 150 else 'white'};"
        )

    def _choose_color(self):
        color = QColorDialog.getColor(
            QColor(getattr(self, "_color_seed", self._color)), self, "Choose color"
        )
        if color.isValid():
            self._set_color_button(color.name().upper())
            self.use_color.setChecked(True)
            self._mark_field_dirty("col")

    def properties(self):
        order = self.targets.ordered_labels()
        out = {}
        if self.kind == "signal":
            if "spacing" in self._dirty_fields:
                out["_spacing"] = self.trace_spacing.value()
            if "scale" in self._dirty_fields:
                out["_scale"] = self.trace_scale.value()
            if "labels" in self._dirty_fields:
                out["_labels"] = self.trace_labels.isChecked()
        if not self.targets.checked_labels():
            out.update({
                "_order": order,
                "_all_order": order,
            } if order != self._initial_target_order else {"_all_order": order})
            return out
        if "col" in self._dirty_fields:
            out["col"] = self._color if self.use_color.isChecked() else None
        if self.kind == "signal":
            if "f" in self._dirty_fields:
                out["f"] = self._filter_bounds(self.filter_combo.currentText())
            if "ylim" in self._dirty_fields:
                if self.use_ylim.isChecked() and self.y_min.value() >= self.y_max.value():
                    raise ValueError("Y-axis minimum must be less than the maximum.")
                out["ylim"] = (
                    (self.y_min.value(), self.y_max.value())
                    if self.use_ylim.isChecked() and self.y_min.value() < self.y_max.value()
                    else None
                )
            if "y" in self._dirty_fields:
                if self.y_lines.text().strip():
                    try:
                        out["y"] = [float(x.strip()) for x in self.y_lines.text().split(",") if x.strip()]
                    except ValueError:
                        raise ValueError("Y reference lines must be comma-separated numbers.")
                else:
                    out["y"] = None
        if order != self._initial_target_order:
            out["_order"] = order
        out["_all_order"] = order
        return out


class DockPropertiesMixin:
    """Adds context-menu properties editing to Dock 3 and Dock 4."""

    def _init_dock_properties(self):
        self._dock_property_views = {
            "signal": self.ui.tbl_desc_signals,
            "annotation": self.ui.tbl_desc_annots,
        }
        for kind, view in self._dock_property_views.items():
            view.setItemDelegateForColumn(0, _ColorCheckDelegate(view))
            view.setContextMenuPolicy(Qt.CustomContextMenu)
            view.customContextMenuRequested.connect(
                lambda pos, k=kind, v=view: self._show_dock_properties_menu(k, v, pos)
            )
            view.installEventFilter(self)
        self._refresh_dock_property_colors()

        # Toolbar "Properties" buttons + shortcuts. Ctrl+Shift+<digit> was
        # tried first but Qt maps Ctrl->Cmd on macOS, so Ctrl+Shift+3/4
        # collided with the OS-level screenshot shortcuts and never reached
        # the app. Ctrl+P / Ctrl+A ("Properties" / "Annotations") are free
        # on every platform; the actions they displaced (Predict, Annotator)
        # now live on Ctrl+Shift+P / Ctrl+Shift+A instead (controller.py).
        button_specs = (
            ("signal", getattr(self.ui, "butt_sig_props", None), "Ctrl+P"),
            ("annotation", getattr(self.ui, "butt_annot_props", None), "Ctrl+A"),
        )
        self._dock_property_actions = {}
        for kind, button, shortcut in button_specs:
            if button is not None:
                button.clicked.connect(lambda _=False, k=kind: self._open_dock_properties_from_button(k))
            action = QAction(f"{kind.title()} Properties…", self.ui)
            action.setShortcut(QKeySequence(shortcut))
            action.setShortcutContext(Qt.ApplicationShortcut)
            action.triggered.connect(lambda _=False, k=kind: self._open_dock_properties_from_button(k))
            self.ui.addAction(action)
            self._dock_property_actions[kind] = action

        # AppFontController.apply_now() (controller.py, runs later during
        # startup) walks every widget and rewrites font/stylesheet from each
        # widget's OWN prior state independently — butt_sig/butt_annot are
        # long-lived widgets that already existed before that pass ever ran,
        # while the *_props buttons are new, so nothing guarantees the two
        # converge on the same result even with identical .ui stylesheets.
        # Force the props button to mirror its sibling explicitly, both now
        # and on every future scale change, so they can't drift apart.
        font_controller = getattr(self, "_font_controller", None)
        if font_controller is not None:
            font_controller.scaleChanged.connect(self._sync_dock_property_button_style)
        self._sync_dock_property_button_style()

    def _sync_dock_property_button_style(self, *_args):
        for reference_name, twin_name in (
            ("butt_sig", "butt_sig_props"),
            ("butt_annot", "butt_annot_props"),
        ):
            reference = getattr(self.ui, reference_name, None)
            twin = getattr(self.ui, twin_name, None)
            if reference is None or twin is None:
                continue
            twin.setFont(reference.font())
            twin.setStyleSheet(reference.styleSheet())

    def _open_dock_properties_from_button(self, kind):
        """Open Properties from the toolbar button/shortcut (no right-clicked row)."""
        view = self._dock_property_views.get(kind)
        model = _source_model(view) if view is not None else None
        if view is None or model is None:
            return
        col = _header_column(model, ("CH", "ANNOTATIONS", "CLASS"))
        selected_labels = []
        selection = view.selectionModel() if hasattr(view, "selectionModel") else None
        if selection is not None:
            for selected_index in selection.selectedRows():
                row = _source_row_for_index(view, selected_index)
                if row is None:
                    continue
                label = str(model.data(model.index(row, col)) or "").strip()
                if label and label not in selected_labels:
                    selected_labels.append(label)
        if not selected_labels and model.rowCount() > 0:
            label = str(model.data(model.index(0, col)) or "").strip()
            if label:
                selected_labels = [label]
        if not selected_labels:
            return
        self._open_dock_properties(kind, selected_labels[0], selected_labels)

    def _show_dock_properties_menu(self, kind, view, pos):
        index = view.indexAt(pos)
        row = _source_row_for_index(view, index)
        model = _source_model(view)
        if row is None or model is None:
            return
        col = _header_column(model, ("CH", "ANNOTATIONS", "CLASS"))
        seed = str(model.data(model.index(row, col)) or "").strip()
        if not seed:
            return

        # Preserve an existing multi-row selection for group editing.  With a
        # single selected row, that row remains the dialog's value seed, so
        # navigating up/down the dock and opening Properties shows its values.
        selected_labels = []
        selection = view.selectionModel() if hasattr(view, "selectionModel") else None
        if selection is not None:
            for selected_index in selection.selectedRows():
                selected_row = _source_row_for_index(view, selected_index)
                if selected_row is None:
                    continue
                label = str(model.data(model.index(selected_row, col)) or "").strip()
                if label and label not in selected_labels:
                    selected_labels.append(label)
        if seed not in selected_labels:
            selected_labels = [seed]

        menu = QMenu(view)
        action = menu.addAction("Properties…")
        if menu.exec(view.viewport().mapToGlobal(pos)) == action:
            self._open_dock_properties(kind, seed, selected_labels)

    def _dock_labels_and_values(self, kind):
        view = self._dock_property_views[kind]
        model = _source_model(view)
        if model is None:
            return [], {}
        col = _header_column(model, ("CH", "ANNOTATIONS", "CLASS"))
        section = "sig" if kind == "signal" else "ann"
        records = getattr(self, "cfg", {}).get(section, {})
        labels = []
        values = {}
        for row in range(model.rowCount()):
            label = str(model.data(model.index(row, col)) or "").strip()
            if label:
                labels.append(label)
                values[label] = dict(records.get(label, {}))
        return labels, values

    def _open_dock_properties(self, kind, seed, initial_targets=None):
        labels, values = self._dock_labels_and_values(kind)
        if seed not in labels:
            labels.insert(0, seed)
        dialog = _PropertiesDialog(
            self.ui,
            kind,
            seed,
            labels,
            values,
            initial_targets=initial_targets,
            fmap_frqs=getattr(self, "fmap_frqs", {}),
            on_trace_view_change=self._on_properties_trace_view_change,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        targets = dialog.targets.checked_labels()
        order = dialog.targets.ordered_labels()
        order_changed = order != dialog._initial_target_order
        try:
            props = dialog.properties()
        except ValueError as exc:
            QMessageBox.warning(self.ui, "Properties", str(exc))
            return
        has_trace_view_change = any(key in props for key in ("_spacing", "_scale", "_labels"))
        if not targets and not order_changed and not has_trace_view_change:
            QMessageBox.information(self.ui, "Properties", "Select at least one target or change a trace-view option.")
            return
        self._apply_dock_properties(kind, targets, props)

    def _on_properties_trace_view_change(self, field, value):
        """Apply global Trace View controls live while Properties is open."""
        if field == "labels":
            self.ui.check_labels.setChecked(bool(value))
        elif field == "spacing":
            self.ui.spin_spacing.setValue(float(value))
        elif field == "scale":
            self.ui.spin_scale.setValue(float(value))

    def _apply_dock_properties(self, kind, targets, props):
        section = "sig" if kind == "signal" else "ann"
        order = props.pop("_order", None)
        dialog_order = props.pop("_all_order", None)
        trace_spacing = props.pop("_spacing", None)
        trace_scale = props.pop("_scale", None)
        trace_labels = props.pop("_labels", None)
        order_only = bool(order) and not targets and not props
        current_labels, _ = self._dock_labels_and_values(kind)
        full_order = list(dialog_order or order or current_labels)
        old_text = self.ui.txt_cmap.toPlainText()
        text = old_text
        try:
            if section == "sig" and props.get("f") is not None:
                low, high = props["f"]
                sample_rates = getattr(self, "srs", {}) or {}
                for label in targets:
                    sr = sample_rates.get(label)
                    if sr is not None and high > float(sr) / 2.0:
                        raise ValueError(
                            f"Filter upper frequency for {label} exceeds its Nyquist frequency."
                        )
            text = _update_config_entries(
                text,
                section,
                targets,
                props,
                order=full_order,
                full_order=full_order,
            )
            from .cmaps import parse_cmap
            parse_cmap(text)
        except ValueError as exc:
            QMessageBox.warning(self.ui, "Properties", str(exc))
            return
        self.ui.txt_cmap.setPlainText(text)
        if not self._apply_cmaps():
            self.ui.txt_cmap.setPlainText(old_text)
            return
        if trace_spacing is not None:
            self.ui.spin_spacing.setValue(float(trace_spacing))
        if trace_scale is not None:
            self.ui.spin_scale.setValue(float(trace_scale))
        if trace_labels is not None:
            self.ui.check_labels.setChecked(bool(trace_labels))
        self._refresh_dock_property_filters()
        if full_order:
            self._reorder_dock_property_view(kind, full_order)
        if not hasattr(self, "p"):
            self._refresh_dock_property_colors()
            return
        try:
            if getattr(self, "rendered", False):
                # Properties edits are presentation/runtime edits to the
                # already-populated study.  Keep Config -> Apply as the
                # explicit full-repopulate path, but do not make a color,
                # order, y-line, fixed-limit, or filter edit repopulate the
                # backend here.
                self._update_dock_properties_runtime(kind, targets, props)
            else:
                self._render_signals_simple()
        finally:
            self._refresh_dock_property_colors()

    def _update_dock_properties_runtime(self, kind, targets, props):
        """Apply Properties changes without rebuilding the rendered backend.

        The Properties dialog only edits signal/annotation presentation
        records.  The existing segsrv can update filters and scaling in place;
        curve construction then consumes the new order, colors, and y-line
        configuration.  Backend population remains owned by Config -> Apply
        and by the normal Render path when the populated data set changes.
        """
        if kind == "signal" and "f" in props:
            fmap = getattr(self, "fmap", None)
            if isinstance(fmap, dict):
                user_fmap_frqs = getattr(self, "user_fmap_frqs", None)
                if not isinstance(user_fmap_frqs, dict):
                    user_fmap_frqs = {}
                    self.user_fmap_frqs = user_fmap_frqs
                named_filters = getattr(self, "fmap_frqs", {}) or {}
                for label in targets:
                    bounds = getattr(self, "cfg", {}).get("sig", {}).get(label, {}).get("f")
                    if bounds is None:
                        fmap.pop(label, None)
                        user_fmap_frqs.pop(label, None)
                        continue
                    bounds = tuple(float(x) for x in bounds)
                    code = next(
                        (
                            name for name, candidate in named_filters.items()
                            if name != "User" and tuple(float(x) for x in candidate) == bounds
                        ),
                        None,
                    )
                    if code is None:
                        fmap[label] = "User"
                        user_fmap_frqs[label] = list(bounds)
                    else:
                        fmap[label] = code
                        user_fmap_frqs.pop(label, None)

            apply_filters = getattr(self, "_apply_backend_filters", None)
            if callable(apply_filters) and getattr(self, "ss", None) is not None:
                apply_filters(getattr(self, "_segsrv_input_throttle_sr", None))

        # _apply_cmaps() intentionally clears the render-stage sigmod map;
        # normally _render_cmaps() rebuilds it during population.  Properties
        # does not change [mod] or signal mod= mappings, so restore only the
        # Python-side channel map and leave the already-registered backend
        # sigmods untouched.
        self.sigmods = {
            label: spec["mod"]
            for label, spec in getattr(self, "cfg", {}).get("sig", {}).items()
            if spec.get("mod") is not None
        }

        # Curve creation consumes order, colors, annotation colors, and the
        # number/placement of reference lines.  Scaling consumes fixed limits
        # and empirical ranges, and also refreshes the plotted data.
        self._initiate_curves()
        self._update_scaling()

    def _refresh_dock_property_colors(self):
        records = {}
        for section in ("sig", "ann"):
            records.update(getattr(self, "cfg", {}).get(section, {}))
        # Build the same label -> colour mapping used by the active plot.
        # `self.colors` is intentionally positional, and rendered/simple
        # views consume it in opposite directions; using a reversed palette
        # here made every table icon fall back to gray when the view changed.
        def checked(view, fallback):
            fn = getattr(view, "checked", None) if view is not None else None
            return list(fn()) if callable(fn) else list(fallback or [])

        signal_names = checked(
            getattr(self.ui, "tbl_desc_signals", None),
            getattr(self, "ss_chs", []),
        )
        ann_names = checked(
            getattr(self.ui, "tbl_desc_annots", None),
            getattr(self, "ss_anns", []),
        )

        def ordered(names, reverse=False):
            names = list(names)
            if getattr(self, "cmap_list", None):
                original_pos = {label: idx for idx, label in enumerate(names)}
                names.sort(
                    key=lambda x: (
                        self.cmap_list.index(x)
                        if x in self.cmap_list
                        else len(self.cmap_list) + original_pos[x]
                    )
                )
            elif hasattr(self, "_order_pp_channels"):
                names = self._order_pp_channels(names)
            if reverse:
                names.reverse()
            return names

        colors = list(getattr(self, "colors", []) or [])
        if getattr(self, "rendered", False):
            plotted_signals = ordered(signal_names)
            signal_palette = {
                label: colors[len(plotted_signals) - idx - 1]
                for idx, label in enumerate(plotted_signals)
                if len(plotted_signals) - idx - 1 < len(colors)
            }
        else:
            plotted_signals = ordered(signal_names, reverse=True)
            signal_palette = {
                label: colors[idx]
                for idx, label in enumerate(plotted_signals)
                if idx < len(colors)
            }
        palette_names = {"signal": signal_palette, "annotation": {}}
        # The trace update records the final resolved colour (after explicit
        # cmap overrides, PP rules, and simple/render slot selection). Prefer
        # that authoritative value whenever it is available.
        for entry in getattr(self, "_pg1_channel_cache", []) or []:
            label = str(entry.get("ch", "")).strip()
            if label and entry.get("color") is not None:
                signal_palette[label] = entry["color"]
        palette_names["annotation"] = {
            label: color
            for label, color in zip(ordered(ann_names), getattr(self, "acolors", []) or [])
        }
        active_ann_names = list(getattr(self, "ss_anns", []) or [])
        for label, curve in zip(active_ann_names, getattr(self, "annot_curves", []) or []):
            try:
                palette_names["annotation"][label] = curve.opts["pen"].color()
            except (AttributeError, KeyError, TypeError):
                pass
        for kind, view in getattr(self, "_dock_property_views", {}).items():
            model = _source_model(view)
            if model is None:
                continue
            label_col = _header_column(model, ("CH", "ANNOTATIONS", "CLASS"))
            for row in range(model.rowCount()):
                label = str(model.data(model.index(row, label_col)) or "").strip()
                color = records.get(label, {}).get("col")
                if not color:
                    color = palette_names[kind].get(label)
                item = model.item(row, label_col) if hasattr(model, "item") else None
                if item is not None:
                    # The Sel cell is both the checkbox and the colour key.
                    # Keep the CH/Annotations label free of a second swatch.
                    item.setForeground(QBrush(QColor("#E7EAEE")))
                    item.setIcon(QIcon())
                    select_item = model.item(row, 0) if hasattr(model, "item") else None
                    if select_item is not None:
                        is_on = _is_checked_state(select_item.checkState())
                        if is_on and color:
                            select_item.setData(QColor(color), _COLOR_ROLE)
                            select_item.setBackground(QBrush(QColor(color)))
                        else:
                            select_item.setData(None, _COLOR_ROLE)
                            select_item.setBackground(QBrush())

    def _refresh_dock_property_filters(self):
        view = getattr(self, "_dock_property_views", {}).get("signal")
        model = _source_model(view) if view is not None else None
        if model is None or not hasattr(model, "setData"):
            return
        label_col = _header_column(model, ("CH",))
        filter_col = _header_column(model, ("FILTER",), fallback=-1)
        if filter_col < 0 or filter_col >= model.columnCount():
            return
        records = getattr(self, "cfg", {}).get("sig", {})
        fmap = getattr(self, "fmap_frqs", {}) or {}
        for row in range(model.rowCount()):
            label = str(model.data(model.index(row, label_col)) or "").strip()
            bounds = records.get(label, {}).get("f")
            name = "None"
            if bounds is not None:
                name = "User"
                for candidate, candidate_bounds in fmap.items():
                    if tuple(candidate_bounds) == tuple(bounds):
                        name = str(candidate)
                        break
            model.setData(model.index(row, filter_col), name, Qt.EditRole)

    def _reorder_dock_property_view(self, kind, order):
        view = getattr(self, "_dock_property_views", {}).get(kind)
        model = _source_model(view) if view is not None else None
        if model is None or not hasattr(model, "takeRow"):
            return
        label_col = _header_column(model, ("CH", "ANNOTATIONS", "CLASS"))
        model.blockSignals(True)
        try:
            for target_row, wanted in enumerate(order):
                current_row = next(
                    (row for row in range(model.rowCount())
                     if str(model.data(model.index(row, label_col)) or "").strip() == wanted),
                    None,
                )
                if current_row is None or current_row == target_row:
                    continue
                items = model.takeRow(current_row)
                model.insertRow(target_row, items)
        finally:
            model.blockSignals(False)


def _entry_fields(record, section):
    fields = []
    if record.get("col"):
        fields.append(f"col={record['col']}")
    if section == "sig":
        if record.get("ylim") is not None:
            fields.append("ylim=" + ",".join(str(float(x)) for x in record["ylim"]))
        if record.get("y") is not None:
            fields.append("y=" + ",".join(str(float(x)) for x in record["y"]))
        if record.get("f") is not None:
            fields.append("f=" + ",".join(str(float(x)) for x in record["f"]))
        if record.get("mod") is not None:
            fields.append("mod=" + ",".join(str(record["mod"].get(x, "")) for x in ("mod", "pal")))
    return fields


def _clean_config_text(text):
    """Remove cmap guide/preamble and comments before a generated write."""
    lines = []
    seen_section = False
    for raw in str(text).splitlines():
        line = raw.split("%", 1)[0].rstrip()
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            seen_section = True
        # Keep valid implicit-[par] assignments, but discard prose before the
        # first section (the bundled cmap guide/preamble).
        if not seen_section and "=" not in line:
            continue
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(line)
    return "\n".join(lines).strip("\n")


def _update_config_entries(text, section, targets, props, order=None, full_order=None):
    """Update target entries while retaining all unrelated config text."""
    from .cmaps import parse_cmap

    text = _clean_config_text(text)
    parsed = parse_cmap(text)
    records = parsed[section]
    for label in targets:
        rec = dict(records.get(label, {}))
        rec.update(props)
        records[label] = rec

    lines = text.splitlines()
    section_start = next((i for i, line in enumerate(lines) if line.strip() == f"[{section}]"), None)
    if section_start is None:
        lines.extend(["", f"[{section}]"])
        section_start = len(lines) - 1
    section_end = len(lines)
    for i in range(section_start + 1, len(lines)):
        if lines[i].strip().startswith("[") and lines[i].strip().endswith("]"):
            section_end = i
            break

    target_set = set(targets)
    out = lines[:section_start + 1]
    found = set()
    i = section_start + 1
    while i < section_end:
        raw = lines[i]
        stripped = raw.split("%", 1)[0].strip()
        if not stripped or stripped.startswith("%") or stripped.startswith("..."):
            out.append(raw)
            i += 1
            continue
        block_end = i + 1
        while block_end < section_end and lines[block_end].split("%", 1)[0].strip().startswith("..."):
            block_end += 1
        tokens = re.sub(r"\s*=\s*", "=", stripped).split()
        old_labels = tokens[0].split(",") if tokens else []
        hit = target_set.intersection(old_labels)
        if not hit:
            out.extend(lines[i:block_end])
        else:
            remaining = [x for x in old_labels if x not in target_set]
            if remaining:
                out.append(" ".join([",".join(remaining)] + tokens[1:]))
            for label in old_labels:
                if label in target_set:
                    # A label should have one authoritative entry. If an
                    # older config contains duplicates, replace the first
                    # occurrence and discard subsequent stale copies.
                    if label in found:
                        continue
                    out.append(" ".join([label] + _entry_fields(records[label], section)))
                    found.add(label)
        i = block_end

    out.extend(lines[section_end:])
    missing = [label for label in targets if label not in found]
    if missing:
        insert_at = len(out)
        for j, line in enumerate(out):
            if j > section_start and line.strip().startswith("[") and line.strip().endswith("]"):
                insert_at = j
                break
        additions = [" ".join([label] + _entry_fields(records[label], section)) for label in missing]
        out[insert_at:insert_at] = additions
    result = "\n".join(out) + ("\n" if text.endswith("\n") else "")
    if order:
        result = _rewrite_section_order(
            result,
            section,
            order,
            drop_unlisted=full_order is not None,
        )
    return result


def _rewrite_section_order(text, section, order, drop_unlisted=False):
    """Reorder complete section entries while leaving other sections intact."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip() == f"[{section}]"), None)
    if start is None:
        return text
    end = next(
        (i for i in range(start + 1, len(lines))
         if lines[i].strip().startswith("[") and lines[i].strip().endswith("]")),
        len(lines),
    )
    entries = {}
    loose = []
    i = start + 1
    while i < end:
        stripped = lines[i].split("%", 1)[0].strip()
        if not stripped or stripped.startswith("%") or stripped.startswith("..."):
            loose.append(lines[i])
            i += 1
            continue
        block_end = i + 1
        while block_end < end and lines[block_end].split("%", 1)[0].strip().startswith("..."):
            block_end += 1
        labels = stripped.split()[0].split(",")
        block = lines[i:block_end]
        for label in labels:
            entries[label] = block if len(labels) == 1 else [
                " ".join([label] + stripped.split()[1:])
            ]
        i = block_end
    ordered = []
    used = set()
    for label in order:
        if label in used:
            continue
        ordered.extend(entries.get(label, [label]))
        used.add(label)
    if not drop_unlisted:
        for label, block in entries.items():
            if label not in used:
                ordered.extend(block)
    body = loose + ordered
    return "\n".join(lines[:start + 1] + body + lines[end:]) + ("\n" if text.endswith("\n") else "")
