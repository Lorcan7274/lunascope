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

import pandas as pd
from os import path
import os
import ntpath
import posixpath
from pathlib import Path
        
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHeaderView,
    QAbstractItemView,
    QMessageBox,
    QSizePolicy,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
)
from PySide6.QtCore import Qt, QDir, QEvent, QRegularExpression, QSortFilterProxyModel, QAbstractTableModel, QModelIndex, QSize, QTimer
from PySide6.QtGui import QStandardItemModel, QStandardItem, QPalette, QIcon

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_integer_dtype

from .tbl_funcs import attach_comma_filter
from ..file_dialogs import (
    attach_annots_directory,
    build_slist_directory,
    normalize_build_ext,
    normalize_build_exts,
    open_file_name,
    save_file_name,
    _annot_stem_for_ext,
)


def _is_absolute_sample_path(value: str) -> bool:
    text = str(value or "").strip()
    if not text or text == ".":
        return False
    return ntpath.isabs(text) or posixpath.isabs(text)


def _resolve_sample_list_path(base_dir: str, value: str) -> str:
    text = str(value or "").strip()
    if not text or text == "." or _is_absolute_sample_path(text):
        return text or "."
    return os.path.normpath(os.path.join(base_dir, text))


def _read_sample_list_rows(filename: str) -> list[list[str]]:
    base_dir = str(Path(filename).expanduser().resolve().parent)
    rows: list[list[str]] = []
    with open(filename, "r", encoding="utf-8-sig") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                continue
            cols = line.split("\t")
            if len(cols) < 2 or len(cols) > 3:
                continue

            sample_id = str(cols[0]).strip()
            edf_path = _resolve_sample_list_path(base_dir, cols[1])

            annot_paths = "."
            if len(cols) == 3:
                annot_bits = []
                for value in str(cols[2]).split(","):
                    resolved = _resolve_sample_list_path(base_dir, value)
                    if resolved and resolved != ".":
                        annot_bits.append(resolved)
                annot_paths = ",".join(annot_bits) if annot_bits else "."

            rows.append([sample_id, edf_path, annot_paths])

    return rows


class NumericSortFilterProxy(QSortFilterProxyModel):
    """QSortFilterProxyModel with numeric sort and fast row filtering.

    When the source is a DataFrameModel, filterAcceptsRow matches against a
    pre-built per-row string (one Python call per row) instead of calling
    data() for every column (ncols calls per row).
    """

    def lessThan(self, left, right):
        lv = left.data(Qt.DisplayRole) or ""
        rv = right.data(Qt.DisplayRole) or ""
        try:
            return float(lv) < float(rv)
        except (TypeError, ValueError):
            return str(lv) < str(rv)

    def filterAcceptsRow(self, source_row, source_parent):
        rx = self.filterRegularExpression()
        if not rx.pattern():
            return True
        src = self.sourceModel()
        if isinstance(src, DataFrameModel) and source_row < len(src._row_text):
            return bool(rx.match(src._row_text[source_row]).hasMatch())
        return super().filterAcceptsRow(source_row, source_parent)


class DataFrameModel(QAbstractTableModel):
    """Read-only model backed by a pandas DataFrame.

    Qt calls data() only for visible cells, so large DataFrames render fast.
    The constructor internally copies the DataFrame (via coerce_numeric_df)
    so the caller's data can be freed without affecting the model.
    """

    def __init__(
        self,
        df,
        float_decimals_default=3,
        float_decimals_per_col=None,
        preserve_text_cols=None,
        parent=None,
    ):
        super().__init__(parent)
        # coerce_numeric_df does df.copy() internally — model owns its data
        # SListMixin is defined later in this same file; resolved at call time
        self._df = SListMixin.coerce_numeric_df(
            df,
            decimals_default=float_decimals_default,
            decimals_per_col=float_decimals_per_col or {},
            preserve_text_cols=preserve_text_cols or set(),
        )
        digs = float_decimals_per_col or {}
        cols = list(self._df.columns)
        self._col_is_int   = [pd.api.types.is_integer_dtype(self._df[c].dtype) for c in cols]
        self._col_is_float = [pd.api.types.is_float_dtype(self._df[c].dtype)   for c in cols]
        self._float_digs   = [digs.get(c, float_decimals_default) for c in cols]
        # Pre-compute a tab-joined search string per row for fast proxy filtering.
        # Built once here; NumericSortFilterProxy.filterAcceptsRow uses it directly.
        self._row_text = self._build_row_text()

    def _build_row_text(self) -> list[str]:
        """Build one tab-joined search string per row using vectorised pandas ops."""
        parts = []
        for c, col in enumerate(self._df.columns):
            s = self._df[col]
            if self._col_is_int[c]:
                parts.append(s.apply(lambda v: "" if pd.isna(v) else str(int(v))))
            elif self._col_is_float[c]:
                digs = self._float_digs[c]
                parts.append(s.apply(lambda v, d=digs: "" if pd.isna(v) else f"{float(v):.{d}f}"))
            else:
                parts.append(s.fillna("").astype(str))
        if not parts:
            return [""] * len(self._df)
        combined = parts[0].astype(object)
        for p in parts[1:]:
            combined = combined + "\t" + p.astype(object)
        return combined.tolist()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        if role in (Qt.DisplayRole, Qt.EditRole):
            v = self._df.iat[r, c]
            try:
                if pd.isna(v):
                    return ""
            except (TypeError, ValueError):
                pass
            if isinstance(v, (list, tuple, set)):
                return ", ".join(map(str, v))
            if self._col_is_int[c]:
                return str(int(v))
            if self._col_is_float[c]:
                return f"{float(v):.{self._float_digs[c]}f}"
            return str(v)
        if role == Qt.TextAlignmentRole:
            if self._col_is_int[c] or self._col_is_float[c]:
                return int(Qt.AlignRight | Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable


class SampleListCompactDelegate(QStyledItemDelegate):
    """Render the sample list as a compact ID-first list with expandable details."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._expanded_keys = set()
        self._meta_rows = {}   # id -> {col: val}
        self._meta_cols = []   # ordered display columns (up to 10)

    @staticmethod
    def _clean_detail(value):
        if value is None:
            return "<none>"
        text = str(value).strip()
        return "<none>" if text in {"", "."} else text

    def _row_key(self, index):
        model = index.model()
        values = []
        for col in range(min(3, model.columnCount())):
            values.append(str(model.index(index.row(), col).data(Qt.DisplayRole) or ""))
        return tuple(values)

    def is_expanded(self, index) -> bool:
        return self._row_key(index) in self._expanded_keys

    def clear_expanded(self):
        self._expanded_keys.clear()

    def set_expanded_only(self, index):
        self._expanded_keys.clear()
        if index.isValid():
            self._expanded_keys.add(self._row_key(index))

    def toggle_index(self, index):
        key = self._row_key(index)
        if key in self._expanded_keys:
            self._expanded_keys.remove(key)
        else:
            self._expanded_keys.add(key)

    def sizeHint(self, option, index):
        base = super().sizeHint(option, index)
        if index.column() != 0:
            return base
        fm = option.fontMetrics
        height = fm.height() + 12
        if self.is_expanded(index):
            n_extra = 2  # EDF + Annot
            if self._meta_cols:
                id_str = str(index.data(Qt.DisplayRole) or "")
                if id_str in self._meta_rows:
                    n_extra += len(self._meta_cols)
            height += (fm.height() + 4) * n_extra
        return QSize(base.width(), max(base.height(), height))

    def paint(self, painter, option, index):
        if index.column() != 0:
            super().paint(painter, option, index)
            return

        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        opt.icon = QIcon()

        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)

        painter.save()

        text_rect = option.rect.adjusted(10, 6, -10, -6)
        fm = option.fontMetrics
        expanded = self.is_expanded(index)

        arrow = "v" if expanded else ">"
        arrow_w = fm.horizontalAdvance(arrow) + 10
        arrow_rect = text_rect.adjusted(0, 0, -(text_rect.width() - arrow_w), 0)
        body_rect = text_rect.adjusted(arrow_w, 0, 0, 0)

        if option.state & QStyle.State_Selected:
            primary = option.palette.color(QPalette.HighlightedText)
            secondary = option.palette.color(QPalette.HighlightedText)
        else:
            primary = option.palette.color(QPalette.Text)
            secondary = option.palette.color(QPalette.Text).darker(135)

        painter.setPen(primary)
        painter.drawText(arrow_rect, Qt.AlignLeft | Qt.AlignTop, arrow)

        id_text = str(index.data(Qt.DisplayRole) or "")
        id_line = fm.elidedText(id_text, Qt.ElideMiddle, max(40, body_rect.width()))
        painter.drawText(body_rect, Qt.AlignLeft | Qt.AlignTop, id_line)

        if expanded:
            edf_text = self._clean_detail(index.siblingAtColumn(1).data(Qt.DisplayRole))
            annot_text = self._clean_detail(index.siblingAtColumn(2).data(Qt.DisplayRole))
            line_h = fm.height() + 4
            detail_rect = body_rect.adjusted(0, line_h, 0, 0)
            painter.setPen(secondary)
            detail1 = fm.elidedText(f"EDF: {edf_text}", Qt.ElideMiddle, max(40, detail_rect.width()))
            detail2 = fm.elidedText(f"Annot: {annot_text}", Qt.ElideMiddle, max(40, detail_rect.width()))
            painter.drawText(detail_rect, Qt.AlignLeft | Qt.AlignTop, detail1)
            painter.drawText(detail_rect.adjusted(0, line_h, 0, 0), Qt.AlignLeft | Qt.AlignTop, detail2)
            if self._meta_cols:
                id_str = str(index.data(Qt.DisplayRole) or "")
                meta = self._meta_rows.get(id_str)
                if meta:
                    for i, col in enumerate(self._meta_cols):
                        val = meta.get(col, "")
                        mr = body_rect.adjusted(0, line_h * (3 + i), 0, 0)
                        painter.drawText(mr, Qt.AlignLeft | Qt.AlignTop,
                                         fm.elidedText(f"{col}: {val}", Qt.ElideMiddle, max(40, mr.width())))

        painter.restore()


class SListMixin:

    def _set_slist_label(self, text: str):
        label = self.ui.lbl_slist
        full_text = str(text or "").strip() or "(none)"
        label.setProperty("fullText", full_text)
        self._refresh_slist_label()

    def _get_slist_label_full_text(self) -> str:
        label = self.ui.lbl_slist
        full_text = label.property("fullText")
        if full_text is None:
            full_text = label.text()
        return str(full_text or "").strip()

    def _refresh_slist_label(self):
        label = self.ui.lbl_slist
        full_text = self._get_slist_label_full_text() or "(none)"
        avail = max(40, label.width() - 12)
        shown = label.fontMetrics().elidedText(full_text, Qt.ElideMiddle, avail)
        label.setText(shown)
        if shown != full_text:
            label.setToolTip(full_text)
        else:
            label.setToolTip("")

    def eventFilter(self, obj, event):
        if hasattr(self, "ui") and obj is getattr(self.ui, "lbl_slist", None):
            if event.type() in (QEvent.Resize, QEvent.Show):
                self._refresh_slist_label()
        return super().eventFilter(obj, event)

    def _annotation_paths_from_cell(self, value):
        text = str(value or "").strip()
        if text in ("", "."):
            return []
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
        return [p.strip() for p in text.split(",") if p.strip() and p.strip() != "."]

    def _current_slist_source_row(self):
        view = self.ui.tbl_slist
        idx = view.currentIndex()
        if not idx.isValid():
            sel = view.selectionModel()
            if sel and sel.currentIndex().isValid():
                idx = sel.currentIndex()
        if not idx.isValid():
            return -1

        model = view.model()
        if hasattr(model, "mapToSource"):
            idx = model.mapToSource(idx)
        return idx.row()

    def _sample_rows_from_source_model(self):
        model = getattr(self, "_proxy", None)
        if model is not None and hasattr(model, "sourceModel"):
            model = model.sourceModel()
        else:
            model = self.ui.tbl_slist.model()
            if hasattr(model, "sourceModel"):
                model = model.sourceModel()

        if model is None:
            return []

        rows = []
        for r in range(model.rowCount()):
            row = []
            for c in range(3):
                idx = model.index(r, c)
                row.append(str(model.data(idx, Qt.DisplayRole) or ""))
            row[2] = ",".join(self._annotation_paths_from_cell(row[2])) or "."
            rows.append(row)
        return rows

    def _sample_row_from_index(self, index):
        if not index.isValid():
            return None

        model = index.model()
        if model is None:
            return None

        src_index = model.mapToSource(index) if hasattr(model, "mapToSource") else index
        src_model = src_index.model()
        if src_model is None or not src_index.isValid():
            return None

        row = []
        for col in range(3):
            idx = src_model.index(src_index.row(), col)
            row.append(str(src_model.data(idx, Qt.DisplayRole) or ""))
        row[2] = ",".join(self._annotation_paths_from_cell(row[2])) or "."
        return row

    def _sync_proj_sample_list_from_view(self):
        self._slist_proj_sync_skipped = False
        if not getattr(self, "_slist_proj_dirty", False):
            self._slist_proj_sync_skipped = True
            return None
        rows = self._sample_rows_from_source_model()
        self.proj.clear()
        if rows:
            self.proj.eng.set_sample_list(rows)
        self._slist_proj_dirty = False
        return rows

    def _replace_sample_rows(self, rows, selected_source_row=0):
        self._slist_loaded_key = None
        self.proj.clear()
        self.proj.eng.set_sample_list(rows)
        self._slist_proj_dirty = False
        df = self.proj.sample_list()
        model = self.sample_list_df_to_model(df)
        self._proxy.setSourceModel(model)
        self._configure_slist_view()

        selected_source_row = max(0, min(selected_source_row, model.rowCount() - 1)) if model.rowCount() else -1
        if selected_source_row >= 0:
            src_idx = model.index(selected_source_row, 0)
            proxy_idx = self._proxy.mapFromSource(src_idx)
            if proxy_idx.isValid():
                self.ui.tbl_slist.setCurrentIndex(proxy_idx)
                self.ui.tbl_slist.selectRow(proxy_idx.row())

    def _find_matching_annotation_file(self, edf_file: str):
        exts = [".annot", ".xml", ".eannot", ".tsv"]
        p = Path(edf_file)
        stem = p.stem.lower()
        nsrr_stem = f"{stem}-nsrr"
        parent = p.parent
        if not parent.exists():
            return None

        by_ext = {e: [] for e in exts}
        for cand in parent.iterdir():
            if not cand.is_file():
                continue
            cand_stem = cand.stem.lower()
            if cand_stem != stem and cand_stem != nsrr_stem:
                continue
            sfx = cand.suffix.lower()
            if sfx in by_ext:
                by_ext[sfx].append(cand)

        for e in exts:
            if by_ext[e]:
                by_ext[e].sort(key=lambda x: x.name.lower())
                return str(by_ext[e][0])
        return None

    def _sample_list_base_dir(self) -> str:
        label = self._get_slist_label_full_text()
        if label and label not in {"<internal>", "(built)"}:
            try:
                p = Path(label).expanduser()
                if p.exists() or p.parent.exists():
                    return str(p.resolve().parent)
            except Exception:
                pass
        try:
            return os.getcwd()
        except Exception:
            return QDir.currentPath()

    def _attach_annotation_folder(self):
        rows = self._sample_rows_from_source_model()
        if not rows:
            QMessageBox.information(self.ui, "Attach Annotation Folder", "No sample list is loaded.")
            return

        ids = [row[0] for row in rows if str(row[0] or "").strip()]
        if not ids:
            QMessageBox.information(self.ui, "Attach Annotation Folder", "The sample list does not contain any IDs.")
            return

        folder, annot_ext, path_mode = attach_annots_directory(
            self.ui,
            "Select Annotation Folder",
            ids,
            QDir.currentPath(),
        )
        if not folder:
            return

        annot_exts = normalize_build_exts(annot_ext)
        if not annot_exts:
            QMessageBox.information(self.ui, "Attach Annotation Folder", "No annotation suffixes were specified.")
            return

        base_dir = self._sample_list_base_dir()

        annot_index: dict[str, list[str]] = {}
        for root, _dirs, files in os.walk(folder):
            for name in sorted(files):
                abs_path = os.path.abspath(os.path.join(root, name))
                stored_path = abs_path
                if path_mode == "relative":
                    try:
                        stored_path = os.path.relpath(abs_path, base_dir)
                    except Exception:
                        stored_path = abs_path
                for ext in annot_exts:
                    annot_stem = _annot_stem_for_ext(name, ext)
                    if annot_stem is None:
                        continue
                    annot_index.setdefault(annot_stem, []).append(stored_path)

        def _norm_path(p: str) -> str:
            try:
                path_str = str(p)
                if not os.path.isabs(path_str):
                    path_str = os.path.join(base_dir, path_str)
                return os.path.normcase(os.path.realpath(os.path.abspath(os.path.expanduser(path_str))))
            except Exception:
                return os.path.normcase(str(p))

        selected_row = self._current_slist_source_row()
        updated_rows = []
        rows_updated = 0
        files_added = 0

        for row in rows:
            id_str = str(row[0] or "").strip()
            existing = self._annotation_paths_from_cell(row[2])
            existing_norm = {_norm_path(path) for path in existing}
            new_paths = []
            for cand in annot_index.get(id_str, []):
                norm = _norm_path(cand)
                if norm in existing_norm:
                    continue
                new_paths.append(cand)
                existing_norm.add(norm)

            if new_paths:
                rows_updated += 1
                files_added += len(new_paths)
                row = list(row)
                row[2] = ",".join([*existing, *new_paths]) or "."
            updated_rows.append(row)

        if files_added == 0:
            QMessageBox.information(
                self.ui,
                "Attach Annotation Folder",
                "No new matching annotation files were found for the current sample list.",
            )
            return

        self._replace_sample_rows(updated_rows, selected_row if selected_row >= 0 else 0)
        QMessageBox.information(
            self.ui,
            "Attach Annotation Folder",
            f"Added {files_added} annotation file(s) across {rows_updated} sample-list row(s).",
        )

    def _init_slist(self):

        # metadata state
        self._meta_rows = {}          # id -> {col: val}
        self._meta_cols = []          # display columns (up to 10, after filter)
        self._meta_file = ""          # path of the currently loaded meta file
        self._meta_vars_filter = []   # from config meta-data-vars
        self._slist_loaded_key = None
        self._slist_proj_dirty = False
        self._slist_proj_sync_skipped = False

        # Flag set while filter text is changing so currentRowChanged is ignored.
        # Connected BEFORE attach_comma_filter so this handler fires first.
        self._slist_filter_changing = False
        self.ui.flt_slist.textChanged.connect(self._on_slist_filter_text_changing)

        # attach comma-delimited OR filter; restrict to col 0 (ID) only
        self._proxy = attach_comma_filter(
            self.ui.tbl_slist,
            self.ui.flt_slist,
            NumericSortFilterProxy(self.ui.tbl_slist),
        )
        self._proxy.setFilterKeyColumn(0)
        self._slist_delegate = SampleListCompactDelegate(self.ui.tbl_slist)
        self.ui.tbl_slist.setItemDelegateForColumn(0, self._slist_delegate)
        self.ui.tbl_slist.clicked.connect(self._on_slist_clicked)
        self.ui.tbl_slist.setToolTip("Click a sample row to show its EDF and annotation paths.")
        self._proxy.modelReset.connect(self._refresh_slist_row_heights)
        self._configure_slist_view()
        self.ui.lbl_slist.setMinimumWidth(0)
        self.ui.lbl_slist.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.ui.lbl_slist.installEventFilter(self)
        self._set_slist_label(self.ui.lbl_slist.text())
        
        # wire buttons
        self.ui.butt_load_slist.clicked.connect(self.open_file)
        self.ui.butt_build_slist.clicked.connect(self.open_folder)
        self.ui.butt_load_edf.clicked.connect(lambda _checked=False: self.open_edf())        
        self.ui.butt_load_annot.clicked.connect(lambda _checked=False: self.open_annot())
        self.ui.butt_refresh.clicked.connect(self._refresh)
        
        # wire select ID from slist --> show details and attach the EDF/annotations
        self.ui.tbl_slist.selectionModel().currentRowChanged.connect(self._on_slist_current_row_changed)
        
        

    # ------------------------------------------------------------
    # Load slist from a file
    # ------------------------------------------------------------
        
    def open_file(self):

        slist, _ = open_file_name(
            self.ui,
            "Open sample-list file",
            "",
            "slist (*.lst *.txt);;All Files (*)",
        )
        self._read_slist_from_file( slist )


    def save_file(self):
        rows = self._sample_rows_from_source_model()
        if not rows:
            QMessageBox.information(self.ui, "Save Sample List", "No sample list is loaded.")
            return

        current_label = self._get_slist_label_full_text()
        default_name = current_label if current_label and current_label not in {"<internal>", "(built)"} else "sample.lst"
        filename, _ = save_file_name(
            self.ui,
            "Save sample-list file",
            default_name,
            "slist (*.lst *.txt);;All Files (*)",
        )
        if not filename:
            return

        try:
            with open(filename, "w", encoding="utf-8") as fh:
                for row in rows:
                    vals = []
                    for value in row[:3]:
                        text = str(value or "").strip()
                        vals.append(text if text else ".")
                    fh.write("\t".join(vals) + "\n")
        except Exception as e:
            QMessageBox.critical(
                self.ui,
                "Error",
                f"Could not save sample list '{filename}':\n{e}",
            )
            return

        self._set_slist_label(filename)


    def _apply_sample_list_df(self, df, label: str):
        self._slist_loaded_key = None
        self._slist_proj_dirty = False
        model = self.sample_list_df_to_model(df)
        self._proxy.setSourceModel(model)
        self._configure_slist_view()
        self._set_slist_label(label)

    def _configure_slist_view(self):
        view = self.ui.tbl_slist
        h = view.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.Interactive)
        view.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.setSelectionMode(QAbstractItemView.SingleSelection)
        view.verticalHeader().setVisible(True)
        if hasattr(self, "_slist_delegate"):
            self._slist_delegate.clear_expanded()
        model = view.model()
        if model is None:
            return
        for col in range(model.columnCount()):
            view.setColumnHidden(col, col != 0)
        if model.columnCount() > 0:
            h.setSectionResizeMode(0, QHeaderView.Stretch)
            h.setStretchLastSection(True)
            sel = view.selectionModel()
            if sel and sel.currentIndex().isValid():
                self._slist_delegate.set_expanded_only(sel.currentIndex())
        self._refresh_slist_row_heights()

    def _refresh_slist_row_heights(self):
        model = self.ui.tbl_slist.model()
        if model is None:
            return
        for row in range(model.rowCount()):
            self.ui.tbl_slist.resizeRowToContents(row)

    def _expand_slist_row(self, index):
        if not index.isValid() or not hasattr(self, "_slist_delegate"):
            return
        self._slist_delegate.set_expanded_only(index)
        self.ui.tbl_slist.resizeRowToContents(index.row())
        self.ui.tbl_slist.viewport().update()

    def _slist_row_key(self, index):
        row = self._sample_row_from_index(index)
        return tuple(row) if row else None

    def _attach_slist_index_if_needed(self, index):
        key = self._slist_row_key(index)
        if key is None:
            return
        if key == getattr(self, "_slist_loaded_key", None):
            return
        self._attach_inst(index, None)
        try:
            loaded_id = str(self.p.id()).strip()
        except Exception:
            loaded_id = ""
        if loaded_id == str(key[0]).strip():
            self._slist_loaded_key = key

    def _on_slist_clicked(self, index):
        self._expand_slist_row(index)
        self._attach_slist_index_if_needed(index)

    def _on_slist_filter_text_changing(self, _text):
        """Called first (before attach_comma_filter's handler) when filter text changes."""
        self._slist_filter_changing = True
        QTimer.singleShot(0, self._on_slist_filter_done)

    def _on_slist_filter_done(self):
        """Reset flag after the proxy has finished re-filtering, then repaint."""
        self._slist_filter_changing = False
        self.ui.tbl_slist.viewport().update()

    def _on_slist_current_row_changed(self, current, previous):
        # Selection changes can be caused by filtering or keyboard movement.
        # Filtering should not implicitly load; direct cursor movement should.
        if self._slist_filter_changing:
            return

        if current.isValid() and hasattr(self, "_slist_delegate"):
            self._slist_delegate.set_expanded_only(current)
            self._refresh_slist_row_heights()
            self.ui.tbl_slist.viewport().update()
            self._attach_slist_index_if_needed(current)


    # ------------------------------------------------------------
    # Metadata file (TSV / .meta)
    # ------------------------------------------------------------

    def _update_meta_delegate(self):
        if hasattr(self, "_slist_delegate"):
            self._slist_delegate._meta_rows = self._meta_rows
            self._slist_delegate._meta_cols = self._meta_cols
            self._refresh_slist_row_heights()
            self.ui.tbl_slist.viewport().update()

    def _load_meta_file(self, path: str) -> bool:
        """Load a TSV/.meta file; first column = ID, remaining = metadata fields.

        Respects self._meta_vars_filter (comma list from config) and limits to
        10 columns.  Returns True on success, False if the file is missing or
        has an unexpected format (caller may warn the user).
        """
        self._meta_rows = {}
        self._meta_cols = []
        if not path:
            self._update_meta_delegate()
            return False
        try:
            p = Path(path).expanduser()
            if not p.exists():
                self._update_meta_delegate()
                return False
            df = pd.read_csv(str(p), sep="\t", dtype=str, encoding="utf-8-sig")
            if df.empty or df.shape[1] < 2:
                self._update_meta_delegate()
                return False
            id_col = df.columns[0]
            data_cols = list(df.columns[1:])
            if self._meta_vars_filter:
                data_cols = [c for c in data_cols if c in self._meta_vars_filter]
            data_cols = data_cols[:10]
            if not data_cols:
                self._update_meta_delegate()
                return False
            rows: dict[str, dict[str, str]] = {}
            for _, row in df.iterrows():
                id_val = str(row[id_col]).strip()
                if id_val and id_val not in ("", ".", "nan", "NaN"):
                    rows[id_val] = {
                        c: (str(row[c]) if pd.notna(row[c]) else "") for c in data_cols
                    }
            self._meta_rows = rows
            self._meta_cols = data_cols
            self._meta_file = str(p)
        except Exception:
            self._meta_rows = {}
            self._meta_cols = []
            self._update_meta_delegate()
            return False
        self._update_meta_delegate()
        return True

    def _clear_meta(self):
        self._meta_rows = {}
        self._meta_cols = []
        self._meta_file = ""
        self._update_meta_delegate()

    def _load_meta_interactive(self):
        from ..file_dialogs import open_file_name
        path, _ = open_file_name(
            self.ui,
            "Open metadata file",
            self._meta_file or "",
            "Metadata (*.tsv *.meta);;All Files (*)",
        )
        if not path:
            return
        ok = self._load_meta_file(path)
        if not ok:
            QMessageBox.warning(
                self.ui,
                "Metadata",
                f"Could not load metadata from:\n{path}\n\n"
                "Expected a tab-separated file with IDs in the first column.",
            )

    def _build_slist_from_folder(self, folder: str, annot_ext: str = ""):
        if not folder:
            return
        annot_ext = normalize_build_ext(annot_ext)
        # Building a new sample list from a folder should replace the current
        # list, not append to whatever the engine singleton already holds.
        self.proj.clear()
        if annot_ext:
            self.proj.build([folder, f"-ext={annot_ext}"])
        else:
            self.proj.build(folder)
        self._slist_proj_dirty = False
        df = self.proj.sample_list()
        self._apply_sample_list_df(df, "(built)")


    # ------------------------------------------------------------
    # Build slist from a folder
    # ------------------------------------------------------------

    def _read_slist_from_file( self, slist : str ):
        if slist:
            try:
                self.proj.clear()
                self.proj.eng.set_sample_list(_read_sample_list_rows(slist))
                self._slist_proj_dirty = False
                df = self.proj.sample_list()
            except Exception as e:
                raise RuntimeError(f"Could not load sample list '{slist}': {e}") from e

            self._apply_sample_list_df(df, slist)

            # Auto-discover companion metadata file (only if not already loaded from config)
            if not self._meta_rows:
                stem = Path(slist).with_suffix("")
                for meta_ext in (".tsv", ".meta"):
                    candidate = stem.with_suffix(meta_ext)
                    if candidate.exists():
                        self._load_meta_file(str(candidate))
                        break

            
    # ------------------------------------------------------------
    # Build slist from a folder
    # ------------------------------------------------------------
        
    def open_folder(self):

        folder, annot_ext = build_slist_directory(self.ui, "Select Folder", QDir.currentPath())

        # update
        if folder != "":
            self._build_slist_from_folder(folder, annot_ext)

            
    # ------------------------------------------------------------
    # Load EDF from a file
    # ------------------------------------------------------------
        
    def open_edf(self , edf_file = None ):
        
        
        if edf_file is None:
            edf_file , _ = open_file_name(
                self.ui,
                "Open EDF file",
                "",
                "EDF (*.edf *.rec);;All Files (*)",
            )

        # update
        if edf_file != "":

            base = path.splitext(path.basename(edf_file))[0]
            annot_file = "."

            matching_annot = self._find_matching_annotation_file(edf_file)
            if matching_annot is not None:
                box = QMessageBox(self.ui)
                box.setIcon(QMessageBox.Question)
                box.setWindowTitle("Load matching annotation?")
                box.setText(
                    "A matching annotation file was found for this EDF.\n\n"
                    f"{matching_annot}"
                )
                box.setInformativeText("Press Return to load the EDF together with this annotation.")
                load_both_btn = box.addButton("Load EDF + Annotation", QMessageBox.AcceptRole)
                edf_only_btn = box.addButton("Load EDF Only", QMessageBox.RejectRole)
                box.setDefaultButton(load_both_btn)
                box.setEscapeButton(edf_only_btn)
                box.exec()
                if box.clickedButton() is load_both_btn:
                    annot_file = matching_annot

            row = [ base , edf_file , annot_file ] 
            
            # specify SL directly
            self.proj.clear()
            self.proj.eng.set_sample_list( [ row ] )
            self._slist_proj_dirty = False

            # get the SL
            df = self.proj.sample_list()

            # assgin to model
            model = self.sample_list_df_to_model( df )
            self._slist_loaded_key = None
            self._proxy.setSourceModel(model)
            self._configure_slist_view()
            # update label to show slist file
            self._set_slist_label('<internal>')

            # and prgrammatically select this first row
            model = self.ui.tbl_slist.model()
            if model and model.rowCount() > 0:
                proxy_idx = model.index(0, 0)
                self.ui.tbl_slist.setCurrentIndex(proxy_idx)
                self.ui.tbl_slist.selectRow(0)              
            

    # ------------------------------------------------------------
    # Reload same EDF, i.e. refresh

    def _refresh(self):

        view = self.ui.tbl_slist
        model = view.model()
        if not model: return

        sel = view.selectionModel()
        row = 0
        if sel and sel.currentIndex().isValid():
            row = sel.currentIndex().row()

        # if the model changed, clamp to bounds
        row = max(0, min(row, model.rowCount() - 1)) if model.rowCount() else -1
        if row < 0: return

        view.selectRow(row)
        idx = model.index(row, 0)
        self._attach_inst(idx, None)
                        

    # ------------------------------------------------------------
    # Load .annot from a file
        
    def open_annot(self,  annot_file = None ):

        interactive = annot_file is None

        if annot_file is None:
            annot_file , _ = open_file_name(
                self.ui,
                "Open annotation file",
                "",
                "EDF (*.annot *.eannot *.xml *.tsv *.txt);;All Files (*)",
            )

        # update
        if annot_file != "":

            # If called interactively and an instance is already attached,
            # offer to append rather than replace.
            if interactive and hasattr(self, "p"):
                box = QMessageBox(self.ui)
                box.setWindowTitle("Load Annotations")
                box.setText("An EDF/annotation is already loaded.")
                box.setInformativeText(
                    "Add this annotation file to the current EDF, or load the "
                    f"annotation file by itself?\n\n{annot_file}"
                )
                add_button = box.addButton("Add to EDF", QMessageBox.AcceptRole)
                load_only_button = box.addButton("Load annotations only", QMessageBox.ActionRole)
                cancel_button = box.addButton(QMessageBox.Cancel)
                box.setDefaultButton(add_button)
                box.exec()

                clicked = box.clickedButton()
                if clicked == cancel_button:
                    return
                if clicked == add_button:
                    rows = self._sample_rows_from_source_model()
                    selected_row = self._current_slist_source_row()
                    if not (0 <= selected_row < len(rows)):
                        return
                    annots = self._annotation_paths_from_cell(rows[selected_row][2])
                    if annot_file not in annots:
                        annots.append(annot_file)
                    rows[selected_row][2] = ",".join(annots) or "."
                    # Update the C++ sample list so the annotation is stored in
                    # this individual's row only.  _replace_sample_rows then
                    # triggers _attach_inst (via setCurrentIndex), which reloads
                    # the individual cleanly from the updated sample list.
                    # Calling attach_annot() directly is intentionally avoided:
                    # it registers the path globally in the engine singleton and
                    # causes every subsequent individual load to also attempt to
                    # open that file, producing "does not exist for EDF X" errors
                    # and writing the wrong ID in project-mode WRITE-ANNOTS runs.
                    self._replace_sample_rows(rows, selected_row)
                    self._set_slist_label("<internal>")
                    return
                if clicked != load_only_button:
                    return

            base = path.splitext(path.basename(annot_file))[0]

            row = [ base ,".", annot_file ]

            # specify SL directly
            self.proj.clear()
            self.proj.eng.set_sample_list( [ row ] )
            self._slist_proj_dirty = False

            # get the SL
            df = self.proj.sample_list()

            # assgin to model
            model = self.sample_list_df_to_model( df )
            self._slist_loaded_key = None
            self._proxy.setSourceModel(model)
            self._configure_slist_view()
            # update label to show slist file
            self._set_slist_label('<internal>')

            # and prgrammatically select this first row
            model = self.ui.tbl_slist.model()
            if model and model.rowCount() > 0:
                proxy_idx = model.index(0, 0)
                self.ui.tbl_slist.setCurrentIndex(proxy_idx)
                self.ui.tbl_slist.selectRow(0)              


                



    # ------------------------------------------------------------
    # Populate sample-list table
    # ------------------------------------------------------------

    @staticmethod
    def OLD_df_to_model(df) -> QStandardItemModel:
        m = QStandardItemModel(df.shape[0], df.shape[1])
        m.setHorizontalHeaderLabels([str(c) for c in df.columns])
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                v = df.iat[r, c]
                # stringify lists/sets for display
                s = ", ".join(map(str, v)) if isinstance(v, (list, tuple, set)) else ("" if pd.isna(v) else str(v))
                m.setItem(r, c, QStandardItem(s))
        #m.setVerticalHeaderLabels([str(i) for i in df.index])
        return m


    @staticmethod
    def coerce_numeric_df(
        df: pd.DataFrame,
        *,
        decimals_default: int = 5,
        decimals_per_col: dict[str, int] | None = None,
        extra_missing: set[str] | None = None,
        preserve_text_cols: set[str] | None = None,
    ) -> pd.DataFrame:
        miss = {"", ".", "NA", "N/A", "NaN", "NAN"}
        if extra_missing:
            miss |= {s.upper() for s in extra_missing}
        decs = decimals_per_col or {}
        preserve_text_cols = {str(c) for c in (preserve_text_cols or set())}

        def is_listy(x): return isinstance(x, (list, tuple, set))

        def clean_cell(x):
            if x is None: return np.nan
            if isinstance(x, float) and np.isnan(x): return np.nan
            if isinstance(x, str):
                xs = x.strip()
                if xs == "" or xs.upper() in miss: return np.nan
                stripped = xs.replace(",", "")
                try:
                    float(stripped)
                    return stripped   # thousands-separator comma — safe to remove
                except ValueError:
                    return xs         # real string content — keep commas
            return x

        def series_to_numeric(s: pd.Series, name: str) -> pd.Series:
            if s.map(is_listy).any():
                return s  # leave list-like columns as-is

            s2 = s.map(clean_cell)
            num = pd.to_numeric(s2, errors="coerce")
            nonmiss = ~s2.isna()

            # some non-missing failed to parse => keep as text
            if nonmiss.any() and num[nonmiss].isna().any():
                return s2.astype(object)

            # all missing => float column
            if not nonmiss.any():
                return num.astype(float)

            # decide int vs float from fractional part
            frac = np.abs(num - np.rint(num))
            z = frac[nonmiss]
            vmax = float(z.max(skipna=True)) if len(z) else 0.0
            if not np.isfinite(vmax):  # all NaN after skipna
                vmax = 0.0

            if vmax == 0.0:
                return num.round().astype("Int64")  # nullable int
            else:
                d = decs.get(name, decimals_default)
                return num.astype(float).round(d)

        out = df.copy()
        for col in out.columns:
            if str(col) in preserve_text_cols:
                out[col] = out[col].apply(
                    lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else v
                ).astype(object)
                continue
            out[col] = series_to_numeric(out[col], col)
        return out

    @staticmethod
    def df_to_model(
        df: pd.DataFrame,
        *,
        float_decimals_default: int = 3,
        float_decimals_per_col: dict[str, int] | None = None,
        preserve_text_cols: set[str] | None = None,
    ) -> DataFrameModel:
        """Virtual model — Qt only renders visible cells. Fast for large tables."""
        return DataFrameModel(
            df,
            float_decimals_default,
            float_decimals_per_col,
            preserve_text_cols,
        )

    @staticmethod
    def sample_list_df_to_model(df: pd.DataFrame) -> DataFrameModel:
        """Sample-list columns are identifiers/paths, so preserve placeholders."""
        return SListMixin.df_to_model(
            df,
            preserve_text_cols={"ID", "EDF", "Annotations"},
        )

    @staticmethod
    def df_to_std_model(
        df: pd.DataFrame,
        *,
        float_decimals_default: int = 3,
        float_decimals_per_col: dict[str, int] | None = None,
    ) -> QStandardItemModel:
        """Eagerly-materialised QStandardItemModel. Use only when the model
        must be mutated after creation (e.g. add_check_column, insertColumn)."""
        clean = SListMixin.coerce_numeric_df(
            df,
            decimals_default=float_decimals_default,
            decimals_per_col=float_decimals_per_col,
        )

        model = QStandardItemModel(clean.shape[0], clean.shape[1])
        model.setHorizontalHeaderLabels([str(c) for c in clean.columns])

        for r in range(clean.shape[0]):
            for c_idx, col in enumerate(clean.columns):
                v = clean.iat[r, c_idx]
                item = QStandardItem()

                if pd.isna(v):
                    item.setText("")
                elif isinstance(v, (list, tuple, set)):
                    item.setText(", ".join(map(str, v)))
                elif pd.api.types.is_integer_dtype(clean[col].dtype):
                    item.setText(str(int(v)))
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                elif pd.api.types.is_float_dtype(clean[col].dtype):
                    digs = (float_decimals_per_col or {}).get(col, float_decimals_default)
                    item.setText(f"{float(v):.{digs}f}")
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    item.setText(str(v))

                model.setItem(r, c_idx, item)

        return model
