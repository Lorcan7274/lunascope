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
from pathlib import Path
        
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHeaderView,
    QAbstractItemView,
    QMessageBox,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
)
from PySide6.QtCore import Qt, QDir, QRegularExpression, QSortFilterProxyModel, QAbstractTableModel, QModelIndex, QSize
from PySide6.QtGui import QStandardItemModel, QStandardItem, QPalette, QIcon

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_integer_dtype

from .tbl_funcs import attach_comma_filter
from ..file_dialogs import existing_directory, open_file_name


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

    def __init__(self, df, float_decimals_default=3, float_decimals_per_col=None, parent=None):
        super().__init__(parent)
        # coerce_numeric_df does df.copy() internally — model owns its data
        # SListMixin is defined later in this same file; resolved at call time
        self._df = SListMixin.coerce_numeric_df(
            df,
            decimals_default=float_decimals_default,
            decimals_per_col=float_decimals_per_col or {},
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
            height += (fm.height() + 4) * 2
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
            detail_rect = body_rect.adjusted(0, fm.height() + 4, 0, 0)
            painter.setPen(secondary)
            detail1 = fm.elidedText(f"EDF: {edf_text}", Qt.ElideMiddle, max(40, detail_rect.width()))
            detail2 = fm.elidedText(f"Annot: {annot_text}", Qt.ElideMiddle, max(40, detail_rect.width()))
            painter.drawText(detail_rect, Qt.AlignLeft | Qt.AlignTop, detail1)
            painter.drawText(detail_rect.adjusted(0, fm.height() + 4, 0, 0), Qt.AlignLeft | Qt.AlignTop, detail2)

        painter.restore()


class SListMixin:

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

    def _init_slist(self):

        # attach comma-delimited OR filter
        self._proxy = attach_comma_filter( self.ui.tbl_slist , self.ui.flt_slist )
        self._slist_delegate = SampleListCompactDelegate(self.ui.tbl_slist)
        self.ui.tbl_slist.setItemDelegateForColumn(0, self._slist_delegate)
        self.ui.tbl_slist.clicked.connect(self._expand_slist_row)
        self.ui.tbl_slist.setToolTip("Click a sample row to show its EDF and annotation paths.")
        self._proxy.modelReset.connect(self._refresh_slist_row_heights)
        self._proxy.layoutChanged.connect(self._refresh_slist_row_heights)
        self._proxy.rowsInserted.connect(lambda *_args: self._refresh_slist_row_heights())
        self._proxy.rowsRemoved.connect(lambda *_args: self._refresh_slist_row_heights())
        self._configure_slist_view()
        
        # wire buttons
        self.ui.butt_load_slist.clicked.connect(self.open_file)
        self.ui.butt_build_slist.clicked.connect(self.open_folder)
        self.ui.butt_load_edf.clicked.connect(lambda _checked=False: self.open_edf())        
        self.ui.butt_load_annot.clicked.connect(lambda _checked=False: self.open_annot())
        self.ui.butt_refresh.clicked.connect(self._refresh)
        
        # wire select ID from slist --> load + show details
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

        # set the path , i.e. to handle relative sample lists

        folder_path = str(Path(slist).parent) + os.sep

        self.proj.var( 'path' , folder_path )
        
        self._read_slist_from_file( slist )


    def _apply_sample_list_df(self, df, label: str):
        model = self.df_to_model(df)
        self._proxy.setSourceModel(model)
        self._configure_slist_view()
        self.ui.lbl_slist.setText(label)

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

    def _on_slist_current_row_changed(self, current, previous):
        if current.isValid() and hasattr(self, "_slist_delegate"):
            self._slist_delegate.set_expanded_only(current)
            self._refresh_slist_row_heights()
            self.ui.tbl_slist.viewport().update()
        self._attach_inst(current, previous)


    def _build_slist_from_folder(self, folder: str):
        if not folder:
            return
        self.proj.build(folder)
        df = self.proj.sample_list()
        self._apply_sample_list_df(df, folder)


    # ------------------------------------------------------------
    # Build slist from a folder
    # ------------------------------------------------------------

    def _read_slist_from_file( self, slist : str ):
        if slist:
            try:
                self.proj.sample_list(slist)
                df = self.proj.sample_list()
            except Exception as e:
                raise RuntimeError(f"Could not load sample list '{slist}': {e}") from e

            self._apply_sample_list_df(df, slist)

            
    # ------------------------------------------------------------
    # Build slist from a folder
    # ------------------------------------------------------------
        
    def open_folder(self):

        folder = existing_directory(self.ui, "Select Folder", QDir.currentPath())

        # update
        if folder != "":
            self._build_slist_from_folder(folder)

            
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
                msg = (
                    f"Found matching annotation file for this EDF:\n\n"
                    f"{matching_annot}\n\n"
                    f"Load it together with the EDF?"
                )
                ans = QMessageBox.question(
                    self.ui,
                    "Load Matching Annotation?",
                    msg,
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if ans == QMessageBox.Yes:
                    annot_file = matching_annot

            row = [ base , edf_file , annot_file ] 
            
            # specify SL directly
            self.proj.clear()
            self.proj.eng.set_sample_list( [ row ] )

            # get the SL
            df = self.proj.sample_list()

            # assgin to model
            model = self.df_to_model( df )
            self._proxy.setSourceModel(model)
            self._configure_slist_view()
            # update label to show slist file
            self.ui.lbl_slist.setText( '<internal>' )

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
                ans = QMessageBox.question(
                    self.ui,
                    "Append or Replace?",
                    f"An EDF/annotation is already loaded.\n\n"
                    f"Append this annotation to the current instance, "
                    f"or replace the current instance?\n\n"
                    f"{annot_file}",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                    QMessageBox.Yes,
                )
                # Yes = Append, No = Replace, Cancel = abort
                if ans == QMessageBox.Cancel:
                    return
                if ans == QMessageBox.Yes:
                    try:
                        self.p.attach_annot(annot_file)
                    except Exception as e:
                        QMessageBox.critical(
                            self.ui,
                            "Error",
                            f"Could not append annotation:\n{e}",
                        )
                        return
                    # Refresh annotation-related displays
                    self._update_metrics()
                    self._render_hypnogram()
                    self._render_signals_simple()
                    return

            base = path.splitext(path.basename(annot_file))[0]

            row = [ base ,".", annot_file ]

            # specify SL directly
            self.proj.clear()
            self.proj.eng.set_sample_list( [ row ] )

            # get the SL
            df = self.proj.sample_list()

            # assgin to model
            model = self.df_to_model( df )
            self._proxy.setSourceModel(model)
            self._configure_slist_view()
            # update label to show slist file
            self.ui.lbl_slist.setText( '<internal>' )

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
    ) -> pd.DataFrame:
        miss = {"", ".", "NA", "N/A", "NaN", "NAN"}
        if extra_missing:
            miss |= {s.upper() for s in extra_missing}
        decs = decimals_per_col or {}

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
            out[col] = series_to_numeric(out[col], col)
        return out

    @staticmethod
    def df_to_model(
        df: pd.DataFrame,
        *,
        float_decimals_default: int = 3,
        float_decimals_per_col: dict[str, int] | None = None,
    ) -> DataFrameModel:
        """Virtual model — Qt only renders visible cells. Fast for large tables."""
        return DataFrameModel(df, float_decimals_default, float_decimals_per_col)

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
