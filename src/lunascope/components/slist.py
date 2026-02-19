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
        
from PySide6.QtWidgets import QFileDialog, QHeaderView, QAbstractItemView, QMessageBox
from PySide6.QtCore import Qt, QDir, QRegularExpression, QSortFilterProxyModel
from PySide6.QtGui import QStandardItemModel, QStandardItem

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype, is_integer_dtype

from .tbl_funcs import attach_comma_filter

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
        
        # wire buttons
        self.ui.butt_load_slist.clicked.connect(self.open_file)
        self.ui.butt_build_slist.clicked.connect(self.open_folder)
        self.ui.butt_load_edf.clicked.connect(lambda _checked=False: self.open_edf())        
        self.ui.butt_load_annot.clicked.connect(lambda _checked=False: self.open_annot())
        self.ui.butt_refresh.clicked.connect(self._refresh)
        
        # wire select ID from slist --> load
        self.ui.tbl_slist.selectionModel().currentRowChanged.connect( self._attach_inst )
        
        

    # ------------------------------------------------------------
    # Load slist from a file
    # ------------------------------------------------------------
        
    def open_file(self):

        slist, _ = QFileDialog.getOpenFileName(
            self.ui,
            "Open sample-list file",
            "",
            "slist (*.lst *.txt);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        # set the path , i.e. to handle relative sample lists

        folder_path = str(Path(slist).parent) + os.sep

        self.proj.var( 'path' , folder_path )
        
        self._read_slist_from_file( slist )


    # ------------------------------------------------------------
    # Build slist from a folder
    # ------------------------------------------------------------

    def _read_slist_from_file( self, slist : str ):
        if slist:
            # load sample list into luna
            self.proj.sample_list( slist )

            # get the SL
            df = self.proj.sample_list()

            # assgin to model
            model = self.df_to_model( df )              
            self._proxy.setSourceModel(model)

            # display options resize
            view = self.ui.tbl_slist
#            view.setSortingEnabled(True)
            h = view.horizontalHeader()
            h.setSectionResizeMode(QHeaderView.Interactive)  # user-resizable
            h.setStretchLastSection(False)                   # no auto-stretch fighting you
            view.resizeColumnsToContents()  
            view.setSelectionBehavior(QAbstractItemView.SelectRows)
            view.setSelectionMode(QAbstractItemView.SingleSelection)
            view.verticalHeader().setVisible(True)
            # update label to show slist file
            self.ui.lbl_slist.setText( slist )

            
    # ------------------------------------------------------------
    # Build slist from a folder
    # ------------------------------------------------------------
        
    def open_folder(self):

        folder = QFileDialog.getExistingDirectory( self.ui , "Select Folder", QDir.currentPath(),
                                                   options=QFileDialog.Option.DontUseNativeDialog )

        # update
        if folder != "":

            # build SL
            self.proj.build( folder )

            # get the SL
            df = self.proj.sample_list()

            # assgin to model
            model = self.df_to_model( df )              
            self._proxy.setSourceModel(model)

            # display options resize
            view = self.ui.tbl_slist
#            view.setSortingEnabled(True)
            h = view.horizontalHeader()
            h.setSectionResizeMode(QHeaderView.Interactive)  # user-resizable
            h.setStretchLastSection(False)                   # no auto-stretch fighting you
            view.resizeColumnsToContents()  
            view.setSelectionBehavior(QAbstractItemView.SelectRows)
            view.setSelectionMode(QAbstractItemView.SingleSelection)
            view.verticalHeader().setVisible(True)
            # update label to show slist file
            self.ui.lbl_slist.setText( folder )

            
    # ------------------------------------------------------------
    # Load EDF from a file
    # ------------------------------------------------------------
        
    def open_edf(self , edf_file = None ):
        
        
        if edf_file is None:
            edf_file , _ = QFileDialog.getOpenFileName(
                self.ui,
                "Open EDF file",
                "",
                "EDF (*.edf *.rec);;All Files (*)",
                options=QFileDialog.Option.DontUseNativeDialog
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

            # display options resize
            view = self.ui.tbl_slist
#            view.setSortingEnabled(True)
            h = view.horizontalHeader()
            h.setSectionResizeMode(QHeaderView.Interactive)  # user-resizable
            h.setStretchLastSection(False)                   # no auto-stretch fighting you
            view.resizeColumnsToContents()  
            view.setSelectionBehavior(QAbstractItemView.SelectRows)
            view.setSelectionMode(QAbstractItemView.SingleSelection)
            view.verticalHeader().setVisible(True)
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

        if annot_file is None:
            annot_file , _ = QFileDialog.getOpenFileName(
                self.ui,
                "Open annotation file",
                "",
                "EDF (*.annot *.eannot *.xml *.tsv *.txt);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog
            )

        # update
        if annot_file != "":

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

            # display options resize
            view = self.ui.tbl_slist
#            view.setSortingEnabled(True)
            h = view.horizontalHeader()
            h.setSectionResizeMode(QHeaderView.Interactive)  # user-resizable
            h.setStretchLastSection(False)                   # no auto-stretch fighting you
            view.resizeColumnsToContents()  
            view.setSelectionBehavior(QAbstractItemView.SelectRows)
            view.setSelectionMode(QAbstractItemView.SingleSelection)
            view.verticalHeader().setVisible(True)
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
                return xs.replace(",", "")
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
    ) -> QStandardItemModel:
        # Clean/round first
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
