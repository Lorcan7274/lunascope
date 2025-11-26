
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

from PySide6.QtWidgets import QPlainTextEdit, QFileDialog
from PySide6.QtWidgets import QVBoxLayout, QHeaderView
from PySide6.QtWidgets import QMessageBox

import pandas as pd

class SettingsMixin:

    def _init_settings(self):

        # tableview formats

        
        
        # wiring

        self.ui.butt_load_param.clicked.connect( self._load_param )
        self.ui.butt_save_param.clicked.connect( self._save_param )
        self.ui.butt_reset_param.clicked.connect( self._reset_param )

    

    # ------------------------------------------------------------
    # load/save functions

    def _load_param(self):

        # if this for cmap or param?
        # determine based on the open tab
        
        idx = self.ui.tab_settings.currentIndex()
        name = self.ui.tab_settings.tabText(idx)
        is_cmap = name == 'Config'

        if is_cmap is False:
            txt_file, _ = QFileDialog.getOpenFileName(
                self.ui,
                "Open a parameter file",
                "",
                "Text (*.txt);;All Files (*)",
                options=QFileDialog.Option.DontUseNativeDialog
            )
        else:
            txt_file, _ = QFileDialog.getOpenFileName(
                self.ui,
                "Open a config file",
                "",
                "Text (*.txt *.cfg);;All Files (*)",
                options=QFileDialog.Option.DontUseNativeDialog
            )
            
        
        if txt_file:
            try:
                text = open(txt_file, "r", encoding="utf-8").read()
                if is_cmap:
                    self.ui.txt_cmap.setPlainText(text)
                else:
                    self.ui.txt_param.setPlainText(text)
            except (UnicodeDecodeError, OSError) as e:
                QMessageBox.critical(
                    None,
                    "Error opening file",
                    f"Could not load {txt_file}\nException: {type(e).__name__}: {e}"
                )


    def _save_param(self):

        # if this for cmap or param?
        # determine based on the open tab
        idx = self.ui.tab_settings.currentIndex()
        name = self.ui.tab_settings.tabText(idx)
        is_cmap = name == 'Config'
        is_param = name == 'Param'

        if is_cmap is False and is_param is False:
            QMessageBox.critical(
                None,
                "Error saving file",
                "Need to select either the Param or Config tab to Save" )
            return
        
        if is_cmap is True:
            new_file = self.ui.txt_cmap.toPlainText()
            filename, selected_filter = QFileDialog.getSaveFileName(
                self.ui,
                "Save file to .cfg",
                "",
                "Config (text) Files (*.cfg *.txt);;All Files (*)",
                options=QFileDialog.Option.DontUseNativeDialog
            )            
        else:
            new_file = self.ui.txt_param.toPlainText()            
            filename, selected_filter = QFileDialog.getSaveFileName(
                self.ui,
                "Save file to .txt",
                "",
                "Text Files (*.txt);;All Files (*)",
                options=QFileDialog.Option.DontUseNativeDialog
            )

        if filename:
            # Ensure .txt extension if none was given
            if selected_filter.startswith("Text") and not filename.lower().endswith(".txt"):
                filename += ".txt"
                
            with open(filename, "w", encoding="utf-8") as f:
                f.write(new_file)



    # ------------------------------------------------------------
    # reset all parameters

    def _reset_param(self):
        
        idx = self.ui.tab_settings.currentIndex()
        name = self.ui.tab_settings.tabText(idx)
        is_cmap = name == 'Config'

        if is_cmap is True:
            self.ui.txt_cmap.clear()
            self._clear_cmap()
            self._apply_cmap()
        else:
            self.ui.txt_param.clear()
            self.proj.clear_vars()
            self.proj.reinit()
            self._update_params()
        
    # ------------------------------------------------------------
    # reset all parameters: called when attaching a new EDF

    def _update_params(self):
        
        # get aliases
        aliases = self.proj.eng.aliases()
        df = pd.DataFrame(aliases, columns=["Type", "Primary", "Secondary"])
        model = self.df_to_model( df )
        self.ui.tbl_aliases.setModel( model )
        view = self.ui.tbl_aliases
        view.verticalHeader().setVisible(False)
        view.resizeColumnsToContents()
        view.setSortingEnabled(False)
        h = view.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.Interactive)
        h.setStretchLastSection(True)
        view.resizeColumnsToContents()
    
        # get special variables
        vars = self.proj.vars()
        df = pd.DataFrame(list(vars.items()), columns=["Variable", "Value"])        
        model = self.df_to_model( df )
        self.ui.tbl_param.setModel( model )
        view = self.ui.tbl_param
        view.verticalHeader().setVisible(False)
        view.resizeColumnsToContents()
        view.setSortingEnabled(False)
        h = view.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.Interactive)
        h.setStretchLastSection(True)
        view.resizeColumnsToContents()
