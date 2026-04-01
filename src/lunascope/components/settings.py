
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


def _append_selected_extension(filename: str, selected_filter: str, allowed_exts: tuple[str, ...]) -> str:
    lower = filename.lower()
    if any(lower.endswith(ext) for ext in allowed_exts):
        return filename

    filt = (selected_filter or "").lower()
    for ext in allowed_exts:
        if f"*{ext}" in filt:
            return filename + ext

    return filename + allowed_exts[0]


class SettingsMixin:

    def _init_settings(self):

        # tableview formats

        
        
        # wiring

        self.ui.txt_cmap.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.ui.butt_load_param.clicked.connect( self._load_param )
        self.ui.butt_save_param.clicked.connect( self._save_param )
        self.ui.butt_reset_param.clicked.connect( self._reset_param )
        self.ui.tab_settings.currentChanged.connect( self._sync_settings_buttons )
        self._sync_settings_buttons()

    def _settings_tab_name(self) -> str:
        idx = self.ui.tab_settings.currentIndex()
        return self.ui.tab_settings.tabText(idx)

    def _sync_settings_buttons(self, *_args):
        name = self._settings_tab_name()
        is_param = name == 'Param'
        is_cmap = name == 'Config'

        self.ui.butt_load_param.setEnabled(is_param or is_cmap)
        self.ui.butt_save_param.setEnabled(is_param or is_cmap)
        self.ui.butt_reset_param.setText("Apply" if is_cmap else "Reset")

    def _apply_current_cmap(self):
        okay = self._apply_cmaps()
        if not okay:
            return

        if not hasattr(self, "p"):
            return

        if getattr(self, "rendered", False):
            self._render_signals()
        else:
            self._render_signals_simple()

    

    # ------------------------------------------------------------
    # load/save functions

    def _load_param(self):

        # if this for cmap or param?
        # determine based on the open tab
        
        name = self._settings_tab_name()
        is_cmap = name == 'Config'

        if is_cmap is False:
            txt_file, _ = QFileDialog.getOpenFileName(
                self.ui,
                "Open a parameter file",
                "",
                "Param Files (*.txt *.par *);;All Files (*)",
                options=QFileDialog.Option.DontUseNativeDialog
            )
        else:
            txt_file, _ = QFileDialog.getOpenFileName(
                self.ui,
                "Open a config file",
                "",
                "Config Files (*.txt *.cfg *);;All Files (*)",
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
        name = self._settings_tab_name()
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
                "Config Files (*.txt *.cfg *);;All Files (*)",
                options=QFileDialog.Option.DontUseNativeDialog
            )            
        else:
            new_file = self.ui.txt_param.toPlainText()            
            filename, selected_filter = QFileDialog.getSaveFileName(
                self.ui,
                "Save file to .par/.txt",
                "",
                "Param Files (*.txt *.par *);;All Files (*)",
                options=QFileDialog.Option.DontUseNativeDialog
            )

        if filename:
            if is_cmap:
                filename = _append_selected_extension(filename, selected_filter, (".txt", ".cfg"))
            else:
                filename = _append_selected_extension(filename, selected_filter, (".txt", ".par"))
                
            with open(filename, "w", encoding="utf-8") as f:
                f.write(new_file)



    # ------------------------------------------------------------
    # reset all parameters

    def _reset_param(self):
        
        name = self._settings_tab_name()
        is_cmap = name == 'Config'

        if is_cmap is True:
            self._apply_current_cmap()
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
