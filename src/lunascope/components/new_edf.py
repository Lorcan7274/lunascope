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

import re

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit,
    QSpinBox, QDialogButtonBox, QMessageBox, QLabel,
)
from PySide6.QtCore import Qt


_TIME_RE = re.compile(r"^\d{2}\.\d{2}\.\d{2}$")
_DATE_RE = re.compile(r"^\d{2}\.\d{2}\.\d{2}$")


class _NewEmptyEDFDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Empty EDF")
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self._id = QLineEdit("empty")
        self._time = QLineEdit("22.00.00")
        self._date = QLineEdit("01.01.00")

        self._rs = QSpinBox()
        self._rs.setRange(1, 60)
        self._rs.setValue(1)
        self._rs.setSuffix(" sec")

        self._nr = QSpinBox()
        self._nr.setRange(1, 86400)
        self._nr.setValue(3600)

        form.addRow("ID:", self._id)
        form.addRow("Start time (HH.MM.SS):", self._time)
        form.addRow("Start date (DD.MM.YY):", self._date)
        form.addRow("Record size:", self._rs)
        form.addRow("Number of records:", self._nr)

        hint = QLabel("Duration: <computed>")
        hint.setObjectName("hint_label")
        form.addRow("", hint)
        self._hint = hint
        self._update_hint()

        self._nr.valueChanged.connect(self._update_hint)
        self._rs.valueChanged.connect(self._update_hint)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Create")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _update_hint(self):
        total = self._nr.value() * self._rs.value()
        h = total // 3600
        m = (total % 3600) // 60
        s = total % 60
        self._hint.setText(f"Duration: {h:02d}:{m:02d}:{s:02d}")

    def _on_accept(self):
        id_str = self._id.text().strip()
        if not id_str:
            QMessageBox.warning(self, "New Empty EDF", "ID cannot be empty.")
            return
        if not _TIME_RE.match(self._time.text().strip()):
            QMessageBox.warning(
                self, "New Empty EDF",
                "Start time must be in HH.MM.SS format (e.g. 22.00.00)."
            )
            return
        if not _DATE_RE.match(self._date.text().strip()):
            QMessageBox.warning(
                self, "New Empty EDF",
                "Start date must be in DD.MM.YY format (e.g. 01.01.00)."
            )
            return
        self.accept()

    def values(self):
        return {
            "id":   self._id.text().strip(),
            "time": self._time.text().strip(),
            "date": self._date.text().strip(),
            "rs":   self._rs.value(),
            "nr":   self._nr.value(),
        }


class NewEDFMixin:

    def _new_empty_edf_dialog(self):
        dlg = _NewEmptyEDFDialog(self.ui)
        if dlg.exec() != QDialog.Accepted:
            return
        v = dlg.values()
        self._launch_new_empty_edf(v["id"], v["nr"], v["rs"], v["date"], v["time"])

    def _launch_new_empty_edf(self, id_str, nr, rs, startdate, starttime):
        # Build an in-memory EDF with user-specified parameters.
        # Store it as a pending instance so _attach_inst uses it
        # instead of calling proj.inst() (which would use defaults).
        inst = self.proj.empty_inst(id_str, nr, rs, startdate, starttime)
        self._pending_empty_inst    = inst
        self._pending_empty_inst_id = id_str

        self.proj.clear()
        self.proj.eng.set_sample_list([[id_str, ".", "."]])
        self._slist_proj_dirty = False

        df    = self.proj.sample_list()
        model = self.sample_list_df_to_model(df)
        self._slist_loaded_key = None
        self._proxy.setSourceModel(model)
        self._configure_slist_view()
        self._set_slist_label("<empty>")

        # Programmatically select row 0; _attach_inst will pick up
        # _pending_empty_inst rather than calling proj.inst().
        view_model = self.ui.tbl_slist.model()
        if view_model and view_model.rowCount() > 0:
            proxy_idx = view_model.index(0, 0)
            self.ui.tbl_slist.setCurrentIndex(proxy_idx)
            self.ui.tbl_slist.selectRow(0)
