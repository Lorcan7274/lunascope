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

import os
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QCheckBox, QRadioButton,
    QButtonGroup, QListWidget, QListWidgetItem, QPlainTextEdit, QDialogButtonBox,
    QMessageBox, QWidget, QAbstractItemView,
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from ..file_dialogs import existing_directory


# ------------------------------------------------------------
# Annotation class picker (multi-select checklist popup)
# ------------------------------------------------------------

class _AnnotPickerDialog(QDialog):
    """Small popup with a checkable list of annotation class names."""

    def __init__(self, parent, classes, current_selection):
        super().__init__(parent)
        self.setWindowTitle("Select annotation classes")
        self.setMinimumWidth(280)

        layout = QVBoxLayout(self)

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.NoSelection)
        for cls in classes:
            item = QListWidgetItem(cls)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if cls in current_selection else Qt.Unchecked
            )
            self._list.addItem(item)
        layout.addWidget(self._list)

        # Select-all / none row
        toggle_row = QHBoxLayout()
        all_btn  = QPushButton("All")
        none_btn = QPushButton("None")
        all_btn.clicked.connect(self._select_all)
        none_btn.clicked.connect(self._select_none)
        toggle_row.addWidget(all_btn)
        toggle_row.addWidget(none_btn)
        toggle_row.addStretch()
        layout.addLayout(toggle_row)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _select_all(self):
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.Checked)

    def _select_none(self):
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.Unchecked)

    def selected(self):
        """Return sorted list of checked class names."""
        return [
            self._list.item(i).text()
            for i in range(self._list.count())
            if self._list.item(i).checkState() == Qt.Checked
        ]


# ------------------------------------------------------------
# Dialog
# ------------------------------------------------------------

class SaveEDFDialog(QDialog):

    def __init__(self, parent, inst_id, edf_file, has_edf, has_annots, annot_classes):
        super().__init__(parent)
        self.setWindowTitle("Export EDF + Annotations")
        self.setMinimumWidth(580)

        self._inst_id    = inst_id
        self._edf_file   = edf_file
        self._has_edf    = has_edf
        self._has_annots = has_annots
        self._annot_classes = annot_classes
        self._manual_edit   = False

        # Source directory — used for the no-overwrite check
        if has_edf and edf_file and edf_file != ".":
            self._src_dir = str(Path(edf_file).parent)
        else:
            self._src_dir = None

        self._build_ui()
        self._connect_signals()
        self._update_script()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # ── Output folder ──────────────────────────────────────────────
        folder_row = QHBoxLayout()
        folder_row.addWidget(QLabel("Output folder:"))
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Select an output folder…")
        folder_row.addWidget(self._folder_edit, 1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(browse_btn)
        root.addLayout(folder_row)

        # ── EDF options ────────────────────────────────────────────────
        if self._has_edf:
            edf_grp  = QGroupBox("EDF")
            edf_form = QFormLayout(edf_grp)
            edf_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

            self._tag_edit = QLineEdit()
            self._tag_edit.setPlaceholderText("e.g. v2   (Luna adds a hyphen: id-tag.edf)")
            edf_form.addRow("Filename tag:", self._tag_edit)

            self._compress_chk = QCheckBox("Compress output (.edf.gz)")
            edf_form.addRow("", self._compress_chk)

            root.addWidget(edf_grp)
        else:
            self._tag_edit     = None
            self._compress_chk = None

        # ── Annotation options ─────────────────────────────────────────
        if self._has_annots:
            ann_grp  = QGroupBox("Annotations")
            ann_form = QFormLayout(ann_grp)
            ann_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

            annot_row = QHBoxLayout()
            self._annot_edit = QLineEdit()
            self._annot_edit.setPlaceholderText("* (all)  or comma-separated class names")
            annot_row.addWidget(self._annot_edit, 1)
            pick_btn = QPushButton("…")
            pick_btn.setFixedWidth(28)
            pick_btn.setToolTip("Pick annotation classes from list")
            pick_btn.clicked.connect(self._pick_annots)
            annot_row.addWidget(pick_btn)
            ann_form.addRow("Include:", annot_row)

            # Time format radio buttons
            fmt_widget = QWidget()
            fmt_layout = QHBoxLayout(fmt_widget)
            fmt_layout.setContentsMargins(0, 0, 0, 0)
            self._fmt_elapsed = QRadioButton("Elapsed seconds")
            self._fmt_hms     = QRadioButton("hh:mm:ss")
            self._fmt_dhms    = QRadioButton("dd-mm-yyyy-hh:mm:ss")
            self._fmt_elapsed.setChecked(True)
            self._fmt_group = QButtonGroup(self)
            for rb in (self._fmt_elapsed, self._fmt_hms, self._fmt_dhms):
                self._fmt_group.addButton(rb)
                fmt_layout.addWidget(rb)
            fmt_layout.addStretch()
            ann_form.addRow("Time format:", fmt_widget)

            self._xml_chk = QCheckBox("XML format (.xml instead of .annot)")
            ann_form.addRow("", self._xml_chk)

            root.addWidget(ann_grp)
        else:
            self._annot_edit  = None
            self._fmt_elapsed = self._fmt_hms = self._fmt_dhms = None
            self._fmt_group   = None
            self._xml_chk     = None

        # ── Script preview ─────────────────────────────────────────────
        script_grp    = QGroupBox("Luna script (editable)")
        script_layout = QVBoxLayout(script_grp)

        self._script_edit = QPlainTextEdit()
        self._script_edit.setFixedHeight(72)
        mono = QFont("Menlo")
        mono.setStyleHint(QFont.Monospace)
        self._script_edit.setFont(mono)
        script_layout.addWidget(self._script_edit)

        hint_row = QHBoxLayout()
        self._manual_label = QLabel("manually edited — widgets detached")
        self._manual_label.setStyleSheet("color: gray; font-size: 11px;")
        self._manual_label.setVisible(False)
        hint_row.addWidget(self._manual_label)
        hint_row.addStretch()
        self._reset_btn = QPushButton("Reset from widgets")
        self._reset_btn.setVisible(False)
        self._reset_btn.clicked.connect(self._reset_script)
        hint_row.addWidget(self._reset_btn)
        script_layout.addLayout(hint_row)

        root.addWidget(script_grp)

        # ── Dialog buttons ─────────────────────────────────────────────
        btn_box = QDialogButtonBox()
        self._run_btn = btn_box.addButton("Run", QDialogButtonBox.AcceptRole)
        btn_box.addButton(QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_run)
        btn_box.rejected.connect(self.reject)
        root.addWidget(btn_box)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self._folder_edit.textChanged.connect(self._on_widget_changed)
        if self._tag_edit:
            self._tag_edit.textChanged.connect(self._on_widget_changed)
        if self._compress_chk:
            self._compress_chk.toggled.connect(self._on_widget_changed)
        if self._annot_edit:
            self._annot_edit.textChanged.connect(self._on_widget_changed)
        if self._fmt_elapsed:
            self._fmt_elapsed.toggled.connect(self._on_widget_changed)
            self._fmt_hms.toggled.connect(self._on_widget_changed)
            self._fmt_dhms.toggled.connect(self._on_widget_changed)
        if self._xml_chk:
            self._xml_chk.toggled.connect(self._on_widget_changed)
        # Manual-edit detection — connected after initial population
        self._script_edit.textChanged.connect(self._on_script_edited)

    # ------------------------------------------------------------------
    # Script building
    # ------------------------------------------------------------------

    def _build_script(self):
        out_dir = self._folder_edit.text().strip()
        if not out_dir:
            out_dir = "/path/to/output/folder"
        if not out_dir.endswith("/"):
            out_dir += "/"

        lines = []

        tag = self._tag_edit.text().strip() if self._tag_edit else ""
        # Luna inserts a "-" before the tag automatically, so the output stem
        # is always "{id}-{tag}".  Mirror that here for the annot filename.
        stem = f"{self._inst_id}-{tag}" if tag else self._inst_id

        if self._has_edf:
            parts = ["WRITE", f"edf-dir={out_dir}"]
            if tag:
                parts.append(f"edf-tag={tag}")
            if self._compress_chk and self._compress_chk.isChecked():
                parts.append("edfz")
            lines.append(" ".join(parts))

        if self._has_annots:
            use_xml = bool(self._xml_chk and self._xml_chk.isChecked())
            ext      = ".xml" if use_xml else ".annot"
            ann_file = f"{out_dir}{stem}{ext}"

            parts = ["WRITE-ANNOTS", f"file={ann_file}"]

            annot_val = (self._annot_edit.text().strip()
                         if self._annot_edit else "")
            if annot_val and annot_val != "* (all)":
                parts.append(f"annot={annot_val}")

            if self._fmt_hms and self._fmt_hms.isChecked():
                parts.append("hms")
            elif self._fmt_dhms and self._fmt_dhms.isChecked():
                parts.append("dhms")

            if use_xml:
                parts.append("xml")

            lines.append(" ".join(parts))

        return "\n".join(lines)

    def _update_script(self):
        self._script_edit.blockSignals(True)
        self._script_edit.setPlainText(self._build_script())
        self._script_edit.blockSignals(False)

    def _on_widget_changed(self, *_):
        if not self._manual_edit:
            self._update_script()

    def _on_script_edited(self):
        if self._manual_edit:
            return
        # Only flag as manual if the text actually differs from what the
        # widgets would generate (guards against the initial population).
        if self._script_edit.toPlainText() != self._build_script():
            self._manual_edit = True
            self._manual_label.setVisible(True)
            self._reset_btn.setVisible(True)

    def _reset_script(self):
        self._manual_edit = False
        self._manual_label.setVisible(False)
        self._reset_btn.setVisible(False)
        self._update_script()

    # ------------------------------------------------------------------
    # Annotation class picker
    # ------------------------------------------------------------------

    def _pick_annots(self):
        current_text = self._annot_edit.text().strip()
        current_sel  = (
            [s.strip() for s in current_text.split(",") if s.strip()]
            if current_text and current_text != "* (all)"
            else []
        )
        dlg = _AnnotPickerDialog(self, self._annot_classes, current_sel)
        if dlg.exec() != QDialog.Accepted:
            return
        chosen = dlg.selected()
        self._annot_edit.setText(",".join(chosen) if chosen else "")

    # ------------------------------------------------------------------
    # Folder browse
    # ------------------------------------------------------------------

    def _browse_folder(self):
        folder = existing_directory(self, "Select output folder")
        if folder:
            self._folder_edit.setText(folder)

    # ------------------------------------------------------------------
    # Run / validation
    # ------------------------------------------------------------------

    def _expected_output_files(self, out_dir):
        """Return list of (Path, label) for files that will be written."""
        tag = self._tag_edit.text().strip() if self._tag_edit else ""
        out = []

        if self._has_edf and self._edf_file and self._edf_file != ".":
            # Luna uses the original EDF's stem, then appends "-tag" if set.
            # Handle .edf.gz: strip both suffixes to get the bare stem.
            src = Path(self._edf_file)
            if src.suffix.lower() == ".gz":
                edf_stem = src.with_suffix("").stem
            else:
                edf_stem = src.stem
            tagged_stem = f"{edf_stem}-{tag}" if tag else edf_stem
            compressed  = bool(self._compress_chk and self._compress_chk.isChecked())
            ext  = ".edf.gz" if compressed else ".edf"
            out.append((Path(out_dir) / f"{tagged_stem}{ext}", "EDF"))

        if self._has_annots:
            use_xml = bool(self._xml_chk and self._xml_chk.isChecked())
            ext      = ".xml" if use_xml else ".annot"
            stem     = f"{self._inst_id}-{tag}" if tag else self._inst_id
            out.append((Path(out_dir) / f"{stem}{ext}", "Annotations"))

        return out

    def _on_run(self):
        out_dir = self._folder_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "No folder selected",
                                "Please select an output folder before running.")
            return

        # Anti-overwrite: reject if same as EDF source folder
        if self._src_dir:
            try:
                if Path(out_dir).resolve() == Path(self._src_dir).resolve():
                    QMessageBox.critical(
                        self,
                        "Cannot overwrite source",
                        "The output folder is the same as the source EDF folder.\n\n"
                        "Please choose a different folder to prevent overwriting "
                        "the original files.",
                    )
                    return
            except Exception:
                pass

        # Offer to create folder if it doesn't exist yet
        if not os.path.isdir(out_dir):
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle("Create output folder?")
            box.setText(f"Create this output folder?\n\n{out_dir}")
            box.setInformativeText("Press Return to create the folder and continue.")
            create_btn = box.addButton("Create Folder", QMessageBox.AcceptRole)
            cancel_btn = box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(create_btn)
            box.setEscapeButton(cancel_btn)
            box.exec()
            if box.clickedButton() is not create_btn:
                return
            try:
                os.makedirs(out_dir, exist_ok=True)
            except OSError as e:
                QMessageBox.critical(self, "Error", f"Could not create folder:\n{e}")
                return

        # Warn if any output file already exists
        existing = [
            f"  [{label}]  {p.name}"
            for p, label in self._expected_output_files(out_dir)
            if p.exists()
        ]
        if existing:
            file_list = "\n".join(existing)
            ans = QMessageBox.question(
                self,
                "File(s) already exist",
                f"The following file(s) already exist in the output folder "
                f"and will be overwritten:\n\n{file_list}\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return

        self.accept()

    def get_script(self):
        """Return the (possibly user-edited) Luna script to execute."""
        return self._script_edit.toPlainText().strip()


# ------------------------------------------------------------
# Mixin
# ------------------------------------------------------------

class SaveEDFMixin:

    def _save_edf_annots(self):
        if not hasattr(self, "p"):
            QMessageBox.information(
                self.ui, "No data loaded",
                "Please load an EDF or annotation file first."
            )
            return

        # --- gather info from current instance ---
        try:
            edf_file = self.p.edf.stat()["edf_file"]
        except Exception:
            edf_file = "."
        has_edf = bool(edf_file and edf_file != ".")

        try:
            annot_classes = self.p.annots()["Annotations"].tolist()
            has_annots    = len(annot_classes) > 0
        except Exception:
            annot_classes = []
            has_annots    = False

        if not has_edf and not has_annots:
            QMessageBox.information(
                self.ui, "Nothing to export",
                "No EDF data or annotations found in the current instance."
            )
            return

        try:
            inst_id = self.p.id()
        except Exception:
            inst_id = "output"

        # --- show dialog ---
        dlg = SaveEDFDialog(
            self.ui, inst_id, edf_file,
            has_edf, has_annots, annot_classes
        )
        if dlg.exec() != QDialog.Accepted:
            return

        script = dlg.get_script()
        if not script:
            return

        # --- execute each line ---
        errors = []
        for line in script.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                self.p.eval(line)
            except Exception as e:
                errors.append(f"{line}\n  → {e}")

        if errors:
            QMessageBox.critical(
                self.ui, "Export errors",
                "One or more commands failed:\n\n" + "\n\n".join(errors),
            )
        else:
            QMessageBox.information(
                self.ui, "Export complete",
                "EDF and/or annotation file(s) written successfully."
            )
