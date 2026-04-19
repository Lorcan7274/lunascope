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

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QCheckBox, QDialogButtonBox, QMessageBox, QScrollArea, QWidget,
    QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

_MAX_NAMES = 12   # truncate lists longer than this in the dialog

def _fmt_names(names):
    """Comma-separated list, truncated with '…' if too long."""
    if not names:
        return "(none)"
    if len(names) <= _MAX_NAMES:
        return ", ".join(names)
    shown = ", ".join(names[:_MAX_NAMES])
    return f"{shown}, … (+{len(names) - _MAX_NAMES} more)"


def _bullet_label(names, color):
    lbl = QLabel(_fmt_names(names))
    lbl.setWordWrap(True)
    lbl.setStyleSheet(f"color: {color};")
    return lbl


def _section_title(verb, n_action, n_total):
    """e.g. 'Drop 7 of 10 channels'  /  'Keep all 10 channels'"""
    n_keep = n_total - n_action
    if n_action == 0:
        return f"Keep all {n_total}"
    if n_keep == 0:
        return f"Drop all {n_total}"
    return f"Drop {n_action} of {n_total}  ·  keep {n_keep}"


# ------------------------------------------------------------
# Confirmation dialog
# ------------------------------------------------------------

class DropDialog(QDialog):

    def __init__(self, parent,
                 keep_chs, drop_chs, all_chs,
                 keep_annots, drop_annots, all_annots):
        super().__init__(parent)
        self.setWindowTitle("Drop channels / annotations")
        self.setMinimumWidth(520)

        self._all_chs         = all_chs
        self._all_annots      = all_annots
        # "selected" = checked in dock; "other" = unchecked
        self._selected_chs    = list(keep_chs)
        self._other_chs       = list(drop_chs)
        self._selected_annots = list(keep_annots)
        self._other_annots    = list(drop_annots)

        root = QVBoxLayout(self)
        root.setSpacing(10)

        note = QLabel(
            "Review what will be <b>permanently removed</b> from the "
            "current instance, then confirm below."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        # ── Channels ────────────────────────────────────────────────
        self._sig_grp = QGroupBox()
        sig_form = QVBoxLayout(self._sig_grp)

        self._sig_keep_lbl = QLabel(); self._sig_keep_lbl.setWordWrap(True)
        self._sig_drop_lbl = QLabel(); self._sig_drop_lbl.setWordWrap(True)
        keep_row = QHBoxLayout()
        keep_row.addWidget(QLabel("<b>Keep:</b>"), 0)
        keep_row.addWidget(self._sig_keep_lbl, 1)
        sig_form.addLayout(keep_row)
        drop_row = QHBoxLayout()
        drop_row.addWidget(QLabel("<b>Drop:</b>"), 0)
        drop_row.addWidget(self._sig_drop_lbl, 1)
        sig_form.addLayout(drop_row)

        bottom_row_chs = QHBoxLayout()
        self._chk_drop_chs  = QCheckBox("Drop these channels")
        self._flip_chs      = QCheckBox("Flip  (selected = drop)")
        self._flip_chs.setStyleSheet("color: #aaaaaa; font-style: italic;")
        bottom_row_chs.addWidget(self._chk_drop_chs)
        bottom_row_chs.addStretch()
        bottom_row_chs.addWidget(self._flip_chs)
        sig_form.addLayout(bottom_row_chs)

        root.addWidget(self._sig_grp)

        # ── Annotations ──────────────────────────────────────────────
        self._ann_grp = QGroupBox()
        ann_form = QVBoxLayout(self._ann_grp)

        self._ann_keep_lbl = QLabel(); self._ann_keep_lbl.setWordWrap(True)
        self._ann_drop_lbl = QLabel(); self._ann_drop_lbl.setWordWrap(True)
        keep_row2 = QHBoxLayout()
        keep_row2.addWidget(QLabel("<b>Keep:</b>"), 0)
        keep_row2.addWidget(self._ann_keep_lbl, 1)
        ann_form.addLayout(keep_row2)
        drop_row2 = QHBoxLayout()
        drop_row2.addWidget(QLabel("<b>Drop:</b>"), 0)
        drop_row2.addWidget(self._ann_drop_lbl, 1)
        ann_form.addLayout(drop_row2)

        bottom_row_ann = QHBoxLayout()
        self._chk_drop_annots = QCheckBox("Drop these annotation classes")
        self._flip_ann        = QCheckBox("Flip  (selected = drop)")
        self._flip_ann.setStyleSheet("color: #aaaaaa; font-style: italic;")
        bottom_row_ann.addWidget(self._chk_drop_annots)
        bottom_row_ann.addStretch()
        bottom_row_ann.addWidget(self._flip_ann)
        ann_form.addLayout(bottom_row_ann)

        root.addWidget(self._ann_grp)

        # ── warning ──────────────────────────────────────────────────
        warn = QLabel("This operation cannot be undone within the current session.")
        warn.setStyleSheet("color: #cc8800; font-style: italic;")
        warn.setWordWrap(True)
        root.addWidget(warn)

        # ── buttons ──────────────────────────────────────────────────
        btn_box = QDialogButtonBox()
        self._run_btn = btn_box.addButton("Drop selected", QDialogButtonBox.AcceptRole)
        btn_box.addButton(QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        root.addWidget(btn_box)

        # connect flip toggles and do initial render
        self._flip_chs.toggled.connect(self._refresh_chs_display)
        self._flip_ann.toggled.connect(self._refresh_ann_display)
        self._refresh_chs_display()
        self._refresh_ann_display()

    # ------------------------------------------------------------------

    def _effective_chs(self):
        """(keep_list, drop_list) respecting the flip toggle."""
        if self._flip_chs.isChecked():
            return self._other_chs, self._selected_chs
        return self._selected_chs, self._other_chs

    def _effective_annots(self):
        """(keep_list, drop_list) respecting the flip toggle."""
        if self._flip_ann.isChecked():
            return self._other_annots, self._selected_annots
        return self._selected_annots, self._other_annots

    def _refresh_chs_display(self):
        keep, drop = self._effective_chs()
        n = len(self._all_chs)
        self._sig_grp.setTitle(f"Channels  —  {_section_title('Drop', len(drop), n)}")
        self._sig_keep_lbl.setText(_fmt_names(keep))
        self._sig_keep_lbl.setStyleSheet("color: #44bb44;")
        self._sig_drop_lbl.setText(_fmt_names(drop))
        self._sig_drop_lbl.setStyleSheet("color: #ee6666;")
        has_drop = bool(drop)
        self._chk_drop_chs.setEnabled(has_drop)
        self._chk_drop_chs.setChecked(has_drop and bool(keep))

    def _refresh_ann_display(self):
        keep, drop = self._effective_annots()
        n = len(self._all_annots)
        self._ann_grp.setTitle(f"Annotation classes  —  {_section_title('Drop', len(drop), n)}")
        self._ann_keep_lbl.setText(_fmt_names(keep))
        self._ann_keep_lbl.setStyleSheet("color: #44bb44;")
        self._ann_drop_lbl.setText(_fmt_names(drop))
        self._ann_drop_lbl.setStyleSheet("color: #ee6666;")
        has_drop = bool(drop)
        self._chk_drop_annots.setEnabled(has_drop)
        self._chk_drop_annots.setChecked(has_drop and bool(keep))

    def _on_accept(self):
        do_drop_chs    = self._chk_drop_chs.isChecked()
        do_drop_annots = self._chk_drop_annots.isChecked()

        if not do_drop_chs and not do_drop_annots:
            QMessageBox.information(
                self, "Nothing to do",
                "Neither 'Drop channels' nor 'Drop annotation classes' is checked."
            )
            return

        _, drop_chs    = self._effective_chs()
        _, drop_annots = self._effective_annots()
        chs_gone    = do_drop_chs    and len(drop_chs)    == len(self._all_chs)
        annots_gone = do_drop_annots and len(drop_annots) == len(self._all_annots)
        if chs_gone and annots_gone:
            QMessageBox.critical(
                self, "Cannot drop everything",
                "This would remove all channels and all annotation classes.\n\n"
                "Keep at least one channel or one annotation class."
            )
            return

        self.accept()

    def drop_channels(self):
        return self._chk_drop_chs.isChecked()

    def drop_annots(self):
        return self._chk_drop_annots.isChecked()

    def get_keep_chs(self):
        keep, _ = self._effective_chs()
        return keep

    def get_drop_annots_list(self):
        _, drop = self._effective_annots()
        return drop


# ------------------------------------------------------------
# Mixin
# ------------------------------------------------------------

class DropSignalsMixin:

    def _drop_signals_annots(self):
        if not hasattr(self, "p"):
            QMessageBox.information(
                self.ui, "No data loaded",
                "Please load an EDF or annotation file first."
            )
            return

        # ── read all channels and annot classes from the instance ──
        try:
            all_chs = self.p.chs()["Channels"].tolist()
        except Exception:
            all_chs = []

        try:
            all_annots = self.p.annots()["Annotations"].tolist()
        except Exception:
            all_annots = []

        # ── read the current dock selection (checked = keep) ──
        try:
            keep_chs = list(self.ui.tbl_desc_signals.checked())
        except Exception:
            keep_chs = list(all_chs)

        try:
            keep_annots = list(self.ui.tbl_desc_annots.checked())
        except Exception:
            keep_annots = list(all_annots)

        # ── derive what will be dropped ──
        keep_chs_set    = set(keep_chs)
        keep_annots_set = set(keep_annots)
        drop_chs    = [c for c in all_chs    if c not in keep_chs_set]
        drop_annots = [a for a in all_annots if a not in keep_annots_set]

        if not drop_chs and not drop_annots:
            QMessageBox.information(
                self.ui, "Nothing to drop",
                "All channels and annotation classes are currently selected.\n\n"
                "Deselect items in the Channels or Annotations dock first."
            )
            return

        # ── show confirmation dialog ──
        dlg = DropDialog(
            self.ui,
            keep_chs=keep_chs,       drop_chs=drop_chs,       all_chs=all_chs,
            keep_annots=keep_annots, drop_annots=drop_annots, all_annots=all_annots,
        )
        if dlg.exec() != QDialog.Accepted:
            return

        # ── build and run Luna commands ──
        errors = []

        if dlg.drop_channels():
            effective_keep = dlg.get_keep_chs()
            if effective_keep:
                cmd = "SIGNALS keep=" + ",".join(effective_keep)
                try:
                    self.p.eval(cmd)
                except Exception as e:
                    errors.append(f"{cmd}\n  → {e}")

        if dlg.drop_annots():
            effective_drop_annots = dlg.get_drop_annots_list()
            if effective_drop_annots:
                cmd = "DROP-ANNOTS annot=" + ",".join(effective_drop_annots)
                try:
                    self.p.eval(cmd)
                except Exception as e:
                    errors.append(f"{cmd}\n  → {e}")

        if errors:
            QMessageBox.critical(
                self.ui, "Errors",
                "One or more commands failed:\n\n" + "\n\n".join(errors)
            )
            return

        # ── refresh UI ──
        self._update_metrics()
        if hasattr(self, "_update_soap_list"):
            self._update_soap_list()
        self._render_hypnogram()
        self._render_signals_simple()
