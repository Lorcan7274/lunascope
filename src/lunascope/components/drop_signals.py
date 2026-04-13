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
        self.setMinimumWidth(500)

        self._all_chs    = all_chs
        self._all_annots = all_annots
        self._drop_chs   = drop_chs
        self._drop_annots = drop_annots

        root = QVBoxLayout(self)
        root.setSpacing(10)

        # ── instruction ──────────────────────────────────────────────
        note = QLabel(
            "Review what will be <b>permanently removed</b> from the "
            "current instance, then confirm below."
        )
        note.setWordWrap(True)
        root.addWidget(note)

        # ── Channels summary ──────────────────────────────────────────
        n_chs   = len(all_chs)
        n_drop_chs = len(drop_chs)
        sig_title = f"Channels  —  {_section_title('Drop', n_drop_chs, n_chs)}"
        sig_grp  = QGroupBox(sig_title)
        sig_form = QVBoxLayout(sig_grp)

        keep_row = QHBoxLayout()
        keep_row.addWidget(QLabel("<b>Keep:</b>"), 0)
        keep_row.addWidget(_bullet_label(keep_chs, "#44bb44"), 1)
        sig_form.addLayout(keep_row)

        drop_row = QHBoxLayout()
        drop_row.addWidget(QLabel("<b>Drop:</b>"), 0)
        drop_row.addWidget(_bullet_label(drop_chs, "#ee6666"), 1)
        sig_form.addLayout(drop_row)

        # Checked only if there is something to drop AND the user had
        # explicitly selected channels to keep (non-empty keep list).
        # If nothing was selected in the dock, default to unchecked —
        # the user probably hasn't reviewed this dimension yet.
        chs_checked = bool(drop_chs) and bool(keep_chs)
        self._chk_drop_chs = QCheckBox("Drop these channels")
        self._chk_drop_chs.setChecked(chs_checked)
        self._chk_drop_chs.setEnabled(bool(drop_chs))
        sig_form.addWidget(self._chk_drop_chs)

        root.addWidget(sig_grp)

        # ── Annotations summary ───────────────────────────────────────
        n_annots      = len(all_annots)
        n_drop_annots = len(drop_annots)
        ann_title = f"Annotation classes  —  {_section_title('Drop', n_drop_annots, n_annots)}"
        ann_grp  = QGroupBox(ann_title)
        ann_form = QVBoxLayout(ann_grp)

        keep_row2 = QHBoxLayout()
        keep_row2.addWidget(QLabel("<b>Keep:</b>"), 0)
        keep_row2.addWidget(_bullet_label(keep_annots, "#44bb44"), 1)
        ann_form.addLayout(keep_row2)

        drop_row2 = QHBoxLayout()
        drop_row2.addWidget(QLabel("<b>Drop:</b>"), 0)
        drop_row2.addWidget(_bullet_label(drop_annots, "#ee6666"), 1)
        ann_form.addLayout(drop_row2)

        annots_checked = bool(drop_annots) and bool(keep_annots)
        self._chk_drop_annots = QCheckBox("Drop these annotation classes")
        self._chk_drop_annots.setChecked(annots_checked)
        self._chk_drop_annots.setEnabled(bool(drop_annots))
        ann_form.addWidget(self._chk_drop_annots)

        root.addWidget(ann_grp)

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

    # ------------------------------------------------------------------

    def _on_accept(self):
        drop_chs    = self._chk_drop_chs.isChecked()
        drop_annots = self._chk_drop_annots.isChecked()

        if not drop_chs and not drop_annots:
            QMessageBox.information(
                self, "Nothing to do",
                "Neither 'Drop channels' nor 'Drop annotation classes' is checked."
            )
            return

        # Block dropping absolutely everything
        chs_gone    = drop_chs    and len(self._drop_chs)    == len(self._all_chs)
        annots_gone = drop_annots and len(self._drop_annots) == len(self._all_annots)
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

        if dlg.drop_channels() and drop_chs:
            # SIGNALS keep=ch1,ch2,... is cleaner than a drop list
            cmd = "SIGNALS keep=" + ",".join(keep_chs)
            try:
                self.p.eval(cmd)
            except Exception as e:
                errors.append(f"{cmd}\n  → {e}")

        if dlg.drop_annots() and drop_annots:
            cmd = "DROP-ANNOTS annot=" + ",".join(drop_annots)
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
        self._render_hypnogram()
        self._render_signals_simple()
