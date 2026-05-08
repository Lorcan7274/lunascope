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
import shutil
import socket
import threading
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QProgressDialog
from ..file_dialogs import existing_directory


TUTORIAL_DOWNLOAD_URL = "https://zzz.nyspi.org/dist/luna/tutorial.zip"
TUTORIAL_TIMEOUT_SECS = 180   # 3 minutes before giving up


class TutorialMixin:

    def _download_tutorial(self) -> None:
        # ── 1. Pick destination folder ────────────────────────────────
        dest_dir = existing_directory(
            self.ui,
            "Select folder to save tutorial into",
            "",
        )
        if not dest_dir:
            return

        dest_path = Path(dest_dir)
        archive_path = dest_path / "tutorial.zip"
        tutorial_subdir = dest_path / "tutorial"

        # ── 2. Warn if archive or extracted folder already exist ──────
        existing = []
        if archive_path.exists():
            existing.append(f"• {archive_path.name} (zip archive)")
        if tutorial_subdir.exists():
            existing.append(f"• {tutorial_subdir.name}/ (extracted folder)")

        if existing:
            answer = QMessageBox.question(
                self.ui,
                "Overwrite existing files?",
                "The following already exist in the selected folder:\n\n"
                + "\n".join(existing)
                + "\n\nOverwrite and re-download?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        # ── 3. Download with a cancellable progress dialog ───────────
        progress = QProgressDialog(
            "Downloading tutorial.zip …\n(this may take a minute on slow connections)",
            "Cancel",
            0,
            0,          # indeterminate bar
            self.ui,
        )
        progress.setWindowTitle("Downloading Tutorial")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        cancelled = [False]
        error = [None]

        def _do_download():
            try:
                with urllib.request.urlopen(TUTORIAL_DOWNLOAD_URL,
                                            timeout=TUTORIAL_TIMEOUT_SECS) as response:
                    with archive_path.open("wb") as fh:
                        while True:
                            if cancelled[0]:
                                return
                            chunk = response.read(65536)
                            if not chunk:
                                break
                            fh.write(chunk)
            except Exception as exc:
                error[0] = exc

        thread = threading.Thread(target=_do_download, daemon=True)
        thread.start()

        while thread.is_alive():
            QApplication.processEvents()
            if progress.wasCanceled():
                cancelled[0] = True
                thread.join(timeout=5)
                # Remove partial download
                if archive_path.exists():
                    archive_path.unlink(missing_ok=True)
                return

        progress.close()

        if error[0] is not None:
            exc = error[0]
            if isinstance(exc, (TimeoutError, socket.timeout, urllib.error.URLError)):
                detail = (
                    "Could not reach the server. "
                    "Please check your network connection and try again.\n\n"
                    f"Detail: {exc}"
                )
            else:
                detail = f"{type(exc).__name__}: {exc}"
            QMessageBox.critical(self.ui, "Tutorial Download Error", detail)
            if archive_path.exists():
                archive_path.unlink(missing_ok=True)
            return

        # ── 4. Extract ────────────────────────────────────────────────
        try:
            if tutorial_subdir.exists():
                shutil.rmtree(tutorial_subdir)
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest_path)
        except Exception as exc:
            QMessageBox.critical(
                self.ui,
                "Tutorial Extract Error",
                f"The archive downloaded but could not be extracted.\n\n{exc}",
            )
            return

        # ── 5. Find s.lst ─────────────────────────────────────────────
        matches = list(tutorial_subdir.rglob("s.lst"))
        if not matches:
            QMessageBox.information(
                self.ui,
                "Tutorial Downloaded",
                f"Tutorial extracted to:\n{tutorial_subdir}\n\n"
                "No s.lst sample-list file was found in the archive.",
            )
            return

        slist_path = str(matches[0])
        slist_dir  = str(Path(slist_path).parent) + os.sep

        # ── 6. Set the path variable so relative EDF refs resolve ─────
        self.proj.var("path", slist_dir)

        # ── 7. Load the sample list ───────────────────────────────────
        try:
            self._read_slist_from_file(slist_path)
            QMessageBox.information(
                self.ui,
                "Tutorial Downloaded",
                f"Tutorial extracted to:\n{tutorial_subdir}\n\n"
                f"Sample list loaded from:\n{slist_path}",
            )
        except Exception as exc:
            QMessageBox.warning(
                self.ui,
                "Tutorial Downloaded",
                f"Tutorial extracted to:\n{tutorial_subdir}\n\n"
                f"s.lst found at:\n{slist_path}\n\n"
                f"Could not load it automatically:\n{exc}",
            )
