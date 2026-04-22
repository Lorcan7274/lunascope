import sys
import subprocess
import urllib.request
import urllib.error
import json

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout,
)

from .tls import create_default_context

_PYPI_URL = "https://pypi.org/pypi/lunascope/json"
_GITHUB_RELEASES_URL = "https://api.github.com/repos/Lorcan7274/lunascope/releases/latest"
_GITHUB_RELEASES_PAGE = "https://github.com/Lorcan7274/lunascope/releases/latest"
_LOCAL_TEST_INDEX = None  # set to a local dir path for offline testing

_IS_FROZEN = getattr(sys, "frozen", False)


class _VersionCheckWorker(QThread):
    """Fetches the latest version silently in the background (PyPI for pip, GitHub for executable)."""
    update_available = Signal(str)  # emits latest version string if newer

    def __init__(self, current_version: str, parent=None):
        super().__init__(parent)
        self._current = current_version

    def run(self):
        try:
            latest = _fetch_latest_version_github() if _IS_FROZEN else _fetch_latest_version_pypi()
            is_newer = (
                tuple(int(x) for x in latest.split("."))
                > tuple(int(x) for x in self._current.split("."))
            )
            if is_newer:
                self.update_available.emit(latest)
        except Exception:
            pass


def start_background_check(current_version: str, on_update_available) -> _VersionCheckWorker:
    """Start a background PyPI check; calls on_update_available(latest) if newer."""
    worker = _VersionCheckWorker(current_version)
    worker.update_available.connect(on_update_available)
    worker.start()
    return worker


def _ssl_context():
    return create_default_context()


def _fetch_latest_version_pypi() -> str:
    """Return the latest lunascope version string from PyPI (or local test index), or raise."""
    if _LOCAL_TEST_INDEX:
        import glob, os, re
        wheels = glob.glob(os.path.join(_LOCAL_TEST_INDEX, "lunascope-*.whl"))
        versions = []
        for w in wheels:
            m = re.search(r"lunascope-(\d+\.\d+\.\d+)", os.path.basename(w))
            if m:
                versions.append(m.group(1))
        if not versions:
            raise RuntimeError("No wheels found in local test index")
        return max(versions, key=lambda v: tuple(int(x) for x in v.split(".")))
    req = urllib.request.Request(_PYPI_URL, headers={"User-Agent": "lunascope-updater"})
    with urllib.request.urlopen(req, timeout=10, context=_ssl_context()) as resp:
        data = json.loads(resp.read().decode())
    return data["info"]["version"]


def _fetch_latest_version_github() -> str:
    """Return the latest lunascope version string from GitHub Releases, or raise."""
    req = urllib.request.Request(
        _GITHUB_RELEASES_URL,
        headers={"User-Agent": "lunascope-updater", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=10, context=_ssl_context()) as resp:
        data = json.loads(resp.read().decode())
    tag = data["tag_name"].lstrip("v")
    return tag


class _PipWorker(QThread):
    output = Signal(str)
    finished = Signal(bool, str)  # success, message

    def run(self):
        try:
            if _LOCAL_TEST_INDEX:
                cmd = [sys.executable, "-m", "pip", "install", "--upgrade",
                       "--find-links", _LOCAL_TEST_INDEX, "--no-index", "lunascope"]
            else:
                cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "lunascope"]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                self.output.emit(line.rstrip())
            proc.wait()
            if proc.returncode == 0:
                self.finished.emit(True, "Update complete.")
            else:
                self.finished.emit(False, f"pip exited with code {proc.returncode}.")
        except Exception as exc:
            self.finished.emit(False, str(exc))


class _UpdateDialog(QDialog):
    def __init__(self, current: str, latest: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Update Lunascope")
        self.setMinimumWidth(480)
        self._worker = None

        layout = QVBoxLayout(self)

        self._label = QLabel(
            f"<b>v{latest}</b> is available &nbsp;(you have v{current}).<br>"
            "Do you want to update now?"
        )
        self._label.setTextFormat(Qt.RichText)
        layout.addWidget(self._label)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.hide()
        layout.addWidget(self._log)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.hide()
        layout.addWidget(self._progress)

        self._buttons = QDialogButtonBox()
        self._update_btn = self._buttons.addButton("Update", QDialogButtonBox.AcceptRole)
        self._cancel_btn = self._buttons.addButton("Cancel", QDialogButtonBox.RejectRole)
        self._update_btn.setDefault(True)
        self._update_btn.setAutoDefault(True)
        self._cancel_btn.setAutoDefault(False)
        self._update_btn.clicked.connect(self._run_update)
        self._cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self._buttons)

    def _run_update(self):
        self._update_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        self._log.show()
        self._progress.show()
        self.adjustSize()

        self._worker = _PipWorker()
        self._worker.output.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _append_log(self, line: str):
        self._log.appendPlainText(line)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _on_finished(self, success: bool, message: str):
        self._progress.hide()
        if success:
            self._label.setText(
                f"<b>{message}</b><br>Please restart Lunascope to use the new version."
            )
            restart_btn = QPushButton("Restart Now")
            restart_btn.clicked.connect(self._restart)
            self._buttons.addButton(restart_btn, QDialogButtonBox.ActionRole)
            self._cancel_btn.setText("Later")
            self._cancel_btn.setEnabled(True)
        else:
            self._label.setText(f"<b>Update failed:</b> {message}")
            self._cancel_btn.setText("Close")
            self._cancel_btn.setEnabled(True)

    def _restart(self):
        self.accept()
        subprocess.Popen([sys.executable] + sys.argv)
        sys.exit(0)


class _ExecUpdateDialog(QDialog):
    """Shown when running as a frozen executable — directs user to GitHub Releases."""
    def __init__(self, current: str, latest: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Update Lunascope")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)

        label = QLabel(
            f"<b>v{latest}</b> is available &nbsp;(you have v{current}).<br><br>"
            "Download the latest installer from the releases page."
        )
        label.setTextFormat(Qt.RichText)
        label.setWordWrap(True)
        layout.addWidget(label)

        buttons = QDialogButtonBox()
        download_btn = buttons.addButton("Open Download Page", QDialogButtonBox.AcceptRole)
        later_btn = buttons.addButton("Later", QDialogButtonBox.RejectRole)
        download_btn.setDefault(True)
        download_btn.setAutoDefault(True)
        later_btn.setAutoDefault(False)
        download_btn.clicked.connect(self._open_releases)
        later_btn.clicked.connect(self.reject)
        layout.addWidget(buttons)

    def _open_releases(self):
        import webbrowser
        webbrowser.open(_GITHUB_RELEASES_PAGE)
        self.accept()


def check_and_prompt(current_version: str, parent=None) -> None:
    """Fetch latest version and show the appropriate update dialog."""
    fetch = _fetch_latest_version_github if _IS_FROZEN else _fetch_latest_version_pypi
    source = "GitHub" if _IS_FROZEN else "PyPI"
    try:
        latest = fetch()
    except urllib.error.URLError:
        QMessageBox.warning(
            parent,
            "Update Check Failed",
            f"Could not reach {source}. Please check your internet connection.",
        )
        return
    except Exception as exc:
        QMessageBox.warning(parent, "Update Check Failed", str(exc))
        return

    try:
        is_newer = tuple(int(x) for x in latest.split(".")) > tuple(int(x) for x in current_version.split("."))
    except Exception:
        is_newer = latest != current_version

    if not is_newer:
        QMessageBox.information(
            parent,
            "Up to Date",
            f"You are already on the latest version (v{current_version}).",
        )
        return

    if _IS_FROZEN:
        dlg = _ExecUpdateDialog(current_version, latest, parent)
    else:
        dlg = _UpdateDialog(current_version, latest, parent)
    dlg.exec()
