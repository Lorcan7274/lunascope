import sys
import subprocess
import urllib.request
import urllib.error
import json

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFrame, QLabel, QMessageBox,
    QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout,
)

from .tls import create_default_context
from .runtime_paths import app_state_file

_PYPI_URL = "https://pypi.org/pypi/lunascope/json"
_GITHUB_RELEASES_URL = "https://api.github.com/repos/Lorcan7274/lunascope/releases/latest"
_GITHUB_RELEASES_PAGE = "https://github.com/Lorcan7274/lunascope/releases/latest"
_CHANGELOG_JSON_URL = "https://raw.githubusercontent.com/Lorcan7274/lunascope/main/CHANGELOG.json"
_LOCAL_TEST_INDEX = None

_IS_FROZEN = getattr(sys, "frozen", False)
_LAST_SEEN_FILE = "last_seen_version.txt"

# Fallback changelog bundled with the app (used if remote fetch fails).
# Add an entry here whenever the version is bumped.
CHANGELOG: dict[str, list[str]] = {
    "1.6.1": [
        "Channel / annotation filter in the sample list — search by channel name or annotation class across your project (requires Scan All in Explorer)",
        "What's New shown in the update dialog so you know what you're getting before updating",
        "GPA dump export and overlap handling improvements",
        "Window geometry is now saved and restored across sessions",
    ],
    "1.6.0": [
        "GPA dump export and overlap handling improvements",
        "Window geometry is now saved and restored across sessions",
    ],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _ssl_context():
    return create_default_context()


def _is_newer(a: str, b: str) -> bool:
    try:
        return tuple(int(x) for x in a.split(".")) > tuple(int(x) for x in b.split("."))
    except Exception:
        return a != b


def _read_last_seen() -> str:
    try:
        return app_state_file(_LAST_SEEN_FILE).read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _write_last_seen(version: str) -> None:
    try:
        app_state_file(_LAST_SEEN_FILE).write_text(version, encoding="utf-8")
    except Exception:
        pass


# ── fetchers ──────────────────────────────────────────────────────────────────

def _fetch_latest_version_pypi() -> str:
    if _LOCAL_TEST_INDEX:
        import glob, os, re as _re
        wheels = glob.glob(os.path.join(_LOCAL_TEST_INDEX, "lunascope-*.whl"))
        versions = []
        for w in wheels:
            m = _re.search(r"lunascope-(\d+\.\d+\.\d+)", os.path.basename(w))
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
    req = urllib.request.Request(
        _GITHUB_RELEASES_URL,
        headers={"User-Agent": "lunascope-updater", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=10, context=_ssl_context()) as resp:
        data = json.loads(resp.read().decode())
    return data["tag_name"].lstrip("v")


def _fetch_remote_changelog() -> dict[str, list[str]]:
    req = urllib.request.Request(
        _CHANGELOG_JSON_URL,
        headers={"User-Agent": "lunascope-updater"},
    )
    with urllib.request.urlopen(req, timeout=10, context=_ssl_context()) as resp:
        return json.loads(resp.read().decode())


def _get_bullets(version: str, remote: dict | None = None) -> list[str]:
    """Return changelog bullets for version, preferring remote over bundled."""
    if remote and version in remote:
        return remote[version]
    return CHANGELOG.get(version, [])


# ── background workers ────────────────────────────────────────────────────────

class _VersionCheckWorker(QThread):
    """Fetches latest version + changelog bullets in the background."""
    update_available = Signal(str, object)  # latest_version, bullets: list[str]

    def __init__(self, current_version: str, parent=None):
        super().__init__(parent)
        self._current = current_version

    def run(self):
        try:
            latest = _fetch_latest_version_github() if _IS_FROZEN else _fetch_latest_version_pypi()
            if not _is_newer(latest, self._current):
                return
            try:
                remote = _fetch_remote_changelog()
            except Exception:
                remote = None
            bullets = _get_bullets(latest, remote)
            self.update_available.emit(latest, bullets)
        except Exception:
            pass


class _PipWorker(QThread):
    output = Signal(str)
    finished = Signal(bool, str)

    def run(self):
        try:
            if _LOCAL_TEST_INDEX:
                cmd = [sys.executable, "-m", "pip", "install", "--upgrade",
                       "--find-links", _LOCAL_TEST_INDEX, "--no-index", "lunascope"]
            else:
                cmd = [sys.executable, "-m", "pip", "install", "--upgrade", "lunascope"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                self.output.emit(line.rstrip())
            proc.wait()
            if proc.returncode == 0:
                self.finished.emit(True, "Update complete.")
            else:
                self.finished.emit(False, f"pip exited with code {proc.returncode}.")
        except Exception as exc:
            self.finished.emit(False, str(exc))


# ── dialogs ───────────────────────────────────────────────────────────────────

def _bullets_widget(bullets: list[str]) -> QLabel:
    label = QLabel()
    label.setTextFormat(Qt.RichText)
    label.setWordWrap(True)
    items = "".join(f"<li style='margin-bottom:4px;'>{b}</li>" for b in bullets)
    label.setText(f"<ul style='margin:0; padding-left:18px;'>{items}</ul>")
    return label


class _UpdateDialog(QDialog):
    def __init__(self, current: str, latest: str, bullets: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Update Lunascope")
        self.setMinimumWidth(520)
        self._worker = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._header = QLabel(
            f"<b>Lunascope v{latest}</b> is available &nbsp;·&nbsp; you have v{current}"
        )
        self._header.setTextFormat(Qt.RichText)
        layout.addWidget(self._header)

        # What's new section
        self._notes_frame = None
        if bullets:
            self._notes_frame = QFrame()
            self._notes_frame.setFrameShape(QFrame.StyledPanel)
            notes_layout = QVBoxLayout(self._notes_frame)
            notes_layout.setContentsMargins(8, 8, 8, 8)
            notes_layout.setSpacing(6)
            whats_new_label = QLabel("<b>What's new:</b>")
            whats_new_label.setTextFormat(Qt.RichText)
            notes_layout.addWidget(whats_new_label)
            notes_layout.addWidget(_bullets_widget(bullets))
            layout.addWidget(self._notes_frame)

        # pip log — hidden until update starts
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.hide()
        layout.addWidget(self._log)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.hide()
        layout.addWidget(self._progress)

        self._buttons = QDialogButtonBox()
        self._update_btn = self._buttons.addButton("Update Now", QDialogButtonBox.AcceptRole)
        self._cancel_btn = self._buttons.addButton("Later", QDialogButtonBox.RejectRole)
        self._update_btn.setDefault(True)
        self._update_btn.setAutoDefault(True)
        self._cancel_btn.setAutoDefault(False)
        self._update_btn.clicked.connect(self._run_update)
        self._cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self._buttons)

    def _run_update(self):
        self._update_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        if self._notes_frame:
            self._notes_frame.hide()
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
            self._header.setText(
                f"<b>{message}</b><br>Please restart Lunascope to use the new version."
            )
            restart_btn = QPushButton("Restart Now")
            restart_btn.clicked.connect(self._restart)
            self._buttons.addButton(restart_btn, QDialogButtonBox.ActionRole)
            self._cancel_btn.setText("Later")
            self._cancel_btn.setEnabled(True)
        else:
            self._header.setText(f"<b>Update failed:</b> {message}")
            self._cancel_btn.setText("Close")
            self._cancel_btn.setEnabled(True)

    def _restart(self):
        self.accept()
        subprocess.Popen([sys.executable] + sys.argv)
        sys.exit(0)


class _ExecUpdateDialog(QDialog):
    """Frozen executable — directs user to GitHub Releases."""
    def __init__(self, current: str, latest: str, bullets: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Update Lunascope")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        header = QLabel(
            f"<b>Lunascope v{latest}</b> is available &nbsp;·&nbsp; you have v{current}<br>"
            "Download the latest installer from the releases page."
        )
        header.setTextFormat(Qt.RichText)
        header.setWordWrap(True)
        layout.addWidget(header)

        if bullets:
            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(8, 8, 8, 8)
            fl.setSpacing(6)
            wn = QLabel("<b>What's new:</b>")
            wn.setTextFormat(Qt.RichText)
            fl.addWidget(wn)
            fl.addWidget(_bullets_widget(bullets))
            layout.addWidget(frame)

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


# ── public API ────────────────────────────────────────────────────────────────

def start_background_check(current_version: str, on_update_available) -> _VersionCheckWorker:
    """Start background check; calls on_update_available(latest, bullets) if newer."""
    worker = _VersionCheckWorker(current_version)
    worker.update_available.connect(on_update_available)
    worker.start()
    return worker


def check_and_prompt(current_version: str, bullets: list | None = None, parent=None) -> None:
    """Fetch latest version and show the update dialog."""
    source = "GitHub" if _IS_FROZEN else "PyPI"
    try:
        latest = _fetch_latest_version_github() if _IS_FROZEN else _fetch_latest_version_pypi()
        if bullets is None:
            try:
                remote = _fetch_remote_changelog()
            except Exception:
                remote = None
            bullets = _get_bullets(latest, remote)
    except urllib.error.URLError:
        QMessageBox.warning(
            parent, "Update Check Failed",
            f"Could not reach {source}. Please check your internet connection.",
        )
        return
    except Exception as exc:
        QMessageBox.warning(parent, "Update Check Failed", str(exc))
        return

    if not _is_newer(latest, current_version):
        QMessageBox.information(
            parent, "Up to Date",
            f"You are already on the latest version (v{current_version}).",
        )
        return

    if _IS_FROZEN:
        dlg = _ExecUpdateDialog(current_version, latest, bullets or [], parent)
    else:
        dlg = _UpdateDialog(current_version, latest, bullets or [], parent)
    dlg.exec()


def show_whats_new_if_needed(current_version: str, parent=None) -> None:
    """Show What's New once on first launch after an update (uses bundled changelog)."""
    last_seen = _read_last_seen()
    if not _is_newer(current_version, last_seen):
        return
    _write_last_seen(current_version)
    bullets = CHANGELOG.get(current_version)
    if not bullets:
        return
    dlg = _WhatsNewDialog(current_version, bullets, parent)
    dlg.exec()


class _WhatsNewDialog(QDialog):
    def __init__(self, version: str, bullets: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"What's New in Lunascope v{version}")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QLabel(f"<b>You're now running Lunascope v{version}</b>")
        header.setTextFormat(Qt.RichText)
        layout.addWidget(header)

        layout.addWidget(_bullets_widget(bullets))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
