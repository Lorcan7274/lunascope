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

import sys
from pathlib import Path
import os
import signal
import traceback
import threading
import faulthandler

from .runtime_paths import app_cache_root
from .tls import configure_tls

def _boot_log(message: str) -> None:
    sys.stderr.write(f"[lunascope] {message}\n")
    sys.stderr.flush()


_boot_log("Initiating startup...")


def _diagnostics_log_path() -> Path:
    return app_cache_root() / "lunascope-diagnostics.log"


def _append_diagnostics_log(message: str) -> None:
    line = f"[lunascope] {message}\n"
    try:
        with _diagnostics_log_path().open("a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        pass
    sys.stderr.write(line)
    sys.stderr.flush()


def _configure_runtime_cache_dirs() -> None:
    cache_root = app_cache_root()
    mpl_cache = cache_root / "matplotlib"
    try:
        mpl_cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        cache_root = app_cache_root()
        mpl_cache = cache_root / "matplotlib"
        mpl_cache.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    bundle_root = Path(sys.executable).resolve().parent
    bundled_mpl_data = bundle_root / "matplotlib" / "mpl-data"
    if bundled_mpl_data.exists():
        os.environ.setdefault("MATPLOTLIBDATA", str(bundled_mpl_data))

    # On Unix, fontconfig typically uses XDG_CACHE_HOME. If it is unset and the
    # default cache location is not writable, Matplotlib can end up rebuilding
    # its font metadata on every launch.
    if os.name != "nt":
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))


_configure_runtime_cache_dirs()
configure_tls()

import argparse

import lunapi as lp

import pyqtgraph as pg

from PySide6.QtCore import QFile, QTimer
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor
from importlib.resources import files, as_file

from .controller import Controller
from .helpers import SmallPlaceholderEdit

# suppress macOS warnings
os.environ["OS_ACTIVITY_MODE"] = "disable"


def _apply_forced_dark_theme(app: QApplication) -> None:
    """Apply a platform-independent dark theme using built-in Qt styling."""
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#242628"))
    pal.setColor(QPalette.WindowText, QColor("#E7EAEE"))
    pal.setColor(QPalette.Base, QColor("#17191C"))
    pal.setColor(QPalette.AlternateBase, QColor("#202328"))
    pal.setColor(QPalette.ToolTipBase, QColor("#1D2024"))
    pal.setColor(QPalette.ToolTipText, QColor("#E7EAEE"))
    pal.setColor(QPalette.Text, QColor("#E7EAEE"))
    pal.setColor(QPalette.Button, QColor("#2B2E33"))
    pal.setColor(QPalette.ButtonText, QColor("#E7EAEE"))
    pal.setColor(QPalette.BrightText, QColor("#FFFFFF"))
    pal.setColor(QPalette.Link, QColor("#7DB7FF"))
    pal.setColor(QPalette.Highlight, QColor("#2F6DB2"))
    pal.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    pal.setColor(QPalette.PlaceholderText, QColor("#8F98A3"))
    pal.setColor(QPalette.Light, QColor("#3A3E44"))
    pal.setColor(QPalette.Midlight, QColor("#31353A"))
    pal.setColor(QPalette.Mid, QColor("#44484F"))
    pal.setColor(QPalette.Dark, QColor("#111315"))
    pal.setColor(QPalette.Shadow, QColor("#080909"))

    disabled_roles = (
        QPalette.WindowText,
        QPalette.Text,
        QPalette.ButtonText,
        QPalette.HighlightedText,
    )
    for role in disabled_roles:
        pal.setColor(QPalette.Disabled, role, QColor("#7B838C"))
    pal.setColor(QPalette.Disabled, QPalette.Base, QColor("#141619"))
    pal.setColor(QPalette.Disabled, QPalette.Button, QColor("#23262A"))
    pal.setColor(QPalette.Disabled, QPalette.Highlight, QColor("#3A4149"))

    app.setPalette(pal)

    app.setStyleSheet(
        """
        QWidget {
            color: #E7EAEE;
            background-color: #242628;
        }
        QMainWindow, QDialog, QFrame, QSplitter, QStatusBar, QMenuBar, QMenu, QToolTip {
            background-color: #242628;
            color: #E7EAEE;
        }
        QStatusBar::item {
            border: none;
        }
        QMenuBar::item:selected, QMenu::item:selected {
            background: #2F6DB2;
            color: #FFFFFF;
        }
        QDockWidget {
            color: #E7EAEE;
        }
        QDockWidget::title {
            background: #2C2F34;
            color: #D7DCE2;
            border: 1px solid #3A3E44;
            padding: 4px 8px;
        }
        QTabWidget::pane, QGroupBox, QAbstractScrollArea, QListView, QTreeView, QTableView {
            border: 1px solid #3A3E44;
            background: #17191C;
        }
        QGroupBox {
            margin-top: 0.7em;
            padding-top: 0.5em;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 4px;
            color: #C4CBD4;
        }
        QHeaderView::section {
            background: #2B2E33;
            color: #D7DCE2;
            border: 1px solid #3A3E44;
            padding: 2px 4px;
        }
        QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QPlainTextEdit, QTextEdit {
            background: #2B2E33;
            color: #E7EAEE;
            border: 1px solid #4A4F56;
            border-radius: 4px;
            selection-background-color: #2F6DB2;
            selection-color: #FFFFFF;
        }
        QPushButton:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
            border: 1px solid #6588B3;
        }
        QPushButton:pressed {
            background: #23262B;
        }
        QPushButton:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {
            background: #23262A;
            color: #7B838C;
            border: 1px solid #32363C;
        }
        QLineEdit[readOnly="true"], QPlainTextEdit[readOnly="true"], QTextEdit[readOnly="true"] {
            background: #1D2024;
        }
        QTabBar::tab {
            background: #2A2D31;
            color: #C9D0D8;
            border: 1px solid #3A3E44;
            padding: 4px 8px;
        }
        QTabBar::tab:selected {
            background: #2F6DB2;
            color: #FFFFFF;
        }
        QTabBar::tab:!selected:hover {
            background: #343940;
        }
        QCheckBox, QRadioButton, QLabel {
            color: #E7EAEE;
        }
        QCheckBox::indicator, QRadioButton::indicator {
            width: 15px;
            height: 15px;
            background: #1A1D21;
            border: 1px solid #747C86;
        }
        QCheckBox::indicator {
            border-radius: 3px;
        }
        QRadioButton::indicator {
            border-radius: 8px;
        }
        QCheckBox::indicator:hover, QRadioButton::indicator:hover {
            border: 1px solid #A6AFBA;
        }
        QCheckBox::indicator:checked {
            background: #2F6DB2;
            border: 1px solid #C9D4E2;
        }
        QRadioButton::indicator:checked {
            background: #2F6DB2;
            border: 1px solid #C9D4E2;
        }
        QTableView::indicator, QListView::indicator, QTreeView::indicator {
            width: 15px;
            height: 15px;
            background: #1A1D21;
            border: 1px solid #747C86;
        }
        QTableView::indicator:hover, QListView::indicator:hover, QTreeView::indicator:hover {
            border: 1px solid #A6AFBA;
        }
        QTableView::indicator:checked, QListView::indicator:checked, QTreeView::indicator:checked {
            background: #2F6DB2;
            border: 1px solid #C9D4E2;
        }
        QScrollBar:vertical, QScrollBar:horizontal {
            background: #1D2024;
            border: none;
            margin: 0;
        }
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background: #4A4F56;
            border-radius: 4px;
            min-height: 18px;
            min-width: 18px;
        }
        QScrollBar::handle:hover:vertical, QScrollBar::handle:hover:horizontal {
            background: #5A6068;
        }
        QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {
            background: none;
            border: none;
        }
        """
    )


def _load_ui():
    candidate_paths = []
    try:
        ui_res = files("lunascope.ui").joinpath("main.ui")
        with as_file(ui_res) as p:
            candidate_paths.append(Path(p))
    except Exception:
        pass

    bundle_root = Path(sys.executable).resolve().parent
    candidate_paths.extend(
        [
            bundle_root / "lunascope" / "ui" / "main.ui",
            bundle_root / "Lunascope" / "ui" / "main.ui",
            Path(__file__).resolve().parent / "ui" / "main.ui",
        ]
    )

    seen = set()
    for path in candidate_paths:
        path = Path(path)
        if path in seen or not path.exists():
            continue
        seen.add(path)
        f = QFile(str(path))
        if not f.open(QFile.ReadOnly):
            continue
        try:
            loader = QUiLoader()
            loader.registerCustomWidget(pg.PlotWidget)
            loader.registerCustomWidget(SmallPlaceholderEdit)
            ui = loader.load(f)
        finally:
            f.close()
        if ui is not None:
            return ui

    raise RuntimeError(
        "Cannot open UI file. Tried: " + ", ".join(str(p) for p in candidate_paths)
    )


def _parse_args(argv):
    ap = argparse.ArgumentParser(prog="lunascope")
    ap.add_argument("slist_file", nargs="?", metavar="FILE",
                    help="a sample list, EDF, .annot, or .lss session file (optional)")
    ap.add_argument("--param", "-p", dest="param_file", metavar="FILE",
                    help="parameter file")
    ap.add_argument("--cmap", "-c", dest="cmap_file", metavar="FILE",
                    help="channel map file")

    # allow options to appear before/after the positional on py>=3.7
    parse = getattr(ap, "parse_intermixed_args", ap.parse_args)
    return parse(argv)


def _install_signal_handlers(app: QApplication, controller=None) -> None:
    def _handle_termination(signum, frame):
        _boot_log(f"Received signal {signum}; shutting down.")
        if controller is not None and getattr(controller, "_busy", False):
            _boot_log("Waiting for background work to finish...")
        app.quit()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_termination)
        except (ValueError, OSError, RuntimeError):
            pass

    # Keep the Python interpreter responsive to signals while Qt owns the main loop.
    heartbeat = QTimer(app)
    heartbeat.setInterval(250)
    heartbeat.timeout.connect(lambda: None)
    heartbeat.start()
    app._signal_heartbeat = heartbeat


def _install_diagnostics(app: QApplication, ui=None) -> None:
    try:
        _diagnostics_log_path().parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    try:
        fh = _diagnostics_log_path().open("a", encoding="utf-8")
    except OSError:
        fh = None

    if fh is not None:
        try:
            faulthandler.enable(file=fh, all_threads=True)
            app._faulthandler_log = fh
            _append_diagnostics_log(f"Faulthandler enabled: {_diagnostics_log_path()}")
        except Exception as exc:
            fh.close()
            _append_diagnostics_log(f"Failed to enable faulthandler: {type(exc).__name__}: {exc}")

    def _log_exception(kind: str, exc_type, exc_value, exc_tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb)).strip()
        # Full traceback → diagnostics log file only
        try:
            with _diagnostics_log_path().open("a", encoding="utf-8") as fh:
                fh.write(f"[lunascope] {kind}:\n{text}\n")
        except OSError:
            pass
        # Console: one-line summary
        sys.stderr.write(f"[lunascope] {kind}: {exc_type.__name__}: {exc_value}\n")
        sys.stderr.flush()

    def _sys_excepthook(exc_type, exc_value, exc_tb):
        _log_exception("Unhandled exception", exc_type, exc_value, exc_tb)

    def _threading_excepthook(args):
        _log_exception(
            f"Unhandled thread exception in {getattr(args.thread, 'name', '<unknown>')}",
            args.exc_type,
            args.exc_value,
            args.exc_traceback,
        )

    sys.excepthook = _sys_excepthook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _threading_excepthook

    app.aboutToQuit.connect(lambda: _append_diagnostics_log("QApplication.aboutToQuit emitted"))
    app.lastWindowClosed.connect(lambda: _append_diagnostics_log("QApplication.lastWindowClosed emitted"))
    if ui is not None:
        ui.destroyed.connect(lambda *_: _append_diagnostics_log("Main window destroyed"))



def main(argv=None) -> int:

#    import faulthandler, sys, signal
#    faulthandler.enable(all_threads=True)
#    if hasattr( faulthandler, "register" ):
#        faulthandler.register(signal.SIGUSR1)  # kill -USR1 <pid> dumps stacks

    args = _parse_args(argv or sys.argv[1:])
    _boot_log("Creating application...")
    app = QApplication(sys.argv)
    _apply_forced_dark_theme(app)

    # initiate silent luna
    _boot_log("Initializing Luna...")
    proj = lp.proj()
    proj.silence( True )
    
    _boot_log("Loading user interface...")
    ui = _load_ui()
    _install_diagnostics(app, ui)
    controller = Controller(ui, proj)
    _install_signal_handlers(app, controller)

    explicit_session = bool(args.slist_file and args.slist_file.lower().endswith(".lss"))
    if not explicit_session:
        controller.load_geometry_cache_silently()

    _boot_log("Showing main window...")
    ui.show()

    # optionally, attach a file list (or .edf or .annot):
    
    if args.slist_file:
        input_path = str(Path(args.slist_file).expanduser())

        # Lunascope session?
        if input_path.lower().endswith(".lss"):
            controller.load_session_state_file(input_path)
        # EDF?
        elif input_path.lower().endswith(".edf"):
            controller.open_edf(input_path)
        # .annot file?
        elif input_path.lower().endswith(".annot"):
            controller.open_annot(input_path)
        # folder? build a sample list
        elif Path(input_path).is_dir():
            controller._build_slist_from_folder(input_path)
        # otherwise, assume a sample list
        else:
            folder_path = str(Path(input_path).parent) + os.sep
            proj.var('path', folder_path)
            controller._read_slist_from_file(input_path)

    # optionally, pre-load a parameter file?
    if args.param_file:
        try:
            text = open( args.param_file , "r", encoding="utf-8").read()
            controller.ui.txt_param.setPlainText(text)
        except (UnicodeDecodeError, OSError) as e:
            print(f"[Error] Could not load {args.param_file}: {type(e).__name__}: {e}", file=sys.stderr)

    # optionally, pre-load a parameter file?
    if args.cmap_file:
        try:
            text = open( args.cmap_file , "r", encoding="utf-8").read()
            controller.ui.txt_cmap.setPlainText(text)
        except (UnicodeDecodeError, OSError) as e:
            print(f"[Error] Could not load {args.cmap_file}: {type(e).__name__}: {e}", file=sys.stderr)


    #
    # run the app
    #
    
    _boot_log("Startup complete.")

    try:
        rc = app.exec()
        _append_diagnostics_log(f"app.exec() returned rc={rc}")
        return rc
    except Exception:
        _append_diagnostics_log("Exception escaped app.exec():\n" + traceback.format_exc().strip())
        return 1


    

if __name__ == "__main__":
    raise SystemExit(main())
