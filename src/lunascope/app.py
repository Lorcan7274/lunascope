import sys
import tempfile
from pathlib import Path
import os
import signal

def _boot_log(message: str) -> None:
    sys.stderr.write(f"[lunascope] {message}\n")
    sys.stderr.flush()


_boot_log("Initiating startup...")


def _user_cache_root() -> Path:
    if sys.platform == "win32":
        for env_var in ("LOCALAPPDATA", "APPDATA"):
            value = os.environ.get(env_var)
            if value:
                return Path(value)
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            return Path(xdg_cache)
        return Path.home() / ".cache"
    return Path(tempfile.gettempdir()) / "lunascope-cache"


def _configure_runtime_cache_dirs() -> None:
    cache_root = _user_cache_root() / "lunascope"
    mpl_cache = cache_root / "matplotlib"
    try:
        mpl_cache.mkdir(parents=True, exist_ok=True)
    except OSError:
        cache_root = Path(tempfile.gettempdir()) / "lunascope-cache"
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

import argparse

import lunapi as lp

import pyqtgraph as pg

from PySide6.QtCore import QFile, QTimer
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication
from importlib.resources import files, as_file

from .controller import Controller

# suppress macOS warnings
os.environ["OS_ACTIVITY_MODE"] = "disable"


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



def main(argv=None) -> int:

#    import faulthandler, sys, signal
#    faulthandler.enable(all_threads=True)
#    if hasattr( faulthandler, "register" ):
#        faulthandler.register(signal.SIGUSR1)  # kill -USR1 <pid> dumps stacks

    args = _parse_args(argv or sys.argv[1:])
    _boot_log("Creating application...")
    app = QApplication(sys.argv)

    # initiate silent luna
    _boot_log("Initializing Luna...")
    proj = lp.proj()
    proj.silence( True )
    
    _boot_log("Loading user interface...")
    ui = _load_ui()
    controller = Controller(ui, proj)
    _install_signal_handlers(app, controller)
    _boot_log("Showing main window...")
    ui.show()

    # optionally, attach a file list (or .edf or .annot):
    
    if args.slist_file:

        # Lunascope session?
        if args.slist_file.lower().endswith(".lss"):
            controller.load_session_state_file(args.slist_file)
        # EDF?
        elif args.slist_file.lower().endswith(".edf"):
            controller.open_edf( args.slist_file )
        # .annot file?
        elif args.slist_file.lower().endswith(".annot"):
            controller.open_annot( args.slist_file )
        # otherwise, assume a sample list
        else:
            folder_path = str(Path( args.slist_file ).parent) + os.sep
            proj.var( 'path' , folder_path )
            controller._read_slist_from_file( args.slist_file )        

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
        return app.exec()
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


    

if __name__ == "__main__":
    raise SystemExit(main())
