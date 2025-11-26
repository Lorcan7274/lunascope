
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

import lunapi as lp

import argparse
from pathlib import Path
import sys, os

import pyqtgraph as pg
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication
from importlib.resources import files, as_file

from .controller import Controller

# suppress macOS warnings
os.environ["OS_ACTIVITY_MODE"] = "disable"


def _load_ui():
    ui_res = files("lunascope.ui").joinpath("main.ui")
    with as_file(ui_res) as p:
        f = QFile(str(p))
        if not f.open(QFile.ReadOnly):
            raise RuntimeError(f"Cannot open UI file: {p}")
        try:
            loader = QUiLoader()
            loader.registerCustomWidget(pg.PlotWidget)
            ui = loader.load(f)
        finally:
            f.close()
    if ui is None:
        raise RuntimeError("Failed to load UI")
    return ui


def _parse_args(argv):
    ap = argparse.ArgumentParser(prog="lunascope")
    ap.add_argument("slist_file", nargs="?", metavar="FILE",
                    help="a sample list, EDF or .annot file (optional)")
    ap.add_argument("--param", "-p", dest="param_file", metavar="FILE",
                    help="parameter file")
    ap.add_argument("--cmap", "-c", dest="cmap_file", metavar="FILE",
                    help="channel map file")

    # allow options to appear before/after the positional on py>=3.7
    parse = getattr(ap, "parse_intermixed_args", ap.parse_args)
    return parse(argv)



def main(argv=None) -> int:

#    import faulthandler, sys, signal
#    faulthandler.enable(all_threads=True)
#    if hasattr( faulthandler, "register" ):
#        faulthandler.register(signal.SIGUSR1)  # kill -USR1 <pid> dumps stacks

    args = _parse_args(argv or sys.argv[1:])
    app = QApplication(sys.argv)

    # initiate silent luna
    proj = lp.proj()
    proj.silence( True )
    
    ui = _load_ui()
    controller = Controller(ui, proj)
    ui.show()

    # optionally, attach a file list (or .edf or .annot):
    
    if args.slist_file:

        # EDF?
        if args.slist_file.lower().endswith(".edf"):
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
            print(f"[Error] Could not load {args.cap_file}: {type(e).__name__}: {e}", file=sys.stderr)

    # [ OLD - this actually set the cols ] optionally pre-load a color map
    # if args.cmap_file:
    #     try:
    #         text = open( args.cmap_file , "r", encoding="utf-8").read()            
    #         controller.cmap = {}
    #         controller.cmap_list = [ ]
    #         for line in text.splitlines():
    #             line = line.strip()
    #             if not line or line.startswith("#"):
    #                 continue
    #             parts = line.replace("=", " ").replace("\t", " ").split()
    #             if len(parts) >= 2:
    #                 controller.cmap[parts[0]] = parts[1]
    #                 controller.cmap_list.append( parts[0] )
    #         controller.cmap_rlist = list(reversed(controller.cmap_list))
    #         controller.palset = 'bespoke'

    #     except (UnicodeDecodeError, OSError) as e:
    #         print(f"[Error] Could not load {args.cmap_file}: {type(e).__name__}: {e}", file=sys.stderr)

                


                

            

    #
    # run the app
    #
    
    try:
        return app.exec()
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


    

if __name__ == "__main__":
    raise SystemExit(main())

