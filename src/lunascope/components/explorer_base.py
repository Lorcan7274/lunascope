
#  --------------------------------------------------------------------
#  Luna / Lunascope  —  Explorer dock: shared base class
#  --------------------------------------------------------------------

"""Base class for all Explorer tab widgets."""

import io
import traceback

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QFrame, QMenu, QVBoxLayout


# Shared dark-theme constants used by all tabs
BG    = "#0d1117"
FG    = "#c9d1d9"
GRID  = "#21262d"
SEP   = "#30363d"


class _ExplorerTab(QtCore.QObject):
    """Base class for Explorer tab panels.

    Subclasses hold a *root* QWidget (the tab's top-level widget) and a
    lazily-created MplCanvas.  Common threading helpers and canvas
    context-menu actions are provided here.
    """

    def __init__(self, ctrl, parent=None):
        super().__init__(parent or ctrl)
        self.ctrl = ctrl
        self._root: QtWidgets.QWidget | None = None
        self._canvas = None
        self._canvas_host: QFrame | None = None  # set by subclass
        self._canvas_scroll: QtWidgets.QScrollArea | None = None

    # ------------------------------------------------------------------
    # Tab widget
    # ------------------------------------------------------------------

    def widget(self) -> QtWidgets.QWidget:
        return self._root

    # ------------------------------------------------------------------
    # Work lifecycle (shared busy / progress / lock)
    # ------------------------------------------------------------------

    def _start_work(self, msg: str = "Working…") -> bool:
        if getattr(self.ctrl, "_busy", False):
            return False
        self.ctrl._busy = True
        _btn = getattr(self.ctrl, "_buttons", None)
        if _btn:
            _btn(False)
        self.ctrl.sb_progress.setVisible(True)
        self.ctrl.sb_progress.setRange(0, 0)
        self.ctrl.sb_progress.setFormat(msg)
        self.ctrl.lock_ui(msg)
        return True

    def _end_work(self):
        self.ctrl.unlock_ui()
        self.ctrl._busy = False
        _btn = getattr(self.ctrl, "_buttons", None)
        if _btn:
            _btn(True)
        self.ctrl.sb_progress.setRange(0, 100)
        self.ctrl.sb_progress.setValue(0)
        self.ctrl.sb_progress.setVisible(False)

    # ------------------------------------------------------------------
    # Canvas (lazy)
    # ------------------------------------------------------------------

    def _ensure_canvas(self):
        if self._canvas is not None:
            self._sync_canvas_width()
            return self._canvas
        if self._canvas_host is None:
            return None

        from .mplcanvas import MplCanvas

        self._canvas = MplCanvas(self._canvas_host)
        lay = self._canvas_host.layout()
        if lay is None:
            lay = QVBoxLayout()
            lay.setContentsMargins(0, 0, 0, 0)
            self._canvas_host.setLayout(lay)
        lay.setAlignment(Qt.AlignTop)
        lay.addWidget(self._canvas)
        self._canvas.installEventFilter(self)
        self._canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self._canvas.customContextMenuRequested.connect(self._context_menu)
        self._sync_canvas_width()
        return self._canvas

    def _on_canvas_scroll_destroyed(self, *_):
        self._canvas_scroll = None

    def _sync_canvas_width(self):
        if self._canvas_scroll is None or self._canvas_host is None:
            return
        try:
            viewport = self._canvas_scroll.viewport()
        except RuntimeError:
            self._canvas_scroll = None
            return
        width = max(320, viewport.width())
        self._canvas_host.setMinimumWidth(width)
        self._canvas_host.setMaximumWidth(width)
        if self._canvas is not None:
            self._canvas.setMinimumWidth(width)
            self._canvas.setMaximumWidth(width)

    def eventFilter(self, obj, event):
        if self._canvas_scroll is not None:
            try:
                viewport = self._canvas_scroll.viewport()
            except RuntimeError:
                self._canvas_scroll = None
                viewport = None
            if viewport is not None and obj is viewport:
                if event.type() in (QtCore.QEvent.Resize, QtCore.QEvent.Show):
                    self._sync_canvas_width()
            if self._canvas is not None and obj is self._canvas and event.type() == QtCore.QEvent.Wheel:
                bar = self._canvas_scroll.verticalScrollBar()
                if bar is not None:
                    pixel_delta = event.pixelDelta().y()
                    if pixel_delta:
                        delta = pixel_delta
                    else:
                        angle_delta = event.angleDelta().y()
                        steps = angle_delta / 120.0 if angle_delta else 0.0
                        delta = steps * max(20, bar.singleStep())
                    if delta:
                        bar.setValue(int(round(bar.value() - delta)))
                        event.accept()
                        return True
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Context menu / export
    # ------------------------------------------------------------------

    def _context_menu(self, pos):
        if self._canvas is None:
            return
        menu = QMenu(self._canvas)
        a_copy = menu.addAction("Copy to Clipboard")
        a_save = menu.addAction("Save Figure…")
        action = menu.exec(self._canvas.mapToGlobal(pos))
        if action == a_copy:
            self._copy_to_clipboard()
        elif action == a_save:
            self._save_figure()

    def _copy_to_clipboard(self):
        if self._canvas is None:
            return
        buf = io.BytesIO()
        self._canvas.figure.savefig(buf, format="png", bbox_inches="tight", facecolor=BG)
        img = QtGui.QImage.fromData(buf.getvalue(), "PNG")
        QtWidgets.QApplication.clipboard().setImage(img)

    def _save_figure(self):
        if self._canvas is None:
            return
        fn, _ = QFileDialog.getSaveFileName(
            self._root, "Save Figure", "figure",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )
        if fn:
            self._canvas.figure.savefig(fn, bbox_inches="tight", facecolor=BG)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _render_empty(self, msg: str = ""):
        canvas = self._ensure_canvas()
        if canvas is None:
            return
        fig = canvas.figure
        fig.clear()
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG)
        ax.text(0.5, 0.5, msg or "No data", color=FG,
                ha="center", va="center", fontsize=10,
                transform=ax.transAxes, wrap=True, multialignment="center")
        ax.set_axis_off()
        canvas.draw()

    def _style_ax(self, ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        if title:
            ax.set_title(title, color=FG, fontsize=9, pad=4)
        if xlabel:
            ax.set_xlabel(xlabel, color=FG, fontsize=8)
        if ylabel:
            ax.set_ylabel(ylabel, color=FG, fontsize=8)
