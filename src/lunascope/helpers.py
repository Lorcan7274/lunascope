
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


from PySide6.QtGui import QAction, QKeySequence, QStandardItemModel
from PySide6.QtGui import QRegularExpressionValidator

from PySide6.QtCore import QModelIndex, QObject, Signal, Qt, QSortFilterProxyModel
from PySide6.QtCore import QRegularExpression, Qt

from PySide6.QtWidgets import QDockWidget, QVBoxLayout, QWidget
from PySide6.QtCore import QSortFilterProxyModel

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QColorDialog, QLabel, QApplication
)

from PySide6.QtGui import QColor, QPainter, QFont
from PySide6.QtWidgets import QPlainTextEdit
from PySide6.QtGui import QPalette
import sys
import random, colorsys
import pyqtgraph as pg
import pandas as pd
import numpy as np


# ------------------------------------------------------------
#
# QPlainTextEdit that renders placeholder text at a reduced,
# non-bold font size so it doesn't overpower the widget.
#
# ------------------------------------------------------------

class SmallPlaceholderEdit(QPlainTextEdit):
    """QPlainTextEdit whose placeholder text is drawn at ~75 % of the
    widget font size and without bold, so it reads clearly as a hint
    rather than as actual content."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_placeholder = super().placeholderText()
        # Keep Qt's built-in placeholder disabled so paintEvent() never has
        # to mutate widget state mid-paint just to suppress the default draw.
        super().setPlaceholderText("")

    def setPlaceholderText(self, text):
        self._custom_placeholder = text or ""
        super().setPlaceholderText("")

    def placeholderText(self):
        return self._custom_placeholder

    def paintEvent(self, event):
        super().paintEvent(event)
        ph = self._custom_placeholder
        if ph and not self.toPlainText():
            # Draw our own smaller, non-bold placeholder on the viewport
            vp = self.viewport()
            painter = QPainter(vp)
            font = QFont(self.font())
            pt = font.pointSizeF()
            font.setPointSizeF(max(7.0, pt * 0.75))
            font.setBold(False)
            painter.setFont(font)
            color = self.palette().color(QPalette.PlaceholderText)
            painter.setPen(color)
            margin = int(self.document().documentMargin())
            ox = max(0, int(self.contentOffset().x()) + margin)
            oy = max(0, int(self.contentOffset().y()) + margin)
            rect = vp.rect().adjusted(ox, oy, -margin, -margin)
            painter.drawText(rect, Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap, ph)
            painter.end()


class AuxiliaryWindow(QWidget):
    """Top-level helper window that preserves a dock-like toggle API.

    This lets legacy code keep calling methods like `toggleViewAction()`,
    `setFloating()`, and `isFloating()` while the actual UI behaves like a
    normal top-level window instead of a floating QDockWidget.
    """

    visibilityChanged = Signal(bool)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.setWindowTitle(title)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self._toggle_action = QAction(title, self)
        self._toggle_action.setCheckable(True)
        self._toggle_action.toggled.connect(self._on_toggle_action)
        self._central_widget = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

    def setWidget(self, widget: QWidget) -> None:
        if self._central_widget is not None:
            self._central_widget.setParent(None)
        self._central_widget = widget
        if widget is not None:
            widget.setParent(self)
            self._layout.addWidget(widget)

    def widget(self) -> QWidget | None:
        return self._central_widget

    def toggleViewAction(self) -> QAction:
        return self._toggle_action

    def setAllowedAreas(self, *_args, **_kwargs) -> None:
        pass

    def setFeatures(self, *_args, **_kwargs) -> None:
        pass

    def isFloating(self) -> bool:
        return False

    def setFloating(self, _floating: bool) -> None:
        pass

    def _on_toggle_action(self, checked: bool) -> None:
        if checked:
            self.show()
            self.raise_()
            self.activateWindow()
        else:
            self.hide()

    def showEvent(self, event):
        super().showEvent(event)
        self._toggle_action.blockSignals(True)
        self._toggle_action.setChecked(True)
        self._toggle_action.blockSignals(False)
        self.visibilityChanged.emit(True)

    def hideEvent(self, event):
        super().hideEvent(event)
        self._toggle_action.blockSignals(True)
        self._toggle_action.setChecked(False)
        self._toggle_action.blockSignals(False)
        self.visibilityChanged.emit(False)

    def closeEvent(self, event):
        event.ignore()
        self.hide()


# ------------------------------------------------------------
#
# clear up tables
#
# ------------------------------------------------------------


def clear_rows(target, *, keep_headers: bool = True) -> None:
    """
    Clear all rows. If keep_headers=False, also clear header labels.
    `target` can be QTableView, QSortFilterProxyModel, or a plain model.
    """
    # Normalize to a model (and remember how to reattach if we rebuild)
    if hasattr(target, "model"):          # QTableView
        view = target
        model = view.model()
        set_model = view.setModel
    else:                                 # model or proxy
        view = None
        model = target
        set_model = None
    if model is None:
        return

    proxy = model if isinstance(model, QSortFilterProxyModel) else None
    src = proxy.sourceModel() if proxy else model
    if src is None:
        return

    rc = src.rowCount()

    # Fast path: QStandardItemModel
    if isinstance(src, QStandardItemModel):
        if rc:
            src.removeRows(0, rc)
        if not keep_headers:
            cols = src.columnCount()
            if cols:
                src.setHorizontalHeaderLabels([""] * cols)
        return

    # Generic path: try to remove rows via API
    ok = True
    if rc and hasattr(src, "removeRows"):
        try:
            ok = bool(src.removeRows(0, rc))
        except Exception:
            ok = False
    if ok:
        if not keep_headers and hasattr(src, "setHeaderData"):
            cols = src.columnCount()
            for c in range(cols):
                try:
                    src.setHeaderData(c, Qt.Horizontal, "")
                except Exception:
                    pass
        return

    # Fallback: rebuild an empty QStandardItemModel, preserving or blanking headers
    cols = src.columnCount()
    headers = [
        src.headerData(c, Qt.Horizontal, Qt.DisplayRole)
        for c in range(cols)
    ]
    new = QStandardItemModel(view or proxy)
    new.setColumnCount(cols)
    if keep_headers:
        new.setHorizontalHeaderLabels([("" if h is None else str(h)) for h in headers])
    else:
        new.setHorizontalHeaderLabels([""] * cols)

    if proxy:
        proxy.setSourceModel(new)
    elif set_model:
        set_model(new)

    

# ------------------------------------------------------------
#
# sort a df
#
# ------------------------------------------------------------

def sort_df_by_list(df, col_idx, order_list):
    """
    Sort DataFrame by the values in a specific column (by index)
    according to a given order list. Case-insensitive.  
    Any rows with values not in order_list are kept at the end,
    preserving their original order.
    """
    col = df.columns[col_idx]
    order_lower = [x.lower() for x in order_list]

    df = df.copy()
    df["_key_lower"] = df[col].astype(str).str.lower()
    df["_pos"] = df["_key_lower"].apply(
        lambda x: order_lower.index(x) if x in order_lower else len(order_lower)
    )

    df_sorted = df.sort_values("_pos", kind="stable").drop(columns=["_key_lower", "_pos"])
    return df_sorted


def winsorize_array(values, limit):
    arr = np.asarray(values, dtype=float).copy()
    lim = float(limit)
    if lim <= 0:
        return arr
    if lim >= 0.5:
        lim = 0.5
    good = np.isfinite(arr)
    if not np.any(good):
        return arr
    lo = np.nanquantile(arr[good], lim)
    hi = np.nanquantile(arr[good], 1.0 - lim)
    arr[good] = np.clip(arr[good], lo, hi)
    return arr


        
# ------------------------------------------------------------
#
# dock menu toggle
#
# ------------------------------------------------------------

def add_dock_shortcuts(win, view_menu, toggle_zero=None, reset_layout=None):

    # hide/show all

    act_show_all = QAction("Show/Hide All Docks", win, checkable=False)
    act_show_all.setShortcut("Ctrl+0")

    if toggle_zero is None:
        def toggle_all():
            docks = win.findChildren(QDockWidget)
            all_hidden = all(not d.isVisible() for d in docks)
            for d in docks:
                d.setVisible(all_hidden)
        act_show_all.triggered.connect(toggle_all)
    else:
        act_show_all.triggered.connect(toggle_zero)
    view_menu.addAction(act_show_all)

    # reset to default layout
    if reset_layout is not None:
        act_reset = QAction("Reset to Default Layout", win, checkable=False)
        act_reset.setShortcut(QKeySequence("Ctrl+R"))
        act_reset.triggered.connect(reset_layout)
        view_menu.addAction(act_reset)

    # control individual docks

    for act in win.menuView.actions():
        if act.text() == "(1) Project sample list":
            act.setShortcut("Ctrl+1")
        elif act.text() == "(2) Parameters":
            act.setShortcut("Ctrl+2")
        elif act.text() == "(3) Signals":
            act.setShortcut("Ctrl+3")
        elif act.text() == "(4) Annotations":
            act.setShortcut("Ctrl+4")
        elif act.text() == "(5) Instances":
            act.setShortcut("Ctrl+5")
        elif act.text() == "(6) Spectrograms":
            act.setShortcut("Ctrl+6")
        elif act.text() == "(7) Hypnograms":
            act.setShortcut("")
        elif act.text() == "(8) Console":
            act.setShortcut("Ctrl+8")
        elif act.text() == "(9) Outputs":
            act.setShortcut("Ctrl+9")
        elif act.text() == "(-) Masks / Subset":
            act.setShortcut("Ctrl+-")
        elif act.text() == "(/) Commands":
            act.setShortcut("Ctrl+/")

    return act_show_all

#
#
# Pick color dialog
#

        
class TwoColorDialog(QDialog):
    def __init__(self, color1=None, color2=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pick background/signal colors")
        self.color1 = QColor(color1 or "#ffffff")
        self.color2 = QColor(color2 or "#000000")

        self.btn1 = QPushButton()
        self.btn2 = QPushButton()
        for b in (self.btn1, self.btn2):
            b.setFixedWidth(80)
        self._update_button_colors()

        self.btn1.clicked.connect(lambda: self.pick_color(1))
        self.btn2.clicked.connect(lambda: self.pick_color(2))

        ok = QPushButton("OK")
        cancel = QPushButton("Cancel")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)

        row = QHBoxLayout()
        row.addWidget(QLabel("Background:"))
        row.addWidget(self.btn1)
        row.addWidget(QLabel("Traces:"))
        row.addWidget(self.btn2)

        row2 = QHBoxLayout()
        row2.addStretch()
        row2.addWidget(ok)
        row2.addWidget(cancel)

        layout = QVBoxLayout(self)
        layout.addLayout(row)
        layout.addLayout(row2)

    def _update_button_colors(self):
        self.btn1.setStyleSheet(f"background-color: {self.color1.name()}")
        self.btn2.setStyleSheet(f"background-color: {self.color2.name()}")

    def pick_color(self, which):
        start = self.color1 if which == 1 else self.color2
        c = QColorDialog.getColor(start, self, "Select Color")
        if c.isValid():
            if which == 1:
                self.color1 = c
            else:
                self.color2 = c
            self._update_button_colors()

def pick_two_colors(c1="#ffffff", c2="#000000"):
    dlg = TwoColorDialog(c1, c2)
    if dlg.exec():
        return dlg.color1, dlg.color2
    return None, None



from PySide6.QtGui import QColor

def _canon(name: str) -> str:
    return name.strip().upper()

def _coerce(color_value, like):
    """Return color_value coerced to the type of 'like' (hex str, tuple, QColor)."""
    if isinstance(like, QColor):
        c = QColor(color_value)
        return c if c.isValid() else like
    if isinstance(like, tuple):  # (r,g,b) or (r,g,b,a)
        c = QColor(color_value)
        return (c.red(), c.green(), c.blue(), c.alpha()) if len(like) == 4 else (c.red(), c.green(), c.blue())
    # default: string hex
    c = QColor(color_value)
    return c.name(QColor.HexArgb if isinstance(like, str) and like.startswith("#") and len(like) == 9 else QColor.HexRgb)

def override_colors(colors, names, overrides: dict):
    """
    colors: list of existing colors (hex str, (r,g,b[,_a]), or QColor)
    names:  list of channel names same length as colors
    overrides: dict like {'Fp1':'#ffee00', ...}
    """
    ov = { _canon(k): v for k, v in overrides.items() }
    out = []
    for col, name in zip(colors, names):
        key = _canon(name)
        if key in ov:
            out.append(_coerce(ov[key], like=col))
        else:
            out.append(col)
    return out


# ------------------------------------------------------------
#
# select N random colors
#
# ------------------------------------------------------------

def random_darkbg_colors(n, seed=None):
    """Return n random pyqtgraph colors (no hue spacing constraints)."""
    rng = random.Random(seed)
    cols = []
    for _ in range(n):
        h = rng.random()                       # full hue range
        s = rng.uniform(0.65, 0.95)            # vivid
        v = rng.uniform(0.78, 0.95)            # bright on dark bg
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        cols.append(pg.mkColor(int(r*255), int(g*255), int(b*255)))
    return cols


# ------------------------------------------------------------
#
# dialog to block GUI 
#
# ------------------------------------------------------------

import weakref
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, QEvent
from PySide6.QtGui import QPainter, QColor


class Blocker(QWidget):
    """
    Child overlay that blocks input and shows a centered message.
    Safe on shutdown (no 'C++ object already deleted' errors).
    """

    def __init__(self, parent, message="Working…", alpha=180, manage_peers=True):
        super().__init__(parent)
        self._parent_ref = weakref.ref(parent)
        self._dead = False
        self._alpha = int(alpha)
        self._manage_peers = bool(manage_peers)
        self._peer_blockers = {}

        # window + event setup
        self.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setObjectName("_busy_blocker")

        # label
        self.label = QLabel(message, self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setAttribute(Qt.WA_TranslucentBackground, True)
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet(
            "QLabel { color: white; font-size: 22px; background: transparent; "
            "background-color: transparent; border: none; }"
        )

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addStretch(1)
        lay.addWidget(self.label, alignment=Qt.AlignCenter)
        lay.addStretch(1)

        # paint-based translucent background
        if parent:
            parent.installEventFilter(self)
            try:
                parent.destroyed.connect(self._on_parent_destroyed)
            except RuntimeError:
                pass

        self.hide()

    def paintEvent(self, _):
        p = QPainter(self)
        alpha = self._alpha
        if is_dark_palette():
            alpha = max(alpha, 170)
        p.fillRect(self.rect(), QColor(28, 28, 32, alpha))

    def eventFilter(self, obj, ev):
        if self._dead:
            return False
        parent = self._parent_ref()
        if not parent:
            return False
        if obj is parent and ev.type() in (
            QEvent.Resize, QEvent.Move, QEvent.Show, QEvent.WindowStateChange
        ):
            self.setGeometry(parent.rect())
        return False

    def show_block(self, msg=None, alpha=None):
        if msg is not None:
            self.label.setText(msg)
        if alpha is not None:
            self._alpha = int(alpha)
        parent = self._parent_ref()
        if parent:
            self.setGeometry(parent.rect())
        self.show()
        self.raise_()
        if self._manage_peers:
            self._sync_peer_blockers()

    def hide_block(self):
        self.hide()
        self._clear_peer_blockers()

    def _on_parent_destroyed(self):
        self._dead = True
        self._clear_peer_blockers()
        self.hide()
        self.deleteLater()

    def _iter_peer_targets(self):
        parent = self._parent_ref()
        if not parent:
            return

        seen = {id(parent)}
        for dock in parent.findChildren(QDockWidget):
            if (
                dock is not self
                and dock.isVisible()
                and dock.isWindow()
                and id(dock) not in seen
            ):
                seen.add(id(dock))
                yield dock

        for widget in QApplication.topLevelWidgets():
            if (
                widget is self
                or widget is parent
                or isinstance(widget, Blocker)
                or not widget.isVisible()
                or not widget.isWindow()
                or widget.parentWidget() is not parent
                or id(widget) in seen
            ):
                continue
            seen.add(id(widget))
            yield widget

    def _sync_peer_blockers(self):
        msg = self.label.text()
        parent = self._parent_ref()
        if not parent:
            self._clear_peer_blockers()
            return

        active = set()
        for widget in self._iter_peer_targets():
            key = id(widget)
            active.add(key)
            peer = self._peer_blockers.get(key)
            if peer is None:
                peer = Blocker(widget, msg, alpha=self._alpha, manage_peers=False)
                self._peer_blockers[key] = peer
            peer.show_block(msg, self._alpha)

        for key in list(self._peer_blockers):
            if key not in active:
                peer = self._peer_blockers.pop(key)
                peer.hide_block()
                peer.deleteLater()

    def _clear_peer_blockers(self):
        for peer in self._peer_blockers.values():
            peer.hide_block()
            peer.deleteLater()
        self._peer_blockers.clear()


# ------------------------------------------------------------
# Screen-aware sizing helpers
# ------------------------------------------------------------

def screen_clamp(w: int, h: int, frac: float = 0.85):
    """Return (w, h) clamped to *frac* × the primary screen's available size."""
    from PySide6.QtGui import QGuiApplication
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        return w, h
    avail = screen.availableGeometry()
    return min(w, int(avail.width() * frac)), min(h, int(avail.height() * frac))


def is_dark_palette() -> bool:
    """Return True when the application palette background is dark."""
    from PySide6.QtGui import QGuiApplication, QPalette
    palette = QGuiApplication.palette()
    return palette.color(QPalette.Window).lightness() < 128
