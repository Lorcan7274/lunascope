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

import numpy as np

import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QModelIndex
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QPushButton, QSizePolicy, QSpinBox, QTableView, QVBoxLayout, QWidget,
)


# ---------------------------------------------------------------------------
# Difference-mode overlay item (XOR-like: inverts whatever is underneath)
# ---------------------------------------------------------------------------

def _to_qcolor(color):
    """Convert any pyqtgraph-compatible color spec to QColor."""
    if isinstance(color, QtGui.QColor):
        return color
    if color is None:
        return QtGui.QColor(255, 255, 255)
    return QtGui.QColor(pg.mkColor(color))


class _DiffOverlayItem(pg.GraphicsObject):
    """Rectangles painted with CompositionMode_Difference.

    Using the channel's own signal color as the Difference source:
      |channel_color - black_bg|   = channel_color  → colored box background
      |channel_color - signal|     ≈ 0 where signal matches  → dark/black trace
    Result: signal appears as dark trace on a colored box, always legible.
    """

    def __init__(self, rects, color=None):
        super().__init__()
        self._rects = rects   # list[QRectF] in data coordinates
        c = _to_qcolor(color)
        self._brush = QtGui.QBrush(c)
        if rects:
            x0 = min(r.left()   for r in rects)
            x1 = max(r.right()  for r in rects)
            y0 = min(r.top()    for r in rects)
            y1 = max(r.bottom() for r in rects)
            self._br = QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)
        else:
            self._br = QtCore.QRectF()

    def boundingRect(self):
        return self._br

    def paint(self, painter, option, widget=None):
        painter.save()
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Difference)
        painter.setBrush(self._brush)
        painter.setPen(QtCore.Qt.NoPen)
        for r in self._rects:
            painter.drawRect(r)
        painter.restore()


# ---------------------------------------------------------------------------
# Colours for CHEP matrix cells
# ---------------------------------------------------------------------------

_C_CHEP_REMOVED = QColor("#111820")   # epoch removed by RE    → near-black
_C_CHEP_CLEAN   = QColor("#1f6640")   # CHEP=1 (clean)         → dark green
_C_CHEP_FLAGGED = QColor("#8b1a1a")   # CHEP=0 (flagged bad)   → dark red


# ---------------------------------------------------------------------------
# CHEP matrix model
# ---------------------------------------------------------------------------

class _ChepModel(QtCore.QAbstractTableModel):
    """Virtual colour model: rows=channels, cols=original epochs."""

    ST_REMOVED = 0   # epoch removed by RE (or no data)
    ST_CLEAN   = 1   # CHEP=1 — present and clean
    ST_FLAGGED = 2   # CHEP=0 — flagged as bad

    _COLOURS = {
        ST_REMOVED: _C_CHEP_REMOVED,
        ST_CLEAN:   _C_CHEP_CLEAN,
        ST_FLAGGED: _C_CHEP_FLAGGED,
    }

    def __init__(self, row_names, n_epochs, matrix, parent=None):
        super().__init__(parent)
        self._rows = row_names          # list[str] — channel names
        self._ne   = n_epochs           # total original epoch count
        self._mat  = matrix             # np.ndarray int8 [n_rows × n_epochs]

    def rowCount(self, parent=QModelIndex()):
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return self._ne

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        if r >= len(self._rows) or c >= self._ne:
            return None
        if role == Qt.BackgroundRole:
            return self._COLOURS.get(int(self._mat[r, c]), _C_CHEP_REMOVED)
        if role == Qt.ToolTipRole:
            label = {0: "removed", 1: "clean", 2: "CHEP-flagged"}.get(int(self._mat[r, c]), "?")
            return f"{self._rows[r]}  ·  E{c + 1}  →  {label}"
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Vertical:
            if section >= len(self._rows):
                return None
            if role in (Qt.DisplayRole, Qt.ToolTipRole):
                return self._rows[section]
        else:
            if section >= self._ne:
                return None
            if role == Qt.ToolTipRole:
                return f"E{section + 1}"
        return None

    def flags(self, index):
        return Qt.ItemIsEnabled


# ---------------------------------------------------------------------------
# MasksMixin
# ---------------------------------------------------------------------------

class MasksMixin:

    def _init_masks(self):

        # overlay state
        self._chep_overlay_items = []
        self._chep_overlay_on    = False
        self._chep_df            = None    # DataFrame with CH, E, CHEP columns
        self._epoch_df           = None    # DataFrame with E, START — maps current E → orig pos

        # wire existing tabs
        self.ui.butt_generic_mask.clicked.connect(self._apply_mask)
        self.ui.butt_drop_subset.clicked.connect(self._drop_signals_annots)

        # add CHEP View as third tab
        self.ui.tab_mask.addTab(self._build_chep_tab(), "CHEP View")

    # ------------------------------------------------------------
    # Build CHEP View tab widget
    # ------------------------------------------------------------

    def _build_chep_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)

        # top bar
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)

        self._chep_butt_update = QPushButton("Update")
        self._chep_butt_update.setFixedWidth(60)
        self._chep_butt_update.setToolTip("Run CHEP dump and refresh the matrix")
        self._chep_butt_update.clicked.connect(self._fetch_chep_data)

        self._chep_butt_show = QPushButton("Show Masks")
        self._chep_butt_show.setCheckable(True)
        self._chep_butt_show.setFixedWidth(95)
        self._chep_butt_show.setToolTip(
            "Overlay CHEP / epoch-mask shading on the signal view.\n"
            "Red = CHEP-flagged channel band; dark = epoch removed by RE.\n"
            "Click Update first to load CHEP data."
        )
        self._chep_butt_show.toggled.connect(self._chep_overlay_toggle)

        self._chep_info_lbl = QLabel("No data")
        self._chep_info_lbl.setStyleSheet("color:#888; font-size:11px;")
        self._chep_info_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        top.addWidget(self._chep_butt_update)
        top.addWidget(self._chep_butt_show)
        top.addWidget(self._chep_info_lbl)
        top.addStretch()

        # column width + row height controls
        top.addWidget(QLabel("W:"))
        self._chep_spin_w = QSpinBox()
        self._chep_spin_w.setRange(1, 40)
        self._chep_spin_w.setValue(4)
        self._chep_spin_w.setSuffix(" px")
        self._chep_spin_w.setFixedWidth(58)
        self._chep_spin_w.setToolTip("Column (epoch) width in pixels")
        self._chep_spin_w.valueChanged.connect(self._apply_chep_cell_size)
        top.addWidget(self._chep_spin_w)

        top.addWidget(QLabel("H:"))
        self._chep_spin_h = QSpinBox()
        self._chep_spin_h.setRange(4, 60)
        self._chep_spin_h.setValue(18)
        self._chep_spin_h.setSuffix(" px")
        self._chep_spin_h.setFixedWidth(58)
        self._chep_spin_h.setToolTip("Row (channel) height in pixels")
        self._chep_spin_h.valueChanged.connect(self._apply_chep_cell_size)
        top.addWidget(self._chep_spin_h)

        lay.addLayout(top)

        # matrix view
        self._chep_view = QTableView()
        self._chep_view.setShowGrid(False)
        self._chep_view.setSelectionMode(QAbstractItemView.NoSelection)
        self._chep_view.setStyleSheet(
            "QTableView { selection-background-color: transparent; }"
        )

        hdr = self._chep_view.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Fixed)
        hdr.setMinimumSectionSize(1)
        hdr.setVisible(False)

        vhdr = self._chep_view.verticalHeader()
        vhdr.setSectionResizeMode(QHeaderView.Fixed)
        vhdr.setMinimumSectionSize(4)

        lay.addWidget(self._chep_view, stretch=1)
        return w

    # ------------------------------------------------------------
    # Update list of potential mask annots
    # ------------------------------------------------------------

    def _update_mask_list(self):

        self.ui.combo_ifnot_mask.clear()
        self.ui.combo_if_mask.clear()

        anns = ['<none>']
        lst  = self.p.edf.annots()
        lst  = [s for s in lst if s != "SleepStage"]
        anns.extend(lst)

        self.ui.combo_ifnot_mask.addItems(anns)
        self.ui.combo_if_mask.addItems(anns)

    # ------------------------------------------------------------
    # Apply MASK
    # ------------------------------------------------------------

    def _apply_mask(self):

        if not hasattr(self, "p"):
            return

        gen_msk   = self.ui.txt_generic_mask.text()
        if_msk    = self.ui.combo_if_mask.currentText()
        ifnot_msk = self.ui.combo_ifnot_mask.currentText()

        n   = 0
        msk = ''
        if gen_msk != "":         n += 1; msk = gen_msk
        if if_msk != "<none>":    n += 1; msk = 'if=' + if_msk
        if ifnot_msk != "<none>": n += 1; msk = 'ifnot=' + ifnot_msk

        if n == 0:
            QMessageBox.warning(None, "Invalid mask", "No mask values specified")
            return
        if n != 1:
            QMessageBox.warning(None, "Invalid mask", "More than one mask set")
            return

        self.curr_chs  = self.ui.tbl_desc_signals.checked()
        self.curr_anns = self.ui.tbl_desc_annots.checked()

        self.p.eval_lunascope('MASK ' + msk + ' & RE ')

        self._set_render_status(self.rendered, False)
        self._update_metrics()
        self._update_pg1()

        self.ui.tbl_desc_signals.set_checked_by_labels(self.curr_chs)
        if hasattr(self.ui.tbl_desc_annots, "set_checked_by_labels_silent"):
            self.ui.tbl_desc_annots.set_checked_by_labels_silent(self.curr_anns)
        else:
            self.ui.tbl_desc_annots.set_checked_by_labels(self.curr_anns)
        if hasattr(self, "_mark_instances_dirty"):
            self._mark_instances_dirty(self.curr_anns)

    # ------------------------------------------------------------
    # CHEP data fetch
    # ------------------------------------------------------------

    def _fetch_chep_data(self):
        if not hasattr(self, "p"):
            return

        self._chep_df  = None
        self._epoch_df = None

        try:
            # Run both in one eval call so the result store contains both.
            # Two separate eval_lunascope calls would clear the store between them.
            self.p.eval_lunascope('CHEP dump & EPOCH')

            strata_df = self.p.strata()
            if strata_df is None or strata_df.empty:
                self._chep_info_lbl.setText("CHEP dump produced no results")
                self._refresh_chep_matrix()
                return

            # Find CHEP CH×E key and EPOCH epoch-level key in one pass
            chep_key  = None
            epoch_key = None
            for _, row in strata_df.iterrows():
                cmd = str(row.get('Command', ''))
                s   = str(row.get('Strata',  ''))
                if cmd == 'CHEP'  and chep_key  is None and 'CH' in s and 'E' in s:
                    chep_key  = s
                if cmd == 'EPOCH' and epoch_key is None and 'E' in s:
                    epoch_key = s
                if chep_key and epoch_key:
                    break

            if chep_key is None:
                self._chep_info_lbl.setText("CHEP dump produced no CH×E data")
                self._refresh_chep_matrix()
                return

            df_chep = self.p.table('CHEP', chep_key)
            if df_chep is None or df_chep.empty:
                self._chep_info_lbl.setText("CHEP table is empty")
                self._refresh_chep_matrix()
                return
            df_chep.columns = [c.upper() for c in df_chep.columns]
            self._chep_df = df_chep

            # Epoch start times — map current E → original 0-based column.
            # After RE, remaining epochs keep their original START seconds, so
            # orig_eidx = round(START / 30).  Falls back to E-1 if unavailable.
            if epoch_key:
                try:
                    df_ep = self.p.table('EPOCH', epoch_key)
                    if df_ep is not None and not df_ep.empty:
                        df_ep.columns = [c.upper() for c in df_ep.columns]
                        if {'E', 'START'}.issubset(df_ep.columns):
                            self._epoch_df = df_ep
                except Exception:
                    pass

        except Exception as exc:
            self._chep_info_lbl.setText(f"Error: {exc}")

        self._refresh_chep_matrix()

    # ------------------------------------------------------------
    # Rebuild the CHEP matrix model and display
    # ------------------------------------------------------------

    def _refresh_chep_matrix(self):
        ne_orig = getattr(self, 'ne', 0)

        if ne_orig == 0:
            self._chep_info_lbl.setText("No EDF attached")
            return

        if self._chep_df is None:
            # label already set in _fetch_chep_data; just show empty grid
            empty = np.zeros((1, ne_orig), dtype=np.int8)
            self._chep_view.setModel(_ChepModel(["—"], ne_orig, empty))
            self._apply_chep_cell_size()
            return

        chs = self.ui.tbl_desc_signals.checked()
        if not chs:
            self._chep_info_lbl.setText("No channels selected in Signals dock")
            return

        df = self._chep_df
        required = {'CH', 'E', 'CHEP'}
        if not required.issubset(set(df.columns)):
            self._chep_info_lbl.setText(
                f"Unexpected CHEP columns: {list(df.columns)}"
            )
            return

        # Build current-E → original-0based-column map from epoch START times.
        # After RE, remaining epochs keep their original START seconds, so
        # orig_eidx = int(START / 30).  Without the map (no EPOCH data) we fall
        # back to E-1, which is correct only before any RE.
        epoch_dur = 30.0
        e_to_orig: dict = {}
        if self._epoch_df is not None:
            for _, erow in self._epoch_df.iterrows():
                try:
                    e_to_orig[int(erow['E'])] = int(round(float(erow['START']) / epoch_dur))
                except (ValueError, TypeError):
                    pass

        n_rows = len(chs)
        mat    = np.zeros((n_rows, ne_orig), dtype=np.int8)   # 0 = removed

        ch_idx = {ch: i for i, ch in enumerate(chs)}

        for _, row in df.iterrows():
            ch     = str(row['CH'])
            e_val  = int(row['E'])
            chep_v = int(row['CHEP'])

            eidx = e_to_orig.get(e_val, e_val - 1)  # fallback: pre-RE case

            if not (0 <= eidx < ne_orig):
                continue

            state = _ChepModel.ST_FLAGGED if chep_v == 1 else _ChepModel.ST_CLEAN

            if ch in ch_idx:
                mat[ch_idx[ch], eidx] = state

        row_names = list(chs)
        model     = _ChepModel(row_names, ne_orig, mat)
        self._chep_view.setModel(model)
        self._apply_chep_cell_size()

        n_flagged = int(np.sum(mat == _ChepModel.ST_FLAGGED))
        n_removed = int(np.sum(np.all(mat == _ChepModel.ST_REMOVED, axis=0)))
        self._chep_info_lbl.setText(
            f"{ne_orig} ep × {len(chs)} ch  |  "
            f"{n_flagged} CHEP-flagged  |  {n_removed} ep removed"
        )

        if self._chep_overlay_on and hasattr(self, '_update_pg1'):
            self._update_pg1()

    # ------------------------------------------------------------
    # Cell-size spinboxes
    # ------------------------------------------------------------

    def _apply_chep_cell_size(self):
        model = self._chep_view.model()
        if model is None:
            return
        pw = int(self._chep_spin_w.value())
        ph = int(self._chep_spin_h.value())

        hdr = self._chep_view.horizontalHeader()
        hdr.setDefaultSectionSize(pw)
        for c in range(model.columnCount()):
            hdr.resizeSection(c, pw)

        vhdr = self._chep_view.verticalHeader()
        vhdr.setDefaultSectionSize(ph)
        for r in range(model.rowCount()):
            vhdr.resizeSection(r, ph)

    # ------------------------------------------------------------
    # Signal overlay — toggle (updates button label as indicator)
    # ------------------------------------------------------------

    def _chep_overlay_toggle(self, checked):
        self._chep_overlay_on = checked
        self._chep_butt_show.setText("Hide Masks" if checked else "Show Masks")

        # reflect in dock title
        dw = getattr(self.ui, 'dock_mask', None)
        if dw is not None:
            base = "(-) Masks / Subset"
            dw.setWindowTitle(base + "  ●" if checked else base)

        if hasattr(self, '_update_pg1'):
            self._update_pg1()

    # ------------------------------------------------------------
    # Signal overlay hook — called by both _update_pg1 and _update_pg1_simple
    # ------------------------------------------------------------

    def _chep_overlay_on_trace_redraw(self):
        self._clear_chep_overlay()
        if not self._chep_overlay_on:
            return
        if self._chep_df is None:
            return
        self._draw_chep_overlay()

    def _clear_chep_overlay(self):
        pw = getattr(self.ui, 'pg1', None)
        if pw is None:
            return
        pi = pw.getPlotItem()
        for item in self._chep_overlay_items:
            try:
                pi.removeItem(item)
            except Exception:
                pass
        self._chep_overlay_items.clear()

    def _draw_chep_overlay(self):

        pw = getattr(self.ui, 'pg1', None)
        if pw is None:
            return
        pi = pw.getPlotItem()

        model = self._chep_view.model()
        if not isinstance(model, _ChepModel):
            return

        # current view window — works for both rendered and non-rendered mode
        vr           = pw.getViewBox().viewRange()
        win_x1, win_x2 = vr[0]

        ne_orig   = model._ne
        row_names = model._rows
        mat       = model._mat
        n_ch_rows = len(row_names)

        # channel → (y_lo, y_hi, color) from _pg1_channel_cache
        ch_bands: dict[str, tuple] = {
            e['ch']: (float(e['band_lo']), float(e['band_hi']), e.get('color'))
            for e in getattr(self, '_pg1_channel_cache', [])
        }

        # epoch E (1-indexed) → x_start: after RE epochs start at 0, 30, 60 …
        # so x_start = (E-1)*30  (E is 1-based column index + 1)
        epoch_dur = 30.0

        # --- collect bars per category ---

        # removed epochs: full-height bars (y0=0, height=1)
        rem_cx:  list = []
        rem_w:   list = []

        # CHEP-flagged per channel: bars within the channel's Y band
        # key = (y0, y1), value = [cx_list, w_list, channel_color]
        flagged_by_band: dict = {}

        for oidx in range(ne_orig):
            xs = float(oidx) * epoch_dur
            xe = xs + epoch_dur
            cx = xs + epoch_dur * 0.5

            # skip entirely off-screen
            if xe < win_x1 or xs > win_x2:
                continue

            epoch_removed = bool(np.all(mat[:n_ch_rows, oidx] == _ChepModel.ST_REMOVED))

            if epoch_removed:
                rem_cx.append(cx)
                rem_w.append(epoch_dur)

            # per-channel CHEP-flagged bars
            for ri in range(n_ch_rows):
                if int(mat[ri, oidx]) != _ChepModel.ST_FLAGGED:
                    continue
                ch   = row_names[ri]
                band = ch_bands.get(ch)
                if band is None:
                    continue
                y0, y1, ch_color = band
                key = (y0, y1)
                if key not in flagged_by_band:
                    flagged_by_band[key] = ([], [], ch_color)
                flagged_by_band[key][0].append(cx)
                flagged_by_band[key][1].append(epoch_dur)

        # --- draw removed-epoch bars ---
        if rem_cx:
            bg = pg.BarGraphItem(
                x=rem_cx, width=rem_w,
                y0=[0.0] * len(rem_cx), height=[1.0] * len(rem_cx),
                brush=QtGui.QColor(20, 20, 20, 80), pen=None,
            )
            bg.setZValue(-5)
            bg.setAcceptedMouseButtons(Qt.NoButton)
            pi.addItem(bg)
            self._chep_overlay_items.append(bg)

        # --- draw CHEP-flagged channel bars (Difference / XOR-like) ---
        for (y0, y1), (cxs, ws, ch_color) in flagged_by_band.items():
            h = max(float(y1) - float(y0), 0.0)
            if h <= 0:
                continue
            rects = [QtCore.QRectF(cx - w * 0.5, float(y0), w, h)
                     for cx, w in zip(cxs, ws)]
            item = _DiffOverlayItem(rects, color=_to_qcolor(ch_color))
            item.setZValue(5)
            item.setAcceptedMouseButtons(Qt.NoButton)
            pi.addItem(item)
            self._chep_overlay_items.append(item)
