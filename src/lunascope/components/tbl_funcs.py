
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


from PySide6.QtCore import Qt, QSize, QObject
from PySide6.QtWidgets import (
    QApplication, QComboBox, QStyledItemDelegate, QTableView,
    QStyle, QStyleOptionViewItem, QHeaderView
)
from PySide6.QtGui import QIcon, QStandardItem
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFileDialog

from typing import Iterable, Optional, Callable, List
from PySide6.QtCore import Qt, QSignalBlocker
from PySide6.QtWidgets import QTableView, QHeaderView
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import QSortFilterProxyModel


class ComboDelegate(QStyledItemDelegate):
    def __init__(self, items, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.items = list(items)

    # do not paint the cell text when using a persistent editor
    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        opt.icon = QIcon()
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter)

    def createEditor(self, parent, option, index):
        w = QComboBox(parent)
        w.addItems(self.items)
        w.setEditable(False)
        w.setFrame(False)
        w.setStyleSheet("QComboBox{padding:0;margin:0;}QComboBox::down-arrow{image:none;}QComboBox::drop-down{width:0;border:0;}")
        # commit to model whenever selection changes
        w.currentIndexChanged.connect(lambda _=None, ed=w: self.commitData.emit(ed))
        return w

    def updateEditorGeometry(self, editor, option, index):
        # leave the right grid line visible
        r = option.rect.adjusted(0, 0, -1, 0)
        editor.setGeometry(r)
                
    def sizeHint(self, option, index):
        return QSize(80, option.rect.height())

    def setEditorData(self, editor, index):
        val = index.data(Qt.EditRole) or index.data(Qt.DisplayRole) or ""
        i = editor.findText(str(val))
        editor.setCurrentIndex(i if i >= 0 else 0)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), Qt.EditRole)


def add_combo_column(
    view: QTableView,
    header_text: str,
    items,
    default_value: str | None = None,
    insert_source_col: int | None = None,
    open_persistent: bool = True,
    on_change=None,
    width: int = 90,
    resize_mode: QHeaderView.ResizeMode = QHeaderView.Fixed,
) -> int:
    """
    Adds a compact combo column to a QTableView that is bound to either:
      - a QSortFilterProxyModel over a QStandardItemModel, or
      - a QStandardItemModel directly.

    Returns the proxy column index for further use.
    """
    proxy = view.model()
    if proxy is None:
        raise RuntimeError("view has no model")

    # get source model if proxy, else the model itself
    try:
        src = proxy.sourceModel()
    except AttributeError:
        src = proxy

    nrows = src.rowCount()
    ncols = src.columnCount()

    col = ncols if insert_source_col is None else int(insert_source_col)
    src.insertColumn(col)
    src.setHeaderData(col, Qt.Horizontal, header_text)

    if default_value is None:
        default_value = next(iter(items), "")

    for r in range(nrows):
        it = QStandardItem(str(default_value))
        it.setEditable(True)
        it.setSelectable(True)
        it.setEnabled(True)
        src.setItem(r, col, it)

    # map to proxy column
    if src is proxy:
        proxy_col = col
    else:
        if nrows > 0:
            proxy_col = proxy.mapFromSource(src.index(0, col)).column()
        else:
            # safe fallback when no rows exist yet
            proxy_col = col

    # install delegate and keep it alive
    delegate = ComboDelegate(items, view)
    view.setItemDelegateForColumn(proxy_col, delegate)
    if not hasattr(view, "_column_delegates"):
        view._column_delegates = {}
    view._column_delegates[proxy_col] = delegate

    # width and resize policy
    view.horizontalHeader().setSectionResizeMode(proxy_col, resize_mode)
    view.setColumnWidth(proxy_col, width)

    # open editors so the combo is always visible
    def _open_all_persistent():
        prow = proxy.rowCount()
        for r in range(prow):
            idx = proxy.index(r, proxy_col)
            view.openPersistentEditor(idx)

    if open_persistent:
        _open_all_persistent()

        # keep new rows opened too
        def _rows_inserted(*_):
            _open_all_persistent()
        src.rowsInserted.connect(_rows_inserted)

    # optional change hook
    if on_change is not None:
        def _emit_change(*_):
            on_change(None)
        src.dataChanged.connect(_emit_change)

    return proxy_col


    
# ------------------------------------------------------------
#
# new add_check_column() -- more robust across platforms
#
# ------------------------------------------------------------


import types
from PySide6.QtCore import QSignalBlocker, Qt, QTimer


def add_check_column(view, channel_col_before_insert, header_text="✔",
                     initial_checked=None, on_change=None, visible_only=False):

    # expects: Qt, QTimer, QSignalBlocker, QStandardItem, QStandardItemModel,
    #          QHeaderView, QSortFilterProxyModel, types
    model = view.model()
    proxy = model if isinstance(model, QSortFilterProxyModel) else None
    src = proxy.sourceModel() if proxy else model
    if not isinstance(src, QStandardItemModel):
        raise TypeError("Expect QStandardItemModel or proxy->QStandardItemModel")

    # detach proxy during structure change
    if proxy:
        proxy.setSourceModel(None)

    # insert column 0
    src.insertColumn(0)
    if header_text:
        src.setHeaderData(0, Qt.Horizontal, header_text)

    checked = set(map(str, (initial_checked or [])))
    chan_col_after = channel_col_before_insert + 1

    # populate without signals
    src.blockSignals(True)
    try:
        for r in range(src.rowCount()):
            it = QStandardItem()
            it.setEditable(False)
            it.setCheckable(True)
            ch = str(src.data(src.index(r, chan_col_after)))
            it.setCheckState(Qt.Checked if ch in checked else Qt.Unchecked)
            it.setDragEnabled(True)
            it.setDropEnabled(True)
            src.setItem(r, 0, it)
    finally:
        src.blockSignals(False)

    # reattach proxy/model
    if proxy:
        proxy.setSourceModel(src)
        view.setModel(proxy)
    else:
        view.setModel(src)

    view.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)

    # ----- state -----
    _squelch = False          # bulk guard
    _in_on_change = False     # re-entrancy guard
    _scheduled = False        # debouncer flag

    _debounce = QTimer(view)
    _debounce.setSingleShot(True)

    def _checked_visible(self=view, _src=src, _proxy=proxy, _cc=chan_col_after):
        return _checked(self, _src=_src, _proxy=_proxy, _vis=True, _cc=_cc)
    
    def _checked(self=view, _src=src, _proxy=proxy, _vis=visible_only, _cc=chan_col_after):
        out = []
        if _proxy and _vis:
            # robust mapping (skip invalids)
            pr = _proxy.rowCount()
            for r in range(pr):
                pix = _proxy.index(r, 0)
                if not pix.isValid():
                    continue
                six = _proxy.mapToSource(pix)
                if not six.isValid():
                    continue
                srow = six.row()
                it = _src.item(srow, 0)
                if it and it.checkState() == Qt.Checked:
                    out.append(str(_src.data(_src.index(srow, _cc))))
        else:
            for r in range(_src.rowCount()):
                it = _src.item(r, 0)
                if it and it.checkState() == Qt.Checked:
                    out.append(str(_src.data(_src.index(r, _cc))))
        return out

    def _emit_now():
        nonlocal _scheduled, _in_on_change
        if not on_change:
            _scheduled = False
            return
        if _squelch or _in_on_change:
            # still unstable; try again next turn
            _debounce.start(0)
            return
        _scheduled = False
        _in_on_change = True
        try:
            on_change(_checked())
        finally:
            _in_on_change = False

    _debounce.timeout.connect(_emit_now)

    def _schedule_emit():
        nonlocal _scheduled
        if _scheduled:
            # restart to coalesce multiple triggers in this tick
            _debounce.start(0)
            return
        _scheduled = True
        _debounce.start(0)

    def _loop_set(state, xs=None, _src=src, _proxy=proxy, _cc=chan_col_after):
        nonlocal _squelch
        _squelch = True

        # freeze UI and signals
        sorting_was = getattr(view, "isSortingEnabled", lambda: False)()
        if sorting_was:
            view.setSortingEnabled(False)
        vb = False
        try:
            view.setUpdatesEnabled(False)
            vb = True
        except Exception:
            pass

        b_src = QSignalBlocker(_src)
        b_prox = QSignalBlocker(_proxy) if _proxy else None

        changed_any = False
        try:
            target = None if xs is None else (xs if isinstance(xs, set) else set(map(str, xs)))

            # choose rows once
#            if _proxy and visible_only:
            if _proxy:
                pr = _proxy.rowCount()
                src_rows = []
                for r in range(pr):
                    pix = _proxy.index(r, 0)
                    if not pix.isValid():
                        continue
                    six = _proxy.mapToSource(pix)
                    if six.isValid():
                        src_rows.append(six.row())
            else:
                src_rows = range(_src.rowCount())

            # apply updates only when needed
            if target is None:
                want = state
                for r in src_rows:
                    it = _src.item(r, 0)
                    if it and it.checkState() != want:
                        it.setCheckState(want)
                        changed_any = True
            else:
                for r in src_rows:
                    it = _src.item(r, 0)
                    if not it:
                        continue
                    ch = str(_src.data(_src.index(r, _cc)))
                    want = Qt.Checked if ch in target else Qt.Unchecked
                    if it.checkState() != want:
                        it.setCheckState(want)
                        changed_any = True

        finally:
            del b_src
            if b_prox is not None:
                del b_prox
            if vb:
                view.setUpdatesEnabled(True)
            if sorting_was:
                view.setSortingEnabled(True)

            # make proxy recompute mappings before we emit
            if proxy:
                proxy.invalidate()

            _squelch = False

        # one repaint over col 0
        rc = _src.rowCount()
        if rc:
            _src.dataChanged.emit(_src.index(0, 0), _src.index(rc - 1, 0), [Qt.CheckStateRole])

        # defer a single logical change until the model/proxy/view are stable
        if changed_any:
            _schedule_emit()

    # bind helpers
    try:
        view.checked = types.MethodType(_checked, view)
        view.checked_visible = types.MethodType(_checked_visible, view )
        view.select_all_checks = types.MethodType(lambda self: _loop_set(Qt.Checked), view)
        view.select_none_checks = types.MethodType(lambda self: _loop_set(Qt.Unchecked), view)
        view.set_checked_by_labels = types.MethodType(lambda self, xs: _loop_set(Qt.PartiallyChecked, xs), view)
    except AttributeError:
        return {
            "checked": _checked,
            "select_all": lambda: _loop_set(Qt.Checked),
            "select_none": lambda: _loop_set(Qt.Unchecked),
            "set_labels": lambda xs: _loop_set(Qt.PartiallyChecked, xs),
        }

    # per-item handler: ignore during bulk, coalesce otherwise
    def _on_item_changed(itm):
        if itm.column() != 0:
            return
        if _squelch:
            return
        _schedule_emit()

    if not getattr(src, "_checkcol_connected", False):
        src.itemChanged.connect(_on_item_changed)
        setattr(src, "_checkcol_connected", True)


            

# ------------------------------------------------------------
#
# filter tabels
#
# ------------------------------------------------------------
        
def attach_comma_filter(table_view, line_edit, proxy=None):
    from PySide6.QtCore import Qt, QRegularExpression, QSortFilterProxyModel

    # create new proxy only if none provided
    if proxy is None:
        proxy = QSortFilterProxyModel(table_view)
        proxy.setSourceModel(table_view.model())
        table_view.setModel(proxy)
    else:
        # already the view's model; do not setSourceModel again
        if table_view.model() is not proxy:
            proxy.setSourceModel(table_view.model())
            table_view.setModel(proxy)

    def on_text_changed(text: str):
        parts = [s.strip() for s in text.split(',') if s.strip()]
        if not parts:
            proxy.setFilterRegularExpression(QRegularExpression())
            return
        esc = [QRegularExpression.escape(p) for p in parts]
        rx = QRegularExpression("(" + "|".join(esc) + ")")
        rx.setPatternOptions(QRegularExpression.CaseInsensitiveOption)
        proxy.setFilterRegularExpression(rx)

    # avoid duplicate connects
    if not hasattr(proxy, "_comma_filter_connected"):
        line_edit.textChanged.connect(on_text_changed)
        proxy._comma_filter_connected = True

    proxy.setFilterKeyColumn(-1)
    proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
    return proxy



# ------------------------------------------------------------
#
# set signal filters programmatically
#
# ------------------------------------------------------------


def set_filter_for_channel(proxy, channel_name: str, filter_code: str):
    """
    Programmatically set the filter for the given channel.
    Updates the source model directly; proxy view & combo boxes update automatically.
    """

    src = proxy.sourceModel() if hasattr(proxy, "sourceModel") else proxy

    ROWS = src.rowCount()
    CH_COL = 1
    FLT_COL = 2

    # Find the source row with matching channel label
    target_row = None
    for r in range(ROWS):
        ch = src.index(r, CH_COL).data(Qt.DisplayRole)
        if ch == channel_name:
            target_row = r
            break

    if target_row is None:
        print(f"[WARN] Channel '{channel_name}' not found in table.")
        return

    # Set filter value
    idx = src.index(target_row, FLT_COL)
    ok = src.setData(idx, filter_code, Qt.EditRole)

    if not ok:
        print(f"[WARN] setData failed for channel '{channel_name}'")
        return

    # nb. persistent editor will update automatically due to _reopen_all_later logic

    
# --------------------------------------------------------------------------------
# copy table


from PySide6.QtWidgets import QTableView, QToolTip
from PySide6.QtGui import QGuiApplication, QCursor
from PySide6.QtCore import Qt

def copy_selection(
    table: QTableView,
    _self,
    with_headers: bool = True,
    show_message: bool = True ):

    use_na_for_empty = _self.cmap_use_na_for_empty
    na_token = _self.cmap_na_token

    sel = table.selectionModel()
    if not sel or not sel.hasSelection():
        return

    model = table.model()
    indexes = sel.selectedIndexes()
    if not indexes:
        return

    # helper to format cell values
    def fmt_val(val):
        if val is None:
            return na_token if use_na_for_empty else ""
        s = str(val)
        if use_na_for_empty and s == "":
            return na_token
        return s

    # Sort and get unique rows/cols
    indexes = sorted(indexes, key=lambda x: (x.row(), x.column()))
    rows = sorted({idx.row() for idx in indexes})
    cols = sorted({idx.column() for idx in indexes})

    # Build lookup for cell values
    values = {}
    for idx in indexes:
        r, c = idx.row(), idx.column()
        val = model.data(idx, Qt.DisplayRole)
        values[(r, c)] = fmt_val(val)

    lines = []

    # Optional header row (leave headers as-is; no NA substitution)
    if with_headers:
        header_cells = []
        for c in cols:
            h = model.headerData(c, Qt.Horizontal, Qt.DisplayRole)
            header_cells.append("" if h is None else str(h))
        lines.append("\t".join(header_cells))

    # Data rows
    for r in rows:
        # for non-selected cells inside the rectangular block, treat as empty/NA
        row_cells = [values.get((r, c), fmt_val(None)) for c in cols]
        lines.append("\t".join(row_cells))

    text = "\n".join(lines)
    QGuiApplication.clipboard().setText(text)

    if show_message:
        QToolTip.showText(
            QCursor.pos(),
            f"Copied {len(rows)}×{len(cols)} cells" + (" (with headers)" if with_headers else ""),
            table,
        )


# ------------------------------------------------------------
# save .tsv

from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import Qt

def save_table_as_tsv(view,_self):

    use_na_for_empty = _self.cmap_use_na_for_empty
    na_token = _self.cmap_na_token
    
    model = view.model()
    if model is None:
        return

    path, _ = QFileDialog.getSaveFileName(
        view,
        "Save as TSV",
        "",
        "TSV files (*.tsv);;All Files (*)",
        options=QFileDialog.Option.DontUseNativeDialog,
    )
    if not path:
        return

    if not path.lower().endswith(".tsv"):
        path += ".tsv"

    rows = model.rowCount()
    cols = model.columnCount()

    def fmt_val(val):
        if val is None:
            return na_token if use_na_for_empty else ""
        s = str(val)
        if use_na_for_empty and s == "":
            return na_token
        return s

    with open(path, "w", encoding="utf-8") as f:

        # --- HEADER ROW (corrected) ---
        header_cells = []
        for c in range(cols):
            hdr = model.headerData(c, Qt.Horizontal, Qt.DisplayRole)
            header_cells.append("" if hdr is None else str(hdr))
        f.write("\t".join(header_cells) + "\n")

        # --- DATA ROWS ---
        for r in range(rows):
            row_cells = []
            for c in range(cols):
                idx = model.index(r, c)
                val = model.data(idx, Qt.DisplayRole)
                row_cells.append(fmt_val(val))
            f.write("\t".join(row_cells) + "\n")
            
