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

from PySide6.QtCore import QModelIndex, QItemSelection, QItemSelectionModel, Qt
from PySide6.QtGui import QAction, QKeySequence, QShortcut, QStandardItemModel, QStandardItem
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QTreeView,
    QWidget,
)


HELP_TREE_PRIMARY_COL_WIDTH = 160

class CTreeMixin:

    def _init_ctree(self):

        # 5 cols
        # ------
        # <domains>
        #  <commands>
        #   Param
        #    <params>
        #   Tables
        #    <tables>
        #     <vars>
        
        # model
        view = self.ui.tree_helper
        
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Element", "Description"])  

        h = view.header()
        h.setSectionResizeMode(QHeaderView.Interactive)
        h.setStretchLastSection(True)
        h.setMinimumSectionSize(120)
        
        def add_root(model: QStandardItemModel, name: str, desc="") -> QStandardItem:
            n = QStandardItem(str(name)); n.setEditable(False)
            d = QStandardItem(str(desc)); d.setEditable(False)
            model.invisibleRootItem().appendRow([n, d])
            return n  # return the column-0 item; use it as parent

        def add_child(parent: QStandardItem, name: str, desc="") -> QStandardItem:
            n = QStandardItem(str(name)); n.setEditable(False)
            d = QStandardItem(str(desc)); d.setEditable(False)
            parent.appendRow([n, d])
            return n
    
        # domains
        doms = lp.fetch_doms()

        for dom in doms:
            l1 = add_root( model, dom ,  lp.fetch_desc_dom( dom ) )

            # get commands
            cmds = lp.fetch_cmds( dom )
            for cmd in cmds:
                l2 = add_child( l1 , cmd , lp.fetch_desc_cmd( cmd ) )

                l3p = add_child( l2 , "Parameters" , "" )
                l3o = add_child( l2 , "Outputs" , "" )

                # parameters
                params = lp.fetch_params( cmd )
                for param in params:
                    add_child( l3p , param , lp.fetch_desc_param( cmd , param ) )


                # tables
                tbls = lp.fetch_tbls( cmd )
                for tbl in tbls:
                    # '{baseline}' is the display sentinel for the root (no-strata) table,
                    # but fetch_vars/fetch_desc_var/fetch_desc_tbl require "" as the key.
                    api_tbl = "" if tbl == "{baseline}" else tbl
                    tbl_label = "(baseline)" if tbl == "{baseline}" else tbl
                    l4 = add_child( l3o , tbl_label , lp.fetch_desc_tbl( cmd , api_tbl ) )

                    vars = lp.fetch_vars( cmd , api_tbl )
                    for var in vars:
                        add_child( l4 , var , lp.fetch_desc_var( cmd , api_tbl , var ) )

        # finish wiring
        view.setModel(model)              
        view.setUniformRowHeights(True)
        view.setAlternatingRowColors(True)
        view.setExpandsOnDoubleClick(False)
        view.collapseAll()
        view.setColumnWidth(0, HELP_TREE_PRIMARY_COL_WIDTH)

        # set filter
        view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        view.doubleClicked.connect(self._toggle_ctree_index)
        view.activated.connect(self._toggle_ctree_index)
        view.setContextMenuPolicy(Qt.CustomContextMenu)
        view.customContextMenuRequested.connect(self._show_ctree_context_menu)

        controls = QWidget(view.parent())
        controls.setObjectName("ctree_controls")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        self._ctree_expand_branch_btn = QPushButton("Expand branch", controls)
        self._ctree_collapse_branch_btn = QPushButton("Collapse branch", controls)
        self._ctree_expand_all_btn = QPushButton("Expand all", controls)
        self._ctree_collapse_all_btn = QPushButton("Collapse all", controls)

        self._ctree_expand_branch_btn.clicked.connect(self._expand_ctree_branch)
        self._ctree_collapse_branch_btn.clicked.connect(self._collapse_ctree_branch)
        self._ctree_expand_all_btn.clicked.connect(view.expandAll)
        self._ctree_collapse_all_btn.clicked.connect(view.collapseAll)

        for button in (
            self._ctree_expand_branch_btn,
            self._ctree_collapse_branch_btn,
            self._ctree_expand_all_btn,
            self._ctree_collapse_all_btn,
        ):
            controls_layout.addWidget(button)
        controls_layout.addStretch(1)

        dock_layout = self.ui.dock_help.widget().layout()
        dock_layout.insertWidget(1, controls)
        self._init_ctree_shortcuts()

        # wire filter
        self.ui.flt_ctree.textChanged.connect(
            lambda txt: expand_and_show_matches(self.ui.tree_helper, txt , partial = True )
        )

    def _init_ctree_shortcuts(self):
        scope = self.ui.dock_help.widget()
        shortcuts = (
            ("Alt+Right", self._expand_ctree_branch),
            ("Alt+Left", self._collapse_ctree_branch),
            ("Ctrl+Alt+Right", self.ui.tree_helper.expandAll),
            ("Ctrl+Alt+Left", self.ui.tree_helper.collapseAll),
        )

        self._ctree_shortcuts = []
        for seq, handler in shortcuts:
            shortcut = QShortcut(QKeySequence(seq), scope)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(handler)
            self._ctree_shortcuts.append(shortcut)

    def _ctree_selected_roots(self):
        view = self.ui.tree_helper
        selection_model = view.selectionModel()
        if selection_model is None:
            return []

        indexes = selection_model.selectedRows(0)
        if not indexes:
            current = view.currentIndex()
            if current.isValid():
                indexes = [current.siblingAtColumn(0)]

        roots = []
        for index in indexes:
            index = index.siblingAtColumn(0)
            if not index.isValid():
                continue
            if any(self._ctree_is_ancestor(existing, index) for existing in roots):
                continue
            roots = [existing for existing in roots if not self._ctree_is_ancestor(index, existing)]
            roots.append(index)
        return roots

    @staticmethod
    def _ctree_is_ancestor(ancestor, child):
        parent = child.parent()
        while parent.isValid():
            if parent == ancestor:
                return True
            parent = parent.parent()
        return False

    def _toggle_ctree_index(self, index):
        index = index.siblingAtColumn(0)
        if not index.isValid() or not self.ui.tree_helper.model().hasChildren(index):
            return
        self.ui.tree_helper.setExpanded(index, not self.ui.tree_helper.isExpanded(index))

    def _expand_ctree_branch(self):
        roots = self._ctree_selected_roots()
        if not roots:
            self.ui.tree_helper.expandAll()
            return
        for index in roots:
            self.ui.tree_helper.expandRecursively(index)

    def _collapse_ctree_branch(self):
        roots = self._ctree_selected_roots()
        if not roots:
            self.ui.tree_helper.collapseAll()
            return
        for index in roots:
            self.ui.tree_helper.collapse(index)
            self._collapse_ctree_children(index)

    def _collapse_ctree_children(self, parent):
        model = self.ui.tree_helper.model()
        for row in range(model.rowCount(parent)):
            child = model.index(row, 0, parent)
            self.ui.tree_helper.collapse(child)
            self._collapse_ctree_children(child)

    def _show_ctree_context_menu(self, pos):
        menu = QMenu(self.ui.tree_helper)
        expand_branch = QAction("Expand branch", menu)
        collapse_branch = QAction("Collapse branch", menu)
        expand_all = QAction("Expand all", menu)
        collapse_all = QAction("Collapse all", menu)

        expand_branch.triggered.connect(self._expand_ctree_branch)
        collapse_branch.triggered.connect(self._collapse_ctree_branch)
        expand_all.triggered.connect(self.ui.tree_helper.expandAll)
        collapse_all.triggered.connect(self.ui.tree_helper.collapseAll)

        menu.addAction(expand_branch)
        menu.addAction(collapse_branch)
        menu.addSeparator()
        menu.addAction(expand_all)
        menu.addAction(collapse_all)
        menu.exec(self.ui.tree_helper.viewport().mapToGlobal(pos))
            

def expand_and_show_matches(view, needle: str, partial=True, case_insensitive=True):
    m = view.model()
    if m is None or needle is None:
        return

    text = needle.strip()
    if not text:
        view.collapseAll()
        return

    needle_cmp = text.lower() if case_insensitive else text

    matches = []  # outer scope list; append is OK (no nonlocal needed)

    def is_match(s: str) -> bool:
        if s is None:
            return False
        a = s.lower() if case_insensitive else s
        return (needle_cmp in a) if partial else (a == needle_cmp)

    def walk(parent: QModelIndex):
        for r in range(m.rowCount(parent)):
            idx = m.index(r, 0, parent)          # column 0
            if is_match(m.data(idx)):
                p = idx
                while p.isValid():
                    view.expand(p)
                    p = p.parent()
                matches.append(idx)
            walk(idx)

    view.setUpdatesEnabled(False)
    view.collapseAll()
    walk(QModelIndex())
    view.setUpdatesEnabled(True)

    sm = view.selectionModel()
    if not matches or sm is None:
        return

    # allow multi-select if you want all matches highlighted
    view.setSelectionMode(QAbstractItemView.ExtendedSelection)

    sel = QItemSelection()
    for idx in matches:
        sel.merge(QItemSelection(idx, idx), QItemSelectionModel.Select)
    sm.select(sel, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
    view.scrollTo(matches[0], QAbstractItemView.PositionAtCenter)

    
