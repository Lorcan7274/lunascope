
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

import io
import os
import pickle
import zipfile

import pandas as pd

from PySide6.QtCore import Qt, QEvent, QObject, QTimer
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QFileDialog, QMessageBox, QHeaderView, QToolTip
from ..file_dialogs import open_file_name, save_file_name

_OUTPUT_DOCK_DRAG_CELL_LIMIT = 25_000


class _OutputsDockDragFilter(QObject):
    """Temporarily lighten large output tables while Qt reparents the dock."""

    def __init__(self, owner):
        super().__init__(owner.ui.dock_outputs)
        self._owner = owner

    def eventFilter(self, obj, event):
        etype = event.type()
        if etype in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.NonClientAreaMouseButtonPress,
        ):
            self._owner._suspend_outputs_table_for_dock_drag()
        elif etype in (
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.NonClientAreaMouseButtonRelease,
            QEvent.Type.Move,
        ):
            self._owner._schedule_outputs_table_restore()
        return False


class HelpHeaderView(QHeaderView):
    """Horizontal header for the output table.
    Hovering a column shows the Luna variable description as a tooltip.
    Results are cached per (cmd, strata, var) so Luna is only queried once.
    """

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.setSectionsClickable(True)   # required for sort-by-column to work
        self._help_cmd = None
        self._help_strata = None
        self._cache: dict[tuple, str] = {}

    def set_help_context(self, cmd: str, strata: str) -> None:
        self._help_cmd = cmd
        self._help_strata = strata

    def event(self, e):
        if e.type() == QEvent.Type.ToolTip and self._help_cmd:
            section = self.logicalIndexAt(e.pos())
            if section >= 0:
                tip = self._tip_for_section(section)
                if tip:
                    QToolTip.showText(e.globalPos(), tip, self)
                    e.accept()
                    return True
        return super().event(e)

    def _tip_for_section(self, section: int) -> str:
        model = self.model()
        if model is None:
            return ""
        var = model.headerData(section, Qt.Horizontal, Qt.DisplayRole)
        if not var:
            return ""
        var = str(var)
        key = (self._help_cmd, self._help_strata, var)
        if key in self._cache:
            return self._cache[key]
        tip = self._lookup(self._help_cmd, self._help_strata, var)
        self._cache[key] = tip
        return tip

    @staticmethod
    def _lookup(cmd: str, strata: str, var: str) -> str:
        try:
            import lunapi as lp
        except ImportError:
            return ""
        # 1. Direct match using strata as the table key
        try:
            desc = lp.fetch_desc_var(cmd, strata, var)
            if desc:
                return f"{cmd} / {var}\n{desc}"
        except Exception:
            pass
        # 2. Scan all tables for this command (strata format may differ)
        try:
            for tbl in (lp.fetch_tbls(cmd) or []):
                try:
                    desc = lp.fetch_desc_var(cmd, tbl, var)
                    if desc:
                        return f"{cmd} / {var}\n{desc}"
                except Exception:
                    pass
        except Exception:
            pass
        return ""


class ResultsIOMixin:

    def _init_results_io(self):
        self._loaded_tsv_csv = False
        self.ui.butt_out_save.clicked.connect(self._save_results)
        self.ui.butt_out_load.clicked.connect(self._load_results)
        self.ui.butt_out_clear.clicked.connect(self._clear_results)
        # install help-aware header on the output table (done once at init)
        self._help_header = HelpHeaderView(self.ui.anal_table)
        self.ui.anal_table.setHorizontalHeader(self._help_header)
        self._outputs_drag_saved_model = None
        self._outputs_drag_saved_sorting = False
        self._outputs_drag_restore_timer = QTimer(self.ui.dock_outputs)
        self._outputs_drag_restore_timer.setSingleShot(True)
        self._outputs_drag_restore_timer.setInterval(250)
        self._outputs_drag_restore_timer.timeout.connect(self._restore_outputs_table_after_dock_drag)
        self._outputs_dock_drag_filter = _OutputsDockDragFilter(self)
        self.ui.dock_outputs.installEventFilter(self._outputs_dock_drag_filter)
        self.ui.dock_outputs.topLevelChanged.connect(
            lambda _floating: self._schedule_outputs_table_restore()
        )
        try:
            self.ui.dock_outputs.dockLocationChanged.connect(
                lambda _area: self._schedule_outputs_table_restore()
            )
        except AttributeError:
            pass

    def _update_table(self, cmd, stratum):
        super()._update_table(cmd, stratum)
        self._help_header.set_help_context(cmd, stratum)

    def _outputs_table_cell_count(self) -> int:
        model = self.ui.anal_table.model()
        if model is None:
            return 0
        try:
            source = model.sourceModel()
        except AttributeError:
            source = model
        try:
            return int(source.rowCount()) * int(source.columnCount())
        except RuntimeError:
            return 0

    def _suspend_outputs_table_for_dock_drag(self):
        if self._outputs_drag_saved_model is not None:
            return
        if self._outputs_table_cell_count() <= _OUTPUT_DOCK_DRAG_CELL_LIMIT:
            return
        table = self.ui.anal_table
        self._outputs_drag_restore_timer.stop()
        self._outputs_drag_saved_model = table.model()
        self._outputs_drag_saved_sorting = table.isSortingEnabled()
        table.setSortingEnabled(False)
        table.setModel(QStandardItemModel(0, 0, table))

    def _schedule_outputs_table_restore(self):
        if self._outputs_drag_saved_model is not None:
            self._outputs_drag_restore_timer.start()

    def _restore_outputs_table_after_dock_drag(self):
        model = self._outputs_drag_saved_model
        if model is None:
            return
        self._outputs_drag_saved_model = None
        table = self.ui.anal_table
        table.setModel(model)
        table.setSortingEnabled(bool(self._outputs_drag_saved_sorting))

    def _render_tables(self, tbls):
        self._set_tsv_csv_mode(False)
        super()._render_tables(tbls)

    def _set_tsv_csv_mode(self, enabled: bool):
        self._loaded_tsv_csv = enabled
        self.ui.butt_out_save.setEnabled(not enabled)

    # ------------------------------------------------------------------
    # Save

    def _save_results(self):
        if not getattr(self, "results", None):
            QMessageBox.information(self.ui, "Nothing to save", "No results to save.")
            return

        filename, selected_filter = save_file_name(
            self.ui,
            "Save Results",
            "",
            "Pickle (*.pkl);;Zip of TSVs (*.zip);;All Files (*)",
        )
        if not filename:
            return

        lower = filename.lower()
        if not (lower.endswith(".pkl") or lower.endswith(".zip")):
            if "pkl" in selected_filter.lower():
                filename += ".pkl"
            elif "zip" in selected_filter.lower():
                filename += ".zip"
            else:
                filename += ".pkl"

        pairs = self._tree_pairs()

        try:
            if filename.lower().endswith(".pkl"):
                self._save_results_pkl(filename, pairs)
            else:
                self._save_results_zip(filename, pairs)
        except Exception as e:
            QMessageBox.critical(self.ui, "Save error", f"Could not save results:\n{e}")

    def _tree_pairs(self):
        """Return list of (command, strata) from the current tree model."""
        pairs = []
        m = self._anal_model
        for row in range(m.rowCount()):
            cmd = m.item(row, 0).text()
            strata_display = m.item(row, 1).text()
            # tree stores strata as "A, B, C"; key uses "A_B_C"
            strata = strata_display.replace(", ", "_")
            pairs.append((cmd, strata))
        return pairs

    def _save_results_pkl(self, path, pairs):
        payload = {"results": self.results, "tree": pairs}
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def _save_results_zip(self, path, pairs):
        from pathlib import Path
        folder = Path(path).stem  # subfolder name = zip stem, e.g. "t1"

        manifest_rows = []
        for cmd, strata in pairs:
            key = f"{cmd}_{strata}"
            df = self.results.get(key)
            cols = " | ".join(df.columns.tolist()) if df is not None else ""
            manifest_rows.append({"key": key, "command": cmd, "strata": strata, "columns": cols})
        manifest_df = pd.DataFrame(manifest_rows)

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            buf = io.StringIO()
            manifest_df.to_csv(buf, sep="\t", index=False)
            zf.writestr("_manifest.tsv", buf.getvalue())

            for cmd, strata in pairs:
                key = f"{cmd}_{strata}"
                df = self.results.get(key)
                if df is not None:
                    buf = io.StringIO()
                    df.to_csv(buf, sep="\t", index=False, na_rep="NA")
                    zf.writestr(f"{folder}/{key}.tsv", buf.getvalue())

    # ------------------------------------------------------------------
    # Load

    def _load_results(self):
        filename, _ = open_file_name(
            self.ui,
            "Load Results",
            "",
            "Results Files (*.pkl *.zip *.db *.tsv *.csv);;TSV/CSV (*.tsv *.csv);;Pickle (*.pkl);;Zip of TSVs (*.zip);;Luna DB (*.db);;All Files (*)",
        )
        if not filename:
            return

        lower = filename.lower()
        if lower.endswith(".tsv") or lower.endswith(".csv"):
            self._load_results_tsv_csv(filename)
            return

        project_mode = False
        try:
            if lower.endswith(".pkl"):
                results, pairs = self._load_results_pkl(filename)
            elif lower.endswith(".zip"):
                results, pairs = self._load_results_zip(filename)
            elif lower.endswith(".db"):
                results, pairs = self._load_results_db(filename)
                project_mode = True
            else:
                QMessageBox.critical(
                    self.ui,
                    "Load error",
                    "Unrecognised file format. Expected .pkl, .zip, .db, .tsv, or .csv.",
                )
                return
        except Exception as e:
            QMessageBox.critical(self.ui, "Load error", f"Could not load results:\n{e}")
            return

        self._set_tsv_csv_mode(False)
        self.project_mode = project_mode
        self.results = dict(results)
        tree_df = pd.DataFrame(pairs, columns=["Command", "Strata"])
        self.set_tree_from_df(tree_df)
        self.ui.dock_outputs.show()
        self.sig_results_changed.emit()

    def _load_results_tsv_csv(self, filename: str):
        lower = filename.lower()
        sep = "\t" if lower.endswith(".tsv") else ","
        file_type = "TSV" if lower.endswith(".tsv") else "CSV"
        try:
            df = pd.read_csv(filename, sep=sep, encoding="utf-8-sig")
        except Exception as e:
            QMessageBox.critical(self.ui, "Load error", f"Could not load file:\n{e}")
            return

        basename = os.path.basename(filename)
        key = f"{basename}_{file_type}"
        self.results = {key: df}
        self._project_results_mode = True  # suppresses ID-column drop in _update_table
        self.project_mode = False
        tree_df = pd.DataFrame([{"Command": basename, "Strata": file_type}])
        self.set_tree_from_df(tree_df)
        self.ui.dock_outputs.show()
        self._set_tsv_csv_mode(True)
        self.sig_results_changed.emit()

        # auto-select the single row so the table appears immediately
        tv = self.ui.anal_tables
        m = tv.model()
        if m and m.rowCount() > 0:
            idx = m.index(0, 0)
            tv.setCurrentIndex(idx)
            sm = tv.selectionModel()
            if sm:
                from PySide6.QtCore import QItemSelectionModel
                sm.select(idx, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)

    def _load_results_db(self, path):
        self.proj.import_db(path)
        tbls = self.proj.strata()
        if tbls is None or getattr(tbls, "empty", True):
            raise ValueError("Database contains no results.")
        results = {}
        pairs = []
        for row in tbls.itertuples(index=False):
            key = f"{row.Command}_{row.Strata}"
            results[key] = self.proj.table(row.Command, row.Strata)
            pairs.append((row.Command, row.Strata))
        return results, pairs

    def _load_results_pkl(self, path):
        with open(path, "rb") as f:
            payload = pickle.load(f)

        if not isinstance(payload, dict):
            raise ValueError("Not a valid results file: expected a dict at top level.")
        for key in ("results", "tree"):
            if key not in payload:
                raise ValueError(f"Not a valid results file: missing '{key}' key.")

        results = payload["results"]
        tree = payload["tree"]

        if not isinstance(results, dict):
            raise ValueError("Not a valid results file: 'results' must be a dict.")
        for k, v in results.items():
            if not isinstance(k, str) or not isinstance(v, pd.DataFrame):
                raise ValueError(
                    f"Not a valid results file: entry {k!r} is not a str→DataFrame mapping."
                )

        if not isinstance(tree, list) or not all(
            isinstance(p, (tuple, list)) and len(p) == 2 for p in tree
        ):
            raise ValueError(
                "Not a valid results file: 'tree' must be a list of (command, strata) pairs."
            )

        return results, [tuple(p) for p in tree]

    def _load_results_zip(self, path):
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())

            if "_manifest.tsv" not in names:
                raise ValueError("Not a valid results zip: missing '_manifest.tsv'.")

            manifest = pd.read_csv(
                io.TextIOWrapper(io.BytesIO(zf.read("_manifest.tsv")), encoding="utf-8-sig"),
                sep="\t",
            )
            required = {"key", "command", "strata"}
            missing = required - set(manifest.columns)
            if missing:
                raise ValueError(
                    f"Not a valid results zip: manifest missing columns: {missing}."
                )

            # build basename -> full zip path, regardless of subfolder name
            tsv_index = {}
            for n in names:
                if n.endswith(".tsv") and n != "_manifest.tsv":
                    basename = n.rsplit("/", 1)[-1]  # works for both flat and subfoldered
                    tsv_index[basename] = n

            results = {}
            pairs = []
            for _, row in manifest.iterrows():
                key = row["key"]
                fname_base = f"{key}.tsv"
                if fname_base not in tsv_index:
                    raise ValueError(
                        f"Not a valid results zip: missing data file '{fname_base}'."
                    )
                results[key] = pd.read_csv(
                    io.TextIOWrapper(io.BytesIO(zf.read(tsv_index[fname_base])), encoding="utf-8-sig"),
                    sep="\t",
                )
                pairs.append((str(row["command"]), str(row["strata"])))

        return results, pairs

    # ------------------------------------------------------------------
    # Clear

    def _clear_results(self):
        from PySide6.QtGui import QStandardItemModel
        self.proj.reinit()
        self.results = {}
        self.project_mode = False
        self.set_tree_from_df(None)
        self.ui.anal_table.setModel(QStandardItemModel(self))
        self._set_tsv_csv_mode(False)
