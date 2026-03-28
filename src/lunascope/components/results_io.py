
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
import pickle
import zipfile

import pandas as pd

from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import QFileDialog, QMessageBox, QHeaderView, QToolTip


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
        self.ui.butt_out_save.clicked.connect(self._save_results)
        self.ui.butt_out_load.clicked.connect(self._load_results)
        self.ui.butt_out_clear.clicked.connect(self._clear_results)
        # install help-aware header on the output table (done once at init)
        self._help_header = HelpHeaderView(self.ui.anal_table)
        self.ui.anal_table.setHorizontalHeader(self._help_header)

    def _update_table(self, cmd, stratum):
        """Wraps AnalMixin._update_table to update the help header context."""
        super()._update_table(cmd, stratum)
        self._help_header.set_help_context(cmd, stratum)

    # ------------------------------------------------------------------
    # Save

    def _save_results(self):
        if not getattr(self, "results", None):
            QMessageBox.information(self.ui, "Nothing to save", "No results to save.")
            return

        filename, selected_filter = QFileDialog.getSaveFileName(
            self.ui,
            "Save Results",
            "",
            "Pickle (*.pkl);;Zip of TSVs (*.zip);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
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
        filename, _ = QFileDialog.getOpenFileName(
            self.ui,
            "Load Results",
            "",
            "Results Files (*.pkl *.zip *.db);;Pickle (*.pkl);;Zip of TSVs (*.zip);;Luna DB (*.db);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not filename:
            return

        lower = filename.lower()
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
                    "Unrecognised file format. Expected .pkl, .zip, or .db.",
                )
                return
        except Exception as e:
            QMessageBox.critical(self.ui, "Load error", f"Could not load results:\n{e}")
            return

        self.project_mode = project_mode
        self.results = {
            k: df.sort_values("ID").reset_index(drop=True) if "ID" in df.columns else df
            for k, df in results.items()
        }
        tree_df = pd.DataFrame(pairs, columns=["Command", "Strata"])
        self.set_tree_from_df(tree_df)
        self.ui.dock_outputs.show()

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

            manifest = pd.read_csv(io.BytesIO(zf.read("_manifest.tsv")), sep="\t")
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
                results[key] = pd.read_csv(io.BytesIO(zf.read(tsv_index[fname_base])), sep="\t")
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
