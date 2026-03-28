
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

from PySide6.QtWidgets import QFileDialog, QMessageBox


class ResultsIOMixin:

    def _init_results_io(self):
        self.ui.butt_out_save.clicked.connect(self._save_results)
        self.ui.butt_out_load.clicked.connect(self._load_results)
        self.ui.butt_out_clear.clicked.connect(self._clear_results)

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
            cols = "\t".join(df.columns.tolist()) if df is not None else ""
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
                    df.to_csv(buf, sep="\t", index=False)
                    zf.writestr(f"{folder}/{key}.tsv", buf.getvalue())

    # ------------------------------------------------------------------
    # Load

    def _load_results(self):
        filename, _ = QFileDialog.getOpenFileName(
            self.ui,
            "Load Results",
            "",
            "Results Files (*.pkl *.zip);;Pickle (*.pkl);;Zip of TSVs (*.zip);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not filename:
            return

        lower = filename.lower()
        try:
            if lower.endswith(".pkl"):
                results, pairs = self._load_results_pkl(filename)
            elif lower.endswith(".zip"):
                results, pairs = self._load_results_zip(filename)
            else:
                QMessageBox.critical(
                    self.ui,
                    "Load error",
                    "Unrecognised file format. Expected .pkl or .zip.",
                )
                return
        except Exception as e:
            QMessageBox.critical(self.ui, "Load error", f"Could not load results:\n{e}")
            return

        self.results = results
        tree_df = pd.DataFrame(pairs, columns=["Command", "Strata"])
        self.set_tree_from_df(tree_df)
        self.ui.dock_outputs.show()

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
        from pathlib import Path
        folder = Path(path).stem  # expected subfolder name

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

            results = {}
            pairs = []
            for _, row in manifest.iterrows():
                key = row["key"]
                # accept files in subfolder (new format) or flat (old format)
                fname_sub = f"{folder}/{key}.tsv"
                fname_flat = f"{key}.tsv"
                if fname_sub in names:
                    fname = fname_sub
                elif fname_flat in names:
                    fname = fname_flat
                else:
                    raise ValueError(
                        f"Not a valid results zip: missing data file '{fname_sub}'."
                    )
                results[key] = pd.read_csv(io.BytesIO(zf.read(fname)), sep="\t")
                pairs.append((str(row["command"]), str(row["strata"])))

        return results, pairs

    # ------------------------------------------------------------------
    # Clear

    def _clear_results(self):
        from PySide6.QtGui import QStandardItemModel
        self.results = {}
        self.set_tree_from_df(None)
        self.ui.anal_table.setModel(QStandardItemModel(self))
