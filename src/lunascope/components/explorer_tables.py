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

#  --------------------------------------------------------------------
#  Luna / Lunascope  - Explorer: summary Tables tab
#  --------------------------------------------------------------------

"""Summary tables for output tables in the Explorer dock."""

import pandas as pd

from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QStackedWidget, QTableView,
    QTextBrowser, QVBoxLayout, QWidget,
)

from .explorer_plotter import (
    _MAX_FILTER_LEVELS, _display_level, _numeric_sort, _table_display_name,
    PlotterTab,
)
from ..helpers import current_font_scale
from .slist import DataFrameModel, NumericSortFilterProxy
from .soappops import MultiSelectComboBox
from .tbl_funcs import copy_selection, save_table_as_tsv


def _coerce_measure_series(series):
    """Coerce a measure column, treating nonnumeric entries as missing."""
    if series.dtype == object:
        series = series.map(
            lambda v: v.replace(",", "").strip() if isinstance(v, str) else v
        )
    return pd.to_numeric(series, errors="coerce")


def _summary_stat_rows(df, measure_cols, group_cols=None, stats=None):
    """Return long-form descriptive summaries for selected numeric measures."""
    if df is None or df.empty:
        return pd.DataFrame()
    measure_cols = [c for c in (measure_cols or []) if c in df.columns]
    group_cols = [c for c in (group_cols or []) if c and c in df.columns]
    stats = list(stats or ("N", "Mean", "SD"))
    if not measure_cols:
        return pd.DataFrame()

    use_cols = list(dict.fromkeys(group_cols + measure_cols))
    work = df[use_cols].copy()
    for col in measure_cols:
        work[col] = _coerce_measure_series(work[col])

    groups = (
        [((), work)]
        if not group_cols
        else work.groupby(group_cols, dropna=False, sort=False)
    )
    rows = []
    for keys, sub in groups:
        if group_cols and not isinstance(keys, tuple):
            keys = (keys,)
        base = {}
        for col, value in zip(group_cols, keys):
            base[col] = "(missing)" if pd.isna(value) else value
        for measure in measure_cols:
            vals = sub[measure].dropna()
            row = dict(base)
            row["Measure"] = measure
            if "N" in stats:
                row["N"] = int(vals.count())
            if "Mean" in stats:
                row["Mean"] = float(vals.mean()) if not vals.empty else pd.NA
            if "SD" in stats:
                row["SD"] = float(vals.std(ddof=1)) if vals.count() > 1 else pd.NA
            if "Median" in stats:
                row["Median"] = float(vals.median()) if not vals.empty else pd.NA
            if "Min" in stats:
                row["Min"] = float(vals.min()) if not vals.empty else pd.NA
            if "Max" in stats:
                row["Max"] = float(vals.max()) if not vals.empty else pd.NA
            rows.append(row)

    out_cols = group_cols + ["Measure"] + stats
    out = pd.DataFrame(rows, columns=out_cols)
    return out.reset_index(drop=True)


def _format_publication_value(value, stat=None):
    """Format one table cell for publication-style display."""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if stat == "N":
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _stat_display_cell(row, stats):
    """Combine common statistics into journal-style cell text."""
    bits = []
    if "N" in stats:
        n = _format_publication_value(row.get("N"), "N")
        if n:
            bits.append(f"n={n}")
    if "Mean" in stats and "SD" in stats:
        mean = _format_publication_value(row.get("Mean"), "Mean")
        sd = _format_publication_value(row.get("SD"), "SD")
        if mean and sd:
            bits.append(f"{mean} ({sd})")
        elif mean:
            bits.append(mean)
    elif "Mean" in stats:
        mean = _format_publication_value(row.get("Mean"), "Mean")
        if mean:
            bits.append(mean)
    elif "SD" in stats:
        sd = _format_publication_value(row.get("SD"), "SD")
        if sd:
            bits.append(f"SD {sd}")
    if "Median" in stats:
        med = _format_publication_value(row.get("Median"), "Median")
        if med:
            bits.append(f"median {med}")
    if "Min" in stats and "Max" in stats:
        mn = _format_publication_value(row.get("Min"), "Min")
        mx = _format_publication_value(row.get("Max"), "Max")
        if mn and mx:
            bits.append(f"{mn}-{mx}")
    elif "Min" in stats:
        mn = _format_publication_value(row.get("Min"), "Min")
        if mn:
            bits.append(f"min {mn}")
    elif "Max" in stats:
        mx = _format_publication_value(row.get("Max"), "Max")
        if mx:
            bits.append(f"max {mx}")
    return "; ".join(bits)


def _html_escape(value):
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _levels_for_column(df, col):
    if col not in df.columns:
        return []
    values = [v for v in pd.unique(df[col]) if not pd.isna(v)]
    return _numeric_sort(values)


def _matching_summary_row(df, group_cols, group_values, measure):
    mask = df["Measure"].eq(measure)
    for col, value in zip(group_cols, group_values):
        mask &= df[col].eq(value)
    rows = df[mask]
    return rows.iloc[0] if not rows.empty else None


def _publication_table_plan(summary_df, group_cols=None, stats=None):
    """Build adaptive section/table descriptors for article-style rendering."""
    if summary_df is None or summary_df.empty:
        return []
    group_cols = [c for c in (group_cols or []) if c in summary_df.columns]
    stats = [s for s in (stats or []) if s in summary_df.columns]
    if not stats:
        return []

    measures = list(summary_df["Measure"].drop_duplicates())
    sections = []
    if not group_cols:
        columns = ["Measure"] + stats
        rows = []
        for measure in measures:
            row = summary_df[summary_df["Measure"].eq(measure)].iloc[0]
            rows.append(
                [measure]
                + [_format_publication_value(row.get(stat), stat) for stat in stats]
            )
        return [{"title": "", "columns": columns, "rows": rows, "layout": "stats"}]

    if len(group_cols) == 1:
        group = group_cols[0]
        levels = _levels_for_column(summary_df, group)
        measure_stat_cols = len(measures) * max(len(stats), 1)
        if 1 < len(measures) <= 4 and measure_stat_cols <= 10 and stats:
            columns = [group] + [
                f"{measure} {stat}"
                for measure in measures
                for stat in stats
            ]
            header_rows = [
                [
                    {"text": group, "rowspan": 2},
                    *[
                        {"text": measure, "colspan": len(stats)}
                        for measure in measures
                    ],
                ],
                [{"text": stat} for _measure in measures for stat in stats],
            ]
            rows = []
            for level in levels:
                out = [_display_level(level)]
                for measure in measures:
                    row = _matching_summary_row(summary_df, [group], [level], measure)
                    out.extend(
                        [_format_publication_value(row.get(stat), stat) for stat in stats]
                        if row is not None else [""] * len(stats)
                    )
                rows.append(out)
            return [
                {
                    "title": group,
                    "columns": columns,
                    "header_rows": header_rows,
                    "rows": rows,
                    "layout": "measure_columns",
                }
            ]

        if len(levels) <= 8 and len(measures) * max(len(levels), 1) <= 80:
            columns = ["Measure"] + [_display_level(level) for level in levels]
            rows = []
            for measure in measures:
                out = [measure]
                for level in levels:
                    row = _matching_summary_row(summary_df, [group], [level], measure)
                    out.append(_stat_display_cell(row, stats) if row is not None else "")
                rows.append(out)
            return [{"title": group, "columns": columns, "rows": rows, "layout": "wide"}]

        columns = [group, "Measure"] + stats
        rows = []
        for level in levels:
            for measure in measures:
                row = _matching_summary_row(summary_df, [group], [level], measure)
                if row is None:
                    continue
                rows.append(
                    [_display_level(level), measure]
                    + [_format_publication_value(row.get(stat), stat) for stat in stats]
                )
        return [{"title": group, "columns": columns, "rows": rows, "layout": "long"}]

    group1, group2 = group_cols[:2]
    levels1 = _levels_for_column(summary_df, group1)
    levels2 = _levels_for_column(summary_df, group2)
    wide_enough = len(levels2) <= 6 and len(measures) * max(len(levels2), 1) <= 90
    if wide_enough:
        for level1 in levels1:
            columns = ["Measure"] + [_display_level(level2) for level2 in levels2]
            rows = []
            for measure in measures:
                out = [measure]
                for level2 in levels2:
                    row = _matching_summary_row(
                        summary_df, [group1, group2], [level1, level2], measure
                    )
                    out.append(_stat_display_cell(row, stats) if row is not None else "")
                rows.append(out)
            sections.append(
                {
                    "title": f"{group1}: {_display_level(level1)}",
                    "columns": columns,
                    "rows": rows,
                    "layout": "sectioned",
                }
            )
        return sections

    columns = [group1, group2, "Measure"] + stats
    rows = []
    for level1 in levels1:
        for level2 in levels2:
            for measure in measures:
                row = _matching_summary_row(
                    summary_df, [group1, group2], [level1, level2], measure
                )
                if row is None:
                    continue
                rows.append(
                    [_display_level(level1), _display_level(level2), measure]
                    + [_format_publication_value(row.get(stat), stat) for stat in stats]
                )
    return [
        {
            "title": f"{group1} x {group2}",
            "columns": columns,
            "rows": rows,
            "layout": "long",
        }
    ]


def _publication_table_html(summary_df, group_cols=None, stats=None, title="", font_scale=None):
    """Render summary rows as readable article-style HTML."""
    sections = _publication_table_plan(summary_df, group_cols, stats)
    if not sections:
        return "<html><body><p class='empty'>No summary rows.</p></body></html>"

    scale = current_font_scale() if font_scale is None else font_scale
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        scale = 1.0
    scale = max(0.8, min(1.4, scale))
    h2_px = 16 * scale
    h3_px = 13 * scale
    cell_px = 12 * scale
    note_px = 11 * scale
    style = """
    body { color:#e8e8e8; background:#14171b; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin:18px; }
    h2 { font-size:%(h2_px).1fpx; font-weight:650; margin:0 0 12px 0; color:#ffffff; }
    h3 { font-size:%(h3_px).1fpx; font-weight:650; margin:18px 0 7px 0; color:#dfe7ee; }
    table { border-collapse:collapse; width:100%%; margin:0 0 16px 0; table-layout:auto; }
    thead th { border-top:2px solid #d8dde3; border-bottom:1px solid #8d98a3; color:#ffffff; font-weight:650; }
    tbody tr:last-child td { border-bottom:2px solid #d8dde3; }
    th, td { padding:6px 9px; vertical-align:top; font-size:%(cell_px).1fpx; line-height:1.35; }
    td { border-bottom:1px solid #313941; }
    th:first-child, td:first-child { text-align:left; }
    th:not(:first-child), td:not(:first-child) { text-align:right; }
    tbody tr:nth-child(even) td { background:#191d22; }
    .note { color:#aab3bb; font-size:%(note_px).1fpx; margin:2px 0 14px 0; }
    .empty { color:#aab3bb; font-size:%(cell_px).1fpx; }
    """ % {
        "h2_px": h2_px,
        "h3_px": h3_px,
        "cell_px": cell_px,
        "note_px": note_px,
    }
    parts = ["<html><head><style>", style, "</style></head><body>"]
    if title:
        parts.append(f"<h2>{_html_escape(title)}</h2>")
    if stats:
        parts.append(
            "<p class='note'>Values shown as "
            f"{_html_escape(', '.join(stats))}; Mean and SD are combined as "
            "mean (SD) where both are selected.</p>"
        )
    for section in sections:
        if section["title"]:
            parts.append(f"<h3>{_html_escape(section['title'])}</h3>")
        parts.append("<table><thead>")
        header_rows = section.get("header_rows")
        if header_rows:
            for header_row in header_rows:
                parts.append("<tr>")
                for cell in header_row:
                    text = cell.get("text", "") if isinstance(cell, dict) else cell
                    attrs = []
                    if isinstance(cell, dict):
                        colspan = int(cell.get("colspan", 1) or 1)
                        rowspan = int(cell.get("rowspan", 1) or 1)
                        if colspan > 1:
                            attrs.append(f"colspan='{colspan}'")
                        if rowspan > 1:
                            attrs.append(f"rowspan='{rowspan}'")
                    attr_text = "" if not attrs else " " + " ".join(attrs)
                    parts.append(f"<th{attr_text}>{_html_escape(text)}</th>")
                parts.append("</tr>")
        else:
            parts.append("<tr>")
            for col in section["columns"]:
                parts.append(f"<th>{_html_escape(col)}</th>")
            parts.append("</tr>")
        parts.append("</thead><tbody>")
        for row in section["rows"]:
            parts.append("<tr>")
            for cell in row:
                parts.append(f"<td>{_html_escape(cell)}</td>")
            parts.append("</tr>")
        parts.append("</tbody></table>")
    parts.append("</body></html>")
    return "".join(parts)


class TablesTab(PlotterTab):
    """Output summary table tab reusing Plotter's data/covariate/filter logic."""

    def _build_widget(self):
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(4)

        # ---- row 1: table selector ------------------------------------
        row1 = QWidget(); rl1 = QHBoxLayout(row1)
        rl1.setContentsMargins(0, 0, 0, 0); rl1.setSpacing(6)

        btn_refresh = QPushButton("↻"); btn_refresh.setFixedWidth(30)
        btn_refresh.setToolTip("Reload available tables from the Outputs dock")
        combo_table = QComboBox()
        combo_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lbl_shape = QLabel("")
        lbl_shape.setStyleSheet("color:#888; font-size:11px;")

        rl1.addWidget(QLabel("Table:")); rl1.addWidget(combo_table, 1)
        rl1.addWidget(lbl_shape); rl1.addWidget(btn_refresh)

        # ---- row 1b: covariate file -----------------------------------
        row1b = QWidget(); rl1b = QHBoxLayout(row1b)
        rl1b.setContentsMargins(0, 0, 0, 0); rl1b.setSpacing(6)

        btn_load_cov = QPushButton("Load covariates…"); btn_load_cov.setFixedWidth(140)
        btn_load_cov.setToolTip("Upload a TSV/CSV file with an ID column to merge as covariates")
        btn_clear_cov = QPushButton("✕"); btn_clear_cov.setFixedWidth(26)
        btn_clear_cov.setToolTip("Remove loaded covariate file")
        lbl_cov_file = QLabel("(none)")
        lbl_cov_file.setStyleSheet("color:#888; font-size:11px;")
        lbl_cov_file.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        rl1b.addWidget(QLabel("Covariates:")); rl1b.addWidget(btn_load_cov)
        rl1b.addWidget(lbl_cov_file, 1); rl1b.addWidget(btn_clear_cov)

        # ---- row 1c: optional row filters ------------------------------
        row1c = QWidget(); rl1c = QHBoxLayout(row1c)
        rl1c.setContentsMargins(0, 0, 0, 0); rl1c.setSpacing(6)

        btn_add_filter = QPushButton("+ Filter")
        btn_add_filter.setFixedWidth(80)
        btn_add_filter.setToolTip("Subset rows before summarising")
        btn_clear_filters = QPushButton("Clear")
        btn_clear_filters.setFixedWidth(60)
        btn_clear_filters.setToolTip("Remove all row filters")
        lbl_filters_hint = QLabel("")
        lbl_filters_hint.setStyleSheet("color:#888; font-size:11px;")
        lbl_filters_hint.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        rl1c.addWidget(QLabel("Subset rows:"))
        rl1c.addWidget(btn_add_filter)
        rl1c.addWidget(btn_clear_filters)
        rl1c.addWidget(lbl_filters_hint, 1)

        filter_host = QWidget()
        filter_host.setLayout(QVBoxLayout())
        filter_host.layout().setContentsMargins(0, 0, 0, 0)
        filter_host.layout().setSpacing(4)

        # ---- row 2: measures and groups --------------------------------
        row2 = QWidget(); rl2 = QHBoxLayout(row2)
        rl2.setContentsMargins(0, 0, 0, 0); rl2.setSpacing(6)

        combo_measures = MultiSelectComboBox()
        combo_measures.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        combo_measures.lineEdit().setPlaceholderText("Select numeric measures")

        combo_group1 = QComboBox(); combo_group1.setMinimumWidth(120)
        combo_group2 = QComboBox(); combo_group2.setMinimumWidth(120)
        for combo in (combo_group1, combo_group2):
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            combo.addItem("(none)", None)

        rl2.addWidget(QLabel("Measures:")); rl2.addWidget(combo_measures, 2)
        rl2.addWidget(QLabel("Group 1:")); rl2.addWidget(combo_group1, 1)
        rl2.addWidget(QLabel("Group 2:")); rl2.addWidget(combo_group2, 1)

        # ---- row 3: stats and export -----------------------------------
        row3 = QWidget(); rl3 = QHBoxLayout(row3)
        rl3.setContentsMargins(0, 0, 0, 0); rl3.setSpacing(6)

        chk_n = QCheckBox("N"); chk_n.setChecked(True)
        chk_mean = QCheckBox("Mean"); chk_mean.setChecked(True)
        chk_sd = QCheckBox("SD"); chk_sd.setChecked(True)
        chk_median = QCheckBox("Median")
        chk_min = QCheckBox("Min")
        chk_max = QCheckBox("Max")
        combo_view_mode = QComboBox()
        combo_view_mode.setFixedWidth(86)
        combo_view_mode.addItem("Grid", "grid")
        combo_view_mode.addItem("Table", "article")
        edit_search = QLineEdit()
        edit_search.setPlaceholderText("filter table…")
        edit_search.setClearButtonEnabled(True)
        edit_search.setFixedWidth(160)
        btn_export = QPushButton("Export…"); btn_export.setFixedWidth(80)
        lbl_status = QLabel("")
        lbl_status.setStyleSheet("color:#888; font-size:11px;")

        rl3.addWidget(QLabel("Stats:"))
        for chk in (chk_n, chk_mean, chk_sd, chk_median, chk_min, chk_max):
            rl3.addWidget(chk)
        rl3.addWidget(lbl_status, 1)
        rl3.addWidget(QLabel("View:"))
        rl3.addWidget(combo_view_mode)
        rl3.addWidget(edit_search)
        rl3.addWidget(btn_export)

        # ---- table view ------------------------------------------------
        view = QTableView()
        view.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        view.setSortingEnabled(True)
        view.horizontalHeader().setStretchLastSection(True)
        view.verticalHeader().setVisible(False)
        view.setAlternatingRowColors(True)

        article = QTextBrowser()
        article.setOpenExternalLinks(False)
        article.setReadOnly(True)
        article.setStyleSheet(
            "QTextBrowser { background:#14171b; border:1px solid #343a40; }"
        )

        stack = QStackedWidget()
        stack.addWidget(view)
        stack.addWidget(article)

        copy_action = QAction("Copy", view)
        copy_action.setShortcut(QKeySequence.Copy)
        copy_action.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        copy_action.triggered.connect(lambda: copy_selection(view, self.ctrl))
        view.addAction(copy_action)

        outer.addWidget(row1)
        outer.addWidget(row1b)
        outer.addWidget(row1c)
        outer.addWidget(filter_host)
        outer.addWidget(row2)
        outer.addWidget(row3)
        outer.addWidget(stack, 1)

        self._root = root
        self._combo_table = combo_table
        self._combo_measures = combo_measures
        self._combo_group1 = combo_group1
        self._combo_group2 = combo_group2
        self._lbl_shape = lbl_shape
        self._lbl_cov_file = lbl_cov_file
        self._lbl_filters_hint = lbl_filters_hint
        self._btn_add_filter = btn_add_filter
        self._btn_clear_filters = btn_clear_filters
        self._filter_host = filter_host
        self._chk_stats = {
            "N": chk_n, "Mean": chk_mean, "SD": chk_sd,
            "Median": chk_median, "Min": chk_min, "Max": chk_max,
        }
        self._combo_view_mode = combo_view_mode
        self._edit_search = edit_search
        self._lbl_status = lbl_status
        self._view = view
        self._article = article
        self._stack = stack
        self._summary_df = pd.DataFrame()
        self._model = None
        self._proxy = None
        self._font_scale = current_font_scale()

        btn_refresh.clicked.connect(self.refresh_tables)
        btn_load_cov.clicked.connect(self._load_aux_file)
        btn_clear_cov.clicked.connect(self._clear_aux_file)
        btn_add_filter.clicked.connect(self._add_filter_row)
        btn_clear_filters.clicked.connect(self._clear_filter_rows)
        combo_table.currentIndexChanged.connect(self._on_table_changed)
        combo_measures.selectionChanged.connect(self._schedule_plot)
        combo_group1.currentIndexChanged.connect(self._schedule_plot)
        combo_group2.currentIndexChanged.connect(self._schedule_plot)
        for chk in self._chk_stats.values():
            chk.stateChanged.connect(self._schedule_plot)
        combo_view_mode.currentIndexChanged.connect(self._sync_view_mode)
        edit_search.textChanged.connect(self._apply_table_filter)
        btn_export.clicked.connect(lambda: save_table_as_tsv(self._view, self.ctrl))
        self._sync_view_mode()

    def refresh_tables(self):
        """Populate the table combo from ctrl.results."""
        results = getattr(self.ctrl, "results", None) or {}
        cur = self._combo_table.currentData()
        self._combo_table.blockSignals(True)
        self._combo_table.clear()
        for key in sorted(results.keys()):
            self._combo_table.addItem(_table_display_name(key), key)
        idx = self._combo_table.findData(cur)
        if idx >= 0:
            self._combo_table.setCurrentIndex(idx)
        self._combo_table.blockSignals(False)
        self._on_table_changed()

    def _current_factor_cols(self):
        cols = []
        for combo in (self._combo_group1, self._combo_group2):
            col = combo.currentData()
            if col:
                cols.append(col)
        return cols

    def _on_table_changed(self, *_):
        key = self._combo_table.currentData()
        results = getattr(self.ctrl, "results", None) or {}
        df = results.get(key) if key else None
        self._df = df if isinstance(df, pd.DataFrame) and not df.empty else None
        old_measures = self._combo_measures.checked_items()
        old_g1 = self._combo_group1.currentData()
        old_g2 = self._combo_group2.currentData()

        self._combo_measures.blockSignals(True)
        self._combo_group1.blockSignals(True)
        self._combo_group2.blockSignals(True)
        self._combo_group1.clear(); self._combo_group2.clear()
        self._combo_group1.addItem("(none)", None)
        self._combo_group2.addItem("(none)", None)

        if self._df is not None:
            eff_df, aux_cols = self._get_effective_df()
            aux_set = set(aux_cols)
            num_cols = []
            for col in eff_df.columns:
                label = f"{col} [cov]" if col in aux_set else col
                nums = _coerce_measure_series(eff_df[col])
                if nums.notna().any():
                    num_cols.append(col)
                self._combo_group1.addItem(label, col)
                self._combo_group2.addItem(label, col)
            default = old_measures or num_cols[: min(3, len(num_cols))]
            self._combo_measures.set_items(num_cols, checked_labels=[c for c in default if c in num_cols])
            self._shape_context = {
                "base_rows": len(self._df),
                "base_cols": len(self._df.columns),
                "n_aux": len(aux_cols),
            }
            self._update_shape_label()
            for combo, old in ((self._combo_group1, old_g1), (self._combo_group2, old_g2)):
                idx = combo.findData(old)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
        else:
            self._combo_measures.set_items([], checked_labels=[])
            self._shape_context = {"base_rows": 0, "base_cols": 0, "n_aux": 0}
            self._lbl_shape.setText("")
            self._filter_candidates = {}

        self._combo_measures.blockSignals(False)
        self._combo_group1.blockSignals(False)
        self._combo_group2.blockSignals(False)
        self._refresh_filter_context()
        self._schedule_plot()

    def _refresh_filter_context(self, df=None):
        if df is None:
            df, _ = self._get_effective_df()
        known_factors = set(self._current_factor_cols())
        extra = getattr(self.ctrl, "results_factor_cols", None)
        if extra:
            known_factors.update(str(col) for col in extra)
        filter_states = self._snapshot_filter_rows()
        self._filter_candidates = self._filterable_columns(df, always_include=known_factors)
        self._restore_filter_rows(filter_states)
        self._sync_filter_controls()

    def _sync_filter_controls(self):
        n_candidates = len(self._filter_candidates)
        self._btn_add_filter.setEnabled(n_candidates > 0)
        self._btn_clear_filters.setEnabled(bool(self._filter_rows))
        self._filter_host.setVisible(bool(self._filter_rows))
        if n_candidates == 0:
            self._lbl_filters_hint.setText("No factor columns available")
        else:
            self._lbl_filters_hint.setText(
                f"{n_candidates} factor columns available (known factors always included)"
            )

    def _selected_stats(self):
        stats = [name for name, chk in self._chk_stats.items() if chk.isChecked()]
        return stats or ["N"]

    def _plot(self):
        df, _ = self._get_effective_df()
        if df is None:
            self._set_summary_df(pd.DataFrame())
            return
        df, active_filter_cols = self._apply_row_filters(df)
        self._update_shape_label(filtered_rows=len(df))
        measures = self._combo_measures.checked_items()
        groups = [
            col for col in (self._combo_group1.currentData(), self._combo_group2.currentData())
            if col
        ]
        if len(groups) == 2 and groups[0] == groups[1]:
            groups = groups[:1]
        out = _summary_stat_rows(df, measures, groups, self._selected_stats())
        self._set_summary_df(out)
        suffix = ""
        if active_filter_cols:
            suffix = f" · filtered by {', '.join(active_filter_cols)}"
        self._lbl_status.setText(f"{len(out)} summary rows{suffix}")
        self._refresh_filter_context(df)

    def _set_summary_df(self, df):
        self._summary_df = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        model = DataFrameModel(self._summary_df, float_decimals_default=3, parent=self._view)
        proxy = NumericSortFilterProxy(self._view)
        proxy.setSourceModel(model)
        proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self._model = model
        self._proxy = proxy
        self._view.setModel(proxy)
        self._apply_table_filter(self._edit_search.text())
        self._view.resizeColumnsToContents()
        self._refresh_article_view()

    def _apply_table_filter(self, text):
        if self._proxy is None:
            return
        pattern = QRegularExpression.escape(text.strip())
        if pattern:
            pattern = f".*{pattern}.*"
        rx = QRegularExpression(pattern)
        rx.setPatternOptions(QRegularExpression.CaseInsensitiveOption)
        self._proxy.setFilterRegularExpression(rx)
        self._refresh_article_view()

    def _article_filtered_df(self):
        df = self._summary_df
        if df is None or df.empty:
            return pd.DataFrame()
        text = self._edit_search.text().strip().casefold()
        if not text:
            return df
        row_text = df.fillna("").astype(str).agg("\t".join, axis=1).str.casefold()
        return df[row_text.str.contains(text, regex=False)].reset_index(drop=True)

    def _refresh_article_view(self):
        if not hasattr(self, "_article"):
            return
        title = _table_display_name(self._combo_table.currentData())
        html = _publication_table_html(
            self._article_filtered_df(),
            self._current_factor_cols(),
            self._selected_stats(),
            title=title,
            font_scale=self._font_scale,
        )
        self._article.setHtml(html)

    def _sync_view_mode(self, *_):
        mode = self._combo_view_mode.currentData() if hasattr(self, "_combo_view_mode") else "grid"
        is_article = mode == "article"
        self._stack.setCurrentWidget(self._article if is_article else self._view)
        self._edit_search.setPlaceholderText("filter article…" if is_article else "filter table…")
        self._refresh_article_view()

    def refresh_font_scale(self, scale=None):
        self._font_scale = current_font_scale() if scale is None else scale
        self._refresh_article_view()

    def _filterable_columns(self, df, always_include=None):
        """Return factor-like columns available for row subsetting.

        Explicitly known factors are always included. Remaining columns fall
        back to the low-cardinality heuristic so Tables still works on ad hoc
        output tables that do not carry metadata.
        """
        out = {}
        if df is None:
            return out
        always_include = {str(c) for c in (always_include or []) if c}
        for col in df.columns:
            if col in always_include:
                vals = [v for v in pd.unique(df[col]) if not pd.isna(v)]
                if len(vals) < 1:
                    continue
                sorted_vals = _numeric_sort(vals)
                levels = [(_display_level(v), v) for v in sorted_vals]
                if len({label for label, _ in levels}) != len(levels):
                    levels = [(repr(v), v) for v in sorted_vals]
                out[col] = levels
                continue
            vals = [v for v in pd.unique(df[col]) if not pd.isna(v)]
            if len(vals) < 2 or len(vals) > _MAX_FILTER_LEVELS:
                continue
            sorted_vals = _numeric_sort(vals)
            levels = [(_display_level(v), v) for v in sorted_vals]
            if len({label for label, _ in levels}) != len(levels):
                levels = [(repr(v), v) for v in sorted_vals]
            out[col] = levels
        return out
