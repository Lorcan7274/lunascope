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

from PySide6.QtWidgets import (
    QApplication,
    QToolButton,
    QVBoxLayout,
    QMessageBox,
    QComboBox,
    QLabel,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QTextBrowser,
    QStyle,
    QFrame,
    QListWidget,
    QListWidgetItem,
)
from PySide6.QtCore import QEvent, QMetaObject, Qt, QTimer, Slot, QSize
from PySide6.QtGui import QStandardItemModel, QStandardItem
import html
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd

from ..runtime_paths import app_cache_root, app_state_file


POPS_DOWNLOAD_URL = "https://zzz.nyspi.org/dist/luna/pops.zip"
POPS_STATE_FILE = "pops_location.json"


class MultiSelectComboBox(QComboBox):
    """QComboBox-like widget with a custom persistent popup for multi-select."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModel(QStandardItemModel(self))
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText("Select one or more channels")
        self._popup_visible = False
        self._press_row = -1

        self._popup = QFrame(None, Qt.Popup)
        self._popup.setFrameShape(QFrame.StyledPanel)
        self._popup.installEventFilter(self)
        popup_layout = QVBoxLayout(self._popup)
        popup_layout.setContentsMargins(0, 0, 0, 0)
        popup_layout.setSpacing(0)

        self._list = QListWidget(self._popup)
        self._list.setSelectionMode(QListWidget.NoSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list.viewport().installEventFilter(self)
        popup_layout.addWidget(self._list)
        self.lineEdit().installEventFilter(self)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._toggle_popup()
            event.accept()
            return
        super().mousePressEvent(event)

    def showPopup(self):
        self._rebuild_popup_geometry()
        self._popup_visible = True
        self._popup.show()
        self._popup.raise_()
        self._list.setFocus(Qt.PopupFocusReason)

    def _force_close(self):
        self._popup_visible = False
        self._popup.hide()
        self._refresh_text()

    def _toggle_popup(self):
        if self._popup_visible:
            self._force_close()
        else:
            self.showPopup()

    def eventFilter(self, obj, event):
        if obj is self.lineEdit():
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._toggle_popup()
                event.accept()
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                event.accept()
                return True
        if obj is self._popup and event.type() == QEvent.Hide:
            self._popup_visible = False
            self._press_row = -1
            self._refresh_text()
            return False
        list_widget = getattr(self, "_list", None)
        if list_widget is not None and obj is list_widget.viewport():
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                index = list_widget.indexAt(event.pos())
                self._press_row = index.row() if index.isValid() else -1
                event.accept()
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                index = list_widget.indexAt(event.pos())
                row = index.row() if index.isValid() else -1
                if row >= 0 and row == self._press_row:
                    self._toggle_row(row)
                self._press_row = -1
                event.accept()
                return True
        return super().eventFilter(obj, event)

    def hidePopup(self):
        self._force_close()

    def set_items(self, labels, checked_labels=None):
        checked = set(checked_labels or [])
        model = self.model()
        model.clear()
        self._list.clear()
        for lab in labels:
            item = QStandardItem(str(lab))
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setData(Qt.Checked if lab in checked else Qt.Unchecked, Qt.CheckStateRole)
            model.appendRow(item)
            popup_item = QListWidgetItem(str(lab), self._list)
            popup_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            popup_item.setCheckState(Qt.Checked if lab in checked else Qt.Unchecked)
        self._refresh_text()

    def checked_items(self):
        out = []
        for r in range(self._list.count()):
            item = self._list.item(r)
            if item and item.checkState() == Qt.Checked:
                out.append(item.text())
        return out

    def _refresh_text(self):
        chs = self.checked_items()
        self.setCurrentIndex(-1)
        if not chs:
            self.lineEdit().setText("")
        elif len(chs) <= 3:
            self.lineEdit().setText(", ".join(chs))
        else:
            self.lineEdit().setText(f"{len(chs)} selected")

    def _toggle_row(self, row: int):
        item = self._list.item(row)
        model_item = self.model().item(row)
        if item is None or model_item is None:
            return
        checked = item.checkState() != Qt.Checked
        state = Qt.Checked if checked else Qt.Unchecked
        item.setCheckState(state)
        model_item.setCheckState(state)
        self._refresh_text()

    def _rebuild_popup_geometry(self):
        row_count = self._list.count()
        row_h = self._list.sizeHintForRow(0)
        if row_h <= 0:
            row_h = self.fontMetrics().height() + 10
        frame = self._popup.frameWidth() * 2
        visible_rows = min(max(row_count, 1), 10)
        popup_h = visible_rows * row_h + frame
        if row_count > visible_rows:
            popup_h += self._list.horizontalScrollBar().sizeHint().height()
        popup_w = max(self.width(), self._list.sizeHintForColumn(0) + 32)
        self._popup.resize(QSize(popup_w, popup_h))
        self._popup.move(self.mapToGlobal(self.rect().bottomLeft()))


def _replace_with_multiselect(combo: QComboBox) -> MultiSelectComboBox:
    parent = combo.parentWidget()
    if parent is None:
        return MultiSelectComboBox()
    layout = parent.layout()
    multi = MultiSelectComboBox(parent)
    multi.setObjectName(combo.objectName())
    if layout is not None and hasattr(layout, "replaceWidget"):
        layout.replaceWidget(combo, multi)
    combo.hide()
    combo.deleteLater()
    return multi
        
class SoapPopsMixin:

    def _show_staging_message(self, message: str, *, title: str = "Error") -> None:
        blocks = []
        for para in str(message).strip().split("\n\n"):
            lines = [html.escape(line.strip()) for line in para.splitlines() if line.strip()]
            if not lines:
                continue
            head, tail = lines[0], lines[1:]
            if tail:
                tail_html = "".join(f"<div>{line}</div>" for line in tail)
                blocks.append(
                    "<div style='margin: 0 0 14px 0;'>"
                    f"<div style='font-weight: 600; margin-bottom: 4px;'>{head}</div>"
                    f"<div style='line-height: 1.45;'>{tail_html}</div>"
                    "</div>"
                )
            else:
                blocks.append(
                    "<div style='margin: 0 0 14px 0; line-height: 1.45;'>"
                    f"{head}"
                    "</div>"
                )

        dlg = QDialog(self.ui)
        dlg.setWindowTitle(title)
        dlg.setModal(True)
        dlg.resize(920, 680)

        outer = QVBoxLayout(dlg)
        outer.setContentsMargins(18, 18, 18, 14)
        outer.setSpacing(14)

        body = QHBoxLayout()
        body.setSpacing(14)

        icon = QLabel(dlg)
        pm = dlg.style().standardIcon(QStyle.SP_MessageBoxWarning).pixmap(48, 48)
        icon.setPixmap(pm)
        icon.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        body.addWidget(icon, 0, Qt.AlignTop)

        viewer = QTextBrowser(dlg)
        viewer.setOpenExternalLinks(False)
        viewer.setReadOnly(True)
        viewer.setFrameShape(QTextBrowser.NoFrame)
        viewer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        viewer.document().setDocumentMargin(0)
        viewer.setStyleSheet("QTextBrowser { background: transparent; border: none; }")
        viewer.setHtml(
            "<div style='font-size: 14px; color: palette(text);'>"
            + "".join(blocks)
            + "</div>"
        )
        body.addWidget(viewer, 1)

        outer.addLayout(body, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok, parent=dlg)
        buttons.accepted.connect(dlg.accept)
        outer.addWidget(buttons)

        dlg.exec()

    def _default_pops_bundle_dir(self) -> Path:
        return app_cache_root() / "pops"

    def _pops_state_path(self) -> Path:
        return app_state_file(POPS_STATE_FILE)

    def _resolve_pops_path(self, value: str | os.PathLike[str]) -> Path:
        return Path(os.path.expandvars(str(Path(value).expanduser()))).resolve()

    def _find_pops_models_dir(self, root: Path) -> Path | None:
        if not root.exists():
            return None
        if any(root.glob("*.mod")):
            return root

        matches = []
        for mod_file in root.rglob("*.mod"):
            try:
                matches.append(mod_file.parent.resolve())
            except OSError:
                continue

        if not matches:
            return None

        matches = sorted(set(matches), key=lambda p: (len(p.parts), str(p)))
        return matches[0]

    def _load_cached_pops_path(self) -> str | None:
        try:
            raw = self._pops_state_path().read_text(encoding="utf-8")
            data = json.loads(raw)
            path = str(data.get("path", "")).strip()
            if not path:
                return None
            resolved = self._resolve_pops_path(path)
            cached = self._find_pops_models_dir(resolved)
            return str(cached or resolved)
        except Exception:
            return None

    def _save_cached_pops_path(self, path: str | os.PathLike[str]) -> None:
        resolved = self._resolve_pops_path(path)
        state_path = self._pops_state_path()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps({"path": str(resolved)}, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def _set_pops_path(self, path: str | os.PathLike[str], *, persist: bool = True) -> str:
        resolved = self._resolve_pops_path(path)
        text = str(resolved)
        self.ui.txt_pops_path.setText(text)
        if persist:
            self._save_cached_pops_path(resolved)
        return text

    def _cache_current_pops_path(self) -> None:
        text = self.ui.txt_pops_path.text().strip()
        if not text:
            return
        try:
            self._save_cached_pops_path(text)
        except Exception:
            pass

    def _download_pops_resources(self) -> None:
        bundle_dir = self._default_pops_bundle_dir()
        archive_path = bundle_dir / "pops.zip"
        extract_dir = bundle_dir / "resources"
        staging_dir = Path(tempfile.mkdtemp(prefix="pops-", dir=str(bundle_dir.parent)))

        try:
            bundle_dir.mkdir(parents=True, exist_ok=True)
            QApplication.setOverrideCursor(Qt.WaitCursor)

            with urllib.request.urlopen(POPS_DOWNLOAD_URL, timeout=120) as response:
                with archive_path.open("wb") as fh:
                    shutil.copyfileobj(response, fh)

            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            staging_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(staging_dir)

            models_dir = self._find_pops_models_dir(staging_dir)
            if models_dir is None:
                raise FileNotFoundError("No POPS model files (*.mod) were found in the downloaded archive.")

            shutil.move(str(staging_dir), str(extract_dir))
            models_dir = self._find_pops_models_dir(extract_dir)
            if models_dir is None:
                raise FileNotFoundError("POPS resources were extracted, but no model directory could be resolved.")

            self._set_pops_path(models_dir)
            QMessageBox.information(
                self.ui,
                "POPS Resources",
                f"Downloaded POPS resources to:\n{models_dir}",
            )
        except Exception as e:
            QMessageBox.critical(
                self.ui,
                "POPS Download Error",
                f"Could not download POPS resources.\n\n{type(e).__name__}: {e}",
            )
        finally:
            QApplication.restoreOverrideCursor()
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

    def _stage_validation_classes(self):
        if hasattr(self, "_navigator_stage_query_classes"):
            return self._navigator_stage_query_classes()
        return ['N1', 'N2', 'N3', 'R', 'W', 'SP', 'WP', '?', 'L']

    def _stage_validation_raw_df(self):
        try:
            df = self.p.fetch_annots(self._stage_validation_classes(), 30)
        except Exception:
            return pd.DataFrame()

        if not isinstance(df, pd.DataFrame) or df.empty or 'Class' not in df.columns:
            return pd.DataFrame()

        if hasattr(self, "_filter_navigator_stage_df"):
            df = self._filter_navigator_stage_df(df, 'Class')

        if df.empty:
            return pd.DataFrame()

        cols = [c for c in ('Start', 'Stop', 'Class') if c in df.columns]
        if len(cols) < 3:
            return pd.DataFrame()
        return df[cols].copy()

    def _stage_validation_df(self):
        try:
            self.p.silent_proc('EPOCH align verbose & STAGE')
        except Exception:
            return pd.DataFrame()

        try:
            tbls = self.p.strata()
        except Exception:
            return pd.DataFrame()

        if not isinstance(tbls, pd.DataFrame) or tbls.empty or "Command" not in tbls.columns:
            return pd.DataFrame()

        if not (tbls["Command"] == "STAGE").any():
            return pd.DataFrame()

        try:
            df_epoch = self.p.table('EPOCH', 'E')
            df_stage = self.p.table('STAGE', 'E')
        except Exception:
            return pd.DataFrame()

        if not isinstance(df_epoch, pd.DataFrame) or not isinstance(df_stage, pd.DataFrame):
            return pd.DataFrame()

        need_epoch = {'E', 'START', 'STOP'}
        need_stage = {'E', 'OSTAGE'}
        if df_epoch.empty or df_stage.empty or not need_epoch.issubset(df_epoch.columns) or not need_stage.issubset(df_stage.columns):
            return pd.DataFrame()

        df = pd.merge(
            df_epoch[['E', 'START', 'STOP']],
            df_stage[['E', 'OSTAGE']],
            on='E',
            how='inner',
        ).rename(columns={'OSTAGE': 'Class'})

        if hasattr(self, "_filter_navigator_stage_df"):
            df = self._filter_navigator_stage_df(df, 'Class')

        if df.empty:
            return pd.DataFrame()

        return df[['START', 'STOP', 'Class']].copy()

    def _stage_validation_format_stage_list(self, values):
        vals = [str(v) for v in values if str(v)]
        return ", ".join(vals) if vals else "(none)"

    def _stage_validation_format_secs(self, value):
        value = float(value)
        if abs(value - round(value)) < 1e-6:
            return f"{int(round(value))}s"
        return f"{value:.3f}s"

    def _stage_validation_overlap_examples(self, df, *, limit=3):
        if df.empty or 'Start' not in df.columns or 'Stop' not in df.columns:
            return {"count": 0, "examples": []}

        ordered = df.sort_values(['Start', 'Stop', 'Class'], kind='stable')
        active = []
        overlaps = []
        total = 0
        for row in ordered.itertuples(index=False):
            start = float(row.Start)
            stop = float(row.Stop)
            active = [prev for prev in active if float(prev.Stop) > start]
            for prev in active:
                total += 1
                if len(overlaps) < limit:
                    overlaps.append({
                        "left_class": str(prev.Class),
                        "left_start": float(prev.Start),
                        "left_stop": float(prev.Stop),
                        "right_class": str(row.Class),
                        "right_start": start,
                        "right_stop": stop,
                    })
            active.append(row)
        return {"count": total, "examples": overlaps}

    def _stage_validation_grid_issue(self, df):
        if df.empty or 'Start' not in df.columns or 'Stop' not in df.columns or 'Class' not in df.columns:
            return None

        valid = {'N1', 'N2', 'N3', 'R', 'W', 'SP', 'WP'}
        probe = df.loc[df['Class'].isin(valid), ['Start', 'Stop', 'Class']].copy()
        if probe.empty:
            return None

        probe['DUR_SEC'] = (probe['Stop'].astype(float) - probe['Start'].astype(float)).round(3)
        dur_counts = probe['DUR_SEC'].value_counts().sort_values(ascending=False)
        epoch_size = float(dur_counts.index[0])
        if epoch_size <= 0:
            return None

        ordered = probe.sort_values(['Start', 'Stop', 'Class'], kind='stable').reset_index(drop=True)
        ordered['NEXT_START'] = ordered['Start'].shift(-1)
        ordered['GAP_SEC'] = (ordered['NEXT_START'] - ordered['Stop']).round(3)
        gap_rows = ordered.loc[ordered['GAP_SEC'] > 1e-6, ['Class', 'Start', 'Stop', 'NEXT_START', 'GAP_SEC']]

        probe['PHASE_SEC'] = probe['Start'].astype(float).map(lambda x: round(x % epoch_size, 3))
        phase_counts = probe['PHASE_SEC'].value_counts().sort_values(ascending=False)

        has_multi_duration = len(dur_counts) > 1
        has_gaps = not gap_rows.empty
        has_multi_phase = len(phase_counts) > 1
        if not (has_multi_duration or has_gaps or has_multi_phase):
            return None

        duration_examples = []
        if has_multi_duration:
            seen_dur = set()
            for row in probe.sort_values('Start').itertuples(index=False):
                dur = float(row.DUR_SEC)
                if dur in seen_dur:
                    continue
                seen_dur.add(dur)
                duration_examples.append({
                    "duration": dur,
                    "start": float(row.Start),
                    "class": str(row.Class),
                })
                if len(duration_examples) >= min(3, len(dur_counts)):
                    break

        gap_examples = []
        for row in gap_rows.itertuples(index=False):
            gap_examples.append({
                "class": str(row.Class),
                "start": float(row.Start),
                "stop": float(row.Stop),
                "next_start": float(row.NEXT_START),
                "gap": float(row.GAP_SEC),
            })
            if len(gap_examples) >= 3:
                break

        phase_examples = []
        if has_multi_phase:
            seen_phase = set()
            for row in probe.sort_values('Start').itertuples(index=False):
                phase = float(row.PHASE_SEC)
                if phase in seen_phase:
                    continue
                seen_phase.add(phase)
                phase_examples.append({
                    "phase": phase,
                    "start": float(row.Start),
                    "class": str(row.Class),
                })
                if len(phase_examples) >= min(3, len(phase_counts)):
                    break

        return {
            "epoch_size": epoch_size,
            "durations": dur_counts.index.tolist(),
            "duration_counts": dur_counts.to_dict(),
            "duration_examples": duration_examples,
            "gap_count": int(len(gap_rows)),
            "gap_examples": gap_examples,
            "phases": phase_counts.index.tolist(),
            "phase_counts": phase_counts.to_dict(),
            "phase_examples": phase_examples,
        }

    def _stage_validation_diagnostics(self, require_multiple=True):
        diag = {
            "ok": False,
            "code": "unknown",
            "message": "No valid staging.",
            "raw_df": pd.DataFrame(),
            "aligned_df": pd.DataFrame(),
            "aligned_unique_count": 0,
        }

        if not hasattr(self, "p"):
            diag["code"] = "no_instance"
            diag["message"] = "No instance attached."
            return diag

        raw_df = self._stage_validation_raw_df()
        aligned_df = self._stage_validation_df()
        aligned_unique = self._stage_validation_unique_count(aligned_df)

        diag["raw_df"] = raw_df
        diag["aligned_df"] = aligned_df
        diag["aligned_unique_count"] = aligned_unique

        if not aligned_df.empty and not self._stage_validation_has_overlap(aligned_df):
            if (not require_multiple) or aligned_unique >= 2:
                diag["ok"] = True
                diag["code"] = "ok"
                diag["message"] = "Valid staging is available after `EPOCH align`."
                return diag

        if raw_df.empty:
            diag["code"] = "none_present"
            diag["message"] = (
                "No stage-like annotations were found.\n\n"
                "Expected one or more of: N1, N2, N3, R, W, SP, WP."
            )
            return diag

        sleep_like = {'N1', 'N2', 'N3', 'R', 'SP'}
        raw_sleep = sorted(raw_df.loc[raw_df['Class'].isin(sleep_like), 'Class'].unique().tolist())
        raw_all = sorted(raw_df['Class'].unique().tolist())
        if not raw_sleep:
            diag["code"] = "wake_only"
            diag["message"] = (
                "Only wake/unknown/light-style staging was found.\n\n"
                f"Observed classes: {self._stage_validation_format_stage_list(raw_all)}\n"
                "No sleep-stage labels such as N1/N2/N3/R (or SP) were present."
            )
            return diag

        issues = []
        overlap_info = self._stage_validation_overlap_examples(raw_df)
        if overlap_info["count"]:
            diag["code"] = "raw_overlap"
            lines = [
                f"Found {overlap_info['count']} true overlap(s) between raw staging annotations; "
                f"showing first {len(overlap_info['examples'])}:"
            ]
            for item in overlap_info["examples"]:
                lines.append(
                    f"{item['left_class']} [{item['left_start']:.3f}, {item['left_stop']:.3f}) "
                    f"overlaps {item['right_class']} [{item['right_start']:.3f}, {item['right_stop']:.3f})"
                )
            issues.append("\n".join(lines))

        grid_issue = self._stage_validation_grid_issue(raw_df)
        if grid_issue:
            if diag["code"] == "unknown":
                diag["code"] = "offset_conflict"
            grid_lines = []
            if len(grid_issue["durations"]) > 1:
                durations = ", ".join(self._stage_validation_format_secs(v) for v in grid_issue["durations"])
                grid_lines.append(
                    f"Found {len(grid_issue['durations'])} epoch-size families in raw staging: {durations}."
                )
                if grid_issue["duration_examples"]:
                    grid_lines.append(
                        f"Showing first {len(grid_issue['duration_examples'])} duration example(s):"
                    )
                    for item in grid_issue["duration_examples"]:
                        grid_lines.append(
                            f"{item['class']} starts at {item['start']:.3f}s with duration "
                            f"{self._stage_validation_format_secs(item['duration'])}"
                        )
            if grid_issue["gap_count"]:
                grid_lines.append(
                    f"Found {grid_issue['gap_count']} gap(s) between consecutive raw staging intervals; "
                    f"showing first {len(grid_issue['gap_examples'])}:"
                )
                for item in grid_issue["gap_examples"]:
                    grid_lines.append(
                        f"{item['class']} ends at {item['stop']:.3f}s and the next stage starts at "
                        f"{item['next_start']:.3f}s (gap {self._stage_validation_format_secs(item['gap'])})"
                    )
            if len(grid_issue["phases"]) > 1:
                phases = ", ".join(self._stage_validation_format_secs(v) for v in grid_issue["phases"])
                grid_lines.append(
                    "Raw stage starts do not stay on one consistent epoch grid.\n"
                    f"Using inferred epoch size {self._stage_validation_format_secs(grid_issue['epoch_size'])}, "
                    f"found {len(grid_issue['phases'])} distinct start-phase families: {phases}."
                )
                if grid_issue["phase_examples"]:
                    grid_lines.append(
                        f"Showing first {len(grid_issue['phase_examples'])} phase example(s):"
                    )
                    for item in grid_issue["phase_examples"]:
                        grid_lines.append(
                            f"{item['class']} starts at {item['start']:.3f}s "
                            f"(phase {self._stage_validation_format_secs(item['phase'])})"
                        )
            issues.append("\n".join(grid_lines))

        if aligned_df.empty:
            if diag["code"] == "unknown":
                diag["code"] = "align_failed"
            issues.append("`EPOCH align & STAGE` did not produce usable aligned staging.")
        elif self._stage_validation_has_overlap(aligned_df):
            if diag["code"] == "unknown":
                diag["code"] = "aligned_overlap"
            issues.append("Aligned staging still contains overlapping epoch spans after `EPOCH align`.")

        if require_multiple and aligned_unique < 2:
            if 'Class' in aligned_df.columns:
                found = sorted(
                    aligned_df.loc[
                        aligned_df['Class'].isin({'N1', 'N2', 'N3', 'R', 'W', 'SP', 'WP'}),
                        'Class',
                    ].unique().tolist()
                )
            else:
                found = []
            if diag["code"] == "unknown":
                diag["code"] = "too_few_stages"
            issues.append(
                "Aligned staging contains fewer than 2 distinct valid stages.\n"
                f"Observed after alignment: {self._stage_validation_format_stage_list(found)}"
            )

        if not issues:
            issues.append("Staging was present, but LunaScope could not validate it after `EPOCH align`.")

        diag["message"] = "\n\n".join(issues)
        return diag

    def _stage_validation_has_overlap(self, df):
        if df.empty or 'START' not in df.columns or 'STOP' not in df.columns:
            return False

        ordered = df.sort_values(['START', 'STOP', 'Class'], kind='stable')
        prev_stop = None
        for row in ordered.itertuples(index=False):
            start = float(row.START)
            stop = float(row.STOP)
            if prev_stop is not None and start < prev_stop:
                return True
            prev_stop = max(prev_stop, stop) if prev_stop is not None else stop
        return False

    def _stage_validation_unique_count(self, df):
        if df.empty or 'Class' not in df.columns:
            return 0
        valid = {'N1', 'N2', 'N3', 'R', 'W', 'SP', 'WP'}
        return int(df.loc[df['Class'].isin(valid), 'Class'].nunique())

    def _ensure_soap_canvas(self):
        if getattr(self, "soapcanvas", None) is not None:
            return self.soapcanvas

        layout = self.ui.host_soap.layout()
        if layout is None:
            layout = QVBoxLayout()
            self.ui.host_soap.setLayout(layout)
        layout.setContentsMargins(0,0,0,0)

        from .mplcanvas import MplCanvas
        self.soapcanvas = MplCanvas(self.ui.host_soap)
        layout.addWidget(self.soapcanvas)
        return self.soapcanvas

    def _ensure_pops_canvas(self):
        if getattr(self, "popscanvas", None) is not None:
            return self.popscanvas

        layout = self.ui.host_pops.layout()
        if layout is None:
            layout = QVBoxLayout()
            self.ui.host_pops.setLayout(layout)
        layout.setContentsMargins(0,0,0,0)

        from .mplcanvas import MplCanvas
        self.popscanvas = MplCanvas(self.ui.host_pops)
        layout.addWidget(self.popscanvas)
        return self.popscanvas

    # valid staging:
    #   - EDF/annotations attached
    #   - EPOCH align & STAGE succeeds
    #   - STAGE output is non-empty after navigator filtering
    #   - aligned epochs do not overlap
    #   - optional: at least 2 unique valid stages

    def _has_staging(self, require_multiple = True ):
        return self._stage_validation_diagnostics(require_multiple=require_multiple)["ok"]

    
    def _init_soap_pops(self):
        self.soapcanvas = None
        self.popscanvas = None
        if self.ui.host_soap.layout() is None:
            self.ui.host_soap.setLayout(QVBoxLayout())
        self.ui.host_soap.layout().setContentsMargins(0,0,0,0)
        if self.ui.host_pops.layout() is None:
            self.ui.host_pops.setLayout(QVBoxLayout())
        self.ui.host_pops.layout().setContentsMargins(0,0,0,0)

        # Replace Designer combo with a checkable multi-select control.
        self.ui.combo_pops = _replace_with_multiselect(self.ui.combo_pops)

        pops_layout = self.ui.butt_pops.parentWidget().layout()
        if pops_layout is not None:
            pops_layout.setColumnStretch(0, 0)
            pops_layout.setColumnStretch(1, 1)
            pops_layout.setColumnStretch(2, 0)
            pops_layout.setColumnStretch(3, 0)
            pops_layout.setColumnStretch(4, 0)
        self.ui.combo_pops.setMinimumWidth(180)
        self.ui.txt_pops_path.setMinimumWidth(360)
        self.ui.txt_pops_model.setMaximumWidth(160)
        if pops_layout is not None and getattr(self.ui, "butt_pops_resource", None) is None:
            butt_pops_resource = QToolButton(self.ui.butt_pops.parentWidget())
            butt_pops_resource.setObjectName("butt_pops_resource")
            butt_pops_resource.setText("Get…")
            butt_pops_resource.setToolTip("Download POPS resources and set the POPS folder path")
            butt_pops_resource.setAutoRaise(True)
            butt_pops_resource.clicked.connect(self._download_pops_resources)
            pops_layout.addWidget(butt_pops_resource, 1, 4)
            self.ui.butt_pops_resource = butt_pops_resource

        cached_pops_path = self._load_cached_pops_path()
        if cached_pops_path:
            self._set_pops_path(cached_pops_path, persist=False)
        
        # wiring
        self.ui.butt_soap.clicked.connect( self._calc_soap )
        self.ui.butt_pops.clicked.connect( self._calc_pops )
        self.ui.txt_pops_path.editingFinished.connect(self._cache_current_pops_path)
        self.ui.radio_pops_hypnodens.toggled.connect( self._render_pops_hypno )

    def _parse_pops_channels(self):
        if hasattr(self.ui.combo_pops, "checked_items"):
            return self.ui.combo_pops.checked_items()
        txt = self.ui.combo_pops.currentText().strip()
        return [txt] if txt else []
        
    def _update_soap_list(self):

        if not hasattr(self, "p"): return

        # first clear
        self.ui.combo_soap.clear()
        prev_checked = []
        if hasattr(self.ui.combo_pops, "checked_items"):
            prev_checked = self.ui.combo_pops.checked_items()
        else:
            prev = self.ui.combo_pops.currentText().strip()
            prev_checked = [prev] if prev else []

        # list all channels with sample frequencies > 32 Hz 
        df = self.p.headers()

        if df is not None:
            chs = df.loc[df['SR'] >= 32, 'CH'].tolist()
        else:
            chs = [ ]

        self.ui.combo_soap.addItems( chs )
        if hasattr(self.ui.combo_pops, "set_items"):
            self.ui.combo_pops.set_items(chs, checked_labels=prev_checked)
        else:
            self.ui.combo_pops.clear()
            self.ui.combo_pops.addItems(chs)
            if prev_checked:
                self.ui.combo_pops.setCurrentText(prev_checked[0])

        
    # ------------------------------------------------------------
    # Run SOAP

    def _calc_soap(self):
        self._ensure_soap_canvas()

        # requires attached individal
        if not hasattr(self, "p"):
            QMessageBox.critical( self.ui , "Error", "No instance attached" )
            return
        
        # requires staging
        diag = self._stage_validation_diagnostics()
        if not diag["ok"]:
            self._show_staging_message(diag["message"])
            return

        # requires 1+ channel
        count = self.ui.combo_soap.model().rowCount()
        if count == 0:
            QMessageBox.critical( self.ui , "Error", "No suitable signal for SOAP" )
            return

        if getattr(self, "_busy", False):
            return

        # parameters
        soap_ch = self.ui.combo_soap.currentText()
        soap_pc = self.ui.spin_soap_pc.value()

        self._busy = True
        self._buttons(False)
        self.sb_progress.setVisible(True)
        self.sb_progress.setRange(0, 0)
        self.sb_progress.setFormat("Running…")
        self.lock_ui()
        QTimer.singleShot(0, lambda: self._start_soap_worker(soap_ch, soap_pc))

    def _start_soap_worker(self, soap_ch, soap_pc):
        if not getattr(self, "_busy", False):
            return

        fut = self._exec.submit(self._derive_soap, self.p, soap_ch, soap_pc)

        def _done(_f=fut):
            try:
                self._last_result = _f.result()
                QMetaObject.invokeMethod(self, "_soap_done_ok", Qt.QueuedConnection)
            except Exception as e:
                self._last_exc = e
                self._last_tb = f"{type(e).__name__}: {e}"
                QMetaObject.invokeMethod(self, "_soap_done_err", Qt.QueuedConnection)

        fut.add_done_callback(_done)

    def _derive_soap(self, p, soap_ch, soap_pc):
        cmd_str = 'EPOCH align & SOAP sig=' + soap_ch + ' epoch pc=' + str(soap_pc)
        p.eval_lunascope(cmd_str)

        df_ch = p.table('SOAP', 'CH')
        df_ch = df_ch[['K', 'K3', 'ACC', 'ACC3']].copy()

        for c in df_ch.columns:
            try:
                df_ch[c] = pd.to_numeric(df_ch[c])
            except Exception:
                pass

        df_epoch = p.table('SOAP', 'CH_E')
        df_epoch = df_epoch[['PRIOR', 'PRED', 'PP_N1', 'PP_N2', 'PP_N3', 'PP_R', 'PP_W', 'DISC']].copy()
        return df_ch, df_epoch

    @Slot()
    def _soap_done_ok(self):
        try:
            df_ch, df_epoch = self._last_result
            k, k3 = df_ch.loc[0, ['K', 'K3']].astype(float)
            self.ui.txt_soap_k.setText(f"K = {k:.2f}")
            self.ui.txt_soap_k3.setText(f"K3 = {k3:.2f}")

            from .plts import hypno_density
            hypno_density(df_epoch, ax=self.soapcanvas.ax)
            self.soapcanvas.draw_idle()
        finally:
            self.unlock_ui()
            self._busy = False
            self._buttons(True)
            self.sb_progress.setRange(0, 100)
            self.sb_progress.setValue(0)
            self.sb_progress.setVisible(False)

    @Slot()
    def _soap_done_err(self):
        try:
            QMessageBox.critical(self.ui, "Error running SOAP", self._last_tb)
        finally:
            self.unlock_ui()
            self._busy = False
            self._buttons(True)
            self.sb_progress.setRange(0, 100)
            self.sb_progress.setValue(0)
            self.sb_progress.setVisible(False)
               
    # ------------------------------------------------------------
    # Run POPS

    def _calc_pops(self):
      
        if not hasattr(self, "p"):
            QMessageBox.critical( self.ui , "Error", "No instance attached" )
            return
        
        # requires 1+ channel
        count = self.ui.combo_pops.model().rowCount()
        if count == 0:
            QMessageBox.critical( self.ui , "Error", "No suitable signal for POPS" )
            return

        if getattr(self, "_busy", False):
            return

        # parameters (single-channel dropdown or manual comma list)
        pops_chs_list = self._parse_pops_channels()
        if not pops_chs_list:
            QMessageBox.critical( self.ui , "Error", "No POPS channel selected" )
            return

        # ensure channels are valid
        valid_chs = set(self.p.edf.channels())
        bad = [c for c in pops_chs_list if c not in valid_chs]
        if bad:
            QMessageBox.critical(
                self.ui,
                "Error",
                "Invalid POPS channel(s): " + ", ".join(bad),
            )
            return

        # refuse to run if POPS-derived channel names already exist —
        # RUN-POPS creates {sig}_F and {sig}_F_N and will segfault if present
        conflicting = [
            ch for sig in pops_chs_list
            for ch in (f"{sig}_F", f"{sig}_F_N")
            if ch in valid_chs
        ]
        if conflicting:
            QMessageBox.critical(
                self.ui,
                "Cannot run POPS",
                "The following channel name(s) conflict with channels that "
                "RUN-POPS needs to create internally:\n\n"
                + "  " + ",  ".join(conflicting) + "\n\n"
                "Please rename or drop these channels first, then run POPS again."
            )
            return

        pops_chs = ",".join(pops_chs_list)

        pops_path = self.ui.txt_pops_path.text()
        pops_model = self.ui.txt_pops_model.text()
        ignore_obs = self.ui.check_pops_ignore_obs.checkState() == Qt.Checked
        
        diag = self._stage_validation_diagnostics()
        has_staging = diag["ok"]
        # requires staging
        if not has_staging:
            ignore_obs = True

        # ignore existing staging
        opts = ""
        if ignore_obs:
            opts += " ignore-obs=T"
            has_staging = False
            

        # test if resource file exists
        base = Path(pops_path).expanduser()
        base = Path(os.path.expandvars(str(base))).resolve()   # absolute
        pops_mod = base / f"{str(pops_model).strip()}.mod"
        if not pops_mod.is_file():
            QMessageBox.critical(
                self.ui,
                "Error",
                "Could not open POPS files; double check file path"
            )
            return None


        # save currents channels/annots selections
        # (needed by _render_tables() used below)
        self.curr_chs = self.ui.tbl_desc_signals.checked()                   
        self.curr_anns = self.ui.tbl_desc_annots.checked()

        self._busy = True
        self._buttons(False)
        self.sb_progress.setVisible(True)
        self.sb_progress.setRange(0, 0)
        self.sb_progress.setFormat("Running…")
        self.lock_ui()
        QTimer.singleShot(
            0,
            lambda: self._start_pops_worker(
                pops_chs,
                pops_path,
                pops_model,
                opts,
                has_staging,
            ),
        )

    def _start_pops_worker(self, pops_chs, pops_path, pops_model, opts, has_staging):
        if not getattr(self, "_busy", False):
            return

        # Release segsrv C++ objects before POPS runs on the background thread.
        # Both self.ss and self.ssa hold internal slot references into the EDF.
        # POPS drops/creates channels during cleanup; if stale slot refs exist
        # in a live segsrv the channel drop segfaults.  Nulling them here lets
        # CPython's reference-counting free the C++ objects immediately.
        # _pops_done_ok recreates self.ssa before _render_tables needs it.
        self.ss  = None
        self.ssa = None

        fut = self._exec.submit(
            self._derive_pops,
            self.p,
            pops_chs,
            pops_path,
            pops_model,
            opts,
            has_staging,
        )

        def _done(_f=fut):
            try:
                self._last_result = _f.result()
                QMetaObject.invokeMethod(self, "_pops_done_ok", Qt.QueuedConnection)
            except Exception as e:
                self._last_exc = e
                self._last_tb = f"{type(e).__name__}: {e}"
                QMetaObject.invokeMethod(self, "_pops_done_err", Qt.QueuedConnection)

        fut.add_done_callback(_done)

    def _derive_pops(self, p, pops_chs, pops_path, pops_model, opts, has_staging):
        cmd_str = 'EPOCH align & RUN-POPS sig=' + pops_chs
        cmd_str += ' path=' + pops_path
        cmd_str += ' model=' + pops_model
        cmd_str += opts

        p.eval_lunascope(cmd_str)

        df = p.table('RUN_POPS', 'E')
        if has_staging:
            df = df[['E', 'START', 'PRIOR', 'PRED', 'PP_N1', 'PP_N2', 'PP_N3', 'PP_R', 'PP_W']].copy()
        else:
            df = df[['E', 'START', 'PRED', 'PP_N1', 'PP_N2', 'PP_N3', 'PP_R', 'PP_W']].copy()

        tbls = p.strata()
        return df, bool(has_staging), tbls

    @Slot()
    def _pops_done_ok(self):
        try:
            df, has_staging, tbls = self._last_result
            self.pops_df = df
            self._render_pops_hypno()
            # Recreate a minimal self.ssa before _render_tables, which calls
            # self.ssa.populate().  _update_metrics() (inside _render_tables)
            # will do a full rebuild, but we need a valid object first.
            try:
                import lunapi as lp
                self.ssa = lp.segsrv(self.p)
            except Exception:
                pass
            self._render_tables(tbls)
            if not has_staging:
                self._render_hypnogram()
                self._update_hypnogram()
        finally:
            self.unlock_ui()
            self._busy = False
            self._buttons(True)
            self.sb_progress.setRange(0, 100)
            self.sb_progress.setValue(0)
            self.sb_progress.setVisible(False)

    @Slot()
    def _pops_done_err(self):
        try:
            QMessageBox.critical(self.ui, "Error running POPS", self._last_tb)
        finally:
            self.unlock_ui()
            self._busy = False
            self._buttons(True)
            self.sb_progress.setRange(0, 100)
            self.sb_progress.setValue(0)
            self.sb_progress.setVisible(False)



    def _render_pops_hypno(self):

        if hasattr(self, 'pops_df') and isinstance(self.pops_df, pd.DataFrame) and not self.pops_df.empty:
            self._ensure_pops_canvas()
            from .plts import hypno_density, hypno

            # either draw hypnodensity or hypnogram
            if self.ui.radio_pops_hypnodens.isChecked():
                hypno_density( self.pops_df , ax=self.popscanvas.ax)
            else:
                hypno( self.pops_df.PRED , ax=self.popscanvas.ax)

            self.popscanvas.draw_idle()        
