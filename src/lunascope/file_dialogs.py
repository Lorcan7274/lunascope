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

"""Shared non-native file dialog helpers with app-wide recent-folder history."""

from __future__ import annotations

import os
import threading
from typing import Iterable

from PySide6 import QtCore, QtWidgets


_ORG = "Lunascope"
_APP = "Lunascope"
_RECENT_KEY = "file_dialogs/recent_dirs"
_BUILD_EXT_KEY = "file_dialogs/build_slist_ext"
_DEFAULT_BUILD_EXTS = ".annot, .xml, .tsv"
_ATTACH_EXT_KEY = "file_dialogs/attach_annots_ext"
_DEFAULT_ATTACH_EXTS = ".annot, .eannot, .xml"
_ATTACH_PATH_MODE_KEY = "file_dialogs/attach_annots_path_mode"
_BUILD_EXT_PREFIXES = (".", "-", "_")
# Keep enough entries to fill the sidebar without feeling stale.
_MAX_RECENT_DIRS = 24
_BUILD_PREVIEW_ROW_LIMIT = 500


def _settings() -> QtCore.QSettings:
    return QtCore.QSettings(_ORG, _APP)


def _cwd() -> str:
    try:
        return os.getcwd()
    except Exception:
        return QtCore.QDir.currentPath()


def _normalize_dir(path: str) -> str:
    if not path:
        return ""
    try:
        path = os.path.abspath(os.path.expanduser(path))
    except Exception:
        return ""
    return path if os.path.isdir(path) else ""


def _dedupe_key(path: str) -> str:
    # Normalize path identity across case-insensitive filesystems and symlinks.
    return os.path.normcase(os.path.realpath(path))


def _read_recent_dirs() -> list[str]:
    try:
        raw = _settings().value(_RECENT_KEY, [])
    except Exception:
        raw = []
    if isinstance(raw, str):
        raw = [raw] if raw else []
    if not isinstance(raw, list):
        raw = list(raw) if raw else []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        folder = _normalize_dir(str(item))
        if not folder:
            continue
        key = _dedupe_key(folder)
        if key in seen:
            continue
        out.append(folder)
        seen.add(key)
        if len(out) >= _MAX_RECENT_DIRS:
            break
    return out


def _write_recent_dirs(paths: Iterable[str]) -> None:
    vals: list[str] = []
    seen: set[str] = set()
    for item in paths:
        folder = _normalize_dir(str(item))
        if not folder:
            continue
        key = _dedupe_key(folder)
        if key in seen:
            continue
        vals.append(folder)
        seen.add(key)
        if len(vals) >= _MAX_RECENT_DIRS:
            break
    try:
        s = _settings()
        s.setValue(_RECENT_KEY, vals)
        s.sync()
    except Exception:
        pass


def remember_dialog_path(path: str) -> None:
    folder = _normalize_dir(path if os.path.isdir(path) else os.path.dirname(path))
    if not folder:
        return
    recent = _read_recent_dirs()
    # Prepend most-recent and drop tail when at capacity.
    _write_recent_dirs([folder, *recent])


def _sidebar_urls(existing: Iterable[QtCore.QUrl] | None = None) -> list[QtCore.QUrl]:
    ordered: list[str] = []
    seen: set[str] = set()

    def _push(candidate: str) -> None:
        folder = _normalize_dir(candidate)
        if not folder:
            return
        key = _dedupe_key(folder)
        if key in seen:
            return
        ordered.append(folder)
        seen.add(key)

    for candidate in [_cwd(), os.path.expanduser("~"), *_read_recent_dirs()]:
        _push(candidate)

    for url in existing or []:
        if url.isLocalFile():
            _push(url.toLocalFile())

    return [QtCore.QUrl.fromLocalFile(folder) for folder in ordered]


def _dialog_start_dir(directory: str = "") -> str:
    return _normalize_dir(directory) or _cwd()


def _first_name_filter(file_filter: str) -> str:
    if not file_filter:
        return ""
    first = str(file_filter).split(";;", 1)[0].strip()
    return first


def _selected_filter_extensions(selected_filter: str) -> list[str]:
    if not selected_filter:
        return []
    start = selected_filter.find("(")
    end = selected_filter.rfind(")")
    if start == -1 or end == -1 or end <= start:
        return []

    exts: list[str] = []
    seen: set[str] = set()
    for token in selected_filter[start + 1:end].split():
        token = token.strip()
        if not token.startswith("*.") or len(token) <= 2:
            continue
        ext = token[1:]
        key = ext.lower()
        if key in seen:
            continue
        exts.append(ext)
        seen.add(key)
    return exts


def _append_implied_extension(path: str, selected_filter: str) -> str:
    if not path:
        return path

    filename = os.path.basename(path)
    if not filename or filename.endswith(os.sep):
        return path

    if "." in filename and not filename.endswith("."):
        return path

    exts = _selected_filter_extensions(selected_filter)
    if not exts:
        return path
    return path + exts[0]


def normalize_build_ext(text: str) -> str:
    return ", ".join(normalize_build_exts(text))


def normalize_build_exts(text: str) -> list[str]:
    raw = (text or "").strip()
    if raw.startswith("-ext="):
        raw = raw[len("-ext="):].strip()
    elif raw.startswith("ext="):
        raw = raw[len("ext="):].strip()
    exts: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        ext = part.strip()
        if not ext or ext in seen:
            continue
        exts.append(ext)
        seen.add(ext)
    return exts


def _edf_stem(filename: str) -> str | None:
    lower = filename.lower()
    for suffix in (".edf.gz", ".edfz", ".edf", ".rec"):
        if lower.endswith(suffix):
            return filename[:-len(suffix)]
    return None


def _annot_stem_for_ext(filename: str, annot_ext: str) -> str | None:
    ext = annot_ext.strip()
    if not ext:
        return None
    lower = filename.lower()
    lower_ext = ext.lower()

    # Luna checks both ".ext" and "ext" forms. Suffixes such as
    # "-nsrr.xml" match without an inserted period.
    candidates = [lower_ext]
    if not ext.startswith((".", "-")):
        candidates.insert(0, f".{lower_ext}")

    for suffix in candidates:
        if lower.endswith(suffix):
            return filename[:-len(suffix)]
    return None


def _infer_build_exts(folder: str) -> list[str]:
    if not folder or not os.path.isdir(folder):
        return []

    inferred: list[str] = []
    seen: set[str] = set()
    stems_by_dir: dict[str, set[str]] = {}
    other_files: list[tuple[str, str]] = []

    for root, _dirs, files in os.walk(folder):
        dir_stems: set[str] = set()
        for name in files:
            stem = _edf_stem(name)
            if stem is not None:
                dir_stems.add(stem)
            else:
                other_files.append((root, name))
        if dir_stems:
            stems_by_dir[root] = dir_stems

    for root, name in other_files:
        stems = stems_by_dir.get(root)
        if not stems:
            continue

        best_suffix = ""
        best_stem_len = -1
        for stem in stems:
            if not name.startswith(stem):
                continue
            suffix = name[len(stem):]
            if not suffix.startswith(_BUILD_EXT_PREFIXES):
                continue
            if len(stem) > best_stem_len:
                best_suffix = suffix
                best_stem_len = len(stem)

        if best_suffix and best_suffix not in seen:
            inferred.append(best_suffix)
            seen.add(best_suffix)

    return sorted(inferred, key=str.lower)


def _count_build_slist_matches(folder: str, annot_ext: str) -> tuple[int, int]:
    edf_count, annot_count, _rows, _limited = _build_slist_link_preview(folder, annot_ext, 0)
    return edf_count, annot_count


def _build_slist_link_preview(
    folder: str,
    annot_ext: str,
    row_limit: int = _BUILD_PREVIEW_ROW_LIMIT,
) -> tuple[int, int, list[tuple[str, str, str]], bool]:
    if not folder or not os.path.isdir(folder):
        return 0, 0, [], False

    edf_count = 0
    annot_count = 0
    rows: list[tuple[str, str, str]] = []
    limited = False
    annot_exts = normalize_build_exts(annot_ext)
    annot_index: dict[str, list[tuple[int, str, str]]] = {}
    edfs: list[tuple[str, str]] = []

    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            stem = _edf_stem(name)
            rel_path = os.path.relpath(os.path.join(root, name), folder)
            if stem is not None:
                edfs.append((stem, rel_path))
                continue

            for ext_index, ext in enumerate(annot_exts):
                annot_stem = _annot_stem_for_ext(name, ext)
                if annot_stem is None:
                    continue
                annot_index.setdefault(annot_stem, []).append((ext_index, ext, rel_path))

    for stem, rel_edf in sorted(edfs, key=lambda item: item[1].lower()):
        edf_count += 1
        matches = sorted(annot_index.get(stem, []), key=lambda item: (item[0], item[2].lower()))
        annot_count += len(matches)
        rel_annot = ""
        status = "No suffixes"
        if annot_exts:
            status = "Missing"
        if matches:
            shown_paths = [path for _idx, _ext, path in matches[:4]]
            rel_annot = ", ".join(shown_paths)
            if len(matches) > len(shown_paths):
                rel_annot += f", +{len(matches) - len(shown_paths)} more"
            matched_exts = []
            seen_exts = set()
            for _idx, ext, _path in matches:
                if ext in seen_exts:
                    continue
                matched_exts.append(ext)
                seen_exts.add(ext)
            status = f"Matched ({', '.join(matched_exts[:3])})"
            if len(matched_exts) > 3:
                status += f" +{len(matched_exts) - 3}"
        if row_limit <= 0:
            continue
        if len(rows) < row_limit:
            rows.append((rel_edf, rel_annot, status))
        else:
            limited = True
    return edf_count, annot_count, rows, limited


def _index_annotation_matches(
    folder: str,
    annot_ext: str,
) -> tuple[dict[str, list[tuple[int, str, str]]], list[str]]:
    annot_exts = normalize_build_exts(annot_ext)
    annot_index: dict[str, list[tuple[int, str, str]]] = {}
    if not folder or not os.path.isdir(folder):
        return annot_index, annot_exts

    for root, _dirs, files in os.walk(folder):
        for name in sorted(files):
            rel_path = os.path.relpath(os.path.join(root, name), folder)
            for ext_index, ext in enumerate(annot_exts):
                annot_stem = _annot_stem_for_ext(name, ext)
                if annot_stem is None:
                    continue
                annot_index.setdefault(annot_stem, []).append((ext_index, ext, rel_path))

    for matches in annot_index.values():
        matches.sort(key=lambda item: (item[0], item[2].lower()))
    return annot_index, annot_exts


def _build_attach_annots_preview(
    folder: str,
    annot_ext: str,
    ids: Iterable[str],
    row_limit: int = _BUILD_PREVIEW_ROW_LIMIT,
) -> tuple[int, int, list[tuple[str, str, str]], bool]:
    ids_norm = [str(id_str or "").strip() for id_str in ids if str(id_str or "").strip()]
    if not ids_norm:
        return 0, 0, [], False

    annot_index, annot_exts = _index_annotation_matches(folder, annot_ext)
    matched_ids = 0
    annot_count = 0
    rows: list[tuple[str, str, str]] = []
    limited = False

    for id_str in ids_norm:
        matches = annot_index.get(id_str, [])
        if matches:
            matched_ids += 1
            annot_count += len(matches)
            shown_paths = [path for _idx, _ext, path in matches[:4]]
            rel_annot = ", ".join(shown_paths)
            if len(matches) > len(shown_paths):
                rel_annot += f", +{len(matches) - len(shown_paths)} more"
            matched_exts = []
            seen_exts = set()
            for _idx, ext, _path in matches:
                if ext in seen_exts:
                    continue
                matched_exts.append(ext)
                seen_exts.add(ext)
            status = f"Matched ({', '.join(matched_exts[:3])})"
            if len(matched_exts) > 3:
                status += f" +{len(matched_exts) - 3}"
        else:
            rel_annot = ""
            status = "Missing" if annot_exts else "No suffixes"

        if row_limit <= 0:
            continue
        if len(rows) < row_limit:
            rows.append((id_str, rel_annot, status))
        else:
            limited = True

    return len(ids_norm), annot_count, rows, limited


def _populate_build_preview_table(
    table: QtWidgets.QTableWidget,
    rows: list[tuple[str, str, str]],
) -> None:
    table.setRowCount(len(rows))
    for r, (edf, annot, status) in enumerate(rows):
        for c, value in enumerate((edf, annot or "-", status)):
            item = QtWidgets.QTableWidgetItem(value)
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            if status == "Missing":
                item.setForeground(QtCore.Qt.darkRed)
            elif status.startswith("Matched"):
                item.setForeground(QtCore.Qt.darkGreen)
            table.setItem(r, c, item)
    table.resizeColumnsToContents()


class _BuildSListPreviewSignals(QtCore.QObject):
    counted = QtCore.Signal(int, int, int, list, bool)


def _show_filtered_files_disabled(dlg: QtWidgets.QFileDialog) -> None:
    """Keep non-matching files visible but disabled in non-native dialogs."""

    def _apply() -> None:
        for model in dlg.findChildren(QtWidgets.QFileSystemModel):
            model.setNameFilterDisables(True)

    _apply()
    QtCore.QTimer.singleShot(0, _apply)
    QtCore.QTimer.singleShot(100, _apply)
    dlg.filterSelected.connect(lambda _filter: QtCore.QTimer.singleShot(0, _apply))
    dlg.directoryEntered.connect(lambda _path: QtCore.QTimer.singleShot(0, _apply))
    dlg.currentChanged.connect(lambda _path: QtCore.QTimer.singleShot(0, _apply))


class _DirectorySelectionProxy(QtCore.QSortFilterProxyModel):
    """Show files while selecting directories, but make files non-selectable."""

    def filterAcceptsRow(self, source_row, source_parent):
        return True

    def flags(self, index):
        flags = super().flags(index)
        if not index.isValid():
            return flags

        source = self.sourceModel()
        source_index = self.mapToSource(index)
        is_dir = False
        if hasattr(source, "isDir"):
            try:
                is_dir = bool(source.isDir(source_index))
            except Exception:
                is_dir = False
        if not is_dir:
            flags &= ~QtCore.Qt.ItemIsEnabled
            flags &= ~QtCore.Qt.ItemIsSelectable
        return flags


def _set_file_dialog_interaction_enabled(dlg: QtWidgets.QFileDialog, enabled: bool) -> None:
    for view_type in (
        QtWidgets.QListView,
        QtWidgets.QTreeView,
        QtWidgets.QColumnView,
    ):
        for view in dlg.findChildren(view_type):
            view.setEnabled(enabled)
    for button_box in dlg.findChildren(QtWidgets.QDialogButtonBox):
        for button in button_box.buttons():
            role = button_box.buttonRole(button)
            if role in (
                QtWidgets.QDialogButtonBox.AcceptRole,
                QtWidgets.QDialogButtonBox.YesRole,
                QtWidgets.QDialogButtonBox.ApplyRole,
            ):
                button.setEnabled(enabled)


def open_file_name(parent, title: str, directory: str = "", file_filter: str = "") -> tuple[str, str]:
    dlg = QtWidgets.QFileDialog(parent, title, "", file_filter)
    dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
    dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
    dlg.setSidebarUrls(_sidebar_urls(dlg.sidebarUrls()))
    dlg.setDirectory(_dialog_start_dir(directory))
    first_filter = _first_name_filter(file_filter)
    if first_filter:
        dlg.selectNameFilter(first_filter)
    _show_filtered_files_disabled(dlg)
    if dlg.exec() != QtWidgets.QDialog.Accepted:
        return "", ""
    files = dlg.selectedFiles()
    path = files[0] if files else ""
    if path:
        remember_dialog_path(path)
    return path, dlg.selectedNameFilter()


def save_file_name(parent, title: str, directory: str = "", file_filter: str = "") -> tuple[str, str]:
    dlg = QtWidgets.QFileDialog(parent, title, "", file_filter)
    dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
    dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
    dlg.setFileMode(QtWidgets.QFileDialog.AnyFile)
    dlg.setSidebarUrls(_sidebar_urls(dlg.sidebarUrls()))
    directory_path = os.path.abspath(os.path.expanduser(directory)) if directory else ""
    if directory_path and os.path.isdir(directory_path):
        start_dir = _dialog_start_dir(directory_path)
        default_name = ""
    else:
        start_dir = _dialog_start_dir(os.path.dirname(directory))
        default_name = os.path.basename(directory)
    dlg.setDirectory(start_dir)
    if default_name:
        dlg.selectFile(default_name)
    first_filter = _first_name_filter(file_filter)
    if first_filter:
        dlg.selectNameFilter(first_filter)
    _show_filtered_files_disabled(dlg)
    if dlg.exec() != QtWidgets.QDialog.Accepted:
        return "", ""
    files = dlg.selectedFiles()
    path = files[0] if files else ""
    selected_filter = dlg.selectedNameFilter()
    path = _append_implied_extension(path, selected_filter)
    if path:
        remember_dialog_path(path)
    return path, selected_filter


def existing_directory(parent, title: str, directory: str = "", show_files: bool = False) -> str:
    start_dir = _dialog_start_dir(directory)
    dlg = QtWidgets.QFileDialog(parent, title)
    dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
    dlg.setFileMode(QtWidgets.QFileDialog.Directory)
    dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, not show_files)
    if show_files:
        proxy = _DirectorySelectionProxy(dlg)
        dlg.setProxyModel(proxy)
        dlg._directory_selection_proxy = proxy
    dlg.setSidebarUrls(_sidebar_urls(dlg.sidebarUrls()))
    dlg.setDirectory(start_dir)
    dlg.selectFile(start_dir)
    _show_filtered_files_disabled(dlg)
    if show_files:
        _set_file_dialog_interaction_enabled(dlg, False)

        def _enable_initial_interaction() -> None:
            try:
                _set_file_dialog_interaction_enabled(dlg, True)
            except RuntimeError:
                pass

        QtCore.QTimer.singleShot(250, _enable_initial_interaction)
    if dlg.exec() != QtWidgets.QDialog.Accepted:
        return ""
    files = dlg.selectedFiles()
    path = files[0] if files else ""
    if path:
        remember_dialog_path(path)
    return path


def build_slist_directory(parent, title: str, directory: str = "") -> tuple[str, str]:
    folder = existing_directory(parent, title, directory, show_files=True)
    if not folder:
        return "", ""

    ext = build_slist_options(parent, folder)
    if ext is None:
        return "", ""
    return folder, ext


def attach_annots_directory(
    parent,
    title: str,
    ids: Iterable[str],
    directory: str = "",
) -> tuple[str, str, str]:
    folder = existing_directory(parent, title, directory, show_files=True)
    if not folder:
        return "", "", ""

    result = attach_annots_options(parent, folder, ids)
    if result is None:
        return "", "", ""
    ext, path_mode = result
    return folder, ext, path_mode


def build_slist_options(parent, folder: str) -> str | None:
    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle("Build Sample List")
    dlg.setModal(True)
    dlg.resize(980, 560)

    outer = QtWidgets.QVBoxLayout(dlg)
    folder_label = QtWidgets.QLabel(folder, dlg)
    folder_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    folder_label.setWordWrap(True)
    outer.addWidget(QtWidgets.QLabel("Selected folder:", dlg))
    outer.addWidget(folder_label)

    help_label = QtWidgets.QLabel(
        "Enter one or more comma-delimited annotation suffixes. "
        "For example, -nsrr.xml links file1.edf to file1-nsrr.xml.",
        dlg,
    )
    help_label.setWordWrap(True)
    outer.addWidget(help_label)

    form = QtWidgets.QFormLayout()
    ext_edit = QtWidgets.QLineEdit(dlg)
    inferred_exts = ", ".join(_infer_build_exts(folder))
    initial_exts = inferred_exts or str(_settings().value(_BUILD_EXT_KEY, _DEFAULT_BUILD_EXTS) or _DEFAULT_BUILD_EXTS)
    ext_edit.setPlaceholderText(inferred_exts or _DEFAULT_BUILD_EXTS)
    ext_edit.setText(initial_exts)
    form.addRow("Annotation suffixes (-ext, comma-delimited):", ext_edit)
    outer.addLayout(form)

    preview = QtWidgets.QLabel("Counting files...", dlg)
    outer.addWidget(preview)

    table = QtWidgets.QTableWidget(0, 3, dlg)
    table.setHorizontalHeaderLabels(["EDF/REC", "Linked annotation", "Status"])
    table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    table.setAlternatingRowColors(True)
    table.setMinimumWidth(900)
    table.setMinimumHeight(240)
    table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
    table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
    table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
    outer.addWidget(table, 1)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
        parent=dlg,
    )
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    outer.addWidget(buttons)

    signals = _BuildSListPreviewSignals(dlg)
    state = {"request_id": 0, "folder": "", "ext": ""}
    timer = QtCore.QTimer(dlg)
    timer.setSingleShot(True)
    timer.setInterval(350)

    def _set_preview(req_id: int, edfs: int, annots: int, rows: list, limited: bool) -> None:
        if req_id != state["request_id"]:
            return
        exts = normalize_build_exts(ext_edit.text())
        if exts:
            suffix = f" (showing first {_BUILD_PREVIEW_ROW_LIMIT})" if limited else ""
            preview.setText(f"{edfs} EDF/REC, {annots} matching annotations{suffix}")
        else:
            suffix = f" (showing first {_BUILD_PREVIEW_ROW_LIMIT})" if limited else ""
            preview.setText(f"{edfs} EDF/REC files{suffix}")
        _populate_build_preview_table(table, rows)

    signals.counted.connect(_set_preview)

    def _start_count() -> None:
        req_id = state["request_id"]
        count_folder = state["folder"]
        ext = state["ext"]

        def _worker() -> None:
            if req_id != state["request_id"]:
                return
            edfs, annots, rows, limited = _build_slist_link_preview(count_folder, ext)
            try:
                signals.counted.emit(req_id, edfs, annots, rows, limited)
            except RuntimeError:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    timer.timeout.connect(_start_count)

    def _queue_preview(path: str = "") -> None:
        state["folder"] = folder
        state["ext"] = normalize_build_ext(ext_edit.text())
        state["request_id"] += 1
        preview.setText("Counting files...")
        table.setRowCount(0)
        timer.start()

    def _update_preview() -> None:
        exts = normalize_build_exts(ext_edit.text())
        if exts:
            preview.setText("Counting files...")
        else:
            preview.setText("Counting EDF/REC files...")
        _queue_preview(folder)

    ext_edit.textChanged.connect(lambda _text: _update_preview())
    QtCore.QTimer.singleShot(0, _update_preview)

    if dlg.exec() != QtWidgets.QDialog.Accepted:
        state["request_id"] += 1
        return None

    ext = normalize_build_ext(ext_edit.text())
    try:
        s = _settings()
        s.setValue(_BUILD_EXT_KEY, ext)
        s.sync()
    except Exception:
        pass
    return ext


def attach_annots_options(parent, folder: str, ids: Iterable[str]) -> tuple[str, str] | None:
    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle("Attach Annotation Folder")
    dlg.setModal(True)
    dlg.resize(980, 560)

    outer = QtWidgets.QVBoxLayout(dlg)
    folder_label = QtWidgets.QLabel(folder, dlg)
    folder_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    folder_label.setWordWrap(True)
    outer.addWidget(QtWidgets.QLabel("Selected folder:", dlg))
    outer.addWidget(folder_label)

    help_label = QtWidgets.QLabel(
        "Match annotation files to the current sample-list IDs. "
        "Standard suffixes such as .annot, .eannot, and .xml work directly; "
        "variants such as -foo.xml can be added here too. "
        "Relative paths are stored against the sample-list file when available, "
        "otherwise against Lunascope's current working folder.",
        dlg,
    )
    help_label.setWordWrap(True)
    outer.addWidget(help_label)

    form = QtWidgets.QFormLayout()
    ext_edit = QtWidgets.QLineEdit(dlg)
    inferred_exts = ", ".join(_infer_build_exts(folder))
    initial_exts = str(_settings().value(_ATTACH_EXT_KEY, _DEFAULT_ATTACH_EXTS) or _DEFAULT_ATTACH_EXTS)
    ext_edit.setPlaceholderText(inferred_exts or _DEFAULT_ATTACH_EXTS)
    ext_edit.setText(initial_exts)
    form.addRow("Annotation suffixes (comma-delimited):", ext_edit)
    path_mode_combo = QtWidgets.QComboBox(dlg)
    path_mode_combo.addItem("Absolute paths", "absolute")
    path_mode_combo.addItem("Relative paths", "relative")
    saved_path_mode = str(_settings().value(_ATTACH_PATH_MODE_KEY, "absolute") or "absolute").strip().lower()
    idx = path_mode_combo.findData(saved_path_mode)
    path_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)
    form.addRow("Stored paths:", path_mode_combo)
    outer.addLayout(form)

    preview = QtWidgets.QLabel("Counting files...", dlg)
    outer.addWidget(preview)

    table = QtWidgets.QTableWidget(0, 3, dlg)
    table.setHorizontalHeaderLabels(["Sample ID", "Matching annotation", "Status"])
    table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    table.setAlternatingRowColors(True)
    table.setMinimumWidth(900)
    table.setMinimumHeight(240)
    table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
    table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
    table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
    outer.addWidget(table, 1)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
        parent=dlg,
    )
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    outer.addWidget(buttons)

    signals = _BuildSListPreviewSignals(dlg)
    state = {"request_id": 0, "folder": "", "ext": ""}
    timer = QtCore.QTimer(dlg)
    timer.setSingleShot(True)
    timer.setInterval(350)
    ids_list = [str(id_str or "").strip() for id_str in ids if str(id_str or "").strip()]

    def _set_preview(req_id: int, sample_count: int, annots: int, rows: list, limited: bool) -> None:
        if req_id != state["request_id"]:
            return
        exts = normalize_build_exts(ext_edit.text())
        suffix = f" (showing first {_BUILD_PREVIEW_ROW_LIMIT})" if limited else ""
        if exts:
            matched = sum(1 for _id, _annot, status in rows if str(status).startswith("Matched"))
            preview.setText(f"{matched}/{sample_count} sample IDs matched, {annots} candidate annotations{suffix}")
        else:
            preview.setText(f"{sample_count} sample IDs loaded{suffix}")
        _populate_build_preview_table(table, rows)

    signals.counted.connect(_set_preview)

    def _start_count() -> None:
        req_id = state["request_id"]
        count_folder = state["folder"]
        ext = state["ext"]

        def _worker() -> None:
            if req_id != state["request_id"]:
                return
            sample_count, annots, rows, limited = _build_attach_annots_preview(count_folder, ext, ids_list)
            try:
                signals.counted.emit(req_id, sample_count, annots, rows, limited)
            except RuntimeError:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    timer.timeout.connect(_start_count)

    def _queue_preview() -> None:
        state["folder"] = folder
        state["ext"] = normalize_build_ext(ext_edit.text())
        state["request_id"] += 1
        preview.setText("Counting files...")
        table.setRowCount(0)
        timer.start()

    def _update_preview() -> None:
        exts = normalize_build_exts(ext_edit.text())
        preview.setText("Counting files..." if exts else "Counting sample IDs...")
        _queue_preview()

    ext_edit.textChanged.connect(lambda _text: _update_preview())
    QtCore.QTimer.singleShot(0, _update_preview)

    if dlg.exec() != QtWidgets.QDialog.Accepted:
        state["request_id"] += 1
        return None

    ext = normalize_build_ext(ext_edit.text())
    try:
        s = _settings()
        s.setValue(_ATTACH_EXT_KEY, ext)
        s.setValue(_ATTACH_PATH_MODE_KEY, str(path_mode_combo.currentData() or "absolute"))
        s.sync()
    except Exception:
        pass
    return ext, str(path_mode_combo.currentData() or "absolute")
