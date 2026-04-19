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
from typing import Iterable

from PySide6 import QtCore, QtWidgets


_ORG = "Lunascope"
_APP = "Lunascope"
_RECENT_KEY = "file_dialogs/recent_dirs"
_MAX_RECENT_DIRS = 8


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
        if folder and folder not in seen:
            out.append(folder)
            seen.add(folder)
    return out


def _write_recent_dirs(paths: Iterable[str]) -> None:
    vals: list[str] = []
    seen: set[str] = set()
    for item in paths:
        folder = _normalize_dir(str(item))
        if folder and folder not in seen:
            vals.append(folder)
            seen.add(folder)
        if len(vals) >= _MAX_RECENT_DIRS:
            break
    try:
        _settings().setValue(_RECENT_KEY, vals)
    except Exception:
        pass


def remember_dialog_path(path: str) -> None:
    folder = _normalize_dir(path if os.path.isdir(path) else os.path.dirname(path))
    if not folder:
        return
    recent = _read_recent_dirs()
    _write_recent_dirs([folder, *recent])


def _sidebar_urls() -> list[QtCore.QUrl]:
    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in [_cwd(), os.path.expanduser("~"), *_read_recent_dirs()]:
        folder = _normalize_dir(candidate)
        if folder and folder not in seen:
            ordered.append(folder)
            seen.add(folder)
    return [QtCore.QUrl.fromLocalFile(folder) for folder in ordered]


def _dialog_start_path(default_name: str = "") -> str:
    cwd = _cwd()
    return os.path.join(cwd, default_name) if default_name else cwd


def open_file_name(parent, title: str, directory: str = "", file_filter: str = "") -> tuple[str, str]:
    dlg = QtWidgets.QFileDialog(parent, title, directory or _dialog_start_path(), file_filter)
    dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
    dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
    dlg.setSidebarUrls(_sidebar_urls())
    if dlg.exec() != QtWidgets.QDialog.Accepted:
        return "", ""
    files = dlg.selectedFiles()
    path = files[0] if files else ""
    if path:
        remember_dialog_path(path)
    return path, dlg.selectedNameFilter()


def save_file_name(parent, title: str, directory: str = "", file_filter: str = "") -> tuple[str, str]:
    dlg = QtWidgets.QFileDialog(parent, title, directory or _dialog_start_path(), file_filter)
    dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
    dlg.setFileMode(QtWidgets.QFileDialog.AnyFile)
    dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
    dlg.setSidebarUrls(_sidebar_urls())
    if dlg.exec() != QtWidgets.QDialog.Accepted:
        return "", ""
    files = dlg.selectedFiles()
    path = files[0] if files else ""
    if path:
        remember_dialog_path(path)
    return path, dlg.selectedNameFilter()


def existing_directory(parent, title: str, directory: str = "") -> str:
    dlg = QtWidgets.QFileDialog(parent, title, directory or _dialog_start_path())
    dlg.setFileMode(QtWidgets.QFileDialog.Directory)
    dlg.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
    dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
    dlg.setSidebarUrls(_sidebar_urls())
    if dlg.exec() != QtWidgets.QDialog.Accepted:
        return ""
    files = dlg.selectedFiles()
    path = files[0] if files else ""
    if path:
        remember_dialog_path(path)
    return path
