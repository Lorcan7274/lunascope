
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
#  along with Luna. If not, see <http://www.gnu.org/licenses/>.
#
#  Please see LICENSE.txt for more details.
#
#  --------------------------------------------------------------------

"""Moonbeam dock – NSRR data browser powered by lunapi.moonbeam."""

import io
import os
import pathlib
import re
import shutil
import sys
import tempfile
import time

from ..tls import configure_tls

configure_tls()

import lunapi as lp
import pandas as pd
from ..file_dialogs import existing_directory
from lunapi.moonbeam import _load_token as _mb_load_tok, _save_token as _mb_save_tok

from ..helpers import screen_clamp, is_dark_palette

from PySide6.QtCore import Qt, QObject, QRegularExpression, QSortFilterProxyModel, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableView,
    QTreeView,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Cache-dir persistence  (~/.config/lunapi/.cdir)
# ---------------------------------------------------------------------------

_CDIR_PATH = pathlib.Path.home() / '.config' / 'lunapi' / '.cdir'


def _save_cdir(path: str) -> None:
    """Persist the chosen cache directory across sessions."""
    _CDIR_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CDIR_PATH.write_text(path)


def _load_cdir() -> str | None:
    """Return the previously saved cache directory, or None."""
    try:
        p = _CDIR_PATH.read_text().strip()
        return p if p else None
    except Exception:
        return None


def _clear_cdir() -> None:
    """Remove the saved cache directory (revert to temp default)."""
    try:
        _CDIR_PATH.unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Stdout/stderr capture for download log
# ---------------------------------------------------------------------------

class _SignalStream(io.RawIOBase):
    """Wraps a Qt Signal so that Python print/tqdm output reaches the UI.

    Buffers until newline; for tqdm's \\r-overwrite lines takes only the
    text after the last \\r (the final update) and strips ANSI codes.
    """
    def __init__(self, signal):
        super().__init__()
        self._sig = signal
        self._buf = ""

    _ANSI = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')

    def write(self, text):
        text = self._ANSI.sub('', text)
        self._buf += text
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            # tqdm uses \r to overwrite; keep only the latest update
            if '\r' in line:
                line = line.rsplit('\r', 1)[1]
            line = line.strip()
            if line:
                self._sig.emit(line)
        return len(text)

    def flush(self):
        if self._buf.strip():
            line = self._buf.rsplit('\r', 1)[-1].strip()
            if line:
                self._sig.emit(line)
            self._buf = ""

    def fileno(self):
        raise io.UnsupportedOperation("fileno")  # tells tqdm: no ANSI


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class _MbConnectWorker(QObject):
    """Validates NSRR token and creates a moonbeam instance off the UI thread."""

    success = Signal(object)   # passes the moonbeam instance
    failure = Signal(str)

    def __init__(self, token, cdir=None):
        super().__init__()
        self._token = token  # may be None → use cached token
        self._cdir  = cdir   # may be None → moonbeam uses its default

    _MANIFEST_MAX_AGE_DAYS = 7

    @Slot()
    def run(self):
        try:
            token = self._token or _mb_load_tok()
            if token is None:
                raise ValueError(
                    "No NSRR token provided and none cached.\n"
                    "Paste a token or obtain one at https://sleepdata.org/token"
                )

            mb = lp.moonbeam.__new__(lp.moonbeam)
            mb.nsrr_tok = token
            mb._last_req = 0.0
            mb.df1 = pd.DataFrame()
            mb.df2 = None
            mb.curr_cohort = None
            mb.curr_subcohort = None
            mb.curr_id = None
            mb.curr_edf = None
            mb.curr_annots = []
            mb._allowed_cohort_slugs = None
            mb._mf = {}

            mb._verify_token()
            _mb_save_tok(token)

            cdir = self._cdir or os.path.join(tempfile.gettempdir(), 'luna-nsrr')
            mb.set_cache(cdir)
            mb._load_or_fetch_manifest()
        except Exception as exc:
            # Give a friendlier message for connectivity failures
            msg = str(exc)
            if "Could not reach" in msg or "ConnectionError" in msg or "Timeout" in msg:
                msg = (
                    "Could not reach sleepdata.org.\n"
                    "Check your internet connection and try again.\n\n"
                    f"Detail: {msg}"
                )
            self.failure.emit(msg)
            return

        # Refresh only when the cached manifest is older than the TTL.
        try:
            p = pathlib.Path(mb.cdir) / '.manifest'
            if p.exists():
                age_days = (time.time() - os.path.getmtime(str(p))) / 86400
                if age_days > self._MANIFEST_MAX_AGE_DAYS:
                    mb.refresh_manifest()   # gracefully no-ops if offline
        except Exception:
            pass  # never block connect for a manifest refresh failure

        self.success.emit(mb)


class _MbUpdateWorker(QObject):
    """Refresh the manifest and cohort-access metadata off the UI thread."""

    success = Signal(object, object)   # (manifest_dict, allowed_cohorts)
    failure = Signal(str)

    def __init__(self, mb):
        super().__init__()
        self._mb = mb

    @Slot()
    def run(self):
        try:
            self._mb.refresh_manifest()
            allowed = self._mb.allowed_cohorts(refresh=True)
        except Exception as exc:
            self.failure.emit(str(exc))
            return
        self.success.emit(getattr(self._mb, "_mf", None), allowed)


class _MbDownloadWorker(QObject):
    """Downloads a list of individuals, emitting per-individual progress."""

    progress  = Signal(int, int)   # (current_index, total)
    item_done = Signal(str, bool)  # (iid, success)
    log_msg   = Signal(str)        # one line of stdout/stderr output
    finished  = Signal()
    error     = Signal(str)        # non-fatal per-item errors

    def __init__(self, mb, iids, cohort, subcohort=None, force_redownload=False):
        super().__init__()
        self._mb        = mb
        self._iids      = list(iids)
        self._cohort    = cohort
        self._subcohort = subcohort
        self._force     = bool(force_redownload)
        self._stop      = False

    def stop(self):
        """Request cancellation between individuals (soft stop)."""
        self._stop = True

    @Slot()
    def run(self):
        total = len(self._iids)

        # Redirect Python stdout + stderr so tqdm/print output reaches the log
        _stream      = _SignalStream(self.log_msg)
        _old_stdout  = sys.stdout
        _old_stderr  = sys.stderr
        sys.stdout   = _stream
        sys.stderr   = _stream

        try:
            # Ensure the cohort context is set on the mb instance
            try:
                self._mb.cohort(self._cohort, self._subcohort)
            except Exception as exc:
                self.error.emit(f"Could not set cohort context: {exc}")
                return

            for i, iid in enumerate(self._iids):
                if self._stop:
                    break
                self.progress.emit(i, total)
                try:
                    if self._force:
                        self._remove_cached_files(iid)
                    self._mb.pull(iid, subcohort=self._subcohort)
                    self.item_done.emit(str(iid), True)
                except Exception as exc:
                    self.error.emit(f"{iid}: {exc}")
                    self.item_done.emit(str(iid), False)

            self.progress.emit(total, total)

        finally:
            sys.stdout = _old_stdout
            sys.stderr = _old_stderr
            self.finished.emit()

    def _remove_cached_files(self, iid):
        """Delete local cached files so moonbeam.pull() re-downloads them."""
        sc, _, info = self._mb._resolve_iid(iid, self._subcohort)
        rel_paths = [info['edf'], *info['annots']]
        if re.search(r'\.(edf\.gz|edfz)$', info['edf'], re.IGNORECASE):
            rel_paths.append(info['edf'] + '.idx')

        removed_any = False
        for rel_path in rel_paths:
            local = pathlib.Path(self._mb._local_path(self._cohort, rel_path.lstrip('/')))
            try:
                if local.exists():
                    local.unlink()
                    removed_any = True
            except Exception as exc:
                raise RuntimeError(f"could not remove cached file {local}: {exc}") from exc

        if removed_any:
            self.log_msg.emit(f"[force] cleared cached files for {iid} [{sc}]")


class _MbCopyWorker(QObject):
    """Copies cached files one-by-one so progress is visible and copy can be cancelled."""

    log_msg  = Signal(str)
    progress = Signal(int, int)   # (files_done, files_total)
    finished = Signal()

    def __init__(self, src: str, dst: str):
        super().__init__()
        self._src  = pathlib.Path(src)
        self._dst  = pathlib.Path(dst)
        self._stop = False

    def stop(self):
        self._stop = True

    @Slot()
    def run(self):
        try:
            # Collect every file to copy upfront so we can show N/total
            all_files = sorted(
                f for f in self._src.rglob('*') if f.is_file()
            )
            total = len(all_files)
            done  = 0
            errors = 0

            self._dst.mkdir(parents=True, exist_ok=True)

            for src_f in all_files:
                if self._stop:
                    self.log_msg.emit("Copy cancelled.")
                    break
                rel    = src_f.relative_to(self._src)
                dst_f  = self._dst / rel
                size   = src_f.stat().st_size
                self.log_msg.emit(f"{rel}  ({_fmt_size(size)})")
                try:
                    dst_f.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src_f), str(dst_f))
                except Exception as exc:
                    self.log_msg.emit(f"  ERROR: {exc}")
                    errors += 1
                done += 1
                self.progress.emit(done, total)

            if not self._stop:
                total_size = sum(
                    (self._dst / f.relative_to(self._src)).stat().st_size
                    for f in all_files
                    if (self._dst / f.relative_to(self._src)).exists()
                )
                self.log_msg.emit(
                    f"Done — {done} file(s) copied, {errors} error(s)."
                    f"  Total: {_fmt_size(total_size)}"
                )
        except Exception as exc:
            self.log_msg.emit(f"Copy failed: {exc}")
        finally:
            self.finished.emit()


def _fmt_size(nbytes: int) -> str:
    nbytes = int(nbytes or 0)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes //= 1024
    return f"{nbytes:.1f} PB"


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class MoonbeamMixin:
    """Adds the Moonbeam floating dock to the Lunascope controller."""

    _MB_FLOAT_SIZE = (1060, 660)

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def _init_moonbeam(self):
        self._mb              = None   # live moonbeam instance
        self._mb_mf           = None   # manifest dict (offline or live)
        self._mb_accessible_cohorts = None
        self._mb_curr_cohort  = None
        self._mb_curr_subcohort = None
        self._mb_df2          = None   # current individual manifest DataFrame
        self._mb_thread       = None
        self._mb_connect_worker = None
        self._mb_update_worker = None
        self._mb_dl_worker    = None
        self._mb_pending_cdir = None   # cdir waiting for copy to finish
        self._mb_tree_sized   = False

        # ----------------------------------------------------------------
        # Build the dock widget
        # ----------------------------------------------------------------
        dock = QDockWidget("Moonbeam (NSRR)", self.ui)
        dock.setObjectName("dock_moonbeam")
        dock.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea | Qt.BottomDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )
        dock.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        dock.setWindowFlag(Qt.WindowMaximizeButtonHint, True)

        root = QWidget(dock)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # ---- Token / connect row ----------------------------------------
        tok_frame = QFrame(root)
        tok_layout = QHBoxLayout(tok_frame)
        tok_layout.setContentsMargins(0, 0, 0, 0)
        tok_layout.setSpacing(6)

        tok_lbl  = QLabel("NSRR Token:")
        tok_edit = QLineEdit()
        tok_edit.setObjectName("mb_token_edit")
        tok_edit.setEchoMode(QLineEdit.Password)
        # Check whether a token is already cached and update the placeholder
        try:
            from lunapi.moonbeam import _load_token as _mb_load_tok
            _has_cached = _mb_load_tok() is not None
        except Exception:
            _has_cached = False
        tok_edit.setPlaceholderText(
            "Cached token found — press Connect (or paste a new token)"
            if _has_cached else
            "Paste NSRR token here (obtain at sleepdata.org/token)"
        )
        tok_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        tok_edit.returnPressed.connect(self._mb_connect)

        connect_btn = QPushButton("Connect")
        connect_btn.setObjectName("mb_connect_btn")
        connect_btn.setFixedWidth(90)

        update_btn = QPushButton("Update cohort permissions")
        update_btn.setObjectName("mb_update_btn")
        update_btn.setToolTip("Refresh cohort permissions and access counts")
        update_btn.setEnabled(False)

        status_lbl = QLabel("Token cached — press Connect" if _has_cached else "Not connected")
        status_lbl.setObjectName("mb_status_lbl")
        status_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        tok_layout.addWidget(tok_lbl)
        tok_layout.addWidget(tok_edit)
        tok_layout.addWidget(connect_btn)
        tok_layout.addWidget(status_lbl, 1)
        tok_layout.addWidget(update_btn)
        outer.addWidget(tok_frame)

        # ---- Cache dir row ----------------------------------------------
        cdir_frame = QFrame(root)
        cdir_layout = QHBoxLayout(cdir_frame)
        cdir_layout.setContentsMargins(0, 0, 0, 0)
        cdir_layout.setSpacing(6)

        cdir_lbl = QLabel("Cache dir:")
        _default_cdir = _load_cdir() or os.path.join(tempfile.gettempdir(), 'luna-nsrr')
        cdir_edit = QLineEdit(_default_cdir)
        cdir_edit.setObjectName("mb_cdir_edit")
        cdir_edit.setReadOnly(True)
        cdir_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        cdir_edit.setToolTip("Directory where downloaded EDF/annotation files are cached")

        cdir_browse_btn = QPushButton("Browse…")
        cdir_browse_btn.setObjectName("mb_cdir_browse_btn")
        cdir_browse_btn.setFixedWidth(80)
        cdir_browse_btn.setToolTip("Choose a persistent folder for the cache")

        cdir_temp_btn = QPushButton("Use Temp")
        cdir_temp_btn.setObjectName("mb_cdir_temp_btn")
        cdir_temp_btn.setFixedWidth(80)
        cdir_temp_btn.setToolTip("Reset to the system temp folder (default, may be purged by OS)")

        cdir_layout.addWidget(cdir_lbl)
        cdir_layout.addWidget(cdir_edit)
        cdir_layout.addWidget(cdir_browse_btn)
        cdir_layout.addWidget(cdir_temp_btn)
        outer.addWidget(cdir_frame)

        # ---- Summary bar ------------------------------------------------
        summ_frame = QFrame(root)
        summ_layout = QHBoxLayout(summ_frame)
        summ_layout.setContentsMargins(0, 0, 0, 0)
        summ_layout.setSpacing(8)

        def _stat(txt, min_w=100):
            lab = QLabel(txt)
            lab.setFrameStyle(QFrame.Panel | QFrame.Sunken)
            lab.setMinimumWidth(min_w)
            lab.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            return lab

        lbl_ncohorts = _stat("Studies: -")
        lbl_ncohorts.setObjectName("mb_lbl_ncohorts")
        lbl_n = _stat("Individuals: -", 120)
        lbl_n.setObjectName("mb_lbl_n")
        lbl_naccess = _stat("Accessible: -", 110)
        lbl_naccess.setObjectName("mb_lbl_naccess")
        lbl_ncached = _stat("Downloaded: -", 110)
        lbl_ncached.setObjectName("mb_lbl_ncached")
        lbl_cdir = QLabel("Cache Folder: -")
        lbl_cdir.setObjectName("mb_lbl_cdir")
        lbl_cdir.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        lbl_cdir.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        for w in (lbl_ncohorts, lbl_n, lbl_naccess, lbl_ncached, lbl_cdir):
            summ_layout.addWidget(w)
        outer.addWidget(summ_frame)

        # ---- Main splitter: tree (left) + table (right) -----------------
        splitter = QSplitter(Qt.Horizontal, root)

        # Left: cohort / subcohort tree
        tree = QTreeView(splitter)
        tree.setObjectName("mb_tree")
        tree.setHeaderHidden(False)
        tree.setSelectionMode(QAbstractItemView.SingleSelection)
        tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        tree.setMinimumWidth(260)
        tree.setMaximumWidth(760)
        tree.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        tree.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Right: filter bar + individuals table
        right_frame = QFrame(splitter)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(3)

        flt_bar = QFrame(right_frame)
        flt_bar_layout = QHBoxLayout(flt_bar)
        flt_bar_layout.setContentsMargins(0, 2, 0, 2)
        flt_bar_layout.setSpacing(4)
        flt_lbl = QLabel("Filter ID:")
        flt_edit = QLineEdit()
        flt_edit.setObjectName("mb_flt")
        flt_edit.setPlaceholderText("comma-separated IDs…")
        flt_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        flt_clear = QPushButton("✕")
        flt_clear.setObjectName("mb_flt_clear")
        flt_clear.setFixedWidth(26)
        flt_clear.setToolTip("Clear filter")
        flt_clear.clicked.connect(flt_edit.clear)
        flt_bar_layout.addWidget(flt_lbl)
        flt_bar_layout.addWidget(flt_edit)
        flt_bar_layout.addWidget(flt_clear)
        right_layout.addWidget(flt_bar)

        table = QTableView(right_frame)
        table.setObjectName("mb_table")
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        table.setSortingEnabled(True)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.horizontalHeader().setStretchLastSection(False)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Persistent proxy – filters on column 1 (ID); source model swapped in
        # each time _mb_populate_table is called.
        mb_proxy = QSortFilterProxyModel(table)
        mb_proxy.setFilterKeyColumn(1)
        mb_proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        table.setModel(mb_proxy)

        def _on_flt_changed(text):
            parts = [s.strip() for s in text.split(',') if s.strip()]
            if not parts:
                mb_proxy.setFilterRegularExpression(QRegularExpression())
                return
            esc = [QRegularExpression.escape(p) for p in parts]
            rx = QRegularExpression("(" + "|".join(esc) + ")",
                                    QRegularExpression.CaseInsensitiveOption)
            mb_proxy.setFilterRegularExpression(rx)

        flt_edit.textChanged.connect(_on_flt_changed)
        right_layout.addWidget(table)

        splitter.addWidget(tree)
        splitter.addWidget(right_frame)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 640])
        outer.addWidget(splitter, 1)

        # ---- Download log -----------------------------------------------
        log_txt = QTextEdit(root)
        log_txt.setObjectName("mb_log")
        log_txt.setReadOnly(True)
        log_txt.setFixedHeight(88)
        log_font = QFont("Courier New")
        log_font.setPointSize(9)
        log_txt.setFont(log_font)
        log_txt.setLineWrapMode(QTextEdit.NoWrap)
        log_txt.setPlaceholderText("Download log…")
        outer.addWidget(log_txt)

        # ---- Bottom bar: action buttons + progress ----------------------
        bot_frame = QFrame(root)
        bot_layout = QHBoxLayout(bot_frame)
        bot_layout.setContentsMargins(0, 0, 0, 0)
        bot_layout.setSpacing(6)

        dl_sel_btn  = QPushButton("Download Selected")
        dl_sel_btn.setObjectName("mb_dl_sel_btn")
        force_redownload_chk = QCheckBox("Force redownload")
        force_redownload_chk.setObjectName("mb_force_redownload_chk")
        force_redownload_chk.setToolTip(
            "Default: if a file already exists in the cache, it is not downloaded again. "
            "Enable this to replace partial or corrupted cached files."
        )
        cancel_btn  = QPushButton("Cancel")
        cancel_btn.setObjectName("mb_cancel_btn")
        cancel_btn.setVisible(False)
        pop_slist_btn = QPushButton("S-List: Cached View")
        pop_slist_btn.setObjectName("mb_pop_slist_btn")
        pop_sel_slist_btn = QPushButton("S-List: Selected")
        pop_sel_slist_btn.setObjectName("mb_pop_sel_slist_btn")

        progress     = QProgressBar()
        progress.setObjectName("mb_progress")
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setVisible(False)
        progress.setFixedHeight(18)
        progress.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        progress_lbl = QLabel("")
        progress_lbl.setObjectName("mb_progress_lbl")
        progress_lbl.setMinimumWidth(70)

        bot_layout.addWidget(dl_sel_btn)
        bot_layout.addWidget(force_redownload_chk)
        bot_layout.addWidget(cancel_btn)
        bot_layout.addStretch(1)
        bot_layout.addWidget(progress_lbl)
        bot_layout.addWidget(progress)
        bot_layout.addWidget(pop_sel_slist_btn)
        bot_layout.addWidget(pop_slist_btn)
        outer.addWidget(bot_frame)

        # ---- Attach dock to main window ---------------------------------
        dock.setWidget(root)
        self.ui.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.setFloating(True)
        _mb_w, _mb_h = screen_clamp(*self._MB_FLOAT_SIZE)
        dock.resize(_mb_w, _mb_h)
        # Position near top-centre of the main window
        parent_geo = self.ui.frameGeometry()
        dock.move(
            parent_geo.x() + (parent_geo.width()  - _mb_w) // 2,
            parent_geo.y() + 60,
        )
        dock.hide()

        # ---- Store references on ui object (actigraphy-style) -----------
        self.ui.dock_moonbeam    = dock
        self.ui.mb_token_edit    = tok_edit
        self.ui.mb_connect_btn   = connect_btn
        self.ui.mb_update_btn    = update_btn
        self.ui.mb_status_lbl    = status_lbl
        self.ui.mb_tree          = tree
        self.ui.mb_table         = table
        self.ui.mb_flt           = flt_edit
        self.ui.mb_cdir_edit     = cdir_edit
        self.ui.mb_cdir_browse_btn = cdir_browse_btn
        self.ui.mb_cdir_temp_btn   = cdir_temp_btn
        self._mb_proxy           = mb_proxy
        self._mb_cdir            = _default_cdir   # tracks user's chosen path
        self.ui.mb_lbl_ncohorts  = lbl_ncohorts
        self.ui.mb_lbl_n         = lbl_n
        self.ui.mb_lbl_naccess   = lbl_naccess
        self.ui.mb_lbl_ncached   = lbl_ncached
        self.ui.mb_lbl_cdir      = lbl_cdir
        self.ui.mb_dl_sel_btn    = dl_sel_btn
        self.ui.mb_force_redownload_chk = force_redownload_chk
        self.ui.mb_cancel_btn    = cancel_btn
        self.ui.mb_pop_slist_btn = pop_slist_btn
        self.ui.mb_pop_sel_slist_btn = pop_sel_slist_btn
        self.ui.mb_progress      = progress
        self.ui.mb_progress_lbl  = progress_lbl
        self.ui.mb_log           = log_txt

        # ---- Wire signals -----------------------------------------------
        connect_btn.clicked.connect(self._mb_connect)
        update_btn.clicked.connect(self._mb_update)
        cdir_browse_btn.clicked.connect(self._mb_browse_cache)
        cdir_temp_btn.clicked.connect(self._mb_use_temp_cache)
        tree.clicked.connect(self._mb_on_tree_click)
        dl_sel_btn.clicked.connect(self._mb_download_selected)
        cancel_btn.clicked.connect(self._mb_cancel_download)
        pop_sel_slist_btn.clicked.connect(self._mb_populate_slist_selected)
        pop_slist_btn.clicked.connect(self._mb_populate_slist)

        # ---- Apply dark-panel styling (only when OS is using a dark theme) ----
        if is_dark_palette():
            root.setStyleSheet("""
                QFrame {
                    color: #d7e3f4;
                }
                QLabel {
                    color: #d7e3f4;
                }
            """)

        self._mb_set_action_enabled(False)
        # Populate from cache immediately — works offline
        self._mb_load_offline()

    # -----------------------------------------------------------------------
    # Thread lifecycle helpers
    # -----------------------------------------------------------------------

    def _mb_clear_thread(self):
        """Called via finished signal so self._mb_thread is never a stale C++ object."""
        self._mb_thread    = None
        self._mb_update_worker = None
        self._mb_dl_worker = None
        self._mb_set_action_enabled(self._mb is not None)

    def _mb_thread_running(self) -> bool:
        return self._mb_thread is not None and self._mb_thread.isRunning()

    # -----------------------------------------------------------------------
    # Helper: enable / disable action buttons
    # -----------------------------------------------------------------------

    def _mb_set_action_enabled(self, enabled: bool):
        """Gate download-only buttons on live connection; others work offline."""
        for name in ("mb_dl_sel_btn", "mb_force_redownload_chk"):
            w = getattr(self.ui, name, None)
            if w:
                w.setEnabled(enabled)
        w = getattr(self.ui, "mb_update_btn", None)
        if w:
            w.setEnabled(enabled and not self._mb_thread_running())
        has_data = self._mb_mf is not None
        slist_btn = getattr(self.ui, "mb_pop_slist_btn", None)
        if slist_btn:
            slist_btn.setEnabled(has_data)
        sel_slist_btn = getattr(self.ui, "mb_pop_sel_slist_btn", None)
        if sel_slist_btn:
            sel_slist_btn.setEnabled(has_data)

    # -----------------------------------------------------------------------
    # Offline manifest helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _mb_parse_manifest(text: str) -> dict:
        """Parse the TSV manifest into the same nested dict used by moonbeam."""
        mf = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) != 5:
                continue
            cohort, subcohort, iid, edf, annots_str = parts
            annots = [] if annots_str == '.' else annots_str.split(',')
            (mf.setdefault(cohort, {})
               .setdefault(subcohort, {}))[iid] = {'edf': edf, 'annots': annots}
        return mf

    def _mb_load_offline(self):
        """Load the cached manifest (if present) without a network connection."""
        manifest = pathlib.Path(self._mb_cdir) / '.manifest'
        if not manifest.exists():
            return
        try:
            mf = self._mb_parse_manifest(manifest.read_text())
        except Exception:
            return
        if not mf:
            return
        self._mb_mf = mf
        self._mb_populate_tree()
        self._mb_set_action_enabled(False)   # download still locked; S-List/Refresh now enabled
        if self._mb is None:
            self.ui.mb_status_lbl.setText("Cached data loaded — connect to download")

    def _mb_cached(self, cohort: str, edf_path: str) -> bool:
        """Check whether a file is on disk; works with or without a live connection."""
        if self._mb is not None:
            return self._mb.cached(f"{cohort}/{edf_path}")
        return (pathlib.Path(self._mb_cdir) / cohort / edf_path.lstrip('/')).exists()

    def _mb_get_cohort_df(self, cohort: str, subcohort=None):
        """Return individuals DataFrame; works with or without a live connection."""
        if self._mb is not None:
            return self._mb.cohort(cohort, subcohort)
        mf = self._mb_mf
        if not mf or cohort not in mf:
            return pd.DataFrame(columns=['Subcohort', 'ID', 'EDF', 'Annot'])
        rows = []
        for sc, subjects in mf[cohort].items():
            if subcohort and sc != subcohort:
                continue
            for iid, info in subjects.items():
                first_annot = info['annots'][0] if info['annots'] else '.'
                rows.append({'Subcohort': sc, 'ID': iid,
                             'EDF': info['edf'], 'Annot': first_annot})
        return pd.DataFrame(rows, columns=['Subcohort', 'ID', 'EDF', 'Annot'])

    # -----------------------------------------------------------------------
    # Connect / authenticate
    # -----------------------------------------------------------------------

    def _mb_connect(self):
        # Guard against double-connect while a thread is running
        if self._mb_thread_running():
            return

        token = self.ui.mb_token_edit.text().strip() or None
        self.ui.mb_status_lbl.setText("Connecting…")
        self.ui.mb_connect_btn.setEnabled(False)

        thread = QThread(self)
        worker = _MbConnectWorker(token, cdir=self._mb_cdir or None)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.success.connect(self._mb_on_connect_success)
        worker.failure.connect(self._mb_on_connect_failure)
        worker.success.connect(thread.quit)
        worker.failure.connect(thread.quit)
        worker.success.connect(worker.deleteLater)
        worker.failure.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._mb_clear_thread)

        self._mb_thread         = thread
        self._mb_connect_worker = worker
        thread.start()

    @staticmethod
    def _mb_truthy(value) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        txt = str(value).strip().lower()
        return txt in {"1", "true", "t", "yes", "y", "access", "accessible"}

    def _mb_get_accessible_cohorts(self, df1=None):
        """Derive the accessible cohort slugs from moonbeam.cohorts()."""
        if df1 is None and self._mb is not None:
            df1 = getattr(self._mb, "df1", None)
        if df1 is None or getattr(df1, "empty", True) or "Cohort" not in df1.columns:
            return None

        for col in ("Accessible", "Access", "HasAccess", "Authorized"):
            if col in df1.columns:
                return {
                    str(row["Cohort"])
                    for _, row in df1.iterrows()
                    if self._mb_truthy(row[col])
                }

        if self._mb is not None and hasattr(self._mb, "allowed_cohorts"):
            try:
                return {str(x) for x in self._mb.allowed_cohorts()}
            except Exception:
                return None

        return None

    def _mb_on_connect_success(self, mb):
        self._mb = mb
        # Merge live manifest into self._mb_mf
        live_mf = getattr(mb, '_mf', None)
        if live_mf:
            self._mb_mf = live_mf
        self._mb_accessible_cohorts = self._mb_get_accessible_cohorts(
            getattr(mb, "df1", None)
        )
        self.ui.mb_connect_btn.setEnabled(True)
        self.ui.mb_status_lbl.setText("Connected")
        self._mb_populate_tree()
        self._mb_set_action_enabled(True)

    def _mb_on_connect_failure(self, msg):
        self._mb = None
        self._mb_accessible_cohorts = None
        self.ui.mb_connect_btn.setEnabled(True)
        self._mb_set_action_enabled(False)
        self.ui.mb_status_lbl.setText(f"Error: {msg[:100]}")
        QMessageBox.critical(self.ui, "Moonbeam – Connection Error", msg)

    def _mb_start_update(self):
        if self._mb is None or self._mb_thread_running():
            return

        thread = QThread(self)
        worker = _MbUpdateWorker(self._mb)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.success.connect(self._mb_on_update_success)
        worker.failure.connect(self._mb_on_update_failure)
        worker.success.connect(thread.quit)
        worker.failure.connect(thread.quit)
        worker.success.connect(worker.deleteLater)
        worker.failure.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._mb_clear_thread)

        self._mb_thread = thread
        self._mb_update_worker = worker
        self._mb_set_action_enabled(False)
        self.ui.mb_connect_btn.setEnabled(False)
        self.ui.mb_status_lbl.setText("Updating studies…")
        thread.start()

    def _mb_on_update_success(self, manifest, allowed):
        if manifest:
            self._mb_mf = manifest
        self._mb_accessible_cohorts = {str(x) for x in allowed} if allowed is not None else None
        if self._mb is not None:
            self._mb.df1 = self._mb.cohorts()
        self.ui.mb_status_lbl.setText("Connected")
        self._mb_populate_tree()
        if self._mb_curr_cohort:
            self._mb_populate_table(self._mb_curr_cohort, self._mb_curr_subcohort)
        self._mb_set_action_enabled(True)
        self.ui.mb_connect_btn.setEnabled(True)

    def _mb_on_update_failure(self, msg):
        self._mb_accessible_cohorts = self._mb_get_accessible_cohorts(
            getattr(self._mb, "df1", None) if self._mb is not None else None
        )
        self.ui.mb_status_lbl.setText("Connected")
        self.ui.mb_log.append(f"Update failed: {msg}")
        self._mb_set_action_enabled(self._mb is not None)
        self.ui.mb_connect_btn.setEnabled(True)

    def _mb_update(self):
        if self._mb is None:
            return
        self._mb_start_update()

    # -----------------------------------------------------------------------
    # Cache directory controls
    # -----------------------------------------------------------------------

    def _mb_browse_cache(self):
        folder = existing_directory(
            self.ui, "Choose cache directory",
            self._mb_cdir,
        )
        if folder:
            self._mb_apply_cache_dir(folder)

    def _mb_use_temp_cache(self):
        _clear_cdir()   # forget any saved persistent path
        default = os.path.join(tempfile.gettempdir(), 'luna-nsrr')
        self._mb_apply_cache_dir(default)

    def _mb_apply_cache_dir(self, new_path: str):
        new_path = str(new_path)
        old_path = self._mb_cdir

        if new_path == old_path:
            return

        if self._mb_thread_running():
            QMessageBox.information(
                self.ui, "Moonbeam",
                "A download or copy is already in progress — please wait."
            )
            return

        # Collect real file paths (not names) in old location, excluding hidden
        old_files = []
        if old_path and os.path.isdir(old_path):
            old_files = [
                f for f in pathlib.Path(old_path).rglob('*')
                if f.is_file() and not f.name.startswith('.')
            ]

        if old_files:
            total_size = sum(f.stat().st_size for f in old_files)
            box = QMessageBox(self.ui)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle("Copy cached files before switching?")
            box.setText(
                f"Found {len(old_files):,} cached file(s) "
                f"({_fmt_size(total_size)}) in:\n  {old_path}\n\n"
                f"Copy them to the new cache location?\n  {new_path}"
            )
            box.setInformativeText(
                "The original files stay in place. "
                "Press Return to copy first, then switch when the copy finishes."
            )
            copy_btn = box.addButton("Copy Then Switch", QMessageBox.AcceptRole)
            switch_btn = box.addButton("Switch Without Copying", QMessageBox.DestructiveRole)
            cancel_btn = box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(copy_btn)
            box.setEscapeButton(cancel_btn)
            box.exec()
            clicked = box.clickedButton()
            if clicked is cancel_btn:
                return
            if clicked is copy_btn:
                # Defer the actual cdir switch until copy finishes
                self._mb_pending_cdir = new_path
                self.ui.mb_cdir_edit.setText(f"{new_path}  ← copying…")
                self._mb_start_copy(old_path, new_path)
                return

        # No copy needed — switch immediately
        self._mb_cdir = new_path
        self.ui.mb_cdir_edit.setText(new_path)
        _save_cdir(new_path)
        if self._mb is not None:
            self._mb.set_cache(new_path)
        # Try loading manifest from new dir; if absent, keep existing manifest
        # but repopulate tree so cached-counts reflect the new directory
        new_manifest = pathlib.Path(new_path) / '.manifest'
        if new_manifest.exists():
            self._mb_load_offline()
        elif self._mb_mf is not None:
            self._mb_populate_tree()
        self.ui.mb_log.append(f"Cache dir → {new_path}")

    def _mb_start_copy(self, src: str, dst: str):
        self.ui.mb_log.clear()
        self.ui.mb_log.append(f"Copying cache:\n  {src}\n→ {dst}\n")
        self.ui.mb_progress.setRange(0, 100)
        self.ui.mb_progress.setValue(0)
        self.ui.mb_progress.setVisible(True)
        self.ui.mb_progress_lbl.setText("0 / ?")
        self.ui.mb_cancel_btn.setVisible(True)
        self._mb_set_action_enabled(False)
        self.ui.mb_connect_btn.setEnabled(False)
        self.ui.mb_cdir_browse_btn.setEnabled(False)
        self.ui.mb_cdir_temp_btn.setEnabled(False)

        thread = QThread(self)
        worker = _MbCopyWorker(src, dst)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log_msg.connect(self._mb_on_log_msg)
        worker.progress.connect(self._mb_on_copy_progress)
        worker.finished.connect(self._mb_on_copy_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._mb_clear_thread)
        self._mb_thread  = thread
        self._mb_dl_worker = worker   # reuse cancel slot
        thread.start()

    def _mb_on_copy_progress(self, done, total):
        if total > 0:
            self.ui.mb_progress.setRange(0, total)
            self.ui.mb_progress.setValue(done)
        self.ui.mb_progress_lbl.setText(f"{done} / {total}")

    def _mb_on_copy_finished(self):
        self.ui.mb_progress.setVisible(False)
        self.ui.mb_cancel_btn.setVisible(False)
        self.ui.mb_progress_lbl.setText("")
        self.ui.mb_cdir_browse_btn.setEnabled(True)
        self.ui.mb_cdir_temp_btn.setEnabled(True)
        self.ui.mb_connect_btn.setEnabled(True)

        # Now apply the deferred cdir switch
        pending = getattr(self, '_mb_pending_cdir', None)
        if pending:
            self._mb_cdir = pending
            self.ui.mb_cdir_edit.setText(pending)
            _save_cdir(pending)
            self._mb_pending_cdir = None
            if self._mb is not None:
                self._mb.set_cache(pending)

        self._mb_set_action_enabled(self._mb is not None)
        if self._mb is not None:
            self._mb_update()
        elif self._mb_mf is not None:
            self._mb_populate_tree()   # refresh cached counts in new dir

    # -----------------------------------------------------------------------
    # Populate cohort/subcohort tree
    # -----------------------------------------------------------------------

    def _mb_populate_tree(self):
        mf = self._mb_mf
        if not mf:
            return

        cdir_path = pathlib.Path(self._mb_cdir)
        accessible = self._mb_accessible_cohorts
        access_known = accessible is not None

        total_n = sum(
            len(subjects)
            for cohort_data in mf.values()
            for subjects in cohort_data.values()
        )
        total_access = sum(
            len(subjects)
            for cohort, cohort_data in mf.items()
            if access_known and cohort in accessible
            for subjects in cohort_data.values()
        )
        total_cached = sum(
            1
            for cohort, cohort_data in mf.items()
            for sc, subjects in cohort_data.items()
            for iid, info in subjects.items()
            if (cdir_path / cohort / info['edf'].lstrip('/')).exists()
        )

        if access_known:
            n_accessible_cohorts = sum(1 for cohort in mf if cohort in accessible)
            self.ui.mb_lbl_ncohorts.setText(
                f"Studies: {len(mf)} ({n_accessible_cohorts:,} accessible)"
            )
            self.ui.mb_lbl_naccess.setText(f"Accessible: {total_access:,}")
        else:
            self.ui.mb_lbl_ncohorts.setText(f"Studies: {len(mf)}")
            self.ui.mb_lbl_naccess.setText("Accessible: ?")
        self.ui.mb_lbl_n.setText(f"Individuals: {total_n:,}")
        self.ui.mb_lbl_ncached.setText(f"Downloaded: {total_cached:,}")
        self.ui.mb_lbl_cdir.setText(f"Cache Folder: {self._mb_cdir}")

        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["Study / Subcohort", "Available", "Access", "Cached"])

        for cohort, subcohort_data in mf.items():
            n_cohort = sum(len(s) for s in subcohort_data.values())
            access_cohort = n_cohort if access_known and cohort in accessible else 0
            cached_cohort = sum(
                1
                for sc, subjects in subcohort_data.items()
                for iid, info in subjects.items()
                if (cdir_path / cohort / info['edf'].lstrip('/')).exists()
            )

            if access_known:
                marker = "[access]" if cohort in accessible else "[no access]"
            else:
                marker = "[access ?]"
            coh_item = QStandardItem(f"{cohort}  {marker}")
            coh_item.setData(('cohort', cohort, None), Qt.UserRole)
            coh_item.setEditable(False)

            for sc, subjects in subcohort_data.items():
                sc_n = len(subjects)
                sc_access = sc_n if access_known and cohort in accessible else 0
                sc_cached = sum(
                    1 for info in subjects.values()
                    if (cdir_path / cohort / info['edf'].lstrip('/')).exists()
                )
                sc_item = QStandardItem(f"  {sc}")
                sc_item.setData(('subcohort', cohort, sc), Qt.UserRole)
                sc_item.setEditable(False)
                coh_item.appendRow([
                    sc_item,
                    _right_item(str(sc_n)),
                    _right_item(str(sc_access) if access_known else "?"),
                    _right_item(str(sc_cached)),
                ])

            model.appendRow([
                coh_item,
                _right_item(str(n_cohort)),
                _right_item(str(access_cohort) if access_known else "?"),
                _right_item(str(cached_cohort)),
            ])

        tree = self.ui.mb_tree
        old_sel = tree.selectionModel()
        if old_sel is not None:
            try:
                old_sel.currentChanged.disconnect(self._mb_on_tree_current_changed)
            except Exception:
                pass
        tree.setModel(model)
        tree.selectionModel().currentChanged.connect(self._mb_on_tree_current_changed)
        h = tree.header()
        h.setSectionResizeMode(0, QHeaderView.Interactive)
        h.setSectionResizeMode(1, QHeaderView.Fixed)
        h.setSectionResizeMode(2, QHeaderView.Fixed)
        h.setSectionResizeMode(3, QHeaderView.Fixed)
        tree.expandAll()
        if not self._mb_tree_sized:
            tree.resizeColumnToContents(0)
            tree.setColumnWidth(0, min(max(tree.columnWidth(0) + 18, 170), 360))
            tree.setColumnWidth(1, 74)
            tree.setColumnWidth(2, 74)
            tree.setColumnWidth(3, 68)
            total_w = sum(tree.columnWidth(col) for col in range(model.columnCount()))
            total_w += tree.frameWidth() * 2 + 28
            if tree.verticalScrollBar().isVisible():
                total_w += tree.verticalScrollBar().sizeHint().width()
            tree.setMinimumWidth(min(max(total_w, 260), 760))
            self._mb_tree_sized = True

    # -----------------------------------------------------------------------
    # Tree click → populate right table
    # -----------------------------------------------------------------------

    def _mb_on_tree_current_changed(self, current, previous):
        del previous
        if current and current.isValid():
            self._mb_on_tree_click(current)

    def _mb_on_tree_click(self, index):
        if self._mb_mf is None:
            return
        item = self.ui.mb_tree.model().itemFromIndex(index.siblingAtColumn(0))
        if item is None:
            return
        data = item.data(Qt.UserRole)
        if data is None:
            return
        kind, cohort, subcohort = data
        self._mb_curr_cohort    = cohort
        self._mb_curr_subcohort = subcohort  # None for cohort-level click
        self._mb_populate_table(cohort, subcohort)

    # -----------------------------------------------------------------------
    # Populate individual table
    # -----------------------------------------------------------------------

    def _mb_populate_table(self, cohort, subcohort=None):
        if self._mb_mf is None:
            return

        df2 = self._mb_get_cohort_df(cohort, subcohort)
        if df2 is None or df2.empty:
            self._mb_proxy.setSourceModel(QStandardItemModel())
            self._mb_df2 = pd.DataFrame()
            return

        self._mb_df2 = df2.copy()

        cols  = ["", "ID", "Subcohort", "Cached", "EDF", "Annot"]
        model = QStandardItemModel(len(df2), len(cols))
        model.setHorizontalHeaderLabels(cols)

        for r, (_, row) in enumerate(df2.iterrows()):
            iid   = str(row['ID'])
            sc    = str(row.get('Subcohort', ''))
            edf   = str(row.get('EDF', ''))
            annot = str(row.get('Annot', '.'))
            is_cached = self._mb_cached(cohort, edf)

            chk = QStandardItem()
            chk.setCheckable(True)
            chk.setCheckState(Qt.Unchecked)
            chk.setEditable(False)
            model.setItem(r, 0, chk)
            model.setItem(r, 1, _plain_item(iid))
            model.setItem(r, 2, _plain_item(sc))
            model.setItem(r, 3, _cached_item(is_cached))
            model.setItem(r, 4, _plain_item(os.path.basename(edf)))
            model.setItem(r, 5, _plain_item(
                os.path.basename(annot) if annot not in ('.', '') else '-'
            ))

        tbl = self.ui.mb_table
        self._mb_proxy.setSourceModel(model)
        h = tbl.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Fixed)
        for c in (1, 2, 4, 5):
            h.setSectionResizeMode(c, QHeaderView.Interactive)
        h.setSectionResizeMode(3, QHeaderView.Fixed)
        h.setStretchLastSection(False)
        tbl.setColumnWidth(0, 28)
        tbl.resizeColumnsToContents()
        tbl.setColumnWidth(0, 28)   # restore after resize
        tbl.setColumnWidth(3, 58)

    # -----------------------------------------------------------------------
    # Refresh cache status (offline-only)
    # -----------------------------------------------------------------------

    def _mb_refresh_cache_status(self):
        if self._mb_mf is None:
            return
        if self._mb_curr_cohort:
            self._mb_populate_table(self._mb_curr_cohort, self._mb_curr_subcohort)
        self._mb_populate_tree()

    # -----------------------------------------------------------------------
    # Get IIDs from table
    # -----------------------------------------------------------------------

    def _mb_src_model(self):
        """Return the source QStandardItemModel behind the proxy."""
        return self._mb_proxy.sourceModel()

    def _mb_get_checked_iids(self):
        src = self._mb_src_model()
        if src is None:
            return []
        iids = []
        for r in range(src.rowCount()):
            chk = src.item(r, 0)
            if chk and chk.checkState() == Qt.Checked:
                it = src.item(r, 1)
                if it:
                    iids.append(it.text())
        return iids

    def _mb_get_all_iids(self):
        src = self._mb_src_model()
        if src is None:
            return []
        return [src.item(r, 1).text()
                for r in range(src.rowCount())
                if src.item(r, 1)]

    def _mb_get_selected_iids(self):
        if self._mb_df2 is None or self._mb_df2.empty:
            return []
        sel = self.ui.mb_table.selectionModel()
        if sel is None:
            return []
        iids = []
        seen = set()
        for idx in sel.selectedRows():
            src_idx = self._mb_proxy.mapToSource(idx)
            if not src_idx.isValid():
                continue
            item = self._mb_src_model().item(src_idx.row(), 1)
            if item is None:
                continue
            iid = item.text()
            if iid and iid not in seen:
                seen.add(iid)
                iids.append(iid)
        return iids

    # -----------------------------------------------------------------------
    # Download: selected
    # -----------------------------------------------------------------------

    def _mb_download_selected(self):
        iids = self._mb_get_checked_iids()
        if not iids:
            QMessageBox.information(
                self.ui, "Moonbeam",
                "No individuals selected.\n"
                "Use the checkboxes in the first column to select individuals."
            )
            return
        self._mb_start_download(iids)

    # -----------------------------------------------------------------------
    # Download: all in current view
    # -----------------------------------------------------------------------

    def _mb_download_all(self):
        if self._mb_curr_cohort is None:
            return
        iids = self._mb_get_all_iids()
        if not iids:
            return
        label = self._mb_curr_subcohort or self._mb_curr_cohort
        force_redownload = bool(self.ui.mb_force_redownload_chk.isChecked())
        behavior_line = (
            "Force redownload is enabled: cached files will be downloaded again."
            if force_redownload else
            "If a file already exists in the cache, it will not be downloaded again."
        )
        reply = QMessageBox.question(
            self.ui, "Download All",
            f"Download all {len(iids):,} individuals from '{label}'?\n"
            f"{behavior_line}",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self._mb_start_download(iids)

    # -----------------------------------------------------------------------
    # Download: start worker thread
    # -----------------------------------------------------------------------

    def _mb_start_download(self, iids):
        if self._mb is None or self._mb_curr_cohort is None:
            return
        if self._mb_thread_running():
            QMessageBox.information(
                self.ui, "Moonbeam", "A download is already in progress."
            )
            return

        total = len(iids)
        force_redownload = bool(self.ui.mb_force_redownload_chk.isChecked())
        verb = "Re-downloading" if force_redownload else "Downloading"
        self.ui.mb_progress.setRange(0, total)
        self.ui.mb_progress.setValue(0)
        self.ui.mb_progress.setVisible(True)
        self.ui.mb_progress_lbl.setText(f"0 / {total}")
        self.ui.mb_cancel_btn.setVisible(True)
        self._mb_set_action_enabled(False)
        self.ui.mb_connect_btn.setEnabled(False)
        self.ui.mb_status_lbl.setText(f"{verb} 0 / {total}…")
        self.ui.mb_log.clear()
        if force_redownload:
            self.ui.mb_log.append(
                "Force redownload enabled: existing cached files will be replaced before download."
            )
        else:
            self.ui.mb_log.append(
                "Default mode: existing cached files will not be downloaded again."
            )

        thread = QThread(self)
        worker = _MbDownloadWorker(
            self._mb, iids,
            self._mb_curr_cohort,
            self._mb_curr_subcohort,
            force_redownload=force_redownload,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._mb_on_dl_progress)
        worker.item_done.connect(self._mb_on_item_done)
        worker.log_msg.connect(self._mb_on_log_msg)
        worker.error.connect(self._mb_on_dl_error)
        worker.finished.connect(self._mb_on_dl_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._mb_clear_thread)

        self._mb_thread    = thread
        self._mb_dl_worker = worker
        self._mb_dl_total  = total
        thread.start()

    # -----------------------------------------------------------------------
    # Cancel download (soft – between individuals)
    # -----------------------------------------------------------------------

    def _mb_cancel_download(self):
        w = getattr(self, "_mb_dl_worker", None)
        if w is not None:
            w.stop()
            self.ui.mb_status_lbl.setText("Cancelling…")

    # -----------------------------------------------------------------------
    # Download progress slots
    # -----------------------------------------------------------------------

    def _mb_on_dl_progress(self, current, total):
        self.ui.mb_progress.setValue(current)
        self.ui.mb_progress_lbl.setText(f"{current} / {total}")
        force_redownload = bool(self.ui.mb_force_redownload_chk.isChecked())
        verb = "Re-downloading" if force_redownload else "Downloading"
        self.ui.mb_status_lbl.setText(f"{verb} {current} / {total}…")

    def _mb_on_item_done(self, iid, success):
        """Update the Cached column in the table for the finished individual."""
        if not success or self._mb_curr_cohort is None:
            return
        if self._mb_df2 is None or self._mb_df2.empty:
            return
        rows = self._mb_df2[self._mb_df2['ID'] == iid]
        if rows.empty:
            return
        edf       = str(rows.iloc[0]['EDF'])
        is_cached = self._mb_cached(self._mb_curr_cohort, edf)
        src = self._mb_src_model()
        if src is None:
            return
        for r in range(src.rowCount()):
            id_it = src.item(r, 1)
            if id_it and id_it.text() == iid:
                src.setItem(r, 5, _cached_item(is_cached))
                break

    def _mb_on_dl_finished(self):
        self.ui.mb_progress.setVisible(False)
        self.ui.mb_cancel_btn.setVisible(False)
        self.ui.mb_progress_lbl.setText("")
        self._mb_set_action_enabled(True)
        self.ui.mb_connect_btn.setEnabled(True)
        self.ui.mb_status_lbl.setText("Connected")
        # Full refresh: tree counts + table cached flags
        self._mb_refresh_cache_status()

    def _mb_on_log_msg(self, line):
        log = self.ui.mb_log
        log.append(line)
        # Keep scrolled to bottom
        sb = log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _mb_on_dl_error(self, msg):
        # Non-fatal; show briefly in status bar
        self.ui.mb_status_lbl.setText(f"Warning: {msg[:100]}")

    def _mb_build_slist_rows(self, df):
        if self._mb_mf is None or self._mb_curr_cohort is None:
            QMessageBox.information(
                self.ui, "Moonbeam", "No cohort selected."
            )
            return None
        if df is None or df.empty:
            QMessageBox.information(
                self.ui, "Moonbeam",
                "The requested set of recordings is empty."
            )
            return None

        cohort  = self._mb_curr_cohort
        cdir    = pathlib.Path(self._mb_cdir)
        rows    = []

        for _, row in df.iterrows():
            iid   = str(row['ID'])
            edf   = str(row['EDF'])
            annot = str(row.get('Annot', '.'))

            if not self._mb_cached(cohort, edf):
                continue

            local_edf = str(cdir / cohort / edf.lstrip('/'))
            if annot in ('.', ''):
                local_annot = '.'
            else:
                candidate = str(cdir / cohort / annot.lstrip('/'))
                local_annot = candidate if os.path.exists(candidate) else '.'

            rows.append([iid, local_edf, local_annot])

        return rows

    def _mb_load_rows_into_slist(self, rows, label, origin_desc):
        if not rows:
            QMessageBox.information(
                self.ui, "Moonbeam",
                f"No cached individuals found in {origin_desc}.\n"
                "Download some files first, then try again."
            )
            return

        box = QMessageBox(self.ui)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Replace S-List?")
        box.setText(
            f"Replace the current S-List with {len(rows):,} cached individual(s) "
            f"from {origin_desc}?"
        )
        box.setInformativeText("Press Return to replace the S-List.")
        replace_btn = box.addButton("Replace S-List", QMessageBox.AcceptRole)
        cancel_btn = box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(replace_btn)
        box.setEscapeButton(cancel_btn)
        box.exec()
        if box.clickedButton() is not replace_btn:
            return

        try:
            self.proj.clear()
            self.proj.eng.set_sample_list(rows)
            df = self.proj.sample_list()
            model = self.df_to_model(df)
            self._proxy.setSourceModel(model)
            self._configure_slist_view()
            self.ui.lbl_slist.setText(f"<moonbeam:{label}>")
            self.ui.dock_moonbeam.hide()
        except Exception as exc:
            QMessageBox.critical(
                self.ui, "Moonbeam – S-List Error", str(exc)
            )

    # -----------------------------------------------------------------------
    # Populate S-List from cached individuals in current view
    # -----------------------------------------------------------------------

    def _mb_populate_slist(self):
        rows = self._mb_build_slist_rows(self._mb_df2)
        if rows is None:
            return
        label = self._mb_curr_subcohort or self._mb_curr_cohort
        self._mb_load_rows_into_slist(rows, label, f"the current view '{label}'")

    def _mb_populate_slist_selected(self):
        if self._mb_df2 is None or self._mb_df2.empty:
            QMessageBox.information(
                self.ui, "Moonbeam",
                "The current cohort view is empty."
            )
            return
        iids = self._mb_get_selected_iids()
        if not iids:
            QMessageBox.information(
                self.ui, "Moonbeam",
                "No recordings selected.\n"
                "Select one or more rows in the table, then try again."
            )
            return
        df = self._mb_df2[self._mb_df2['ID'].astype(str).isin(iids)].copy()
        rows = self._mb_build_slist_rows(df)
        if rows is None:
            return
        label = self._mb_curr_subcohort or self._mb_curr_cohort
        self._mb_load_rows_into_slist(rows, label, f"the selected recordings in '{label}'")


# ---------------------------------------------------------------------------
# Item helpers
# ---------------------------------------------------------------------------

def _plain_item(text: str) -> QStandardItem:
    it = QStandardItem(text)
    it.setEditable(False)
    return it


def _right_item(text: str) -> QStandardItem:
    it = QStandardItem(text)
    it.setEditable(False)
    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return it


def _cached_item(is_cached: bool) -> QStandardItem:
    it = QStandardItem("✓" if is_cached else "-")
    it.setEditable(False)
    it.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
    if is_cached:
        it.setForeground(QColor("#5dca7a"))
    return it
