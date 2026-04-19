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

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def user_cache_root() -> Path:
    if sys.platform == "win32":
        for env_var in ("LOCALAPPDATA", "APPDATA"):
            value = os.environ.get(env_var)
            if value:
                return Path(value)
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache:
            return Path(xdg_cache)
        return Path.home() / ".cache"
    return Path(tempfile.gettempdir()) / "lunascope-cache"


def app_cache_root() -> Path:
    preferred = user_cache_root() / "lunascope"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "lunascope-cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def app_state_file(*parts: str) -> Path:
    return app_cache_root().joinpath(*parts)
