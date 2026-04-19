#!/usr/bin/env python3

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

import sys
from pathlib import Path


BROKEN_BLOCK = """- module-name: 'scipy.optimize._cobyla_py' # checksum: 3ea0b5b1
  implicit-imports:
    - post-import-code:
        - |
          import scipy.optimize._cobyla
          orig_minimize=scipy.optimize._cobyla.minimize

          def nuitka_compatible_minimize(*args, **kwargs):
            if len(args) > 0:
              arg0 = args[0]
              wrapper = eval(\"\"\"lambda x, con: arg0(x, con)\"\"\", locals())
              args = list(args)
              args[0] = wrapper
            if \"callback\" in kwargs:
              kw_callback = kwargs[\"callback\"]
              callback_wrapper = eval(\"\"\"lambda x: kw_callback(x)\"\"\", locals())
              kwargs[\"callback\"] = callback_wrapper
            return orig_minimize(*args, **kwargs)
          scipy.optimize._cobyla.minimize=nuitka_compatible_minimize
"""


def main() -> int:
    try:
        import nuitka  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"Nuitka is not installed in this environment: {exc}")

    config_path = (
        Path(nuitka.__file__).resolve().parent
        / "plugins"
        / "standard"
        / "standard.nuitka-package.config.yml"
    )

    text = config_path.read_text(encoding="utf-8")

    if BROKEN_BLOCK not in text:
        print(f"No SciPy COBYLA patch needed in {config_path}")
        return 0

    config_path.write_text(
        text.replace(BROKEN_BLOCK, "", 1),
        encoding="utf-8",
    )
    print(f"Patched broken SciPy COBYLA Nuitka rule in {config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
