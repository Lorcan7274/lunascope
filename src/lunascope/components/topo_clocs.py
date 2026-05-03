
#  --------------------------------------------------------------------
#
#  This file is part of Luna.
#
#  LUNA is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  LUNA is distributed in the hope that it will be useful,
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

"""Channel location data and 2-D projection for EEG topoplot rendering.

Coordinates ported from luna-base/clocs/clocs.cpp::set_default().
All labels stored upper-case; old 10-20 aliases (T3→T7, T5→P7, etc.) included.

Projection: azimuthal equidistant from the vertex (Cz).
  - nose at top (+y_plot), right hemisphere at right (+x_plot)
  - head boundary circle at radius ≈ 0.5 (equatorial channels)
"""

import math

# 3-D Cartesian coordinates (x=anterior, y=left, z=up) from luna-base clocs.cpp
_CLOCS_DEFAULT_3D: dict[str, tuple[float, float, float]] = {
    "FP1":  ( 80.7840,  26.1330,  -4.0011),
    "AF7":  ( 68.6911,  49.7094,  -5.9589),
    "AF3":  ( 76.1528,  31.4828,  20.8468),
    "F1":   ( 59.9127,  26.0421,  54.3808),
    "F3":   ( 57.5511,  48.2004,  39.8697),
    "F5":   ( 54.0379,  63.0582,  18.1264),
    "F7":   ( 49.8714,  68.4233,  -7.4895),
    "FT7":  ( 26.2075,  80.4100,  -8.5086),
    "FC5":  ( 28.7628,  76.2474,  24.1669),
    "FC3":  ( 30.9553,  59.2750,  52.4714),
    "FC1":  ( 32.4362,  32.3514,  71.5981),
    "C1":   (  0.0000,  34.5374,  77.6670),
    "C3":   (  0.0000,  63.1713,  56.8717),
    "C5":   (  0.0000,  80.8315,  26.2918),
    "T7":   (  0.0000,  84.5385,  -8.8451),
    "T3":   (  0.0000,  84.5385,  -8.8451),   # old → T7
    "TP7":  (-26.2075,  80.4100,  -8.5086),
    "CP5":  (-28.7628,  76.2474,  24.1669),
    "CP3":  (-30.9553,  59.2750,  52.4714),
    "CP1":  (-32.4362,  32.3514,  71.5981),
    "P1":   (-59.9127,  26.0421,  54.3808),
    "P3":   (-57.5511,  48.2004,  39.8697),
    "P5":   (-54.0379,  63.0582,  18.1264),
    "P7":   (-49.8714,  68.4233,  -7.4895),
    "T5":   (-49.8714,  68.4233,  -7.4895),   # old → P7
    "P9":   (-44.4841,  59.7083, -41.0011),
    "PO7":  (-68.6911,  49.7094,  -5.9589),
    "PO3":  (-76.1528,  31.4828,  20.8468),
    "O1":   (-80.7840,  26.1330,  -4.0011),
    "IZ":   (-77.6333,   0.0000, -34.6133),
    "OZ":   (-84.9812,   0.0000,  -1.7860),
    "POZ":  (-79.0255,   0.0000,  31.3044),
    "PZ":   (-60.7385,   0.0000,  59.4629),
    "CPZ":  (-32.9279,   0.0000,  78.3630),
    "FPZ":  ( 84.9812,   0.0000,  -1.7860),
    "FP2":  ( 80.7840, -26.1330,  -4.0011),
    "AF8":  ( 68.7209, -49.6689,  -5.9530),
    "AF4":  ( 76.1528, -31.4828,  20.8468),
    "AFZ":  ( 79.0255,   0.0000,  31.3044),
    "FZ":   ( 60.7385,   0.0000,  59.4629),
    "F2":   ( 59.8744, -26.0254,  54.4310),
    "F4":   ( 57.5840, -48.1426,  39.8920),
    "F6":   ( 54.0263, -63.0447,  18.2076),
    "F8":   ( 49.9265, -68.3836,  -7.4851),
    "FT8":  ( 26.2075, -80.4100,  -8.5086),
    "FC6":  ( 28.7628, -76.2474,  24.1669),
    "FC4":  ( 30.9553, -59.2750,  52.4714),
    "FC2":  ( 32.4362, -32.3514,  71.5981),
    "FCZ":  ( 32.9279,   0.0000,  78.3630),
    "CZ":   (  0.0000,   0.0000,  85.0000),
    "C2":   (  0.0000, -34.6092,  77.6351),
    "C4":   (  0.0000, -63.1673,  56.8761),
    "C6":   (  0.0000, -80.8315,  26.2918),
    "T8":   (  0.0000, -84.5385,  -8.8451),
    "T4":   (  0.0000, -84.5385,  -8.8451),   # old → T8
    "TP8":  (-26.2848, -80.3851,  -8.5057),
    "CP6":  (-28.7628, -76.2474,  24.1669),
    "CP4":  (-30.9553, -59.2750,  52.4714),
    "CP2":  (-32.4362, -32.3514,  71.5981),
    "P2":   (-59.8744, -26.0254,  54.4310),
    "P4":   (-57.5840, -48.1426,  39.8920),
    "P6":   (-54.0263, -63.0447,  18.2076),
    "P8":   (-49.9265, -68.3836,  -7.4851),
    "T6":   (-49.9265, -68.3836,  -7.4851),   # old → P8
    "P10":  (-44.4841, -59.7083, -41.0011),
    "PO8":  (-68.7209, -49.6689,  -5.9530),
    "PO4":  (-76.1528, -31.4828,  20.8468),
    "O2":   (-80.7840, -26.1330,  -4.0011),
}


def cart_to_plot2d(x: float, y: float, z: float) -> tuple[float, float]:
    """Project 3-D Cartesian electrode position to 2-D topoplot coordinates.

    Uses the azimuthal equidistant projection from the vertex (Cz).
    Convention matches EEGLAB topoplot:
      - nose at top (+py), right hemisphere at right (+px)
      - head boundary circle ≈ radius 0.5 (equatorial plane)
    """
    r = math.sqrt(x * x + y * y + z * z)
    if r < 1e-10:
        return 0.0, 0.0
    xn, yn, zn = x / r, y / r, z / r
    colatitude = math.acos(max(-1.0, min(1.0, zn)))   # 0 at vertex (Cz)
    azimuth    = math.atan2(yn, xn)                    # 0 = anterior
    r2d = colatitude / math.pi                         # ≈0.5 at equator
    px  = -r2d * math.sin(azimuth)                     # left (y>0) → left on screen
    py  =  r2d * math.cos(azimuth)                     # anterior (x>0) → top
    return px, py


def get_positions(
    labels: list[str],
    user_overrides: dict[str, tuple[float, float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    """Return 2-D plot positions for the given channel labels.

    Labels without a known coordinate (default or override) are silently omitted.
    Keys in the returned dict match the case of the input *labels*.
    """
    overrides = {k.upper(): v for k, v in (user_overrides or {}).items()}
    result: dict[str, tuple[float, float]] = {}
    for lab in labels:
        up = lab.upper()
        if up in overrides:
            result[lab] = cart_to_plot2d(*overrides[up])
        elif up in _CLOCS_DEFAULT_3D:
            result[lab] = cart_to_plot2d(*_CLOCS_DEFAULT_3D[up])
    return result


def load_clocs_file(path: str) -> dict[str, tuple[float, float, float]]:
    """Load 3-D channel coordinates from a whitespace/comma-delimited LABEL X Y Z file.

    Returns a dict suitable for passing as *user_overrides* to :func:`get_positions`.
    Lines starting with # or % are treated as comments.
    """
    coords: dict[str, tuple[float, float, float]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in ("#", "%"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 4:
                continue
            lab = parts[0].upper()
            try:
                coords[lab] = (float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError:
                continue
    return coords


def all_known_labels() -> list[str]:
    """Return sorted list of all labels with default coordinates."""
    return sorted(_CLOCS_DEFAULT_3D.keys())
