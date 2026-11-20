"""Small surface-atlas resource contract; no inverse-model dependencies."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Parcel:
    id: int
    name: str
    hemisphere: str
    signal: str


@dataclass
class Atlas:
    directory: Path
    metadata: dict
    parcels: tuple[Parcel, ...]
    surfaces: dict[str, np.ndarray]
    faces: np.ndarray
    vertex_parcels: np.ndarray
    hemisphere: np.ndarray

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        with (directory / "parcels.tsv").open(newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        hemis = {"lh": "left", "rh": "right", "left": "left", "right": "right"}
        parcels = tuple(Parcel(int(r["parcel_id"]), r["name"],
                               hemis[r["hemisphere"].lower()], r["luna_signal"])
                        for r in rows)
        if not parcels or len({p.id for p in parcels}) != len(parcels):
            raise ValueError("Atlas requires unique parcel IDs")
        if len({p.signal.casefold() for p in parcels}) != len(parcels):
            raise ValueError("Atlas requires unique luna_signal names (ignoring case)")
        if any(p.id < 0 or not p.signal.strip() for p in parcels):
            raise ValueError("Parcel IDs must be nonnegative and signals nonempty")
        with np.load(directory / "cortex_mesh.npz", allow_pickle=False) as mesh:
            surfaces = {"pial": np.asarray(mesh["pial_vertices"], dtype=np.float32)}
            if "inflated_vertices" in mesh:
                surfaces["inflated"] = np.asarray(mesh["inflated_vertices"], dtype=np.float32)
            faces = np.asarray(mesh["faces"])
            hemisphere = np.asarray(mesh["hemisphere"])
        labels = np.load(directory / "vertex_parcels.npy", allow_pickle=False)
        n = len(surfaces["pial"])
        if any(v.shape != (n, 3) or not np.isfinite(v).all() for v in surfaces.values()):
            raise ValueError("Surfaces must contain finite N × 3 vertices with shared topology")
        if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= n):
            raise ValueError("Faces must be zero-based M × 3 vertex indices")
        if labels.shape != (n,) or labels.dtype.kind not in "iu":
            raise ValueError("vertex_parcels must contain one integer ID per vertex")
        if not set(np.unique(labels)) <= {-1, *(p.id for p in parcels)}:
            raise ValueError("Unknown vertex parcel ID (neutral/medial wall must be -1)")
        if hemisphere.shape != (n,) or not set(np.unique(hemisphere)) <= {0, 1}:
            raise ValueError("Mesh hemisphere must be 0=left, 1=right per vertex")
        for p in parcels:
            if np.any(hemisphere[labels == p.id] != (0 if p.hemisphere == "left" else 1)):
                raise ValueError(f"Hemisphere mismatch for {p.name}")
        # A declared transform permits other coordinate frames without guessing axes.
        geometry = metadata.get("geometry", {})
        transform = metadata.get("display", {}).get("to_ras")
        if transform is not None:
            matrix = np.asarray(transform, dtype=float)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError("display.to_ras must be a finite 4 × 4 affine")
            surfaces = {k: (v @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)
                        for k, v in surfaces.items()}
        elif "ras" not in str(geometry.get("frame", "")).casefold():
            raise ValueError("Declare geometry.frame as RAS or supply display.to_ras")
        return cls(directory, metadata, parcels, surfaces, faces.astype(np.int32),
                   labels, hemisphere)

    def match(self, names, *, prefix="", mapping=None):
        """Match exact names, then unique 16-byte EDF representations.

        Aliases come only from this atlas's expected labels (including explicit
        renaming). Never use arbitrary prefix/fuzzy matching. Check ambiguity
        against the entire model, even if one colliding region is absent.
        """
        actual = {}
        for name in names:
            key = str(name).casefold()
            if key in actual:
                raise ValueError(f"Duplicate signal name: {name}")
            actual[key] = str(name)
        mapping = mapping or {}
        expected = [str(mapping.get(p.id, mapping.get(str(p.id), prefix + p.signal)))
                    for p in self.parcels]
        if len({s.casefold() for s in expected}) != len(expected):
            raise ValueError("Signal mapping assigns multiple parcels to the same signal")
        matched = [actual.get(s.casefold()) for s in expected]
        aliases = {}
        for index, signal in enumerate(expected):
            # EDF header labels are 16 ASCII bytes, not arbitrary-length prefixes.
            if signal.isascii() and len(signal) >= 16:
                aliases.setdefault(signal[:16].casefold(), []).append(index)
        for index, signal in enumerate(expected):
            if matched[index] is not None or not signal.isascii() or len(signal) <= 16:
                continue
            alias = signal[:16].casefold()
            if alias not in actual:
                continue
            owners = aliases[alias]
            if len(owners) != 1:
                conflicts = ", ".join(expected[i] for i in owners)
                raise ValueError(f"Ambiguous EDF label {actual[alias]!r}: {conflicts}. "
                                 "Supply an explicit signal mapping")
            matched[index] = actual[alias]
        assigned = [s.casefold() for s in matched if s is not None]
        if len(set(assigned)) != len(assigned):
            raise ValueError("Signal mapping assigns multiple parcels to the same recording channel")
        return matched

    def face_parcels(self):
        """Majority vertex label per face, ties neutral; return parcel-vector indices."""
        labels = self.vertex_parcels[self.faces]
        majority = np.full(len(labels), -1, dtype=np.int32)
        for a, b in ((0, 1), (0, 2), (1, 2)):
            equal = labels[:, a] == labels[:, b]
            majority[equal] = labels[equal, a]
        indices = np.full(len(labels), -1, dtype=np.int32)
        for i, parcel in enumerate(self.parcels):
            indices[majority == parcel.id] = i
        return indices


def pick_triangle(origin, direction, triangles):
    """Two-sided vectorized Möller–Trumbore intersection, nearest face or None."""
    if not len(triangles):
        return None
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(det) > 1e-10
    inv = np.divide(1., det, out=np.zeros_like(det), where=valid)
    s = origin - triangles[:, 0]
    u = inv * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inv * (q @ direction)
    distance = inv * np.einsum("ij,ij->i", edge2, q)
    valid &= (u >= 0) & (v >= 0) & (u + v <= 1) & (distance > 0)
    distance[~valid] = np.inf
    return int(np.argmin(distance)) if np.any(valid) else None
