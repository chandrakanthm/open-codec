"""Mesh I/O: OBJ read/write, raw float32 vertex blobs, and a Wavefront texture export.

Multiface tracked meshes ship as raw little-endian float32 ``.bin`` blobs of
shape (V*3,); the shared topology (faces, UVs) lives in an accompanying ``.obj``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_bin_vertices(path: str | Path) -> np.ndarray:
    """Read a Multiface ``.bin`` vertex blob -> (V, 3) float32."""
    arr = np.fromfile(str(path), dtype="<f4")
    if arr.size % 3 != 0:
        raise ValueError(f"{path}: {arr.size} floats is not divisible by 3")
    return arr.reshape(-1, 3).astype(np.float32)


def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Minimal OBJ reader.

    Returns ``(verts (V,3), faces (F,3) int64, uvs (T,2)|None, face_uvs (F,3)|None)``.
    Faces are triangulated by a simple fan and converted to 0-based indices.
    """
    verts: list[list[float]] = []
    uvs: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    face_uvs: list[tuple[int, int, int]] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                uvs.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("f "):
                toks = line.split()[1:]
                vi, ti = [], []
                for t in toks:
                    parts = t.split("/")
                    vi.append(int(parts[0]))
                    if len(parts) > 1 and parts[1]:
                        ti.append(int(parts[1]))
                # negative indices are relative; OBJ is 1-based
                vi = [(i - 1) if i > 0 else (len(verts) + i) for i in vi]
                for k in range(1, len(vi) - 1):  # fan triangulation
                    faces.append((vi[0], vi[k], vi[k + 1]))
                    if len(ti) == len(vi):
                        ti0 = [(i - 1) if i > 0 else (len(uvs) + i) for i in ti]
                        face_uvs.append((ti0[0], ti0[k], ti0[k + 1]))
    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64) if faces else np.empty((0, 3), np.int64)
    vt = np.asarray(uvs, dtype=np.float32) if uvs else None
    fvt = np.asarray(face_uvs, dtype=np.int64) if face_uvs else None
    return v, f, vt, fvt


def save_obj(
    path: str | Path,
    verts: np.ndarray,
    faces: np.ndarray | None = None,
    uvs: np.ndarray | None = None,
    face_uvs: np.ndarray | None = None,
    texture_name: str | None = None,
) -> None:
    """Write an OBJ. If ``texture_name`` is given, also writes a sibling ``.mtl``."""
    path = Path(path)
    verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    lines: list[str] = []
    mtl_name = None
    if texture_name is not None:
        mtl_name = path.with_suffix(".mtl").name
        lines.append(f"mtllib {mtl_name}")
        lines.append("usemtl avatar")
    for x, y, z in verts:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    if uvs is not None:
        for u, w in np.asarray(uvs, dtype=np.float32).reshape(-1, 2):
            lines.append(f"vt {u:.6f} {w:.6f}")
    if faces is not None and len(faces) > 0:
        faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3) + 1  # OBJ is 1-based
        if face_uvs is not None and len(face_uvs) == len(faces):
            fuv = np.asarray(face_uvs, dtype=np.int64).reshape(-1, 3) + 1
            for (a, b, c), (ta, tb, tc) in zip(faces, fuv):
                lines.append(f"f {a}/{ta} {b}/{tb} {c}/{tc}")
        else:
            for a, b, c in faces:
                lines.append(f"f {a} {b} {c}")
    path.write_text("\n".join(lines) + "\n")

    if texture_name is not None and mtl_name is not None:
        mtl = (
            "newmtl avatar\n"
            "Ka 1.000 1.000 1.000\n"
            "Kd 1.000 1.000 1.000\n"
            "illum 1\n"
            f"map_Kd {texture_name}\n"
        )
        (path.parent / mtl_name).write_text(mtl)
