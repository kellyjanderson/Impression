from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from impression.mesh import Mesh


def _face_normals(mesh: Mesh) -> np.ndarray:
    if mesh.n_faces == 0:
        return np.zeros((0, 3), dtype=float)
    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.linalg.norm(normals, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        normals = np.divide(normals, lengths[:, np.newaxis], where=lengths[:, np.newaxis] > 0)
    normals[~np.isfinite(normals)] = 0.0
    return normals


def write_stl(mesh: Mesh, path: Path, ascii: bool = False) -> None:
    path = Path(path)
    normals = _face_normals(mesh)
    faces = mesh.faces
    vertices = mesh.vertices

    if ascii:
        lines = ["solid impression"]
        for idx, tri in enumerate(faces):
            nx, ny, nz = normals[idx]
            lines.append(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}")
            lines.append("    outer loop")
            for vidx in tri:
                vx, vy, vz = vertices[vidx]
                lines.append(f"      vertex {vx:.6e} {vy:.6e} {vz:.6e}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append("endsolid impression")
        path.write_text("\n".join(lines))
        return

    header = b"Impression STL".ljust(80, b"\0")
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<I", faces.shape[0]))
        for idx, tri in enumerate(faces):
            nx, ny, nz = normals[idx]
            v0 = vertices[tri[0]]
            v1 = vertices[tri[1]]
            v2 = vertices[tri[2]]
            handle.write(
                struct.pack(
                    "<12fH",
                    float(nx),
                    float(ny),
                    float(nz),
                    float(v0[0]),
                    float(v0[1]),
                    float(v0[2]),
                    float(v1[0]),
                    float(v1[1]),
                    float(v1[2]),
                    float(v2[0]),
                    float(v2[1]),
                    float(v2[2]),
                    0,
                )
            )
