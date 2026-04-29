from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from impression.mesh import Mesh

COLOR_FIELD = "__impression_color__"
COLOR_CELL_DATA = "__impression_cell_color__"


def set_mesh_color(mesh: Mesh, color: Sequence[float] | str) -> Mesh:
    rgba = _normalize_color(color)
    mesh.color = rgba
    return mesh


def get_mesh_color(mesh: Mesh) -> Optional[Tuple[Tuple[float, float, float], float]]:
    if mesh.color is None:
        return None
    rgba = mesh.color
    rgb = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
    alpha = float(rgba[3])
    return rgb, alpha


def get_mesh_rgba(mesh: Mesh) -> Tuple[float, float, float, float]:
    info = get_mesh_color(mesh)
    if info is None:
        return (0.8, 0.8, 0.8, 1.0)
    rgb, alpha = info
    return (rgb[0], rgb[1], rgb[2], alpha)


def transfer_mesh_color(target: Mesh, source: Mesh) -> Mesh:
    info = get_mesh_color(source)
    if info is None:
        return target
    rgb, alpha = info
    target.color = (rgb[0], rgb[1], rgb[2], alpha)
    return target


def set_cell_colors(mesh: Mesh, rgba: np.ndarray) -> None:
    if mesh.n_faces == 0:
        return
    if rgba.shape[0] != mesh.n_faces:
        raise ValueError("rgba array must match number of cells")
    mesh.face_colors = rgba.astype(float)


def _normalize_color(color: Sequence[float] | str) -> Tuple[float, float, float, float]:
    if isinstance(color, str):
        value = color.strip().lower()
        if value.startswith("#"):
            hex_value = value[1:]
            if len(hex_value) == 3:
                hex_value = "".join(ch * 2 for ch in hex_value)
            if len(hex_value) == 6:
                r, g, b = hex_value[0:2], hex_value[2:4], hex_value[4:6]
                return (int(r, 16) / 255.0, int(g, 16) / 255.0, int(b, 16) / 255.0, 1.0)
            if len(hex_value) == 8:
                r, g, b, a = hex_value[0:2], hex_value[2:4], hex_value[4:6], hex_value[6:8]
                return (
                    int(r, 16) / 255.0,
                    int(g, 16) / 255.0,
                    int(b, 16) / 255.0,
                    int(a, 16) / 255.0,
                )
            raise ValueError("Hex colors must be #RGB, #RRGGBB, or #RRGGBBAA.")
        named = {
            "black": (0.0, 0.0, 0.0),
            "white": (1.0, 1.0, 1.0),
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "gray": (0.5, 0.5, 0.5),
            "grey": (0.5, 0.5, 0.5),
            "orange": (1.0, 0.65, 0.0),
            "purple": (0.5, 0.0, 0.5),
            "pink": (1.0, 0.75, 0.8),
        }
        if value in named:
            rgb = named[value]
            return (rgb[0], rgb[1], rgb[2], 1.0)
        raise ValueError("Named colors must be a hex string or one of: " + ", ".join(sorted(named)))

    arr = np.asarray(color, dtype=float).flatten()
    if arr.size not in (3, 4):
        raise ValueError("Color must be RGB or RGBA.")
    if arr.max() > 1.0:
        arr = arr / 255.0
    rgb = tuple(float(c) for c in arr[:3])
    alpha = float(arr[3]) if arr.size == 4 else 1.0
    return (rgb[0], rgb[1], rgb[2], alpha)
