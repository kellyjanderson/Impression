from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pyvista as pv

COLOR_FIELD = "__impression_color__"


def set_mesh_color(mesh: pv.PolyData, color: Sequence[float] | str) -> pv.PolyData:
    rgb, alpha = _normalize_color(color)
    data = np.array(list(rgb) + [alpha], dtype=float)[np.newaxis, :]
    mesh.field_data[COLOR_FIELD] = data
    return mesh


def get_mesh_color(mesh: pv.DataObject) -> Optional[Tuple[Tuple[float, float, float], float]]:
    if COLOR_FIELD not in mesh.field_data:
        return None
    arr = np.array(mesh.field_data[COLOR_FIELD])
    if arr.size == 0:
        return None
    rgba = arr[0]
    if rgba.size < 3:
        return None
    rgb = tuple(float(c) for c in rgba[:3])
    alpha = float(rgba[3]) if rgba.size >= 4 else 1.0
    return rgb, alpha


def transfer_mesh_color(target: pv.PolyData, source: pv.DataObject) -> pv.PolyData:
    info = get_mesh_color(source)
    if info is None:
        return target
    rgb, alpha = info
    data = np.array(list(rgb) + [alpha], dtype=float)[np.newaxis, :]
    target.field_data[COLOR_FIELD] = data
    return target


def _normalize_color(color: Sequence[float] | str) -> Tuple[Tuple[float, float, float], float]:
    if isinstance(color, str):
        col = pv.Color(color)
        rgb = tuple(col.float_rgb)
        alpha = 1.0
        return rgb, alpha

    arr = np.asarray(color, dtype=float).flatten()
    if arr.size not in (3, 4):
        raise ValueError("Color must be RGB or RGBA.")
    if arr.max() > 1.0:
        arr = arr / 255.0
    rgb = tuple(float(c) for c in arr[:3])
    alpha = float(arr[3]) if arr.size == 4 else 1.0
    return rgb, alpha
