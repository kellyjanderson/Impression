from __future__ import annotations

from typing import Iterable, Literal

from impression._vtk_runtime import ensure_vtk_runtime

ensure_vtk_runtime()

import numpy as np
import pyvista as pv

from pyvista import _vtk

from ._color import get_mesh_color, get_mesh_rgba, set_cell_colors, set_mesh_color


BooleanBackend = Literal["mesh"]


def _check_mesh(mesh: pv.DataSet) -> pv.PolyData:
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.cast_to_polydata()
    return mesh.triangulate().clean()


def boolean_union(
    meshes: Iterable[pv.DataSet],
    tolerance: float = 1e-5,
    backend: BooleanBackend = "mesh",
) -> pv.PolyData:
    _ensure_backend(backend)
    iterator = iter(meshes)
    try:
        first = _check_mesh(next(iterator))
    except StopIteration:
        raise ValueError("boolean_union requires at least one mesh.")

    sources = [first] + [_check_mesh(mesh) for mesh in iterator]
    result = sources[0]
    for mesh in sources[1:]:
        result = result.boolean_union(mesh, tolerance=tolerance)
    result = result.clean()
    _assign_boolean_colors("union", result, sources)
    return result


def boolean_difference(
    base: pv.DataSet,
    cutters: Iterable[pv.DataSet],
    tolerance: float = 1e-5,
    backend: BooleanBackend = "mesh",
) -> pv.PolyData:
    _ensure_backend(backend)
    sources = [_check_mesh(base)] + [_check_mesh(mesh) for mesh in cutters]
    result = sources[0]
    for mesh in sources[1:]:
        result = result.boolean_difference(mesh, tolerance=tolerance)
    result = result.clean()
    _assign_boolean_colors("difference", result, sources)
    return result


def boolean_intersection(
    meshes: Iterable[pv.DataSet],
    tolerance: float = 1e-5,
    backend: BooleanBackend = "mesh",
) -> pv.PolyData:
    _ensure_backend(backend)
    sources = [_check_mesh(mesh) for mesh in meshes]
    if not sources:
        raise ValueError("boolean_intersection requires at least one mesh.")

    result = sources[0]

    for mesh in sources[1:]:
        result = result.boolean_intersection(mesh, tolerance=tolerance)
    result = result.clean()
    _assign_boolean_colors("intersection", result, sources)
    return result


def _ensure_backend(backend: BooleanBackend) -> None:
    if backend != "mesh":
        raise ValueError(f"Unsupported backend '{backend}'. Only 'mesh' is available at the moment.")


def _assign_boolean_colors(mode: str, result: pv.PolyData, sources: list[pv.PolyData]) -> None:
    if result.n_cells == 0 or not sources:
        return

    centers = result.cell_centers().points
    if centers.size == 0:
        return

    implicit_distances = []
    colors = []
    for mesh in sources:
        rgba = get_mesh_rgba(mesh)
        colors.append(np.array(rgba, dtype=float))
        implicit = np.abs(_implicit_distance(mesh, centers))
        implicit_distances.append(implicit)

    implicit_distances = np.vstack(implicit_distances)
    rgba_array = np.tile(colors[0], (result.n_cells, 1))
    tol = max(result.length / 500.0, 1e-4)

    if mode == "difference":
        for idx in range(1, len(sources)):
            dist = implicit_distances[idx]
            mask = dist <= tol
            if np.any(mask):
                rgba_array[mask] = colors[idx]
    elif mode in {"union", "intersection"}:
        nearest = np.argmin(implicit_distances, axis=0)
        for idx in range(len(sources)):
            mask = nearest == idx
            if np.any(mask):
                rgba_array[mask] = colors[idx]

    set_cell_colors(result, rgba_array)
    # default mesh color remains first source
    set_mesh_color(result, colors[0])


def _implicit_distance(mesh: pv.PolyData, points: np.ndarray) -> np.ndarray:
    func = _vtk.vtkImplicitPolyDataDistance()
    func.SetInput(mesh)
    distances = [func.EvaluateFunction(point) for point in points]
    return np.array(distances, dtype=float)
