from __future__ import annotations

from typing import Iterable, Literal

import pyvista as pv

from ._color import get_mesh_color, set_mesh_color


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
        result = _check_mesh(next(iterator))
    except StopIteration:
        raise ValueError("boolean_union requires at least one mesh.")

    base_color = get_mesh_color(result)
    for mesh in iterator:
        result = result.boolean_union(_check_mesh(mesh), tolerance=tolerance)
    result = result.clean()
    if base_color:
        rgb, alpha = base_color
        set_mesh_color(result, (*rgb, alpha))
    return result


def boolean_difference(
    base: pv.DataSet,
    cutters: Iterable[pv.DataSet],
    tolerance: float = 1e-5,
    backend: BooleanBackend = "mesh",
) -> pv.PolyData:
    _ensure_backend(backend)
    result = _check_mesh(base)
    for mesh in cutters:
        result = result.boolean_difference(_check_mesh(mesh), tolerance=tolerance)
    result = result.clean()
    base_color = get_mesh_color(base)
    if base_color:
        rgb, alpha = base_color
        set_mesh_color(result, (*rgb, alpha))
    return result


def boolean_intersection(
    meshes: Iterable[pv.DataSet],
    tolerance: float = 1e-5,
    backend: BooleanBackend = "mesh",
) -> pv.PolyData:
    _ensure_backend(backend)
    iterator = iter(meshes)
    try:
        result = _check_mesh(next(iterator))
    except StopIteration:
        raise ValueError("boolean_intersection requires at least one mesh.")

    base_color = get_mesh_color(result)
    for mesh in iterator:
        result = result.boolean_intersection(_check_mesh(mesh), tolerance=tolerance)
    result = result.clean()
    if base_color:
        rgb, alpha = base_color
        set_mesh_color(result, (*rgb, alpha))
    return result


def _ensure_backend(backend: BooleanBackend) -> None:
    if backend != "mesh":
        raise ValueError(f"Unsupported backend '{backend}'. Only 'mesh' is available at the moment.")
