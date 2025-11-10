from __future__ import annotations

from functools import reduce
from typing import Iterable, Literal

import pyvista as pv


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

    for mesh in iterator:
        result = result.boolean_union(_check_mesh(mesh), tolerance=tolerance)
    return result.clean()


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
    return result.clean()


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

    for mesh in iterator:
        result = result.boolean_intersection(_check_mesh(mesh), tolerance=tolerance)
    return result.clean()


def _ensure_backend(backend: BooleanBackend) -> None:
    if backend != "mesh":
        raise ValueError(f"Unsupported backend '{backend}'. Only 'mesh' is available at the moment.")
