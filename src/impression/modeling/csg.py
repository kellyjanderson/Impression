from __future__ import annotations

from typing import Iterable, Literal, Mapping, Union

import warnings

import numpy as np

from impression.mesh import Mesh, analyze_mesh
from impression.modeling.group import MeshGroup

from ._color import get_mesh_color, set_mesh_color

BooleanBackend = Literal["manifold"]


class BooleanOperationError(RuntimeError):
    """Raised when a boolean operation cannot produce a valid solid."""


def _ensure_backend(backend: BooleanBackend) -> None:
    if backend != "manifold":
        raise ValueError(f"Unsupported backend '{backend}'. Only 'manifold' is available right now.")


def _load_manifold():
    try:
        from manifold3d import Manifold, Mesh as ManifoldMesh
    except ImportError as exc:  # pragma: no cover - runtime dep
        raise BooleanOperationError(
            "manifold3d is required for boolean operations. Install it with `pip install manifold3d`."
        ) from exc
    return Manifold, ManifoldMesh


def _mesh_from_manifold(manifold_mesh) -> Mesh:
    vertices = None
    faces = None
    for attr in ("vertices", "vert_properties", "verts"):
        if hasattr(manifold_mesh, attr):
            vertices = np.asarray(getattr(manifold_mesh, attr), dtype=float)
            break
    for attr in ("triangles", "tri_verts", "faces"):
        if hasattr(manifold_mesh, attr):
            faces = np.asarray(getattr(manifold_mesh, attr), dtype=int)
            break
    if vertices is None or faces is None:
        raise BooleanOperationError("manifold3d returned an unexpected mesh format.")
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        raise BooleanOperationError("manifold3d returned invalid vertex data.")
    if vertices.shape[1] > 3:
        vertices = vertices[:, :3]
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise BooleanOperationError("manifold3d returned non-triangular faces.")
    return Mesh(vertices, faces)


def _manifold_from_mesh(mesh: Mesh):
    Manifold, ManifoldMesh = _load_manifold()
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    try:
        manifold_mesh = ManifoldMesh(vertices, faces)
    except TypeError:
        try:
            manifold_mesh = ManifoldMesh(vertices=vertices, triangles=faces)
        except TypeError as exc:  # pragma: no cover - defensive
            raise BooleanOperationError("Unable to build manifold mesh from input data.") from exc
    try:
        return Manifold(manifold_mesh)
    except Exception as exc:
        raise BooleanOperationError("manifold3d failed to create a solid from the provided mesh.") from exc


def _flatten_meshes(meshes: Iterable[Mesh | MeshGroup]) -> list[Mesh]:
    flattened: list[Mesh] = []
    for mesh in meshes:
        if isinstance(mesh, MeshGroup):
            flattened.extend(mesh.to_meshes())
        elif isinstance(mesh, Mesh):
            flattened.append(mesh)
        else:
            raise TypeError("Boolean operations require Mesh or MeshGroup inputs.")
    return flattened


def _check_mesh(mesh: Mesh) -> Mesh:
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError("Mesh faces must be a (N, 3) triangle array.")
    if mesh.n_faces == 0:
        raise ValueError("Mesh has no faces.")
    if mesh.faces.min() < 0 or mesh.faces.max() >= mesh.n_vertices:
        raise ValueError("Mesh faces contain out-of-range vertex indices.")
    analyze_mesh(mesh)
    issues = mesh.analysis.issues() if mesh.analysis else []
    if issues:
        warnings.warn(
            f"Mesh analysis warnings: {', '.join(issues)}",
            RuntimeWarning,
        )
    return mesh


def _combine_color(result: Mesh, sources: list[Mesh]) -> None:
    for mesh in sources:
        color = get_mesh_color(mesh)
        if color is not None:
            set_mesh_color(result, (*color[0], color[1]))
            return


def _apply_boolean(
    meshes: Iterable[Mesh],
    operation: str,
) -> Mesh:
    meshes_list = [_check_mesh(mesh) for mesh in meshes]
    if not meshes_list:
        raise ValueError(f"boolean_{operation} requires at least one mesh.")

    base = _manifold_from_mesh(meshes_list[0])
    for mesh in meshes_list[1:]:
        other = _manifold_from_mesh(mesh)
        if hasattr(base, operation):
            base = getattr(base, operation)(other)
        elif operation == "union" and hasattr(base, "__add__"):
            base = base + other
        elif operation == "difference" and hasattr(base, "__sub__"):
            base = base - other
        elif operation == "intersection":
            if hasattr(base, "__and__"):
                base = base & other
            elif hasattr(base, "__sub__"):
                # Intersection = A - (A - B) when direct op is unavailable.
                base = base - (base - other)
            else:
                raise BooleanOperationError(
                    "manifold3d does not support intersection on this version."
                )
        else:
            raise BooleanOperationError(f"manifold3d does not support '{operation}' on this version.")

    if hasattr(base, "to_mesh"):
        result_mesh = base.to_mesh()
    elif hasattr(base, "mesh"):
        result_mesh = base.mesh
    else:
        raise BooleanOperationError("manifold3d returned an unexpected result type.")

    result = _mesh_from_manifold(result_mesh)
    _combine_color(result, meshes_list)
    return result


def boolean_union(
    meshes: Iterable[Mesh | MeshGroup],
    tolerance: float = 1e-4,
    backend: BooleanBackend = "manifold",
) -> Mesh:
    _ensure_backend(backend)
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    return _apply_boolean(_flatten_meshes(meshes), "union")


def boolean_difference(
    base: Mesh | MeshGroup,
    cutters: Iterable[Mesh | MeshGroup],
    tolerance: float = 1e-4,
    backend: BooleanBackend = "manifold",
) -> Mesh:
    _ensure_backend(backend)
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    if isinstance(base, MeshGroup):
        base_mesh = base.to_mesh()
    else:
        base_mesh = base
    meshes = _flatten_meshes([base_mesh]) + _flatten_meshes(cutters)
    return _apply_boolean(meshes, "difference")


def boolean_intersection(
    meshes: Iterable[Mesh | MeshGroup],
    tolerance: float = 1e-4,
    backend: BooleanBackend = "manifold",
) -> Mesh:
    _ensure_backend(backend)
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")
    return _apply_boolean(_flatten_meshes(meshes), "intersection")


def union_meshes(
    meshes: Union[Iterable[Mesh | MeshGroup], Mapping[object, Mesh | MeshGroup]],
    tolerance: float = 1e-4,
    backend: BooleanBackend = "manifold",
) -> Mesh:
    if isinstance(meshes, Mapping):
        meshes = meshes.values()
    return boolean_union(meshes, tolerance=tolerance, backend=backend)
