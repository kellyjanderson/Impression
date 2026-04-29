from __future__ import annotations

from typing import Iterable

import numpy as np

from impression.mesh import Mesh
from impression.modeling.group import MeshGroup


def hull_mesh(shapes: Iterable[Mesh | MeshGroup]) -> Mesh:
    """Compute convex hull for mesh inputs using manifold3d."""

    manifolds = [manifold_from_mesh_group(item) for item in shapes]
    if len(manifolds) == 1:
        result = manifolds[0].hull()
    else:
        from manifold3d import Manifold

        result = Manifold.batch_hull(manifolds)
    return mesh_from_manifold(result)


def manifold_from_mesh_group(mesh: Mesh | MeshGroup):
    from manifold3d import Manifold, Mesh as ManifoldMesh

    if isinstance(mesh, MeshGroup):
        mesh = mesh.to_mesh()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    try:
        manifold_mesh = ManifoldMesh(vertices, faces)
    except TypeError:
        manifold_mesh = ManifoldMesh(vertices=vertices, triangles=faces)
    return Manifold(manifold_mesh)


def mesh_from_manifold(manifold) -> Mesh:
    mesh = manifold.to_mesh() if hasattr(manifold, "to_mesh") else manifold.mesh
    vertices = np.asarray(getattr(mesh, "vertices", mesh.vert_properties), dtype=float)
    faces = np.asarray(getattr(mesh, "triangles", mesh.tri_verts), dtype=int)
    return Mesh(vertices, faces)

