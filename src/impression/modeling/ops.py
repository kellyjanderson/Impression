from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from impression.mesh import Mesh
from impression.modeling.drawing2d import Path2D, Profile2D
from impression.modeling._profile2d import _profile_loops, _signed_area
from impression.modeling.group import MeshGroup


def offset(
    shape: Profile2D | Path2D,
    r: float | None = None,
    delta: float | None = None,
    chamfer: bool = False,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
) -> list[Profile2D]:
    """Offset a 2D profile/path using manifold3d CrossSection."""

    if r is None and delta is None:
        raise ValueError("offset requires r or delta.")
    if r is not None and delta is not None:
        raise ValueError("Provide only one of r or delta.")
    dist = float(r if r is not None else delta)

    CrossSection, JoinType = _load_cross_section()
    contours = _shape_to_contours(shape, segments_per_circle, bezier_samples)
    cross = CrossSection(contours)
    join_type = JoinType.Miter if chamfer else JoinType.Round
    result = cross.offset(dist, join_type=join_type)
    return _cross_section_to_profiles(result)


def hull(shapes: Iterable[Mesh | MeshGroup | Profile2D | Path2D]) -> Mesh | list[Profile2D]:
    """Compute convex hull for 3D meshes or 2D profiles."""

    items = list(shapes)
    if not items:
        raise ValueError("hull requires at least one shape.")

    if isinstance(items[0], (Profile2D, Path2D)):
        CrossSection, _ = _load_cross_section()
        cross_sections = [
            CrossSection(_shape_to_contours(item, 64, 32)) for item in items  # default sampling
        ]
        result = CrossSection.batch_hull(cross_sections)
        return _cross_section_to_profiles(result)

    manifolds = [_manifold_from_mesh_group(item) for item in items]
    if len(manifolds) == 1:
        result = manifolds[0].hull()
    else:
        from manifold3d import Manifold
        result = Manifold.batch_hull(manifolds)
    return _mesh_from_manifold(result)


def minkowski(*_args, **_kwargs):  # pragma: no cover - explicit stub
    raise NotImplementedError(
        "minkowski is not available with the current manifold3d backend. "
        "Use an external geometry kernel (e.g., CGAL/libigl) to support 3D Minkowski sums."
    )


def _shape_to_contours(
    shape: Profile2D | Path2D,
    segments_per_circle: int,
    bezier_samples: int,
) -> list[np.ndarray]:
    if isinstance(shape, Profile2D):
        return _profile_loops(shape, segments_per_circle, bezier_samples, enforce_winding=True)
    if isinstance(shape, Path2D):
        if not shape.closed:
            raise ValueError("Path2D must be closed for offset/hull.")
        pts = shape.sample(segments_per_circle=segments_per_circle, bezier_samples=bezier_samples)
        if pts.shape[0] > 1 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        return [pts]
    raise TypeError("offset/hull expects Profile2D or Path2D.")


def _cross_section_to_profiles(cross_section) -> list[Profile2D]:
    profiles: list[Profile2D] = []
    for part in cross_section.decompose():
        contours = [np.asarray(poly, dtype=float) for poly in part.to_polygons()]
        if not contours:
            continue
        outers: list[np.ndarray] = []
        holes: list[np.ndarray] = []
        for contour in contours:
            area = _signed_area(contour)
            if area >= 0:
                outers.append(contour)
            else:
                holes.append(contour)
        if not outers:
            continue
        outer = _largest_area(outers)
        hole_paths = [_path_from_contour(hole) for hole in holes if _point_in_poly(hole.mean(axis=0), outer)]
        profiles.append(Profile2D(outer=_path_from_contour(outer), holes=hole_paths))
    return profiles


def _path_from_contour(contour: np.ndarray) -> Path2D:
    pts = np.asarray(contour, dtype=float)
    return Path2D.from_points(pts, closed=True)


def _largest_area(contours: list[np.ndarray]) -> np.ndarray:
    return max(contours, key=lambda c: abs(_signed_area(c)))


def _point_in_poly(point: np.ndarray, poly: np.ndarray) -> bool:
    x, y = point
    inside = False
    n = poly.shape[0]
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0):
            inside = not inside
    return inside


def _load_cross_section():
    try:
        from manifold3d import CrossSection, JoinType
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("manifold3d is required for 2D offset/hull.") from exc
    return CrossSection, JoinType


def _manifold_from_mesh_group(mesh: Mesh | MeshGroup):
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


def _mesh_from_manifold(manifold) -> Mesh:
    mesh = manifold.to_mesh() if hasattr(manifold, "to_mesh") else manifold.mesh
    vertices = np.asarray(getattr(mesh, "vertices", mesh.vert_properties), dtype=float)
    faces = np.asarray(getattr(mesh, "triangles", mesh.tri_verts), dtype=int)
    return Mesh(vertices, faces)


__all__ = ["offset", "hull", "minkowski"]
