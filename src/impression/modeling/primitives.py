from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Sequence, Tuple, cast

import numpy as np

from impression.mesh import Mesh, triangulate_faces

from ._color import set_mesh_color
if TYPE_CHECKING:
    from .surface import SurfaceBody

Backend = Literal["mesh", "surface"]


@dataclass(frozen=True)
class PrimitiveCSGRouteRecord:
    """Primitive authored route covered by the no-hidden-mesh CSG policy."""

    caller_id: str
    surface_constructor: str
    explicit_mesh_constructor: str
    csg_gate: str = "assert_no_hidden_surface_csg_mesh_fallback"

    def canonical_payload(self) -> dict[str, object]:
        return {
            "caller_id": self.caller_id,
            "csg_gate": self.csg_gate,
            "explicit_mesh_constructor": self.explicit_mesh_constructor,
            "surface_constructor": self.surface_constructor,
        }


PRIMITIVE_CSG_ROUTE_INVENTORY: tuple[PrimitiveCSGRouteRecord, ...] = (
    PrimitiveCSGRouteRecord("primitive.make_box", "make_box", "make_box_mesh"),
    PrimitiveCSGRouteRecord("primitive.make_cylinder", "make_cylinder", "make_cylinder_mesh"),
    PrimitiveCSGRouteRecord("primitive.make_ngon", "make_ngon", "make_ngon_mesh"),
    PrimitiveCSGRouteRecord("primitive.make_polyhedron", "make_polyhedron", "make_polyhedron_mesh"),
    PrimitiveCSGRouteRecord("primitive.make_nhedron", "make_nhedron", "make_nhedron_mesh"),
    PrimitiveCSGRouteRecord("primitive.make_sphere", "make_sphere", "make_sphere_mesh"),
    PrimitiveCSGRouteRecord("primitive.make_torus", "make_torus", "make_torus_mesh"),
    PrimitiveCSGRouteRecord("primitive.make_cone", "make_cone", "make_cone_mesh"),
    PrimitiveCSGRouteRecord("primitive.make_prism", "make_prism", "make_prism_mesh"),
)


def primitive_csg_route_inventory() -> tuple[PrimitiveCSGRouteRecord, ...]:
    """Return primitive authored routes guarded against hidden mesh fallback."""

    return PRIMITIVE_CSG_ROUTE_INVENTORY


def _ensure_backend(backend: Backend) -> None:
    if backend not in {"mesh", "surface"}:
        raise ValueError(
            f"Unsupported backend '{backend}'. Only 'mesh' and 'surface' are available right now."
        )


def _surface_metadata(*, color: Sequence[float] | str | None) -> dict[str, object] | None:
    if color is None:
        return None
    return {"consumer": {"color": color}}


def _surface_primitive_result(caller_id: str, result: SurfaceBody) -> SurfaceBody:
    from .csg import assert_no_hidden_surface_csg_mesh_fallback

    return cast("SurfaceBody", assert_no_hidden_surface_csg_mesh_fallback(caller_id, result))


def make_box(
    size: Sequence[float] = (1.0, 1.0, 1.0),
    center: Sequence[float] = (0.0, 0.0, 0.0),
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
) -> Mesh | SurfaceBody:
    """Axis-aligned box specified by size (dx, dy, dz) and center."""

    _ensure_backend(backend)
    if backend == "surface":
        from ._surface_primitives import make_surface_box

        return _surface_primitive_result(
            "primitive.make_box",
            make_surface_box(size=size, center=center, metadata=_surface_metadata(color=color)),
        )

    from ._legacy_mesh_primitives import box_mesh

    mesh = box_mesh(size, center)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_cylinder(
    radius: float = 0.5,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    resolution: int = 128,
    capping: bool = True,
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
) -> Mesh | SurfaceBody:
    """Right circular cylinder aligned with `direction`."""

    _ensure_backend(backend)
    if backend == "surface":
        from ._surface_primitives import make_surface_cylinder

        return _surface_primitive_result(
            "primitive.make_cylinder",
            make_surface_cylinder(
                radius=radius,
                height=height,
                center=center,
                direction=direction,
                resolution=resolution,
                capping=capping,
                metadata=_surface_metadata(color=color),
            ),
        )

    direction = _normalize(direction)
    from ._legacy_mesh_primitives import circular_frustum_mesh, orient_mesh

    mesh = circular_frustum_mesh(radius, radius, height, resolution, capping=capping)
    mesh = orient_mesh(mesh, direction)
    mesh.translate(center, inplace=True)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_ngon(
    sides: int = 6,
    radius: float = 0.5,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
    *,
    side_length: float | None = None,
) -> Mesh | SurfaceBody:
    """Regular n-gon prism aligned to `direction`."""

    _ensure_backend(backend)
    sides = int(sides)
    if sides < 3:
        raise ValueError("sides must be >= 3.")
    if side_length is not None:
        inferred = float(side_length) / (2.0 * np.sin(np.pi / sides))
        if radius != 0.5 and not np.isclose(radius, inferred):
            raise ValueError("Specify either radius or side_length, not both.")
        radius = inferred

    if backend == "surface":
        from ._surface_primitives import make_surface_ngon

        return _surface_primitive_result(
            "primitive.make_ngon",
            make_surface_ngon(
                sides=sides,
                radius=radius,
                height=height,
                center=center,
                direction=direction,
                side_length=side_length,
                metadata=_surface_metadata(color=color),
            ),
        )

    direction = _normalize(direction)
    from ._legacy_mesh_primitives import circular_frustum_mesh, orient_mesh

    mesh = circular_frustum_mesh(radius, radius, height, sides, capping=True)
    mesh = orient_mesh(mesh, direction)
    mesh.translate(center, inplace=True)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_polyhedron(
    faces: int = 6,
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
) -> Mesh | SurfaceBody:
    """Regular polyhedron specified by number of faces (4, 6, 8, 12, 20)."""

    _ensure_backend(backend)
    faces = int(faces)
    if radius <= 0:
        raise ValueError("radius must be positive.")
    if backend == "surface":
        from ._surface_primitives import make_surface_polyhedron

        return _surface_primitive_result(
            "primitive.make_polyhedron",
            make_surface_polyhedron(
                faces=faces,
                radius=radius,
                center=center,
                metadata=_surface_metadata(color=color),
            ),
        )

    vertices, face_list = _regular_polyhedron_data(faces)
    vertices = np.asarray(vertices, dtype=float)
    faces_arr = triangulate_faces(face_list)
    if faces_arr.size:
        faces_arr = _orient_faces_outward(vertices, faces_arr)

    max_norm = np.linalg.norm(vertices, axis=1).max(initial=0.0)
    if max_norm > 0:
        vertices = vertices * (radius / max_norm)
    vertices = vertices + np.asarray(center, dtype=float).reshape(3)

    mesh = Mesh(vertices, faces_arr)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_nhedron(
    faces: int = 6,
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
) -> Mesh | SurfaceBody:
    """Compatibility wrapper for make_polyhedron."""

    result = make_polyhedron(
        faces=faces,
        radius=radius,
        center=center,
        backend=backend,
        color=color,
    )
    if backend == "surface":
        return _surface_primitive_result("primitive.make_nhedron", cast("SurfaceBody", result))
    return result


def make_sphere(
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    theta_resolution: int = 64,
    phi_resolution: int = 64,
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
) -> Mesh | SurfaceBody:
    _ensure_backend(backend)
    if backend == "surface":
        from ._surface_primitives import make_surface_sphere

        return _surface_primitive_result(
            "primitive.make_sphere",
            make_surface_sphere(
                radius=radius,
                center=center,
                theta_resolution=theta_resolution,
                phi_resolution=phi_resolution,
                metadata=_surface_metadata(color=color),
            ),
        )
    from ._legacy_mesh_primitives import sphere_mesh

    mesh = sphere_mesh(radius, theta_resolution, phi_resolution)
    mesh.translate(center, inplace=True)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_torus(
    major_radius: float = 1.0,
    minor_radius: float = 0.25,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    n_theta: int = 64,
    n_phi: int = 32,
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
) -> Mesh | SurfaceBody:
    """Generate a torus (donut) with given major/minor radii."""

    _ensure_backend(backend)
    if backend == "surface":
        from ._surface_primitives import make_surface_torus

        return _surface_primitive_result(
            "primitive.make_torus",
            make_surface_torus(
                major_radius=major_radius,
                minor_radius=minor_radius,
                center=center,
                direction=direction,
                n_theta=n_theta,
                n_phi=n_phi,
                metadata=_surface_metadata(color=color),
            ),
        )

    direction = _normalize(direction)
    from ._legacy_mesh_primitives import orient_mesh, torus_mesh

    base = torus_mesh(major_radius, minor_radius, n_theta, n_phi)
    aligned = orient_mesh(base, direction)
    aligned.translate(center, inplace=True)
    if color is not None:
        set_mesh_color(aligned, color)
    return aligned


def make_cone(
    bottom_diameter: float = 1.0,
    top_diameter: float = 0.0,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    resolution: int = 64,
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
    *,
    radius: float | None = None,
) -> Mesh | SurfaceBody:
    """Circular frustum. Set top_diameter=0 for a classic cone."""

    _ensure_backend(backend)
    if radius is not None:
        inferred_bottom = 2.0 * float(radius)
        if bottom_diameter != 1.0 and not np.isclose(bottom_diameter, inferred_bottom):
            raise ValueError("Specify either bottom_diameter or radius, not both.")
        bottom_diameter = inferred_bottom

    bottom_radius = bottom_diameter / 2.0
    top_radius = top_diameter / 2.0
    if bottom_radius <= 0 and top_radius <= 0:
        raise ValueError("At least one of bottom_diameter or top_diameter must be > 0.")
    if backend == "surface":
        from ._surface_primitives import make_surface_cone

        return _surface_primitive_result(
            "primitive.make_cone",
            make_surface_cone(
                bottom_diameter=bottom_diameter,
                top_diameter=top_diameter,
                height=height,
                center=center,
                direction=direction,
                resolution=resolution,
                metadata=_surface_metadata(color=color),
            ),
        )

    from ._legacy_mesh_primitives import circular_frustum_mesh, orient_mesh

    mesh = circular_frustum_mesh(bottom_radius, top_radius, height, resolution)
    mesh = orient_mesh(mesh, direction)
    mesh.translate(center, inplace=True)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_prism(
    base_size: Sequence[float] = (1.0, 1.0),
    top_size: Sequence[float] | None = None,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    backend: Backend = "surface",
    color: Sequence[float] | str | None = None,
) -> Mesh | SurfaceBody:
    """
    Rectangular frustum (pyramid/prism). Set top_size=(0,0) for a pyramid, or None to match base.
    """

    _ensure_backend(backend)
    if backend == "surface":
        from ._surface_primitives import make_surface_prism

        return _surface_primitive_result(
            "primitive.make_prism",
            make_surface_prism(
                base_size=base_size,
                top_size=top_size,
                height=height,
                center=center,
                direction=direction,
                metadata=_surface_metadata(color=color),
            ),
        )

    if top_size is None:
        top_size = tuple(base_size)
    from ._legacy_mesh_primitives import orient_mesh, rectangular_frustum_mesh

    mesh = rectangular_frustum_mesh(tuple(base_size), tuple(top_size), height)
    mesh = orient_mesh(mesh, direction)
    mesh.translate(center, inplace=True)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_box_mesh(
    size: Sequence[float] = (1.0, 1.0, 1.0),
    center: Sequence[float] = (0.0, 0.0, 0.0),
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_box`."""

    return cast(Mesh, make_box(size=size, center=center, backend="mesh", color=color))


def make_cylinder_mesh(
    radius: float = 0.5,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    resolution: int = 128,
    capping: bool = True,
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_cylinder`."""

    return cast(
        Mesh,
        make_cylinder(
            radius=radius,
            height=height,
            center=center,
            direction=direction,
            resolution=resolution,
            capping=capping,
            backend="mesh",
            color=color,
        ),
    )


def make_ngon_mesh(
    sides: int = 6,
    radius: float = 0.5,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    color: Sequence[float] | str | None = None,
    *,
    side_length: float | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_ngon`."""

    return cast(
        Mesh,
        make_ngon(
            sides=sides,
            radius=radius,
            height=height,
            center=center,
            direction=direction,
            backend="mesh",
            color=color,
            side_length=side_length,
        ),
    )


def make_polyhedron_mesh(
    faces: int = 6,
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_polyhedron`."""

    return cast(Mesh, make_polyhedron(faces=faces, radius=radius, center=center, backend="mesh", color=color))


def make_nhedron_mesh(
    faces: int = 6,
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_nhedron`."""

    return cast(Mesh, make_nhedron(faces=faces, radius=radius, center=center, backend="mesh", color=color))


def make_sphere_mesh(
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    theta_resolution: int = 64,
    phi_resolution: int = 64,
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_sphere`."""

    return cast(
        Mesh,
        make_sphere(
            radius=radius,
            center=center,
            theta_resolution=theta_resolution,
            phi_resolution=phi_resolution,
            backend="mesh",
            color=color,
        ),
    )


def make_torus_mesh(
    major_radius: float = 1.0,
    minor_radius: float = 0.25,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    n_theta: int = 64,
    n_phi: int = 32,
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_torus`."""

    return cast(
        Mesh,
        make_torus(
            major_radius=major_radius,
            minor_radius=minor_radius,
            center=center,
            direction=direction,
            n_theta=n_theta,
            n_phi=n_phi,
            backend="mesh",
            color=color,
        ),
    )


def make_cone_mesh(
    bottom_diameter: float = 1.0,
    top_diameter: float = 0.0,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    resolution: int = 64,
    color: Sequence[float] | str | None = None,
    *,
    radius: float | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_cone`."""

    return cast(
        Mesh,
        make_cone(
            bottom_diameter=bottom_diameter,
            top_diameter=top_diameter,
            height=height,
            center=center,
            direction=direction,
            resolution=resolution,
            backend="mesh",
            color=color,
            radius=radius,
        ),
    )


def make_prism_mesh(
    base_size: Sequence[float] = (1.0, 1.0),
    top_size: Sequence[float] | None = None,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Explicit mesh compatibility constructor for :func:`make_prism`."""

    return cast(
        Mesh,
        make_prism(
            base_size=base_size,
            top_size=top_size,
            height=height,
            center=center,
            direction=direction,
            backend="mesh",
            color=color,
        ),
    )


def _normalize(vector: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Direction vector must be non-zero.")
    arr = arr / norm
    return float(arr[0]), float(arr[1]), float(arr[2])


def _orient_faces_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return faces
    oriented = faces.copy()
    centers = vertices[oriented].mean(axis=1)
    v1 = vertices[oriented[:, 1]] - vertices[oriented[:, 0]]
    v2 = vertices[oriented[:, 2]] - vertices[oriented[:, 0]]
    normals = np.cross(v1, v2)
    dots = np.einsum("ij,ij->i", normals, centers)
    flip = dots < 0
    if np.any(flip):
        oriented[flip] = oriented[flip][:, [0, 2, 1]]
    return oriented


def _regular_polyhedron_data(face_count: int) -> tuple[np.ndarray, list[list[int]]]:
    if face_count == 4:
        vertices = np.array(
            [
                (1.0, 1.0, 1.0),
                (-1.0, -1.0, 1.0),
                (-1.0, 1.0, -1.0),
                (1.0, -1.0, -1.0),
            ]
        )
        faces = [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ]
        return vertices, faces

    if face_count == 6:
        vertices = np.array(
            [
                (-1.0, -1.0, -1.0),
                (1.0, -1.0, -1.0),
                (1.0, 1.0, -1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, -1.0, 1.0),
                (1.0, -1.0, 1.0),
                (1.0, 1.0, 1.0),
                (-1.0, 1.0, 1.0),
            ]
        )
        faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ]
        return vertices, faces

    if face_count == 8:
        vertices = np.array(
            [
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ]
        )
        faces = [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ]
        return vertices, faces

    if face_count == 20:
        return _icosahedron_data()

    if face_count == 12:
        ico_vertices, ico_faces = _icosahedron_data()
        dodeca_vertices, dodeca_faces = _dodecahedron_from_icosa(ico_vertices, ico_faces)
        return dodeca_vertices, dodeca_faces

    raise ValueError("faces must be one of: 4, 6, 8, 12, 20.")


def _icosahedron_data() -> tuple[np.ndarray, list[list[int]]]:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            (-1.0, phi, 0.0),
            (1.0, phi, 0.0),
            (-1.0, -phi, 0.0),
            (1.0, -phi, 0.0),
            (0.0, -1.0, phi),
            (0.0, 1.0, phi),
            (0.0, -1.0, -phi),
            (0.0, 1.0, -phi),
            (phi, 0.0, -1.0),
            (phi, 0.0, 1.0),
            (-phi, 0.0, -1.0),
            (-phi, 0.0, 1.0),
        ]
    )
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]
    return vertices, faces


def _dodecahedron_from_icosa(
    ico_vertices: np.ndarray, ico_faces: list[list[int]]
) -> tuple[np.ndarray, list[list[int]]]:
    ico_vertices = np.asarray(ico_vertices, dtype=float)
    centroids = np.asarray([ico_vertices[face].mean(axis=0) for face in ico_faces], dtype=float)
    norms = np.linalg.norm(centroids, axis=1)
    centroids = centroids / norms[:, None]

    face_map: dict[int, list[int]] = {idx: [] for idx in range(len(ico_vertices))}
    for face_idx, face in enumerate(ico_faces):
        for vert_idx in face:
            face_map[vert_idx].append(face_idx)

    faces: list[list[int]] = []
    for vert_idx, face_indices in face_map.items():
        normal = ico_vertices[vert_idx]
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            continue
        normal = normal / normal_norm
        axis = np.array([0.0, 0.0, 1.0])
        if abs(normal[2]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        basis_u = np.cross(axis, normal)
        basis_norm = np.linalg.norm(basis_u)
        if basis_norm == 0:
            basis_u = np.array([1.0, 0.0, 0.0])
        else:
            basis_u = basis_u / basis_norm
        basis_v = np.cross(normal, basis_u)

        angles: list[tuple[float, int]] = []
        for face_idx in face_indices:
            vec = centroids[face_idx]
            vec = vec - normal * np.dot(vec, normal)
            x = float(np.dot(vec, basis_u))
            y = float(np.dot(vec, basis_v))
            angles.append((np.arctan2(y, x), face_idx))
        ordered = [idx for _, idx in sorted(angles)]
        faces.append(ordered)

    return centroids, faces
