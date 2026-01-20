from __future__ import annotations

from typing import Literal, Sequence, Tuple

import numpy as np

from impression.mesh import Mesh, triangulate_faces

from ._color import set_mesh_color

Backend = Literal["mesh"]


def _ensure_backend(backend: Backend) -> None:
    if backend != "mesh":
        raise ValueError(f"Unsupported backend '{backend}'. Only 'mesh' is available right now.")


def make_box(
    size: Sequence[float] = (1.0, 1.0, 1.0),
    center: Sequence[float] = (0.0, 0.0, 0.0),
    backend: Backend = "mesh",
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Axis-aligned box specified by size (dx, dy, dz) and center."""

    _ensure_backend(backend)
    mesh = _box_mesh(size, center)
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
    backend: Backend = "mesh",
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Right circular cylinder aligned with `direction`."""

    _ensure_backend(backend)
    direction = _normalize(direction)
    mesh = _circular_frustum_mesh(radius, radius, height, resolution, capping=capping)
    mesh = _orient_mesh(mesh, direction)
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
    backend: Backend = "mesh",
    color: Sequence[float] | str | None = None,
    *,
    side_length: float | None = None,
) -> Mesh:
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

    direction = _normalize(direction)
    mesh = _circular_frustum_mesh(radius, radius, height, sides, capping=True)
    mesh = _orient_mesh(mesh, direction)
    mesh.translate(center, inplace=True)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def make_nhedron(
    faces: int = 6,
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    backend: Backend = "mesh",
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Regular polyhedron specified by number of faces (4, 6, 8, 12, 20)."""

    _ensure_backend(backend)
    faces = int(faces)
    if radius <= 0:
        raise ValueError("radius must be positive.")

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


def make_sphere(
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    theta_resolution: int = 64,
    phi_resolution: int = 64,
    backend: Backend = "mesh",
    color: Sequence[float] | str | None = None,
) -> Mesh:
    _ensure_backend(backend)
    mesh = _sphere_mesh(radius, theta_resolution, phi_resolution)
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
    backend: Backend = "mesh",
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Generate a torus (donut) with given major/minor radii."""

    _ensure_backend(backend)
    direction = _normalize(direction)
    base = _torus_mesh(major_radius, minor_radius, n_theta, n_phi)
    aligned = _orient_mesh(base, direction)
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
    backend: Backend = "mesh",
    color: Sequence[float] | str | None = None,
    *,
    radius: float | None = None,
) -> Mesh:
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

    mesh = _circular_frustum_mesh(bottom_radius, top_radius, height, resolution)
    mesh = _orient_mesh(mesh, direction)
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
    backend: Backend = "mesh",
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """
    Rectangular frustum (pyramid/prism). Set top_size=(0,0) for a pyramid, or None to match base.
    """

    _ensure_backend(backend)
    if top_size is None:
        top_size = tuple(base_size)
    mesh = _rectangular_frustum_mesh(tuple(base_size), tuple(top_size), height)
    mesh = _orient_mesh(mesh, direction)
    mesh.translate(center, inplace=True)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def _normalize(vector: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        raise ValueError("Direction vector must be non-zero.")
    arr = arr / norm
    return float(arr[0]), float(arr[1]), float(arr[2])


def _orient_mesh(mesh: Mesh, direction: Sequence[float]) -> Mesh:
    target = np.asarray(direction, dtype=float)
    target_norm = np.linalg.norm(target)
    if target_norm == 0:
        raise ValueError("Direction vector must be non-zero.")
    target = target / target_norm
    default = np.array([0.0, 0.0, 1.0])
    if np.allclose(target, default):
        return mesh.copy()
    axis = np.cross(default, target)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        # opposite direction; rotate 180 around X
        axis = np.array([1.0, 0.0, 0.0])
        angle_deg = 180.0
    else:
        axis = axis / axis_norm
        angle_rad = np.arccos(np.clip(np.dot(default, target), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
    rotated = mesh.copy()
    rotated.rotate_vector(axis, angle_deg, point=(0.0, 0.0, 0.0), inplace=True)
    return rotated


def _box_mesh(size: Sequence[float], center: Sequence[float]) -> Mesh:
    sx, sy, sz = size
    cx, cy, cz = center
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    points = np.array(
        [
            (-hx, -hy, -hz),
            (hx, -hy, -hz),
            (hx, hy, -hz),
            (-hx, hy, -hz),
            (-hx, -hy, hz),
            (hx, -hy, hz),
            (hx, hy, hz),
            (-hx, hy, hz),
        ]
    )
    points += np.array([cx, cy, cz], dtype=float)
    faces = np.array(
        [
            [0, 2, 1],  # bottom (-Z)
            [0, 3, 2],
            [4, 5, 6],  # top (+Z)
            [4, 6, 7],
            [0, 1, 5],  # -Y
            [0, 5, 4],
            [1, 2, 6],  # +X
            [1, 6, 5],
            [2, 3, 7],  # +Y
            [2, 7, 6],
            [0, 4, 7],  # -X
            [0, 7, 3],
        ],
        dtype=int,
    )
    return Mesh(points, faces)


def _sphere_mesh(radius: float, theta_resolution: int, phi_resolution: int) -> Mesh:
    theta_steps = max(int(theta_resolution), 3)
    phi_points = max(int(phi_resolution), 3)

    points = [
        (0.0, 0.0, radius),
        (0.0, 0.0, -radius),
    ]

    ring_count = phi_points - 2
    for i in range(1, phi_points - 1):
        phi = np.pi * i / (phi_points - 1)
        z = radius * np.cos(phi)
        ring_radius = radius * np.sin(phi)
        for j in range(theta_steps):
            theta = 2 * np.pi * j / theta_steps
            points.append((ring_radius * np.cos(theta), ring_radius * np.sin(theta), z))

    faces: list[list[int]] = []
    ring_start = 2
    if ring_count > 0:
        for j in range(theta_steps):
            j_next = (j + 1) % theta_steps
            faces.append([0, ring_start + j, ring_start + j_next])

        for i in range(ring_count - 1):
            ring0 = ring_start + i * theta_steps
            ring1 = ring_start + (i + 1) * theta_steps
            for j in range(theta_steps):
                j_next = (j + 1) % theta_steps
                faces.append([ring0 + j, ring1 + j, ring1 + j_next, ring0 + j_next])

        last_ring = ring_start + (ring_count - 1) * theta_steps
        for j in range(theta_steps):
            j_next = (j + 1) % theta_steps
            faces.append([1, last_ring + j_next, last_ring + j])

    points_arr = np.asarray(points, dtype=float)
    faces_arr = triangulate_faces(faces)
    return Mesh(points_arr, faces_arr)


def _torus_mesh(major_radius: float, minor_radius: float, n_theta: int, n_phi: int) -> Mesh:
    theta_steps = max(int(n_theta), 3)
    phi_steps = max(int(n_phi), 3)
    points = []
    for i in range(theta_steps):
        u = 2 * np.pi * i / theta_steps
        cos_u = np.cos(u)
        sin_u = np.sin(u)
        for j in range(phi_steps):
            v = 2 * np.pi * j / phi_steps
            cos_v = np.cos(v)
            sin_v = np.sin(v)
            radial = major_radius + minor_radius * cos_v
            points.append((radial * cos_u, radial * sin_u, minor_radius * sin_v))

    faces: list[list[int]] = []
    for i in range(theta_steps):
        i_next = (i + 1) % theta_steps
        for j in range(phi_steps):
            j_next = (j + 1) % phi_steps
            idx0 = i * phi_steps + j
            idx1 = i * phi_steps + j_next
            idx2 = i_next * phi_steps + j_next
            idx3 = i_next * phi_steps + j
            faces.append([idx0, idx1, idx2, idx3])

    points_arr = np.asarray(points, dtype=float)
    faces_arr = triangulate_faces(faces)
    if faces_arr.size:
        faces_arr = faces_arr[:, [0, 2, 1]]
    return Mesh(points_arr, faces_arr)


def _circular_frustum_mesh(
    bottom_radius: float,
    top_radius: float,
    height: float,
    resolution: int,
    capping: bool = True,
) -> Mesh:
    resolution = max(int(resolution), 3)
    bottom_radius = max(bottom_radius, 0.0)
    top_radius = max(top_radius, 0.0)
    z_bottom = -height / 2.0
    z_top = height / 2.0
    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)

    points = []
    faces: list[list[int]] = []

    def ring_points(radius: float, z: float) -> np.ndarray:
        return np.column_stack(
            [
                radius * np.cos(angles),
                radius * np.sin(angles),
                np.full_like(angles, z),
            ]
        )

    bottom_has_ring = bottom_radius > 0
    top_has_ring = top_radius > 0

    if bottom_has_ring:
        bottom = ring_points(bottom_radius, z_bottom)
        bottom_indices = np.arange(len(bottom))
        points.append(bottom)
    else:
        bottom = np.array([[0.0, 0.0, z_bottom]])
        bottom_indices = np.array([0])
        points.append(bottom)

    if top_has_ring:
        top = ring_points(top_radius, z_top)
        top_indices = np.arange(len(points[0]), len(points[0]) + len(top))
        points.append(top)
    else:
        top = np.array([[0.0, 0.0, z_top]])
        top_indices = np.array([len(points[0])])
        points.append(top)

    points_arr = np.vstack(points)

    if bottom_has_ring and top_has_ring:
        count = resolution
        for i in range(count):
            j = (i + 1) % count
            faces.append([bottom_indices[i], bottom_indices[j], top_indices[j], top_indices[i]])
        if capping:
            faces.append(list(bottom_indices[::-1]))
            faces.append(list(top_indices))
    elif bottom_has_ring:
        apex = int(top_indices[0])
        count = resolution
        for i in range(count):
            j = (i + 1) % count
            faces.append([bottom_indices[i], bottom_indices[j], apex])
        if capping:
            faces.append(list(bottom_indices[::-1]))
    else:
        # inverted cone (top ring, bottom apex)
        apex = int(bottom_indices[0])
        count = resolution
        for i in range(count):
            j = (i + 1) % count
            faces.append([apex, top_indices[j], top_indices[i]])
        if capping:
            faces.append(list(top_indices))

    faces_arr = triangulate_faces(faces)
    return Mesh(points_arr, faces_arr)


def _rectangular_frustum_mesh(
    base_size: Tuple[float, float],
    top_size: Tuple[float, float],
    height: float,
) -> Mesh:
    hx, hy = base_size[0] / 2.0, base_size[1] / 2.0
    tx, ty = top_size[0] / 2.0, top_size[1] / 2.0
    z_bottom = -height / 2.0
    z_top = height / 2.0

    bottom_pts = np.array(
        [
            (-hx, -hy, z_bottom),
            (hx, -hy, z_bottom),
            (hx, hy, z_bottom),
            (-hx, hy, z_bottom),
        ]
    )

    if top_size[0] == 0 and top_size[1] == 0:
        top_pts = np.array([[0.0, 0.0, z_top]])
        apex_only = True
    else:
        top_pts = np.array(
            [
                (-tx, -ty, z_top),
                (tx, -ty, z_top),
                (tx, ty, z_top),
                (-tx, ty, z_top),
            ]
        )
        apex_only = False

    points = np.vstack([bottom_pts, top_pts])
    faces: list[list[int]] = []
    bottom_indices = np.arange(4)

    if apex_only:
        apex_idx = 4
        for i in range(4):
            j = (i + 1) % 4
            faces.append([bottom_indices[i], bottom_indices[j], apex_idx])
        faces.append(list(bottom_indices[::-1]))
    else:
        top_indices = np.arange(4, 8)
        for i in range(4):
            j = (i + 1) % 4
            faces.append([bottom_indices[i], bottom_indices[j], top_indices[j], top_indices[i]])
        faces.append(list(bottom_indices[::-1]))
        faces.append(list(top_indices))

    faces_arr = triangulate_faces(faces)
    return Mesh(points, faces_arr)


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
