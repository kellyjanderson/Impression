from __future__ import annotations

from typing import Sequence

import numpy as np

from ._surface_ops import make_surface_linear_extrude
from .drawing2d import make_ngon as make_ngon_2d, make_rect
from .primitives import _regular_polyhedron_data
from .surface import (
    ParameterDomain,
    PlanarSurfacePatch,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    SurfaceBody,
    SurfaceBoundaryRef,
    SurfaceSeam,
    TrimLoop,
    make_surface_body,
    make_surface_shell,
)


def _as_vec3(value: Sequence[float], *, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{name} must contain only finite values.")
    return vec


def _normalize(value: Sequence[float], *, name: str) -> np.ndarray:
    vec = _as_vec3(value, name=name)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        raise ValueError(f"{name} must be non-zero.")
    return vec / norm


def _attached_transform(center: Sequence[float], direction: Sequence[float]) -> np.ndarray:
    origin = _as_vec3(center, name="center")
    z_axis = _normalize(direction, name="direction")
    reference = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(z_axis, reference))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0], dtype=float)
    x_axis = np.cross(reference, z_axis)
    x_axis = x_axis / float(np.linalg.norm(x_axis))
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / float(np.linalg.norm(y_axis))

    transform = np.eye(4, dtype=float)
    transform[:3, 0] = x_axis
    transform[:3, 1] = y_axis
    transform[:3, 2] = z_axis
    transform[:3, 3] = origin
    return transform


def _circle_loop(radius: float, *, samples: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=False)
    return np.column_stack((np.cos(angles) * radius, np.sin(angles) * radius)).astype(float)


def _torus_profile(major_radius: float, minor_radius: float, *, samples: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=False)
    x = major_radius + (np.cos(angles) * minor_radius)
    z = np.sin(angles) * minor_radius
    profile = np.column_stack((x, np.zeros_like(x), z)).astype(float)
    return np.vstack([profile, profile[0]])


def _sphere_profile(radius: float, *, samples: int) -> np.ndarray:
    angles = np.linspace(0.0, np.pi, int(samples))
    angles = np.unique(np.concatenate([angles, np.asarray([np.pi / 2.0])]))
    x = np.sin(angles) * radius
    z = np.cos(angles) * radius
    return np.column_stack((x, np.zeros_like(x), z)).astype(float)


def _planar_patch_from_face(points: np.ndarray, *, metadata: dict[str, object]) -> PlanarSurfacePatch:
    face_points = np.asarray(points, dtype=float).reshape(-1, 3)
    if face_points.shape[0] < 3:
        raise ValueError("Polyhedron face requires at least three points.")
    origin = face_points[0]
    u_seed = face_points[1] - origin
    u_norm = float(np.linalg.norm(u_seed))
    if u_norm == 0.0:
        raise ValueError("Polyhedron face has degenerate first edge.")
    u_dir = u_seed / u_norm
    normal = np.cross(face_points[1] - origin, face_points[2] - origin)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm == 0.0:
        raise ValueError("Polyhedron face points must not be collinear.")
    normal = normal / normal_norm
    v_dir = np.cross(normal, u_dir)
    v_dir = v_dir / float(np.linalg.norm(v_dir))

    uv_points = np.asarray(
        [
            (
                float(np.dot(point - origin, u_dir)),
                float(np.dot(point - origin, v_dir)),
            )
            for point in face_points
        ],
        dtype=float,
    )
    xmin, ymin = uv_points.min(axis=0)
    xmax, ymax = uv_points.max(axis=0)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Polyhedron face trim domain must have positive span.")
    return PlanarSurfacePatch(
        family="planar",
        domain=ParameterDomain((xmin, xmax), (ymin, ymax)),
        trim_loops=(TrimLoop(uv_points, category="outer"),),
        origin=origin,
        u_axis=u_dir,
        v_axis=v_dir,
        metadata=metadata,
    )


def make_surface_box(
    size: Sequence[float] = (1.0, 1.0, 1.0),
    center: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    metadata: dict[str, object] | None = None,
) -> SurfaceBody:
    """Internal surface-native box constructor.

    This builder is intentionally private until the public primitive API
    migration is ready. It returns one closed shell made from six planar patches
    with explicit shared seams.
    """

    sx, sy, sz = _as_vec3(size, name="size")
    if sx <= 0.0 or sy <= 0.0 or sz <= 0.0:
        raise ValueError("size components must all be > 0.")
    cx, cy, cz = _as_vec3(center, name="center")
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0

    patches = (
        PlanarSurfacePatch(  # front
            family="planar",
            origin=(cx - hx, cy - hy, cz - hz),
            u_axis=(sx, 0.0, 0.0),
            v_axis=(0.0, sy, 0.0),
            metadata={"kernel": {"primitive_family": "box", "face": "front"}},
        ),
        PlanarSurfacePatch(  # right
            family="planar",
            origin=(cx + hx, cy - hy, cz - hz),
            u_axis=(0.0, 0.0, sz),
            v_axis=(0.0, sy, 0.0),
            metadata={"kernel": {"primitive_family": "box", "face": "right"}},
        ),
        PlanarSurfacePatch(  # back
            family="planar",
            origin=(cx + hx, cy - hy, cz + hz),
            u_axis=(-sx, 0.0, 0.0),
            v_axis=(0.0, sy, 0.0),
            metadata={"kernel": {"primitive_family": "box", "face": "back"}},
        ),
        PlanarSurfacePatch(  # left
            family="planar",
            origin=(cx - hx, cy - hy, cz + hz),
            u_axis=(0.0, 0.0, -sz),
            v_axis=(0.0, sy, 0.0),
            metadata={"kernel": {"primitive_family": "box", "face": "left"}},
        ),
        PlanarSurfacePatch(  # top
            family="planar",
            origin=(cx - hx, cy + hy, cz - hz),
            u_axis=(sx, 0.0, 0.0),
            v_axis=(0.0, 0.0, sz),
            metadata={"kernel": {"primitive_family": "box", "face": "top"}},
        ),
        PlanarSurfacePatch(  # bottom
            family="planar",
            origin=(cx - hx, cy - hy, cz + hz),
            u_axis=(sx, 0.0, 0.0),
            v_axis=(0.0, 0.0, -sz),
            metadata={"kernel": {"primitive_family": "box", "face": "bottom"}},
        ),
    )

    seams = (
        SurfaceSeam("front-right", (SurfaceBoundaryRef(0, "right"), SurfaceBoundaryRef(1, "left"))),
        SurfaceSeam("right-back", (SurfaceBoundaryRef(1, "right"), SurfaceBoundaryRef(2, "left"))),
        SurfaceSeam("back-left", (SurfaceBoundaryRef(2, "right"), SurfaceBoundaryRef(3, "left"))),
        SurfaceSeam("left-front", (SurfaceBoundaryRef(3, "right"), SurfaceBoundaryRef(0, "left"))),
        SurfaceSeam("front-top", (SurfaceBoundaryRef(0, "top"), SurfaceBoundaryRef(4, "bottom"))),
        SurfaceSeam("right-top", (SurfaceBoundaryRef(1, "top"), SurfaceBoundaryRef(4, "right"))),
        SurfaceSeam("back-top", (SurfaceBoundaryRef(2, "top"), SurfaceBoundaryRef(4, "top"))),
        SurfaceSeam("left-top", (SurfaceBoundaryRef(3, "top"), SurfaceBoundaryRef(4, "left"))),
        SurfaceSeam("front-bottom", (SurfaceBoundaryRef(0, "bottom"), SurfaceBoundaryRef(5, "top"))),
        SurfaceSeam("right-bottom", (SurfaceBoundaryRef(1, "bottom"), SurfaceBoundaryRef(5, "right"))),
        SurfaceSeam("back-bottom", (SurfaceBoundaryRef(2, "bottom"), SurfaceBoundaryRef(5, "bottom"))),
        SurfaceSeam("left-bottom", (SurfaceBoundaryRef(3, "bottom"), SurfaceBoundaryRef(5, "left"))),
    )

    shell = make_surface_shell(patches, seams=seams, metadata={"kernel": {"primitive_family": "box"}})
    return make_surface_body((shell,), metadata=metadata)


def make_surface_ngon(
    sides: int = 6,
    radius: float = 0.5,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    *,
    side_length: float | None = None,
    metadata: dict[str, object] | None = None,
) -> SurfaceBody:
    """Internal surface-native regular n-gon prism constructor."""

    shape = make_ngon_2d(sides=sides, radius=radius, side_length=side_length)
    body = make_surface_linear_extrude(
        shape,
        height=height,
        direction=direction,
        center=center,
        metadata=metadata,
    )
    return body


def make_surface_cylinder(
    radius: float = 0.5,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    resolution: int = 128,
    capping: bool = True,
    *,
    metadata: dict[str, object] | None = None,
) -> SurfaceBody:
    """Internal surface-native cylinder constructor.

    This stays private until primitive API migration is ready. The sidewall is
    modeled as four quarter-turn revolution patches so shared seams stay
    explicit. Caps remain planar trimmed patches and are not yet seam-connected
    to the sidewall ring.
    """

    radius = float(radius)
    height = float(height)
    resolution = int(resolution)
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if height <= 0.0:
        raise ValueError("height must be positive.")
    if resolution < 8:
        raise ValueError("resolution must be >= 8.")

    z0 = -height / 2.0
    z1 = height / 2.0
    quarter_sweeps = (0.0, 90.0, 180.0, 270.0)
    side_patches = tuple(
        RevolutionSurfacePatch(
            family="revolution",
            profile_curve=((radius, 0.0, z0), (radius, 0.0, z1)),
            axis_origin=(0.0, 0.0, 0.0),
            axis_direction=(0.0, 0.0, 1.0),
            start_angle_deg=start_angle,
            sweep_angle_deg=90.0,
            metadata={"kernel": {"primitive_family": "cylinder", "surface_role": "sidewall", "quadrant": quadrant}},
        )
        for quadrant, start_angle in enumerate(quarter_sweeps)
    )
    seams = tuple(
        SurfaceSeam(
            seam_id=f"sidewall-{index}",
            boundaries=(
                SurfaceBoundaryRef(index, "right"),
                SurfaceBoundaryRef((index + 1) % 4, "left"),
            ),
            metadata={"kernel": {"primitive_family": "cylinder", "surface_role": "sidewall-seam"}},
        )
        for index in range(4)
    )

    patches = list(side_patches)
    if capping:
        domain = ParameterDomain((-radius, radius), (-radius, radius))
        circle_trim = TrimLoop(_circle_loop(radius, samples=max(16, resolution)), category="outer")
        patches.extend(
            [
                PlanarSurfacePatch(
                    family="planar",
                    domain=domain,
                    trim_loops=(circle_trim,),
                    origin=(0.0, 0.0, z0),
                    u_axis=(1.0, 0.0, 0.0),
                    v_axis=(0.0, 1.0, 0.0),
                    metadata={"kernel": {"primitive_family": "cylinder", "surface_role": "bottom-cap"}},
                ),
                PlanarSurfacePatch(
                    family="planar",
                    domain=domain,
                    trim_loops=(circle_trim,),
                    origin=(0.0, 0.0, z1),
                    u_axis=(1.0, 0.0, 0.0),
                    v_axis=(0.0, 1.0, 0.0),
                    metadata={"kernel": {"primitive_family": "cylinder", "surface_role": "top-cap"}},
                ),
            ]
        )

    shell = make_surface_shell(
        tuple(patches),
        connected=False if capping else True,
        seams=seams,
        metadata={"kernel": {"primitive_family": "cylinder"}},
    )
    body = make_surface_body((shell,), metadata=metadata)
    return body.with_transform(_attached_transform(center, direction))


def make_surface_cone(
    bottom_diameter: float = 1.0,
    top_diameter: float = 0.0,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    resolution: int = 64,
    *,
    radius: float | None = None,
    metadata: dict[str, object] | None = None,
) -> SurfaceBody:
    """Internal surface-native cone/frustum constructor.

    This stays private until primitive API migration is ready. The sidewall is
    modeled as four quarter-turn revolution patches with an optional planar
    base cap when the bottom radius is non-zero.
    """

    if radius is not None:
        inferred_bottom = 2.0 * float(radius)
        if bottom_diameter != 1.0 and not np.isclose(bottom_diameter, inferred_bottom):
            raise ValueError("Specify either bottom_diameter or radius, not both.")
        bottom_diameter = inferred_bottom

    bottom_radius = float(bottom_diameter) / 2.0
    top_radius = float(top_diameter) / 2.0
    height = float(height)
    resolution = int(resolution)
    if bottom_radius <= 0.0 and top_radius <= 0.0:
        raise ValueError("At least one of bottom_diameter or top_diameter must be > 0.")
    if height <= 0.0:
        raise ValueError("height must be positive.")
    if resolution < 8:
        raise ValueError("resolution must be >= 8.")

    z0 = -height / 2.0
    z1 = height / 2.0
    quarter_sweeps = (0.0, 90.0, 180.0, 270.0)
    side_patches = tuple(
        RevolutionSurfacePatch(
            family="revolution",
            profile_curve=((bottom_radius, 0.0, z0), (top_radius, 0.0, z1)),
            axis_origin=(0.0, 0.0, 0.0),
            axis_direction=(0.0, 0.0, 1.0),
            start_angle_deg=start_angle,
            sweep_angle_deg=90.0,
            metadata={"kernel": {"primitive_family": "cone", "surface_role": "sidewall", "quadrant": quadrant}},
        )
        for quadrant, start_angle in enumerate(quarter_sweeps)
    )
    seams = tuple(
        SurfaceSeam(
            seam_id=f"sidewall-{index}",
            boundaries=(
                SurfaceBoundaryRef(index, "right"),
                SurfaceBoundaryRef((index + 1) % 4, "left"),
            ),
            metadata={"kernel": {"primitive_family": "cone", "surface_role": "sidewall-seam"}},
        )
        for index in range(4)
    )

    patches = list(side_patches)
    if bottom_radius > 0.0:
        domain = ParameterDomain((-bottom_radius, bottom_radius), (-bottom_radius, bottom_radius))
        circle_trim = TrimLoop(_circle_loop(bottom_radius, samples=max(16, resolution)), category="outer")
        patches.append(
            PlanarSurfacePatch(
                family="planar",
                domain=domain,
                trim_loops=(circle_trim,),
                origin=(0.0, 0.0, z0),
                u_axis=(1.0, 0.0, 0.0),
                v_axis=(0.0, 1.0, 0.0),
                metadata={"kernel": {"primitive_family": "cone", "surface_role": "bottom-cap"}},
            )
        )

    shell = make_surface_shell(
        tuple(patches),
        connected=False if bottom_radius > 0.0 else True,
        seams=seams,
        metadata={"kernel": {"primitive_family": "cone"}},
    )
    body = make_surface_body((shell,), metadata=metadata)
    return body.with_transform(_attached_transform(center, direction))


def make_surface_prism(
    base_size: Sequence[float] = (1.0, 1.0),
    top_size: Sequence[float] | None = None,
    height: float = 1.0,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    *,
    metadata: dict[str, object] | None = None,
) -> SurfaceBody:
    """Internal surface-native rectangular frustum constructor."""

    bx, by = float(base_size[0]), float(base_size[1])
    if bx <= 0.0 or by <= 0.0:
        raise ValueError("base_size must be positive.")
    if top_size is None:
        top_size = tuple(base_size)
    tx, ty = float(top_size[0]), float(top_size[1])
    if tx < 0.0 or ty < 0.0:
        raise ValueError("top_size must be non-negative.")
    height = float(height)
    if height <= 0.0:
        raise ValueError("height must be positive.")

    if np.isclose(tx, bx) and np.isclose(ty, by):
        shape = make_rect(size=(bx, by))
        return make_surface_linear_extrude(
            shape,
            height=height,
            direction=direction,
            center=center,
            metadata=metadata,
        )

    hx, hy = bx / 2.0, by / 2.0
    txh, tyh = tx / 2.0, ty / 2.0
    z0 = -height / 2.0
    z1 = height / 2.0

    bottom = np.asarray(
        [
            (-hx, -hy),
            (hx, -hy),
            (hx, hy),
            (-hx, hy),
        ],
        dtype=float,
    )
    top = np.asarray(
        [
            (-txh, -tyh),
            (txh, -tyh),
            (txh, tyh),
            (-txh, tyh),
        ],
        dtype=float,
    )

    side_patches = []
    seams = []
    for index in range(4):
        next_index = (index + 1) % 4
        b0 = bottom[index]
        b1 = bottom[next_index]
        t0 = top[index]
        t1 = top[next_index]
        side_patches.append(
            RuledSurfacePatch(
                family="ruled",
                start_curve=((b0[0], b0[1], z0), (b1[0], b1[1], z0)),
                end_curve=((t0[0], t0[1], z1), (t1[0], t1[1], z1)),
                metadata={"kernel": {"primitive_family": "prism", "surface_role": "sidewall", "edge_index": index}},
            )
        )
        seams.append(
            SurfaceSeam(
                seam_id=f"sidewall-{index}",
                boundaries=(
                    SurfaceBoundaryRef(index, "top"),
                    SurfaceBoundaryRef((index + 1) % 4, "bottom"),
                ),
                metadata={"kernel": {"primitive_family": "prism", "surface_role": "vertex-seam"}},
            )
        )

    patches: list[object] = list(side_patches)
    bottom_domain = ParameterDomain((-hx, hx), (-hy, hy))
    bottom_trim = TrimLoop(bottom, category="outer")
    patches.append(
        PlanarSurfacePatch(
            family="planar",
            domain=bottom_domain,
            trim_loops=(bottom_trim,),
            origin=(0.0, 0.0, z0),
            u_axis=(1.0, 0.0, 0.0),
            v_axis=(0.0, 1.0, 0.0),
            metadata={"kernel": {"primitive_family": "prism", "surface_role": "bottom-cap"}},
        )
    )
    connected = False
    if tx > 0.0 and ty > 0.0:
        top_domain = ParameterDomain((-txh, txh), (-tyh, tyh))
        top_trim = TrimLoop(top, category="outer")
        patches.append(
            PlanarSurfacePatch(
                family="planar",
                domain=top_domain,
                trim_loops=(top_trim,),
                origin=(0.0, 0.0, z1),
                u_axis=(1.0, 0.0, 0.0),
                v_axis=(0.0, 1.0, 0.0),
                metadata={"kernel": {"primitive_family": "prism", "surface_role": "top-cap"}},
            )
        )
    shell = make_surface_shell(tuple(patches), connected=connected, seams=tuple(seams), metadata={"kernel": {"primitive_family": "prism"}})
    body = make_surface_body((shell,), metadata=metadata)
    return body.with_transform(_attached_transform(center, direction))


def make_surface_torus(
    major_radius: float = 1.0,
    minor_radius: float = 0.25,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    n_theta: int = 64,
    n_phi: int = 32,
    *,
    metadata: dict[str, object] | None = None,
) -> SurfaceBody:
    """Internal surface-native torus constructor.

    This stays private until primitive API migration is ready. The torus is one
    closed revolution patch with explicit self-seams in both parameter
    directions.
    """

    major_radius = float(major_radius)
    minor_radius = float(minor_radius)
    n_theta = int(n_theta)
    n_phi = int(n_phi)
    if major_radius <= 0.0:
        raise ValueError("major_radius must be positive.")
    if minor_radius <= 0.0:
        raise ValueError("minor_radius must be positive.")
    if n_theta < 8 or n_phi < 8:
        raise ValueError("n_theta and n_phi must both be >= 8.")

    patch = RevolutionSurfacePatch(
        family="revolution",
        profile_curve=_torus_profile(major_radius, minor_radius, samples=n_phi),
        axis_origin=(0.0, 0.0, 0.0),
        axis_direction=(0.0, 0.0, 1.0),
        start_angle_deg=0.0,
        sweep_angle_deg=360.0,
        metadata={"kernel": {"primitive_family": "torus", "surface_role": "body"}},
    )
    seams = (
        SurfaceSeam(
            seam_id="u-wrap",
            boundaries=(
                SurfaceBoundaryRef(0, "left"),
                SurfaceBoundaryRef(0, "right"),
            ),
            metadata={"kernel": {"primitive_family": "torus", "surface_role": "u-wrap"}},
        ),
        SurfaceSeam(
            seam_id="v-wrap",
            boundaries=(
                SurfaceBoundaryRef(0, "bottom"),
                SurfaceBoundaryRef(0, "top"),
            ),
            metadata={"kernel": {"primitive_family": "torus", "surface_role": "v-wrap"}},
        ),
    )
    shell = make_surface_shell((patch,), seams=seams, metadata={"kernel": {"primitive_family": "torus"}})
    body = make_surface_body((shell,), metadata=metadata)
    return body.with_transform(_attached_transform(center, direction))


def make_surface_sphere(
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    theta_resolution: int = 64,
    phi_resolution: int = 64,
    *,
    metadata: dict[str, object] | None = None,
) -> SurfaceBody:
    """Internal surface-native sphere constructor.

    This stays private until primitive API migration is ready. The sphere is
    one closed revolution patch with a self-seam in the sweep direction and
    collapsed pole boundaries in the profile direction.
    """

    radius = float(radius)
    theta_resolution = int(theta_resolution)
    phi_resolution = int(phi_resolution)
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if theta_resolution < 8 or phi_resolution < 8:
        raise ValueError("theta_resolution and phi_resolution must both be >= 8.")

    patch = RevolutionSurfacePatch(
        family="revolution",
        profile_curve=_sphere_profile(radius, samples=phi_resolution),
        axis_origin=(0.0, 0.0, 0.0),
        axis_direction=(0.0, 0.0, 1.0),
        start_angle_deg=0.0,
        sweep_angle_deg=360.0,
        metadata={"kernel": {"primitive_family": "sphere", "surface_role": "body"}},
    )
    seams = (
        SurfaceSeam(
            seam_id="u-wrap",
            boundaries=(
                SurfaceBoundaryRef(0, "left"),
                SurfaceBoundaryRef(0, "right"),
            ),
            metadata={"kernel": {"primitive_family": "sphere", "surface_role": "u-wrap"}},
        ),
    )
    shell = make_surface_shell((patch,), seams=seams, metadata={"kernel": {"primitive_family": "sphere"}})
    body = make_surface_body((shell,), metadata=metadata)
    return body.with_transform(_attached_transform(center, (0.0, 0.0, 1.0)))


def make_surface_polyhedron(
    faces: int = 6,
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    metadata: dict[str, object] | None = None,
) -> SurfaceBody:
    """Internal surface-native regular polyhedron constructor.

    This remains private until primitive API migration is ready. Faces are
    modeled as trimmed planar patches. Shared seams are not yet expressed for
    arbitrary polygonal face boundaries, so the resulting shell is surface-
    native but not currently classified closed-valid.
    """

    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("radius must be positive.")

    vertices, face_list = _regular_polyhedron_data(int(faces))
    vertices = np.asarray(vertices, dtype=float)
    max_norm = float(np.linalg.norm(vertices, axis=1).max(initial=0.0))
    if max_norm > 0.0:
        vertices = vertices * (radius / max_norm)

    patches = tuple(
        _planar_patch_from_face(
            vertices[np.asarray(face, dtype=int)],
            metadata={"kernel": {"primitive_family": "polyhedron", "face_count": int(faces), "face_index": face_index}},
        )
        for face_index, face in enumerate(face_list)
    )
    shell = make_surface_shell(
        patches,
        connected=False,
        metadata={"kernel": {"primitive_family": "polyhedron", "face_count": int(faces)}},
    )
    body = make_surface_body((shell,), metadata=metadata)
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = _as_vec3(center, name="center")
    return body.with_transform(transform)


def make_surface_nhedron(
    faces: int = 6,
    radius: float = 0.5,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    *,
    metadata: dict[str, object] | None = None,
) -> SurfaceBody:
    """Internal surface-native compatibility wrapper for polyhedron."""

    return make_surface_polyhedron(
        faces=faces,
        radius=radius,
        center=center,
        metadata=metadata,
    )


__all__ = [
    "make_surface_box",
    "make_surface_ngon",
    "make_surface_cylinder",
    "make_surface_cone",
    "make_surface_prism",
    "make_surface_sphere",
    "make_surface_torus",
    "make_surface_polyhedron",
    "make_surface_nhedron",
]
