from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

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
from .topology import Loop, Region, Section, as_section


def _as_vec3(value: Sequence[float], *, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{name} must contain only finite values.")
    return vec


def _normalize(vector: Sequence[float], *, name: str) -> np.ndarray:
    vec = _as_vec3(vector, name=name)
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
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm == 0.0:
        raise ValueError("Could not construct a stable attached transform frame.")
    x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / float(np.linalg.norm(y_axis))

    transform = np.eye(4, dtype=float)
    transform[:3, 0] = x_axis
    transform[:3, 1] = y_axis
    transform[:3, 2] = z_axis
    transform[:3, 3] = origin
    return transform


def _rotation_transform(
    axis_origin: Sequence[float],
    axis_direction: Sequence[float],
    plane_normal: Sequence[float],
) -> np.ndarray:
    origin = _as_vec3(axis_origin, name="axis_origin")
    z_axis = _normalize(axis_direction, name="axis_direction")
    y_axis = _normalize(plane_normal, name="plane_normal")
    if abs(float(np.dot(z_axis, y_axis))) > 1e-6:
        raise ValueError("plane_normal must be perpendicular to axis_direction.")
    x_axis = np.cross(y_axis, z_axis)
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm == 0.0:
        raise ValueError("Could not construct a stable rotation frame.")
    x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / float(np.linalg.norm(y_axis))

    transform = np.eye(4, dtype=float)
    transform[:3, 0] = x_axis
    transform[:3, 1] = y_axis
    transform[:3, 2] = z_axis
    transform[:3, 3] = origin
    return transform


def _domain_for_region(region: Region) -> ParameterDomain:
    xmin, xmax, ymin, ymax = region.outer.bbox
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Extrude region must have positive planar span.")
    return ParameterDomain((xmin, xmax), (ymin, ymax))


def _trim_loops_for_region(region: Region) -> tuple[TrimLoop, ...]:
    return (
        TrimLoop(region.outer.points, category="outer"),
        *(TrimLoop(hole.points, category="inner") for hole in region.holes),
    )


def _closed_loop_points(loop: Loop) -> np.ndarray:
    points = np.asarray(loop.points, dtype=float).reshape(-1, 2)
    if points.shape[0] < 3:
        raise ValueError("rotate_extrude requires loops with at least three points.")
    return np.vstack([points, points[0]])


def _validate_rotate_loop(loop_points: np.ndarray, *, loop_name: str) -> None:
    radii = np.asarray(loop_points[:, 0], dtype=float)
    min_radius = float(radii.min())
    if min_radius < -1e-9:
        raise ValueError(f"{loop_name} crosses the revolution axis; private surface rotate_extrude requires non-negative radial coordinates.")


def _profile_curve_for_rotate_loop(loop: Loop) -> np.ndarray:
    points = _closed_loop_points(loop)
    _validate_rotate_loop(points, loop_name="rotate_extrude loop")
    return np.column_stack((points[:, 0], np.zeros(points.shape[0], dtype=float), points[:, 1])).astype(float)


def _rotate_cap_patch(
    region: Region,
    *,
    angle_deg: float,
    role: str,
) -> PlanarSurfacePatch:
    domain = _domain_for_region(region)
    angle_rad = float(np.deg2rad(angle_deg))
    radial_axis = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0], dtype=float)
    return PlanarSurfacePatch(
        family="planar",
        domain=domain,
        trim_loops=_trim_loops_for_region(region),
        origin=(0.0, 0.0, 0.0),
        u_axis=tuple(radial_axis),
        v_axis=(0.0, 0.0, 1.0),
        metadata={"kernel": {"operation": "rotate_extrude", "surface_role": role}},
    )


def _loop_sidewall_patches(loop: Loop, *, height: float, category: str, loop_index: int) -> tuple[list[RuledSurfacePatch], list[SurfaceSeam]]:
    points = np.asarray(loop.points, dtype=float).reshape(-1, 2)
    patches: list[RuledSurfacePatch] = []
    seams: list[SurfaceSeam] = []
    patch_count = int(points.shape[0])
    for point_index in range(patch_count):
        next_index = (point_index + 1) % patch_count
        p0 = points[point_index]
        p1 = points[next_index]
        patches.append(
            RuledSurfacePatch(
                family="ruled",
                start_curve=((p0[0], p0[1], 0.0), (p1[0], p1[1], 0.0)),
                end_curve=((p0[0], p0[1], height), (p1[0], p1[1], height)),
                metadata={"kernel": {"operation": "linear_extrude", "surface_role": "sidewall", "loop_category": category, "loop_index": loop_index, "edge_index": point_index}},
            )
        )
    for patch_index in range(patch_count):
        next_patch = (patch_index + 1) % patch_count
        seams.append(
            SurfaceSeam(
                seam_id=f"{category}-{loop_index}-vertex-{patch_index}",
                boundaries=(
                    SurfaceBoundaryRef(patch_index, "top"),
                    SurfaceBoundaryRef(next_patch, "bottom"),
                ),
                metadata={"kernel": {"operation": "linear_extrude", "surface_role": "sidewall-vertex", "loop_category": category, "loop_index": loop_index}},
            )
        )
    return patches, seams


def _build_region_shell(region: Region, *, height: float, region_index: int) -> object:
    domain = _domain_for_region(region)
    sidewall_patches: list[RuledSurfacePatch] = []
    seams: list[SurfaceSeam] = []

    outer_patches, outer_seams = _loop_sidewall_patches(region.outer, height=height, category="outer", loop_index=0)
    sidewall_patches.extend(outer_patches)
    seams.extend(outer_seams)

    hole_offset = 1
    for hole_index, hole in enumerate(region.holes):
        patches, hole_seams = _loop_sidewall_patches(hole, height=height, category="inner", loop_index=hole_offset + hole_index)
        base_index = len(sidewall_patches)
        for seam in hole_seams:
            remapped = tuple(
                SurfaceBoundaryRef(boundary.patch_index + base_index, boundary.boundary_id)
                for boundary in seam.boundaries
            )
            seams.append(
                SurfaceSeam(
                    seam_id=f"hole-{hole_index}-{seam.seam_id}",
                    boundaries=remapped,
                    continuity=seam.continuity,
                    metadata=seam.metadata,
                )
            )
        sidewall_patches.extend(patches)

    bottom_cap = PlanarSurfacePatch(
        family="planar",
        domain=domain,
        trim_loops=_trim_loops_for_region(region),
        origin=(0.0, 0.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
        metadata={"kernel": {"operation": "linear_extrude", "surface_role": "bottom-cap", "region_index": region_index}},
    )
    top_cap = PlanarSurfacePatch(
        family="planar",
        domain=domain,
        trim_loops=_trim_loops_for_region(region),
        origin=(0.0, 0.0, height),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 1.0, 0.0),
        metadata={"kernel": {"operation": "linear_extrude", "surface_role": "top-cap", "region_index": region_index}},
    )

    return make_surface_shell(
        tuple(sidewall_patches) + (bottom_cap, top_cap),
        connected=False,
        seams=tuple(seams),
        metadata={"kernel": {"operation": "linear_extrude", "region_index": region_index}},
    )


def make_surface_linear_extrude(
    shape: Section | Region | object,
    *,
    height: float = 1.0,
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    center: Sequence[float] = (0.0, 0.0, 0.0),
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
    metadata: dict[str, object] | None = None,
) -> SurfaceBody:
    """Internal surface-native linear extrude constructor.

    This builder intentionally remains private until the public modeling-op
    migration is ready. It returns a `SurfaceBody` with ruled sidewalls and
    planar cap patches, using attached transforms for placement.
    """

    height = float(height)
    if height <= 0.0:
        raise ValueError("height must be positive.")

    section = as_section(
        shape,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
    ).normalized()
    if not section.regions:
        raise ValueError("linear extrude requires at least one non-empty region.")

    shells = tuple(
        _build_region_shell(region, height=height, region_index=region_index)
        for region_index, region in enumerate(section.regions)
    )
    body = make_surface_body(shells, metadata=metadata)
    return body.with_transform(_attached_transform(center, np.asarray(direction, dtype=float) * height))


def make_surface_rotate_extrude(
    shape: Section | Region | object,
    *,
    angle_deg: float = 360.0,
    axis_origin: Sequence[float] = (0.0, 0.0, 0.0),
    axis_direction: Sequence[float] = (0.0, 0.0, 1.0),
    plane_normal: Sequence[float] = (0.0, 1.0, 0.0),
    segments: int = 64,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
    cap_ends: bool = True,
    metadata: dict[str, object] | None = None,
) -> SurfaceBody:
    """Internal surface-native rotate-extrude constructor.

    This builder intentionally remains private until the public modeling-op
    migration is ready. It currently supports one connected region and expects
    loop coordinates to stay on the non-negative radial side of the revolution
    axis.
    """

    angle_deg = float(angle_deg)
    if angle_deg == 0.0:
        raise ValueError("angle_deg must be non-zero.")
    segments = int(segments)
    if segments < 3:
        raise ValueError("segments must be >= 3.")

    section = as_section(
        shape,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
    ).normalized()
    if len(section.regions) != 1:
        raise ValueError("private surface rotate_extrude currently requires a single connected region.")
    region = section.regions[0]

    patches: list[object] = []
    seams: list[SurfaceSeam] = []
    loops = [region.outer, *region.holes]
    for loop_index, loop in enumerate(loops):
        patch_index = len(patches)
        profile_curve = _profile_curve_for_rotate_loop(loop)
        loop_category = "outer" if loop_index == 0 else "inner"
        patches.append(
            RevolutionSurfacePatch(
                family="revolution",
                profile_curve=profile_curve,
                axis_origin=(0.0, 0.0, 0.0),
                axis_direction=(0.0, 0.0, 1.0),
                start_angle_deg=0.0,
                sweep_angle_deg=angle_deg,
                metadata={"kernel": {"operation": "rotate_extrude", "surface_role": "sidewall", "loop_category": loop_category, "loop_index": loop_index}},
            )
        )
        seams.append(
            SurfaceSeam(
                seam_id=f"{loop_category}-{loop_index}-profile-wrap",
                boundaries=(
                    SurfaceBoundaryRef(patch_index, "bottom"),
                    SurfaceBoundaryRef(patch_index, "top"),
                ),
                metadata={"kernel": {"operation": "rotate_extrude", "surface_role": "profile-wrap", "loop_category": loop_category, "loop_index": loop_index}},
            )
        )
        if np.isclose(abs(angle_deg), 360.0):
            seams.append(
                SurfaceSeam(
                    seam_id=f"{loop_category}-{loop_index}-revolve-wrap",
                    boundaries=(
                        SurfaceBoundaryRef(patch_index, "left"),
                        SurfaceBoundaryRef(patch_index, "right"),
                    ),
                    metadata={"kernel": {"operation": "rotate_extrude", "surface_role": "revolve-wrap", "loop_category": loop_category, "loop_index": loop_index}},
                )
            )

    connected = True
    if not np.isclose(abs(angle_deg), 360.0) and cap_ends:
        connected = False
        patches.extend(
            [
                _rotate_cap_patch(region, angle_deg=0.0, role="start-cap"),
                _rotate_cap_patch(region, angle_deg=angle_deg, role="end-cap"),
            ]
        )

    shell = make_surface_shell(
        tuple(patches),
        connected=connected,
        seams=tuple(seams),
        metadata={"kernel": {"operation": "rotate_extrude"}},
    )
    body = make_surface_body((shell,), metadata=metadata)
    return body.with_transform(_rotation_transform(axis_origin, axis_direction, plane_normal))


__all__ = ["make_surface_linear_extrude", "make_surface_rotate_extrude"]
