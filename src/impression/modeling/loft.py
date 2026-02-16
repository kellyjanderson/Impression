from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from impression.mesh import Mesh
from impression.mesh_quality import MeshQuality, apply_lod

from ._color import set_mesh_color
from ._profile2d import _loops_resampled, _profile_loops, _resample_loop, _triangulate_loops, _signed_area, _ensure_winding
from .drawing2d import Profile2D
from .path3d import Path3D
from .paths import Path as PolyPath


def loft_profiles(
    profiles: Sequence[Profile2D],
    path: Path3D | PolyPath | Sequence[Sequence[float]] | None = None,
    samples: int = 200,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
    quality: MeshQuality | None = None,
    cap_ends: bool = False,
    start_cap: str = "none",
    end_cap: str = "none",
    cap_steps: int = 6,
    start_cap_length: float | None = None,
    end_cap_length: float | None = None,
    cap_scale_dims: str = "both",
) -> Mesh:
    """Loft a sequence of profiles, optionally along a path."""

    if quality is not None:
        quality = apply_lod(quality)
        samples = _apply_quality_samples(samples, quality)
        segments_per_circle = _apply_quality_samples(segments_per_circle, quality)
        bezier_samples = _apply_quality_samples(bezier_samples, quality)

    if len(profiles) < 2:
        raise ValueError("loft_profiles requires at least two profiles.")
    hole_count = len(profiles[0].holes)
    for profile in profiles[1:]:
        if len(profile.holes) != hole_count:
            raise ValueError("All profiles must have the same number of holes.")

    positions = _resolve_positions(path, len(profiles))
    if start_cap != "none" or end_cap != "none":
        cap_ends = True
    if cap_ends and start_cap == "none":
        start_cap = "flat"
    if cap_ends and end_cap == "none":
        end_cap = "flat"
    _validate_scale_dims(cap_scale_dims)
    _validate_caps(start_cap, end_cap)
    loops_per_profile, positions = _apply_caps(
        profiles=profiles,
        positions=positions,
        start_cap=start_cap,
        end_cap=end_cap,
        cap_steps=cap_steps,
        start_cap_length=start_cap_length,
        end_cap_length=end_cap_length,
        cap_scale_dims=cap_scale_dims,
        samples=samples,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
    )
    normals, binormals, tangents = _compute_frames(positions)

    loop_count = len(loops_per_profile[0]) if loops_per_profile else 0

    vertices = []
    offsets = []
    for profile_idx, loops in enumerate(loops_per_profile):
        profile_offsets = []
        normal = normals[profile_idx]
        binormal = binormals[profile_idx]
        tangent = tangents[profile_idx]
        for loop in loops:
            profile_offsets.append(len(vertices))
            pts3 = (
                positions[profile_idx]
                + loop[:, 0:1] * normal
                + loop[:, 1:2] * binormal
            )
            vertices.extend(pts3)
        offsets.append(profile_offsets)

    vertices = np.asarray(vertices, dtype=float)

    faces = []
    for idx in range(len(loops_per_profile) - 1):
        for loop_idx in range(loop_count):
            start_a = offsets[idx][loop_idx]
            start_b = offsets[idx + 1][loop_idx]
            for i in range(samples):
                j = (i + 1) % samples
                a0 = start_a + i
                a1 = start_a + j
                b0 = start_b + i
                b1 = start_b + j
                faces.append([a0, a1, b1])
                faces.append([a0, b1, b0])

    if cap_ends:
        base_vertices, base_faces = _triangulate_loops(loops_per_profile[0])
        if base_faces.size:
            start_offset = offsets[0][0]
            end_offset = offsets[-1][0]
            faces.extend((base_faces + start_offset).tolist())
            faces.extend((base_faces[:, [0, 2, 1]] + end_offset).tolist())

    mesh = Mesh(vertices, np.asarray(faces, dtype=int))
    color = profiles[0].color
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def loft(
    profiles: Sequence[Profile2D],
    path: Path3D | PolyPath | Sequence[Sequence[float]] | None = None,
    samples: int = 200,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
    quality: MeshQuality | None = None,
    cap_ends: bool = False,
    start_cap: str = "none",
    end_cap: str = "none",
    cap_steps: int = 6,
    start_cap_length: float | None = None,
    end_cap_length: float | None = None,
    cap_scale_dims: str = "both",
) -> Mesh:
    """Alias for loft_profiles."""

    return loft_profiles(
        profiles=profiles,
        path=path,
        samples=samples,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
        quality=quality,
        cap_ends=cap_ends,
        start_cap=start_cap,
        end_cap=end_cap,
        cap_steps=cap_steps,
        start_cap_length=start_cap_length,
        end_cap_length=end_cap_length,
        cap_scale_dims=cap_scale_dims,
    )


def loft_endcaps(
    profiles: Sequence[Profile2D],
    path: Path3D | PolyPath | Sequence[Sequence[float]] | None = None,
    samples: int = 200,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
    quality: MeshQuality | None = None,
    endcap_mode: str = "FLAT",
    endcap_amount: float = 0.0,
    endcap_steps: int = 12,
    endcap_placement: str = "BOTH",
) -> Mesh:
    """Loft profiles with experimental endcap generation (spec mode)."""

    if quality is not None:
        quality = apply_lod(quality)
        samples = _apply_quality_samples(samples, quality)
        segments_per_circle = _apply_quality_samples(segments_per_circle, quality)
        bezier_samples = _apply_quality_samples(bezier_samples, quality)

    if len(profiles) < 2:
        raise ValueError("loft_endcaps requires at least two profiles.")

    hole_count = len(profiles[0].holes)
    for profile in profiles[1:]:
        if len(profile.holes) != hole_count:
            raise ValueError("All profiles must have the same number of holes.")

    _validate_endcap_mode(endcap_mode)
    _validate_endcap_placement(endcap_placement)
    if endcap_mode != "FLAT" and endcap_amount <= 0:
        raise ValueError("endcap_amount must be > 0 for CHAMFER or ROUND.")
    if endcap_mode == "ROUND" and endcap_steps < 2:
        raise ValueError("endcap_steps must be >= 2 for ROUND.")

    positions = _resolve_positions(path, len(profiles))
    normals, binormals, tangents = _compute_frames(positions)

    loops_per_profile = [
        _loops_resampled_anchored(
            profile,
            samples,
            segments_per_circle=segments_per_circle,
            bezier_samples=bezier_samples,
        )
        for profile in profiles
    ]

    start_enabled = endcap_placement in {"START", "BOTH"} and endcap_mode != "FLAT"
    end_enabled = endcap_placement in {"END", "BOTH"} and endcap_mode != "FLAT"

    new_loops: list[list[np.ndarray]] = []
    new_positions: list[np.ndarray] = []
    cap_indices: list[tuple[int, str]] = []

    start_body_index = 0
    end_body_index = len(profiles)

    if start_enabled:
        cap_sections = _build_endcap_sections(
            profile=profiles[0],
            base_loops=loops_per_profile[0],
            position=positions[0],
            tangent=tangents[0],
            direction=-1.0,
            mode=endcap_mode,
            amount=endcap_amount,
            steps=endcap_steps,
            samples=samples,
            segments_per_circle=segments_per_circle,
            bezier_samples=bezier_samples,
            hole_count=hole_count,
            reverse=True,
        )
        cap_start_index = len(new_loops)
        for loops, pos in cap_sections:
            new_loops.append(loops)
            new_positions.append(pos)
        cap_indices.append((cap_start_index, "start"))
        start_body_index = 1

    if end_enabled:
        end_body_index = len(profiles) - 1

    for idx in range(start_body_index, end_body_index):
        new_loops.append(loops_per_profile[idx])
        new_positions.append(positions[idx])

    if end_enabled:
        cap_sections = _build_endcap_sections(
            profile=profiles[-1],
            base_loops=loops_per_profile[-1],
            position=positions[-1],
            tangent=tangents[-1],
            direction=1.0,
            mode=endcap_mode,
            amount=endcap_amount,
            steps=endcap_steps,
            samples=samples,
            segments_per_circle=segments_per_circle,
            bezier_samples=bezier_samples,
            hole_count=hole_count,
        )
        cap_start_index = len(new_loops)
        for loops, pos in cap_sections:
            new_loops.append(loops)
            new_positions.append(pos)
        cap_indices.append((cap_start_index + len(cap_sections) - 1, "end"))

    if endcap_mode == "FLAT":
        if endcap_placement in {"START", "BOTH"}:
            cap_indices.append((0, "start"))
        if endcap_placement in {"END", "BOTH"}:
            cap_indices.append((len(loops_per_profile) - 1, "end"))

    loops_per_profile = new_loops
    positions = np.asarray(new_positions, dtype=float)
    normals, binormals, tangents = _compute_frames(positions)

    loop_count = len(loops_per_profile[0]) if loops_per_profile else 0
    vertices: list[np.ndarray] = []
    offsets: list[list[int]] = []
    for profile_idx, loops in enumerate(loops_per_profile):
        profile_offsets = []
        normal = normals[profile_idx]
        binormal = binormals[profile_idx]
        for loop in loops:
            profile_offsets.append(len(vertices))
            pts3 = positions[profile_idx] + loop[:, 0:1] * normal + loop[:, 1:2] * binormal
            vertices.extend(pts3)
        offsets.append(profile_offsets)

    vertices = np.asarray(vertices, dtype=float)

    faces: list[list[int]] = []
    for idx in range(len(loops_per_profile) - 1):
        for loop_idx in range(loop_count):
            start_a = offsets[idx][loop_idx]
            start_b = offsets[idx + 1][loop_idx]
            for i in range(samples):
                j = (i + 1) % samples
                a0 = start_a + i
                a1 = start_a + j
                b0 = start_b + i
                b1 = start_b + j
                faces.append([a0, a1, b1])
                faces.append([a0, b1, b0])

    for cap_idx, side in cap_indices:
        base_vertices, base_faces = _triangulate_loops(loops_per_profile[cap_idx])
        if base_faces.size == 0:
            continue
        offset = offsets[cap_idx][0]
        if side == "start":
            faces.extend((base_faces + offset).tolist())
        else:
            faces.extend((base_faces[:, [0, 2, 1]] + offset).tolist())

    mesh = Mesh(vertices, np.asarray(faces, dtype=int))
    color = profiles[0].color
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def _resolve_positions(
    path: Path3D | PolyPath | Sequence[Sequence[float]] | None,
    count: int,
) -> np.ndarray:
    if path is None:
        return np.column_stack([np.zeros(count), np.zeros(count), np.linspace(0.0, 1.0, count)])

    if isinstance(path, Path3D):
        pts = path.sample()
    elif isinstance(path, PolyPath):
        pts = np.asarray(path._effective_points(), dtype=float)
    else:
        pts = np.asarray(path, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("path must be a sequence of 3D points.")

    return _resample_path(pts, count)


def _apply_quality_samples(value: int, quality: MeshQuality) -> int:
    if quality.lod == "preview":
        return max(6, int(value * 0.5))
    return value


def _resample_path(points: np.ndarray, count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("path sample count must be >= 2.")
    pts = np.asarray(points, dtype=float)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if pts.shape[0] == count:
        return pts.copy()
    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg_lengths.sum())
    if total == 0:
        return np.tile(pts[0], (count, 1))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    targets = np.linspace(0.0, total, count)
    result = []
    seg_index = 0
    for t in targets:
        while seg_index < len(seg_lengths) - 1 and cumulative[seg_index + 1] < t:
            seg_index += 1
        seg_start = cumulative[seg_index]
        seg_end = cumulative[seg_index + 1]
        p0 = pts[seg_index]
        p1 = pts[seg_index + 1]
        if seg_end == seg_start:
            result.append(p0)
        else:
            alpha = (t - seg_start) / (seg_end - seg_start)
            result.append((1 - alpha) * p0 + alpha * p1)
    return np.asarray(result, dtype=float)


def _validate_caps(start_cap: str, end_cap: str) -> None:
    allowed = {"none", "flat", "taper", "dome", "slope"}
    if start_cap not in allowed:
        raise ValueError(f"start_cap must be one of {sorted(allowed)}.")
    if end_cap not in allowed:
        raise ValueError(f"end_cap must be one of {sorted(allowed)}.")


def _validate_endcap_mode(mode: str) -> None:
    allowed = {"FLAT", "CHAMFER", "ROUND", "COVE"}
    if mode not in allowed:
        raise ValueError(f"endcap_mode must be one of {sorted(allowed)}.")


def _validate_endcap_placement(placement: str) -> None:
    allowed = {"START", "END", "BOTH"}
    if placement not in allowed:
        raise ValueError(f"endcap_placement must be one of {sorted(allowed)}.")


def _apply_caps(
    profiles: Sequence[Profile2D],
    positions: np.ndarray,
    start_cap: str,
    end_cap: str,
    cap_steps: int,
    start_cap_length: float | None,
    end_cap_length: float | None,
    cap_scale_dims: str,
    samples: int,
    segments_per_circle: int,
    bezier_samples: int,
) -> tuple[list[list[np.ndarray]], np.ndarray]:
    loops_per_profile: list[list[np.ndarray]] = []
    for profile in profiles:
        loops = _loops_resampled(
            profile,
            samples,
            segments_per_circle=segments_per_circle,
            bezier_samples=bezier_samples,
        )
        loops_per_profile.append(loops)

    if start_cap == "none" and end_cap == "none":
        return loops_per_profile, positions

    if cap_steps < 1:
        raise ValueError("cap_steps must be >= 1.")

    step_dist = 1.0
    if positions.shape[0] > 1:
        step_dist = float(np.linalg.norm(positions[1] - positions[0])) or 1.0

    normals, binormals, tangents = _compute_frames(positions)
    new_loops: list[list[np.ndarray]] = []
    new_positions: list[np.ndarray] = []

    _validate_scale_dims(cap_scale_dims)

    if start_cap != "none":
        half_dims = _loop_half_dims(loops_per_profile[0][0])
        offsets, scales = _cap_profile_series(
            mode=start_cap,
            step_dist=step_dist,
            steps=cap_steps,
            length=start_cap_length,
            reverse=True,
            scale_dims=cap_scale_dims,
            half_dims=half_dims,
        )
        if scales:
            center = loops_per_profile[0][0].mean(axis=0)
            for offset, scale in zip(offsets, scales, strict=False):
                cap_loops = [_scale_loop(loop, center, scale) for loop in loops_per_profile[0]]
                new_loops.append(cap_loops)
                new_positions.append(positions[0] - tangents[0] * offset)

    new_loops.extend(loops_per_profile)
    new_positions.extend(list(positions))

    if end_cap != "none":
        half_dims = _loop_half_dims(loops_per_profile[-1][0])
        offsets, scales = _cap_profile_series(
            mode=end_cap,
            step_dist=step_dist,
            steps=cap_steps,
            length=end_cap_length,
            reverse=False,
            scale_dims=cap_scale_dims,
            half_dims=half_dims,
        )
        if scales:
            center = loops_per_profile[-1][0].mean(axis=0)
            for offset, scale in zip(offsets, scales, strict=False):
                cap_loops = [_scale_loop(loop, center, scale) for loop in loops_per_profile[-1]]
                new_loops.append(cap_loops)
                new_positions.append(positions[-1] + tangents[-1] * offset)

    return new_loops, np.asarray(new_positions, dtype=float)


def _cap_profile_series(
    mode: str,
    step_dist: float,
    steps: int,
    length: float | None,
    reverse: bool,
    scale_dims: str,
    half_dims: np.ndarray,
) -> tuple[list[float], list[np.ndarray]]:
    if mode in {"flat", "none"}:
        return [], []
    half_dims = np.asarray(half_dims, dtype=float).reshape(2)
    if np.any(half_dims <= 1e-9):
        return [], []
    min_axis = int(np.argmin(half_dims))
    min_half = float(half_dims[min_axis])
    if min_half <= 1e-9:
        return [], []
    requested_steps = max(1, steps)
    if length is not None:
        if length <= 0:
            return [], []
        steps = max(requested_steps, int(np.ceil(length / step_dist)))
        total_length = float(length)
    else:
        steps = requested_steps
        total_length = float(step_dist * steps)

    if steps < 1:
        return [], []

    step_size = total_length / steps
    offsets = np.linspace(step_size, total_length, steps)
    if reverse:
        offsets = offsets[::-1]

    scales: list[np.ndarray] = []
    resolved_offsets: list[float] = []
    min_scale = 1e-3
    for offset in offsets:
        t = min(max(offset / total_length, 0.0), 1.0)
        eased = _cap_ease(mode, t)
        scale_factor = max(1.0 - eased, 0.0)
        if scale_dims == "smallest":
            scale = np.ones(2, dtype=float)
            scale[min_axis] = max(scale_factor, min_scale)
        else:
            ratios = min_half / half_dims
            scale = 1.0 - eased * ratios
            scale = np.clip(scale, min_scale, 1.0)
        scales.append(scale)
        resolved_offsets.append(float(offset))
    return resolved_offsets, scales


def _cap_ease(mode: str, t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    if mode == "taper":
        return t
    if mode == "dome":
        return float(1.0 - np.sqrt(max(1.0 - t * t, 0.0)))
    if mode == "slope":
        return float(np.sin(t * np.pi / 2.0))
    return t


def _scale_loop(loop: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale = np.asarray(scale, dtype=float).reshape(2)
    return center + (loop - center) * scale


def _loop_half_dims(loop: np.ndarray) -> np.ndarray:
    bbox_min = loop.min(axis=0)
    bbox_max = loop.max(axis=0)
    return (bbox_max - bbox_min) / 2.0


def _validate_scale_dims(scale_dims: str) -> None:
    if scale_dims not in {"smallest", "both"}:
        raise ValueError("cap_scale_dims must be 'smallest' or 'both'.")


def _loops_resampled_anchored(
    profile: Profile2D,
    count: int,
    segments_per_circle: int,
    bezier_samples: int,
) -> list[np.ndarray]:
    loops = _profile_loops(
        profile,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
        enforce_winding=True,
    )
    return [_resample_loop(_anchor_loop(loop), count) for loop in loops]


def _anchor_loop(loop: np.ndarray) -> np.ndarray:
    if loop.shape[0] == 0:
        return loop
    max_x = np.max(loop[:, 0])
    candidates = np.where(np.isclose(loop[:, 0], max_x))[0]
    if candidates.size == 1:
        idx = int(candidates[0])
    else:
        max_y = np.max(loop[candidates, 1])
        candidates = candidates[np.where(np.isclose(loop[candidates, 1], max_y))[0]]
        idx = int(candidates[0])
    return np.roll(loop, -idx, axis=0)


def _build_endcap_sections(
    profile: Profile2D,
    base_loops: list[np.ndarray],
    position: np.ndarray,
    tangent: np.ndarray,
    direction: float,
    mode: str,
    amount: float,
    steps: int,
    samples: int,
    segments_per_circle: int,
    bezier_samples: int,
    hole_count: int,
    reverse: bool = False,
) -> list[tuple[list[np.ndarray], np.ndarray]]:
    schedule = _endcap_schedule(mode, amount, steps)
    sections: list[tuple[list[np.ndarray], np.ndarray]] = []
    if mode == "COVE":
        center = base_loops[0].mean(axis=0)
        for scale, d in schedule:
            scale_vec = np.array([scale, scale], dtype=float)
            loops = [_scale_loop(loop, center, scale_vec) for loop in base_loops]
            loops = [_resample_loop(_anchor_loop(loop), samples) for loop in loops]
            pos = position + tangent * (direction * d)
            sections.append((loops, pos))
    else:
        for t, d in schedule:
            if np.isclose(t, 0.0):
                loops = base_loops
            else:
                loops = _inset_profile_loops(
                    profile,
                    t,
                    join_type=mode,
                    hole_count=hole_count,
                )
                loops = [_resample_loop(_anchor_loop(loop), samples) for loop in loops]
            pos = position + tangent * (direction * d)
            sections.append((loops, pos))
    if reverse:
        if len(sections) > 1:
            sections = list(reversed(sections))
    return sections


def _endcap_schedule(mode: str, amount: float, steps: int) -> list[tuple[float, float]]:
    if mode == "CHAMFER":
        return [(0.0, 0.0), (amount, amount)]
    if mode == "ROUND":
        count = max(1, steps)
        schedule: list[tuple[float, float]] = []
        for i in range(count + 1):
            theta = (i / count) * (np.pi / 2.0)
            t = amount * np.sin(theta)
            d = amount * (1.0 - np.cos(theta))
            schedule.append((float(t), float(d)))
        return schedule
    if mode == "COVE":
        count = max(1, steps)
        schedule = []
        for i in range(count + 1):
            theta = (i / count) * (np.pi / 2.0)
            scale = max(np.cos(theta), 1e-3)
            d = amount * np.sin(theta)
            schedule.append((float(scale), float(d)))
        return schedule
    return [(0.0, 0.0)]


def _inset_profile_loops(
    profile: Profile2D,
    inset: float,
    join_type: str,
    hole_count: int,
) -> list[np.ndarray]:
    try:
        import pyclipper
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError("pyclipper is required for endcap inset operations.") from exc

    if inset <= 0:
        loops = _profile_loops(
            profile,
            segments_per_circle=64,
            bezier_samples=32,
            enforce_winding=True,
        )
        return loops

    scale = 1_000_000.0
    loops = _profile_loops(
        profile,
        segments_per_circle=64,
        bezier_samples=32,
        enforce_winding=True,
    )
    if not loops:
        raise ValueError("Profile has no loops to inset.")

    join_key = "ROUND" if join_type in {"ROUND", "COVE"} else join_type
    jt = pyclipper.JT_ROUND if join_key == "ROUND" else pyclipper.JT_MITER

    def offset_single(path_pts: np.ndarray, delta: float) -> list[np.ndarray]:
        pco = pyclipper.PyclipperOffset(miter_limit=2.0, arc_tolerance=0.25 * scale)
        path = np.round(path_pts * scale).astype(np.int64).tolist()
        pco.AddPath(path, jt, pyclipper.ET_CLOSEDPOLYGON)
        result = pco.Execute(delta * scale)
        return [np.asarray(p, dtype=float) / scale for p in result]

    outer = loops[0]
    holes = loops[1:]

    outer_result = offset_single(outer, -inset)
    if len(outer_result) != 1:
        raise ValueError("endcap_amount too large for profile; inset collapsed.")
    outer = _ensure_winding(outer_result[0], clockwise=False)

    inset_holes: list[np.ndarray] = []
    for hole in holes:
        hole_result = offset_single(hole, inset)
        if len(hole_result) != 1:
            raise ValueError("endcap_amount too large for profile; inset collapsed.")
        hole_loop = _ensure_winding(hole_result[0], clockwise=True)
        if not _point_in_polygon(hole_loop[0], outer):
            raise ValueError("endcap_amount too large for profile; hole collapsed.")
        inset_holes.append(hole_loop)

    if hole_count and len(inset_holes) != hole_count:
        raise ValueError("endcap_amount too large for profile; hole topology changed.")
    return [outer] + inset_holes


def _classify_loops(loops: list[np.ndarray], hole_count: int) -> list[np.ndarray]:
    if not loops:
        raise ValueError("Inset produced no geometry.")
    areas = [_signed_area(loop) for loop in loops]
    abs_areas = [abs(a) for a in areas]
    outer_idx = int(np.argmax(abs_areas))
    outer = loops[outer_idx]
    holes = [loops[i] for i in range(len(loops)) if i != outer_idx]
    for hole in holes:
        if not _point_in_polygon(hole[0], outer):
            raise ValueError("Inset produced split geometry; endcap_amount too large.")
    outer = _ensure_winding(outer, clockwise=False)
    holes = [_ensure_winding(hole, clockwise=True) for hole in holes]
    if hole_count and len(holes) != hole_count:
        raise ValueError("Inset changed hole count; endcap_amount too large.")
    return [outer] + holes


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x = float(point[0])
    y = float(point[1])
    inside = False
    n = polygon.shape[0]
    j = n - 1
    for i in range(n):
        xi = polygon[i, 0]
        yi = polygon[i, 1]
        xj = polygon[j, 0]
        yj = polygon[j, 1]
        intersect = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        if intersect:
            inside = not inside
        j = i
    return inside


def _compute_frames(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute parallel-transport frames along the path."""
    pts = np.asarray(positions, dtype=float)
    count = pts.shape[0]
    if count < 2:
        tangent = np.array([[0.0, 0.0, 1.0]])
        normal = np.array([[1.0, 0.0, 0.0]])
        binormal = np.array([[0.0, 1.0, 0.0]])
        return normal, binormal, tangent

    tangents = np.zeros((count, 3), dtype=float)
    for i in range(count):
        if i == 0:
            delta = pts[1] - pts[0]
        elif i == count - 1:
            delta = pts[-1] - pts[-2]
        else:
            delta = pts[i + 1] - pts[i - 1]
        norm = np.linalg.norm(delta)
        if norm == 0:
            tangents[i] = tangents[i - 1] if i > 0 else np.array([0.0, 0.0, 1.0])
        else:
            tangents[i] = delta / norm

    normals = np.zeros((count, 3), dtype=float)
    binormals = np.zeros((count, 3), dtype=float)

    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, tangents[0])) > 0.9:
        up = np.array([1.0, 0.0, 0.0])
    n0 = np.cross(tangents[0], up)
    if np.linalg.norm(n0) < 1e-9:
        n0 = np.array([1.0, 0.0, 0.0])
    n0 = n0 / np.linalg.norm(n0)
    b0 = np.cross(tangents[0], n0)
    b0 = b0 / np.linalg.norm(b0)
    normals[0] = n0
    binormals[0] = b0

    for i in range(1, count):
        v1 = tangents[i - 1]
        v2 = tangents[i]
        axis = np.cross(v1, v2)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-9:
            normals[i] = normals[i - 1]
            binormals[i] = binormals[i - 1]
            continue
        axis = axis / axis_norm
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        n_prev = normals[i - 1]
        n_rot = _rotate_vector(n_prev, axis, angle)
        b_rot = np.cross(v2, n_rot)
        if np.linalg.norm(b_rot) < 1e-9:
            normals[i] = n_rot
            binormals[i] = binormals[i - 1]
        else:
            b_rot = b_rot / np.linalg.norm(b_rot)
            n_rot = np.cross(b_rot, v2)
            normals[i] = n_rot / np.linalg.norm(n_rot)
            binormals[i] = b_rot

    return normals, binormals, tangents


def _rotate_vector(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotate vector around axis by angle (radians)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    vec = np.asarray(vec, dtype=float)
    c = np.cos(angle)
    s = np.sin(angle)
    return vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1.0 - c)


__all__ = ["loft_profiles", "loft", "loft_endcaps"]
