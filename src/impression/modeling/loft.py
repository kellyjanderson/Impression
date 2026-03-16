from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from impression.mesh import Mesh
from impression.mesh_quality import MeshQuality, apply_lod

from ._color import set_mesh_color
from .topology import (
    loops_resampled as _loops_resampled,
    profile_loops as _profile_loops,
    resample_loop as _resample_loop,
    triangulate_loops as _triangulate_loops,
    anchor_loop as _anchor_loop,
    inset_profile_loops as _inset_profile_loops,
    classify_loops as _classify_loops,
    signed_area as _signed_area,
    minimum_cost_loop_assignment as _minimum_cost_loop_assignment_topology,
    stable_loop_transition as _stable_loop_transition,
    split_merge_ambiguous as _split_merge_ambiguous,
    point_in_polygon as _point_in_polygon,
    Loop,
    Section,
    Region,
    as_section,
)
from .drawing2d import Path2D
from .path3d import Path3D
from .paths import Path as PolyPath


@dataclass(frozen=True)
class Station:
    """A loft station with explicit 3D frame data."""

    t: float
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    n: np.ndarray
    section: Section | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", float(self.t))
        object.__setattr__(self, "origin", np.asarray(self.origin, dtype=float).reshape(3))
        object.__setattr__(self, "u", np.asarray(self.u, dtype=float).reshape(3))
        object.__setattr__(self, "v", np.asarray(self.v, dtype=float).reshape(3))
        object.__setattr__(self, "n", np.asarray(self.n, dtype=float).reshape(3))


@dataclass(frozen=True)
class _LoopRef:
    kind: str  # "actual" | "synthetic"
    index: int


@dataclass(frozen=True)
class _RegionRef:
    kind: str  # "actual" | "synthetic"
    index: int


@dataclass(frozen=True)
class _PairedRegionLoops:
    prev_loops: tuple[np.ndarray, ...]
    curr_loops: tuple[np.ndarray, ...]
    prev_sources: tuple[_LoopRef, ...]
    curr_sources: tuple[_LoopRef, ...]
    closures: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _PairedSectionTransition:
    prev_region_ref: _RegionRef
    curr_region_ref: _RegionRef
    region: _PairedRegionLoops
    region_closures: tuple[str, ...]
    action: str


@dataclass(frozen=True)
class PlannedStation:
    """Planner output for one normalized station."""

    station_index: int
    t: float
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    n: np.ndarray
    regions: tuple[tuple[np.ndarray, ...], ...]


@dataclass(frozen=True)
class PlannedLoopRef:
    """Planner-native loop reference."""

    kind: str  # actual | synthetic
    index: int


@dataclass(frozen=True)
class PlannedRegionRef:
    """Planner-native region reference."""

    kind: str  # actual | synthetic
    index: int


@dataclass(frozen=True)
class PlannedLoopPair:
    """Planner output for one loop correspondence."""

    prev_loop_ref: PlannedLoopRef
    curr_loop_ref: PlannedLoopRef
    prev_loop: np.ndarray
    curr_loop: np.ndarray
    role: str  # stable | synthetic_birth | synthetic_death


@dataclass(frozen=True)
class PlannedClosure:
    """Planner output for closure ownership."""

    side: str  # prev | curr
    scope: str  # loop | region
    loop_index: int | None = None


@dataclass(frozen=True)
class PlannedRegionPair:
    """Planner output for one region-level transition pairing."""

    prev_region_ref: PlannedRegionRef
    curr_region_ref: PlannedRegionRef
    loop_pairs: tuple[PlannedLoopPair, ...]
    closures: tuple[PlannedClosure, ...]
    action: str  # stable | split_match | split_birth | merge_match | merge_death
    branch_id: str


@dataclass(frozen=True)
class PlannedTransition:
    """Planner output for one station interval transition."""

    interval: tuple[int, int]
    region_pairs: tuple[PlannedRegionPair, ...]
    branch_order: tuple[str, ...]
    topology_case: str  # one_to_one | one_to_many | many_to_one | many_to_many_expand | many_to_many_collapse
    ambiguity_class: str = "none"  # none | permutation | containment | symmetry | closure


@dataclass(frozen=True)
class LoftPlan:
    """Deterministic planner output consumed by the loft executor."""

    samples: int
    stations: tuple[PlannedStation, ...]
    transitions: tuple[PlannedTransition, ...]
    metadata: dict[str, object]


def loft_profiles(
    profiles: Sequence[Section | Region | Path2D | object],
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
    """Loft a sequence of planar sections/profiles, optionally along a path."""

    if quality is not None:
        quality = apply_lod(quality)
        samples = _apply_quality_samples(samples, quality)
        segments_per_circle = _apply_quality_samples(segments_per_circle, quality)
        bezier_samples = _apply_quality_samples(bezier_samples, quality)

    normalized_profiles = _normalize_profile_inputs(
        profiles,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
    )

    _validate_profile_topology_input(
        normalized_profiles,
        fn_name="loft_profiles",
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
    )

    positions = _resolve_positions(path, len(normalized_profiles))
    if start_cap != "none" or end_cap != "none":
        cap_ends = True
    if cap_ends and start_cap == "none":
        start_cap = "flat"
    if cap_ends and end_cap == "none":
        end_cap = "flat"
    _validate_scale_dims(cap_scale_dims)
    _validate_caps(start_cap, end_cap)
    loops_per_profile, positions = _apply_caps(
        profiles=normalized_profiles,
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
    loops_per_profile = _align_loops_for_loft(loops_per_profile)
    stations = _build_stations(positions)

    loop_count = len(loops_per_profile[0]) if loops_per_profile else 0

    vertices = []
    offsets = []
    for profile_idx, loops in enumerate(loops_per_profile):
        profile_offsets = []
        station = stations[profile_idx]
        for loop in loops:
            profile_offsets.append(len(vertices))
            pts3 = (
                station.origin
                + loop[:, 0:1] * station.u
                + loop[:, 1:2] * station.v
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
    color = getattr(normalized_profiles[0], "color", None)
    if color is not None:
        set_mesh_color(mesh, color)
    return mesh


def loft(
    profiles: Sequence[Section | Region | Path2D | object],
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


def loft_sections(
    stations: Sequence[Station],
    samples: int = 200,
    quality: MeshQuality | None = None,
    cap_ends: bool = False,
    split_merge_mode: str = "fail",
    split_merge_steps: int = 8,
    split_merge_bias: float = 0.5,
    ambiguity_mode: str | None = None,
    ambiguity_cost_profile: str = "balanced",
    ambiguity_max_branches: int = 64,
    fairness_mode: str = "local",
    fairness_weight: float = 0.2,
    skeleton_mode: str = "auto",
    fairness_iterations: int = 12,
) -> Mesh:
    """Loft topology-native sections using a planner/executor pipeline."""

    plan = loft_plan_sections(
        stations,
        samples=samples,
        quality=quality,
        split_merge_mode=split_merge_mode,
        split_merge_steps=split_merge_steps,
        split_merge_bias=split_merge_bias,
        ambiguity_mode=ambiguity_mode,
        ambiguity_cost_profile=ambiguity_cost_profile,
        ambiguity_max_branches=ambiguity_max_branches,
        fairness_mode=fairness_mode,
        fairness_weight=fairness_weight,
        skeleton_mode=skeleton_mode,
        fairness_iterations=fairness_iterations,
    )
    return loft_execute_plan(plan, cap_ends=cap_ends)


def loft_plan_sections(
    stations: Sequence[Station],
    samples: int = 200,
    quality: MeshQuality | None = None,
    split_merge_mode: str = "fail",
    split_merge_steps: int = 8,
    split_merge_bias: float = 0.5,
    ambiguity_mode: str | None = None,
    ambiguity_cost_profile: str = "balanced",
    ambiguity_max_branches: int = 64,
    fairness_mode: str = "local",
    fairness_weight: float = 0.2,
    skeleton_mode: str = "auto",
    fairness_iterations: int = 12,
) -> LoftPlan:
    """Build a deterministic loft execution plan from section stations."""

    if quality is not None:
        quality = apply_lod(quality)
        samples = _apply_quality_samples(samples, quality)

    _validate_split_merge_mode(split_merge_mode)
    _validate_split_merge_controls(split_merge_steps, split_merge_bias)
    resolved_ambiguity_mode = _resolve_ambiguity_mode(split_merge_mode, ambiguity_mode)
    _validate_ambiguity_cost_profile(ambiguity_cost_profile)
    _validate_ambiguity_max_branches(ambiguity_max_branches)
    _validate_fairness_mode(fairness_mode)
    _validate_fairness_weight(fairness_weight)
    _validate_skeleton_mode(skeleton_mode)
    _validate_fairness_iterations(fairness_iterations)
    _validate_section_stations(stations)

    effective_stations = list(stations)
    if split_merge_mode == "resolve":
        effective_stations = _expand_split_merge_stations(
            stations=effective_stations,
            samples=samples,
            split_merge_steps=split_merge_steps,
            split_merge_bias=split_merge_bias,
        )
        _validate_section_stations(effective_stations)

    planned_stations: list[PlannedStation] = []
    for idx, station in enumerate(effective_stations):
        section = station.section
        if section is None:
            raise ValueError(f"Station at index {idx} is missing section data.")
        regions = _section_to_region_loops(section, samples=samples)
        planned_stations.append(
            PlannedStation(
                station_index=idx,
                t=station.t,
                origin=station.origin.copy(),
                u=station.u.copy(),
                v=station.v.copy(),
                n=station.n.copy(),
                regions=tuple(tuple(np.asarray(loop, dtype=float) for loop in region) for region in regions),
            )
        )

    skeleton_available = _skeleton_guidance_available(planned_stations)
    if skeleton_mode == "required" and not skeleton_available:
        raise ValueError(
            "Unsupported topology transition: skeleton_required_unavailable "
            "(skeleton_mode='required' but skeleton guidance is unavailable)."
        )

    planned_transitions: list[PlannedTransition] = []
    ambiguity_class_counts: dict[str, int] = {
        "permutation": 0,
        "containment": 0,
        "symmetry": 0,
        "closure": 0,
    }
    ambiguity_resolved_intervals_count = 0
    global_optimizer_ran = False
    global_optimizer_hit_iteration_cap = False
    previous_interval_vectors: dict[int, np.ndarray] | None = None
    for idx in range(len(planned_stations) - 1):
        prev_station = planned_stations[idx]
        curr_station = planned_stations[idx + 1]
        prev_regions = [list(region) for region in planned_stations[idx].regions]
        curr_regions = [list(region) for region in planned_stations[idx + 1].regions]
        topology_case = _classify_region_topology_case(
            len(prev_regions),
            len(curr_regions),
        )
        ambiguity_class = _classify_region_transition_ambiguity(
            prev_regions=[region[0] for region in prev_regions],
            curr_regions=[region[0] for region in curr_regions],
            ambiguity_max_branches=int(ambiguity_max_branches),
            ambiguity_cost_profile=ambiguity_cost_profile,
        )
        interval = (idx, idx + 1)
        if split_merge_mode == "resolve" and resolved_ambiguity_mode == "fail" and ambiguity_class != "none":
            _raise_structured_ambiguity_error(
                interval=interval,
                ambiguity_class=ambiguity_class,
                tie_break_stage="ambiguity_mode_fail",
                candidate_count_after_pruning="unknown",
                detail=(
                    "ambiguity_mode='fail' rejects ambiguous interval; "
                    "use ambiguity_mode='auto' to enable deterministic auto-resolution."
                ),
            )
        if ambiguity_class != "none" and split_merge_mode == "resolve" and resolved_ambiguity_mode == "auto":
            ambiguity_class_counts[ambiguity_class] = ambiguity_class_counts.get(ambiguity_class, 0) + 1
            ambiguity_resolved_intervals_count += 1
        region_order_override: tuple[int, ...] | None = None
        if fairness_mode == "global" and topology_case == "one_to_one":
            (
                region_order_override,
                interval_optimizer_ran,
                interval_hit_iteration_cap,
            ) = _select_global_region_order(
                prev_regions=[region[0] for region in prev_regions],
                curr_regions=[region[0] for region in curr_regions],
                prev_station=prev_station,
                curr_station=curr_station,
                previous_interval_vectors=previous_interval_vectors,
                ambiguity_cost_profile=ambiguity_cost_profile,
                ambiguity_max_branches=int(ambiguity_max_branches),
                fairness_weight=float(fairness_weight),
                fairness_iterations=int(fairness_iterations),
            )
            global_optimizer_ran = global_optimizer_ran or interval_optimizer_ran
            global_optimizer_hit_iteration_cap = (
                global_optimizer_hit_iteration_cap or interval_hit_iteration_cap
            )
        try:
            transitions = _pair_sections_for_transition(
                prev_regions,
                curr_regions,
                split_merge_mode=split_merge_mode,
                split_merge_steps=split_merge_steps,
                split_merge_bias=split_merge_bias,
                ambiguity_mode=resolved_ambiguity_mode,
                ambiguity_cost_profile=ambiguity_cost_profile,
                ambiguity_max_branches=int(ambiguity_max_branches),
                fairness_mode=fairness_mode,
                fairness_weight=fairness_weight,
                skeleton_mode=skeleton_mode,
                fairness_iterations=fairness_iterations,
                region_order_override=region_order_override,
            )
        except ValueError as exc:
            message = str(exc)
            if "unsupported_topology_ambiguity" not in message:
                raise
            _raise_structured_ambiguity_error(
                interval=interval,
                ambiguity_class=ambiguity_class,
                tie_break_stage=_ambiguity_failure_stage(message),
                candidate_count_after_pruning=_ambiguity_failure_candidate_count(message),
                detail=message,
            )
        previous_interval_vectors = _transition_curr_vectors(
            prev_station=prev_station,
            curr_station=curr_station,
            transitions=transitions,
        )
        planned_pairs = tuple(
            _as_planned_region_pair(
                transition,
                interval=interval,
                pair_index=pair_index,
            )
            for pair_index, transition in enumerate(transitions)
        )
        planned_transitions.append(
            PlannedTransition(
                interval=interval,
                region_pairs=planned_pairs,
                branch_order=tuple(pair.branch_id for pair in planned_pairs),
                topology_case=topology_case,
                ambiguity_class=ambiguity_class,
            )
        )

    fairness_objective_pre = _compute_fairness_objective_terms(planned_stations, planned_transitions)
    fairness_objective_post = dict(fairness_objective_pre)
    fairness_diagnostics = {
        "branch_crossing_count": float(fairness_objective_post["branch_crossing"]),
        "continuity_score": float(fairness_objective_post["curvature_continuity"]),
        "closure_distortion_score": float(fairness_objective_post["closure_stress"]),
    }
    fairness_convergence_status = "not_run"
    if fairness_mode == "global" and global_optimizer_ran:
        fairness_convergence_status = (
            "max_iterations" if global_optimizer_hit_iteration_cap else "converged"
        )
    plan = LoftPlan(
        samples=samples,
        stations=tuple(planned_stations),
        transitions=tuple(planned_transitions),
        metadata={
            "plan_schema_version": 1,
            "planner": "loft_plan_sections",
            "split_merge_mode": split_merge_mode,
            "split_merge_steps": split_merge_steps,
            "split_merge_bias": split_merge_bias,
            "ambiguity_mode": resolved_ambiguity_mode,
            "ambiguity_cost_profile": ambiguity_cost_profile,
            "ambiguity_max_branches": int(ambiguity_max_branches),
            "ambiguity_resolved_intervals_count": ambiguity_resolved_intervals_count,
            "ambiguity_failed_intervals_count": 0,
            "ambiguity_class_counts": ambiguity_class_counts,
            "region_topology_case_counts": _count_region_topology_cases(planned_transitions),
            "region_action_counts": _count_region_actions(planned_transitions),
            "fairness_mode": fairness_mode,
            "fairness_weight": float(fairness_weight),
            "fairness_iterations": int(fairness_iterations),
            "skeleton_mode": skeleton_mode,
            "fairness_objective_pre": fairness_objective_pre,
            "fairness_objective_post": fairness_objective_post,
            "fairness_diagnostics": fairness_diagnostics,
            "fairness_optimization_convergence_status": fairness_convergence_status,
        },
    )
    _validate_loft_plan(plan)
    return plan


def loft_execute_plan(
    plan: LoftPlan,
    *,
    cap_ends: bool = False,
) -> Mesh:
    """Execute a loft plan into a deterministic mesh."""

    _validate_loft_plan(plan)

    vertices: list[np.ndarray] = []
    offsets: list[list[list[int]]] = []
    for station in plan.stations:
        station_offsets: list[list[int]] = []
        for loops in station.regions:
            region_offsets: list[int] = []
            for loop in loops:
                region_offsets.append(len(vertices))
                pts3 = station.origin + loop[:, 0:1] * station.u + loop[:, 1:2] * station.v
                vertices.extend(pts3)
            station_offsets.append(region_offsets)
        offsets.append(station_offsets)

    faces: list[list[int]] = []
    loop_start_cache: dict[tuple[int, str, int, str, int], int] = {}
    for transition_block in plan.transitions:
        prev_idx, curr_idx = transition_block.interval
        prev_station = plan.stations[prev_idx]
        curr_station = plan.stations[curr_idx]
        region_pairs_by_branch = {pair.branch_id: pair for pair in transition_block.region_pairs}
        for branch_id in transition_block.branch_order:
            region_pair = region_pairs_by_branch[branch_id]
            for loop_pair in region_pair.loop_pairs:
                prev_start = _resolve_transition_loop_start(
                    station_index=prev_idx,
                    station_origin=prev_station.origin,
                    station_u=prev_station.u,
                    station_v=prev_station.v,
                    region_ref=region_pair.prev_region_ref,
                    ref=loop_pair.prev_loop_ref,
                    loop=loop_pair.prev_loop,
                    offsets=offsets,
                    vertices=vertices,
                    loop_start_cache=loop_start_cache,
                )
                curr_start = _resolve_transition_loop_start(
                    station_index=curr_idx,
                    station_origin=curr_station.origin,
                    station_u=curr_station.u,
                    station_v=curr_station.v,
                    region_ref=region_pair.curr_region_ref,
                    ref=loop_pair.curr_loop_ref,
                    loop=loop_pair.curr_loop,
                    offsets=offsets,
                    vertices=vertices,
                    loop_start_cache=loop_start_cache,
                )
                for i in range(plan.samples):
                    j = (i + 1) % plan.samples
                    a0 = prev_start + i
                    a1 = prev_start + j
                    b0 = curr_start + i
                    b1 = curr_start + j
                    faces.append([a0, a1, b1])
                    faces.append([a0, b1, b0])

            for closure in region_pair.closures:
                if closure.scope == "loop":
                    if closure.loop_index is None:
                        raise ValueError("Invalid plan closure: loop scope requires loop_index.")
                    loop_pair = region_pair.loop_pairs[closure.loop_index]
                    if closure.side == "prev":
                        closure_start = _resolve_transition_loop_start(
                            station_index=prev_idx,
                            station_origin=prev_station.origin,
                            station_u=prev_station.u,
                            station_v=prev_station.v,
                            region_ref=region_pair.prev_region_ref,
                            ref=loop_pair.prev_loop_ref,
                            loop=loop_pair.prev_loop,
                            offsets=offsets,
                            vertices=vertices,
                            loop_start_cache=loop_start_cache,
                        )
                        _, closure_faces = _triangulate_loops([loop_pair.prev_loop])
                        faces.extend((closure_faces + closure_start).tolist())
                    elif closure.side == "curr":
                        closure_start = _resolve_transition_loop_start(
                            station_index=curr_idx,
                            station_origin=curr_station.origin,
                            station_u=curr_station.u,
                            station_v=curr_station.v,
                            region_ref=region_pair.curr_region_ref,
                            ref=loop_pair.curr_loop_ref,
                            loop=loop_pair.curr_loop,
                            offsets=offsets,
                            vertices=vertices,
                            loop_start_cache=loop_start_cache,
                        )
                        _, closure_faces = _triangulate_loops([loop_pair.curr_loop])
                        faces.extend((closure_faces[:, [0, 2, 1]] + closure_start).tolist())
                    else:
                        raise ValueError(f"Invalid plan closure side: {closure.side!r}")
                elif closure.scope == "region" and closure.side == "prev":
                    loop_starts = [
                        _resolve_transition_loop_start(
                            station_index=prev_idx,
                            station_origin=prev_station.origin,
                            station_u=prev_station.u,
                            station_v=prev_station.v,
                            region_ref=region_pair.prev_region_ref,
                            ref=loop_pair.prev_loop_ref,
                            loop=loop_pair.prev_loop,
                            offsets=offsets,
                            vertices=vertices,
                            loop_start_cache=loop_start_cache,
                        )
                        for loop_pair in region_pair.loop_pairs
                    ]
                    closure_faces = _triangulate_loopset_faces_with_starts(
                        [loop_pair.prev_loop for loop_pair in region_pair.loop_pairs],
                        loop_starts,
                    )
                    faces.extend(closure_faces.tolist())
                elif closure.scope == "region" and closure.side == "curr":
                    loop_starts = [
                        _resolve_transition_loop_start(
                            station_index=curr_idx,
                            station_origin=curr_station.origin,
                            station_u=curr_station.u,
                            station_v=curr_station.v,
                            region_ref=region_pair.curr_region_ref,
                            ref=loop_pair.curr_loop_ref,
                            loop=loop_pair.curr_loop,
                            offsets=offsets,
                            vertices=vertices,
                            loop_start_cache=loop_start_cache,
                        )
                        for loop_pair in region_pair.loop_pairs
                    ]
                    closure_faces = _triangulate_loopset_faces_with_starts(
                        [loop_pair.curr_loop for loop_pair in region_pair.loop_pairs],
                        loop_starts,
                    )
                    faces.extend(closure_faces[:, [0, 2, 1]].tolist())
                else:
                    raise ValueError(
                        "Invalid plan closure record: "
                        f"scope={closure.scope!r} side={closure.side!r}"
                    )

    if cap_ends:
        for region_idx in range(len(plan.stations[0].regions)):
            _, base_faces_start = _triangulate_loops(list(plan.stations[0].regions[region_idx]))
            if base_faces_start.size:
                faces.extend((base_faces_start + offsets[0][region_idx][0]).tolist())
        for region_idx in range(len(plan.stations[-1].regions)):
            _, base_faces_end = _triangulate_loops(list(plan.stations[-1].regions[region_idx]))
            if base_faces_end.size:
                faces.extend((base_faces_end[:, [0, 2, 1]] + offsets[-1][region_idx][0]).tolist())

    return Mesh(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int))


def _as_planned_region_pair(
    transition: _PairedSectionTransition,
    *,
    interval: tuple[int, int],
    pair_index: int,
) -> PlannedRegionPair:
    loop_pairs: list[PlannedLoopPair] = []
    for prev_loop, curr_loop, prev_ref, curr_ref in zip(
        transition.region.prev_loops,
        transition.region.curr_loops,
        transition.region.prev_sources,
        transition.region.curr_sources,
        strict=True,
    ):
        loop_pairs.append(
            PlannedLoopPair(
                prev_loop_ref=_to_planned_loop_ref(prev_ref),
                curr_loop_ref=_to_planned_loop_ref(curr_ref),
                prev_loop=np.array(prev_loop, dtype=float, copy=True),
                curr_loop=np.array(curr_loop, dtype=float, copy=True),
                role=_planned_loop_pair_role(
                    _to_planned_loop_ref(prev_ref),
                    _to_planned_loop_ref(curr_ref),
                ),
            )
        )

    closures: list[PlannedClosure] = [
        PlannedClosure(side=side, scope="loop", loop_index=loop_idx)
        for side, loop_idx in transition.region.closures
    ]
    closures.extend(
        PlannedClosure(side=side, scope="region", loop_index=None)
        for side in transition.region_closures
    )

    return PlannedRegionPair(
        prev_region_ref=_to_planned_region_ref(transition.prev_region_ref),
        curr_region_ref=_to_planned_region_ref(transition.curr_region_ref),
        loop_pairs=tuple(loop_pairs),
        closures=tuple(closures),
        action=transition.action,
        branch_id=_make_branch_id(interval, pair_index, transition),
    )


def _make_branch_id(
    interval: tuple[int, int],
    pair_index: int,
    transition: _PairedSectionTransition,
) -> str:
    prev_ref = transition.prev_region_ref
    curr_ref = transition.curr_region_ref
    return (
        f"i{interval[0]}_{interval[1]}:"
        f"p{pair_index}:"
        f"{transition.action}:"
        f"{prev_ref.kind}{prev_ref.index}_to_{curr_ref.kind}{curr_ref.index}"
    )


def _raise_structured_ambiguity_error(
    *,
    interval: tuple[int, int],
    ambiguity_class: str,
    tie_break_stage: str,
    candidate_count_after_pruning: str,
    detail: str,
) -> None:
    raise ValueError(
        "Unsupported topology transition: unsupported_topology_ambiguity "
        f"(interval {interval[0]}->{interval[1]}; "
        f"ambiguity_class={ambiguity_class}; "
        f"tie_break_stage={tie_break_stage}; "
        f"candidate_count_after_pruning={candidate_count_after_pruning}; "
        f"detail={detail})"
    )


def _ambiguity_failure_stage(message: str) -> str:
    if "tie_break_stage=" in message:
        token = message.split("tie_break_stage=", 1)[1]
        return token.split(";", 1)[0].split(")", 1)[0].strip()
    if "candidate count exceeds ambiguity_max_branches" in message:
        return "candidate_enumeration_limit"
    if "residual indeterminate ambiguity" in message:
        return "residual_tie_break"
    if "ambiguity detected" in message:
        return "ambiguity_gate"
    return "unknown"


def _ambiguity_failure_candidate_count(message: str) -> str:
    if "candidate_count_after_pruning=" in message:
        token = message.split("candidate_count_after_pruning=", 1)[1]
        return token.split(";", 1)[0].split(")", 1)[0].strip()
    return "unknown"


def _planned_loop_pair_role(prev_ref: PlannedLoopRef, curr_ref: PlannedLoopRef) -> str:
    if prev_ref.kind == "actual" and curr_ref.kind == "actual":
        return "stable"
    if prev_ref.kind == "synthetic" and curr_ref.kind == "actual":
        return "synthetic_birth"
    if prev_ref.kind == "actual" and curr_ref.kind == "synthetic":
        return "synthetic_death"
    return "stable"


def _to_planned_loop_ref(ref: _LoopRef) -> PlannedLoopRef:
    return PlannedLoopRef(kind=ref.kind, index=ref.index)


def _to_planned_region_ref(ref: _RegionRef) -> PlannedRegionRef:
    return PlannedRegionRef(kind=ref.kind, index=ref.index)


def _validate_loft_plan(plan: LoftPlan) -> None:
    if plan.samples < 3:
        raise ValueError("Invalid loft plan: samples must be >= 3.")
    if len(plan.stations) < 2:
        raise ValueError("Invalid loft plan: requires at least two planned stations.")
    if len(plan.transitions) != len(plan.stations) - 1:
        raise ValueError("Invalid loft plan: transition count must be stations-1.")

    prev_t: float | None = None
    for idx, station in enumerate(plan.stations):
        if station.station_index != idx:
            raise ValueError(
                f"Invalid loft plan station index: expected {idx}, got {station.station_index}."
            )
        if prev_t is not None and station.t <= prev_t:
            raise ValueError("Invalid loft plan: station t values must be strictly increasing.")
        prev_t = station.t
        if (
            station.origin.shape != (3,)
            or station.u.shape != (3,)
            or station.v.shape != (3,)
            or station.n.shape != (3,)
        ):
            raise ValueError(f"Invalid loft plan station {idx}: frame vectors must be shape (3,).")
        if not (
            np.all(np.isfinite(station.origin))
            and np.all(np.isfinite(station.u))
            and np.all(np.isfinite(station.v))
            and np.all(np.isfinite(station.n))
        ):
            raise ValueError(f"Invalid loft plan station {idx}: non-finite frame values.")

    if not isinstance(plan.metadata, dict):
        raise ValueError("Invalid loft plan: metadata must be a dict.")
    if "plan_schema_version" not in plan.metadata:
        raise ValueError("Invalid loft plan metadata: missing plan_schema_version.")
    if "ambiguity_mode" not in plan.metadata:
        raise ValueError("Invalid loft plan metadata: missing ambiguity_mode.")
    if "ambiguity_cost_profile" not in plan.metadata:
        raise ValueError("Invalid loft plan metadata: missing ambiguity_cost_profile.")
    if "ambiguity_max_branches" not in plan.metadata:
        raise ValueError("Invalid loft plan metadata: missing ambiguity_max_branches.")
    if "ambiguity_resolved_intervals_count" not in plan.metadata:
        raise ValueError("Invalid loft plan metadata: missing ambiguity_resolved_intervals_count.")
    if "ambiguity_failed_intervals_count" not in plan.metadata:
        raise ValueError("Invalid loft plan metadata: missing ambiguity_failed_intervals_count.")
    if "ambiguity_class_counts" not in plan.metadata:
        raise ValueError("Invalid loft plan metadata: missing ambiguity_class_counts.")
    if "fairness_mode" not in plan.metadata:
        raise ValueError("Invalid loft plan metadata: missing fairness_mode.")
    if "fairness_weight" not in plan.metadata:
        raise ValueError("Invalid loft plan metadata: missing fairness_weight.")
    if "fairness_iterations" not in plan.metadata:
        raise ValueError("Invalid loft plan metadata: missing fairness_iterations.")
    if "skeleton_mode" not in plan.metadata:
        raise ValueError("Invalid loft plan metadata: missing skeleton_mode.")
    if "fairness_objective_pre" not in plan.metadata:
        raise ValueError("Invalid loft plan metadata: missing fairness_objective_pre.")
    if "fairness_objective_post" not in plan.metadata:
        raise ValueError("Invalid loft plan metadata: missing fairness_objective_post.")
    if "fairness_diagnostics" not in plan.metadata:
        raise ValueError("Invalid loft plan metadata: missing fairness_diagnostics.")
    if "fairness_optimization_convergence_status" not in plan.metadata:
        raise ValueError("Invalid loft plan metadata: missing fairness_optimization_convergence_status.")

    _validate_ambiguity_mode(str(plan.metadata["ambiguity_mode"]))
    _validate_ambiguity_cost_profile(str(plan.metadata["ambiguity_cost_profile"]))
    _validate_ambiguity_max_branches(int(plan.metadata["ambiguity_max_branches"]))
    _validate_fairness_mode(str(plan.metadata["fairness_mode"]))
    _validate_fairness_weight(float(plan.metadata["fairness_weight"]))
    _validate_fairness_iterations(int(plan.metadata["fairness_iterations"]))
    _validate_skeleton_mode(str(plan.metadata["skeleton_mode"]))
    if int(plan.metadata["ambiguity_resolved_intervals_count"]) < 0:
        raise ValueError(
            "Invalid loft plan metadata: ambiguity_resolved_intervals_count must be >= 0."
        )
    if int(plan.metadata["ambiguity_failed_intervals_count"]) < 0:
        raise ValueError(
            "Invalid loft plan metadata: ambiguity_failed_intervals_count must be >= 0."
        )
    class_counts = plan.metadata["ambiguity_class_counts"]
    if not isinstance(class_counts, dict):
        raise ValueError("Invalid loft plan metadata: ambiguity_class_counts must be a dict.")
    for key in ("permutation", "containment", "symmetry", "closure"):
        if key not in class_counts:
            raise ValueError(
                f"Invalid loft plan metadata: ambiguity_class_counts missing key {key!r}."
            )
        if int(class_counts[key]) < 0:
            raise ValueError(
                f"Invalid loft plan metadata: ambiguity_class_counts[{key!r}] must be >= 0."
            )
    for key in ("fairness_objective_pre", "fairness_objective_post"):
        terms = plan.metadata[key]
        if not isinstance(terms, dict):
            raise ValueError(f"Invalid loft plan metadata: {key} must be a dict.")
        for term in (
            "curvature_continuity",
            "branch_crossing",
            "branch_acceleration",
            "synthetic_harshness",
            "closure_stress",
        ):
            if term not in terms:
                raise ValueError(f"Invalid loft plan metadata: {key} missing term {term!r}.")
            if float(terms[term]) < 0.0:
                raise ValueError(f"Invalid loft plan metadata: {key}[{term!r}] must be >= 0.")
    fairness_diagnostics = plan.metadata["fairness_diagnostics"]
    if not isinstance(fairness_diagnostics, dict):
        raise ValueError("Invalid loft plan metadata: fairness_diagnostics must be a dict.")
    for key in ("branch_crossing_count", "continuity_score", "closure_distortion_score"):
        if key not in fairness_diagnostics:
            raise ValueError(f"Invalid loft plan metadata: fairness_diagnostics missing key {key!r}.")
        if float(fairness_diagnostics[key]) < 0.0:
            raise ValueError(
                f"Invalid loft plan metadata: fairness_diagnostics[{key!r}] must be >= 0."
            )
    if str(plan.metadata["fairness_optimization_convergence_status"]) not in {
        "not_run",
        "converged",
        "max_iterations",
        "failed",
    }:
        raise ValueError(
            "Invalid loft plan metadata: fairness_optimization_convergence_status "
            "must be one of ['converged', 'failed', 'max_iterations', 'not_run']."
        )

    valid_roles = {"stable", "synthetic_birth", "synthetic_death"}
    valid_scopes = {"loop", "region"}
    valid_sides = {"prev", "curr"}
    valid_ref_kinds = {"actual", "synthetic"}
    valid_region_actions = {"stable", "split_match", "split_birth", "merge_match", "merge_death"}
    valid_topology_cases = {
        "one_to_one",
        "one_to_many",
        "many_to_one",
        "many_to_many_expand",
        "many_to_many_collapse",
    }
    valid_ambiguity_classes = {"none", "permutation", "containment", "symmetry", "closure"}

    for transition_idx, transition in enumerate(plan.transitions):
        expected_interval = (transition_idx, transition_idx + 1)
        if transition.interval != expected_interval:
            raise ValueError(
                "Invalid loft plan transition interval: "
                f"expected {expected_interval}, got {transition.interval}."
            )
        if transition.topology_case not in valid_topology_cases:
            raise ValueError(
                f"Invalid loft plan transition interval {expected_interval}: "
                f"invalid topology_case {transition.topology_case!r}."
            )
        if transition.ambiguity_class not in valid_ambiguity_classes:
            raise ValueError(
                f"Invalid loft plan transition interval {expected_interval}: "
                f"invalid ambiguity_class {transition.ambiguity_class!r}."
            )
        if len(transition.branch_order) != len(transition.region_pairs):
            raise ValueError(
                f"Invalid loft plan transition interval {expected_interval}: "
                "branch_order length must match region_pairs length."
            )
        if len(set(transition.branch_order)) != len(transition.branch_order):
            raise ValueError(
                f"Invalid loft plan transition interval {expected_interval}: "
                "branch_order contains duplicate branch IDs."
            )
        prev_idx, curr_idx = transition.interval
        prev_station = plan.stations[prev_idx]
        curr_station = plan.stations[curr_idx]
        collected_branch_order: list[str] = []
        for region_pair_idx, region_pair in enumerate(transition.region_pairs):
            if region_pair.action not in valid_region_actions:
                raise ValueError(
                    f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx}: "
                    f"invalid action {region_pair.action!r}."
                )
            if not isinstance(region_pair.branch_id, str) or not region_pair.branch_id.strip():
                raise ValueError(
                    f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx}: "
                    "branch_id must be a non-empty string."
                )
            collected_branch_order.append(region_pair.branch_id)
            if region_pair.prev_region_ref.kind not in valid_ref_kinds:
                raise ValueError(
                    f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx}: "
                    f"invalid prev region kind {region_pair.prev_region_ref.kind!r}."
                )
            if region_pair.curr_region_ref.kind not in valid_ref_kinds:
                raise ValueError(
                    f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx}: "
                    f"invalid curr region kind {region_pair.curr_region_ref.kind!r}."
                )
            if region_pair.prev_region_ref.index < 0:
                raise ValueError(
                    f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx}: "
                    "prev region ref index must be >= 0."
                )
            if region_pair.curr_region_ref.index < 0:
                raise ValueError(
                    f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx}: "
                    "curr region ref index must be >= 0."
                )
            if region_pair.prev_region_ref.kind == "actual" and region_pair.prev_region_ref.index >= len(prev_station.regions):
                raise ValueError(
                    f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx}: "
                    f"prev region ref out of range ({region_pair.prev_region_ref.index})."
                )
            if region_pair.curr_region_ref.kind == "actual" and region_pair.curr_region_ref.index >= len(curr_station.regions):
                raise ValueError(
                    f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx}: "
                    f"curr region ref out of range ({region_pair.curr_region_ref.index})."
                )
            if not region_pair.loop_pairs:
                raise ValueError(
                    f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx}: "
                    "requires at least one loop pair."
                )
            expected_action = _expected_region_pair_action(
                transition.topology_case,
                region_pair.prev_region_ref,
                region_pair.curr_region_ref,
            )
            if region_pair.action != expected_action:
                raise ValueError(
                    f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx}: "
                    f"action {region_pair.action!r} does not match refs/topology_case "
                    f"(expected {expected_action!r})."
                )
            for loop_pair_idx, loop_pair in enumerate(region_pair.loop_pairs):
                if loop_pair.role not in valid_roles:
                    raise ValueError(
                        f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx} "
                        f"loop pair {loop_pair_idx}: invalid role {loop_pair.role!r}."
                    )
                expected_role = _planned_loop_pair_role(loop_pair.prev_loop_ref, loop_pair.curr_loop_ref)
                if loop_pair.role != expected_role:
                    raise ValueError(
                        f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx} "
                        f"loop pair {loop_pair_idx}: role {loop_pair.role!r} does not match refs."
                    )
                if loop_pair.prev_loop_ref.kind not in valid_ref_kinds:
                    raise ValueError(
                        f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx} "
                        f"loop pair {loop_pair_idx}: invalid prev loop kind "
                        f"{loop_pair.prev_loop_ref.kind!r}."
                    )
                if loop_pair.curr_loop_ref.kind not in valid_ref_kinds:
                    raise ValueError(
                        f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx} "
                        f"loop pair {loop_pair_idx}: invalid curr loop kind "
                        f"{loop_pair.curr_loop_ref.kind!r}."
                    )
                if loop_pair.prev_loop_ref.index < 0 or loop_pair.curr_loop_ref.index < 0:
                    raise ValueError(
                        f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx} "
                        f"loop pair {loop_pair_idx}: loop ref indices must be >= 0."
                    )
                if (
                    region_pair.prev_region_ref.kind == "actual"
                    and loop_pair.prev_loop_ref.kind == "actual"
                    and loop_pair.prev_loop_ref.index >= len(prev_station.regions[region_pair.prev_region_ref.index])
                ):
                    raise ValueError(
                        f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx} "
                        f"loop pair {loop_pair_idx}: prev loop ref out of range "
                        f"({loop_pair.prev_loop_ref.index})."
                    )
                if (
                    region_pair.curr_region_ref.kind == "actual"
                    and loop_pair.curr_loop_ref.kind == "actual"
                    and loop_pair.curr_loop_ref.index >= len(curr_station.regions[region_pair.curr_region_ref.index])
                ):
                    raise ValueError(
                        f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx} "
                        f"loop pair {loop_pair_idx}: curr loop ref out of range "
                        f"({loop_pair.curr_loop_ref.index})."
                    )
                if loop_pair.prev_loop.ndim != 2 or loop_pair.prev_loop.shape[1] != 2:
                    raise ValueError(
                        f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx} "
                        f"loop pair {loop_pair_idx}: prev loop must be Nx2."
                    )
                if loop_pair.curr_loop.ndim != 2 or loop_pair.curr_loop.shape[1] != 2:
                    raise ValueError(
                        f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx} "
                        f"loop pair {loop_pair_idx}: curr loop must be Nx2."
                    )
                if loop_pair.prev_loop.shape[0] != plan.samples or loop_pair.curr_loop.shape[0] != plan.samples:
                    raise ValueError(
                        f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx} "
                        f"loop pair {loop_pair_idx}: loop vertex count must equal samples={plan.samples}."
                    )
            for closure_idx, closure in enumerate(region_pair.closures):
                if closure.scope not in valid_scopes:
                    raise ValueError(
                        f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx} "
                        f"closure {closure_idx}: invalid scope {closure.scope!r}."
                    )
                if closure.side not in valid_sides:
                    raise ValueError(
                        f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx} "
                        f"closure {closure_idx}: invalid side {closure.side!r}."
                    )
                if closure.scope == "loop":
                    if closure.loop_index is None:
                        raise ValueError(
                            f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx} "
                            f"closure {closure_idx}: loop closure missing loop_index."
                        )
                    if closure.loop_index < 0 or closure.loop_index >= len(region_pair.loop_pairs):
                        raise ValueError(
                            f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx} "
                            f"closure {closure_idx}: loop_index {closure.loop_index} out of range."
                        )
                else:
                    if closure.loop_index is not None:
                        raise ValueError(
                            f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx} "
                            f"closure {closure_idx}: region closure must not define loop_index."
                        )
            seen_closure_keys: set[tuple[str, str, int | None]] = set()
            for closure_idx, closure in enumerate(region_pair.closures):
                closure_key = (closure.side, closure.scope, closure.loop_index)
                if closure_key in seen_closure_keys:
                    raise ValueError(
                        f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx} "
                        f"closure {closure_idx}: duplicate closure ownership for {closure_key}."
                    )
                seen_closure_keys.add(closure_key)
            _validate_region_pair_closure_ownership(
                region_pair=region_pair,
                interval=(prev_idx, curr_idx),
                region_pair_idx=region_pair_idx,
            )
        if tuple(collected_branch_order) != transition.branch_order:
            raise ValueError(
                f"Invalid loft plan transition interval {expected_interval}: "
                "branch_order must match region_pairs emission order."
            )


def _validate_region_pair_closure_ownership(
    *,
    region_pair: PlannedRegionPair,
    interval: tuple[int, int],
    region_pair_idx: int,
) -> None:
    prev_idx, curr_idx = interval
    region_prev = sum(1 for c in region_pair.closures if c.scope == "region" and c.side == "prev")
    region_curr = sum(1 for c in region_pair.closures if c.scope == "region" and c.side == "curr")
    if region_pair.action == "split_birth":
        if region_prev != 1 or region_curr != 0:
            raise ValueError(
                f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx}: "
                "split_birth requires exactly one prev region closure and zero curr region closures."
            )
    elif region_pair.action == "merge_death":
        if region_curr != 1 or region_prev != 0:
            raise ValueError(
                f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx}: "
                "merge_death requires exactly one curr region closure and zero prev region closures."
            )
    else:
        if region_prev != 0 or region_curr != 0:
            raise ValueError(
                f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx}: "
                f"action {region_pair.action!r} must not define region closures."
            )

    loop_prev = {c.loop_index for c in region_pair.closures if c.scope == "loop" and c.side == "prev"}
    loop_curr = {c.loop_index for c in region_pair.closures if c.scope == "loop" and c.side == "curr"}
    for loop_idx, loop_pair in enumerate(region_pair.loop_pairs):
        if loop_pair.role == "synthetic_birth":
            if loop_idx not in loop_prev or loop_idx in loop_curr:
                raise ValueError(
                    f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx}: "
                    f"synthetic_birth loop pair {loop_idx} must be closed on prev side only."
                )
        elif loop_pair.role == "synthetic_death":
            if loop_idx not in loop_curr or loop_idx in loop_prev:
                raise ValueError(
                    f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx}: "
                    f"synthetic_death loop pair {loop_idx} must be closed on curr side only."
                )
        else:
            if loop_idx in loop_prev or loop_idx in loop_curr:
                raise ValueError(
                    f"Invalid loft plan interval {prev_idx}->{curr_idx} region pair {region_pair_idx}: "
                    f"stable loop pair {loop_idx} must not define loop closures."
                )


def loft_endcaps(
    profiles: Sequence[Section | Region | Path2D | object],
    path: Path3D | PolyPath | Sequence[Sequence[float]] | None = None,
    samples: int = 200,
    segments_per_circle: int = 64,
    bezier_samples: int = 32,
    quality: MeshQuality | None = None,
    endcap_mode: str = "FLAT",
    endcap_amount: float | None = None,
    endcap_depth: float | None = None,
    endcap_radius: float | None = None,
    endcap_parameter_mode: str = "independent",
    endcap_steps: int = 12,
    endcap_placement: str = "BOTH",
) -> Mesh:
    """Loft profiles with experimental endcap generation (spec mode)."""

    if quality is not None:
        quality = apply_lod(quality)
        samples = _apply_quality_samples(samples, quality)
        segments_per_circle = _apply_quality_samples(segments_per_circle, quality)
        bezier_samples = _apply_quality_samples(bezier_samples, quality)

    normalized_profiles = _normalize_profile_inputs(
        profiles,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
    )

    _validate_profile_topology_input(
        normalized_profiles,
        fn_name="loft_endcaps",
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
    )
    hole_count = len(normalized_profiles[0].regions[0].holes)

    _validate_endcap_mode(endcap_mode)
    _validate_endcap_placement(endcap_placement)
    _validate_endcap_parameter_mode(endcap_parameter_mode)
    depth, radius = _resolve_endcap_amounts(
        endcap_mode=endcap_mode,
        endcap_amount=endcap_amount,
        endcap_depth=endcap_depth,
        endcap_radius=endcap_radius,
        parameter_mode=endcap_parameter_mode,
    )
    if endcap_mode != "FLAT" and (depth <= 0 or radius <= 0):
        raise ValueError("endcap_depth and endcap_radius must be > 0 for non-flat endcaps.")
    if endcap_mode == "ROUND" and endcap_steps < 2:
        raise ValueError("endcap_steps must be >= 2 for ROUND.")

    positions = _resolve_positions(path, len(normalized_profiles))
    stations = _build_stations(positions)

    loops_per_profile = [
        _loops_resampled_anchored(
            profile,
            samples,
            segments_per_circle=segments_per_circle,
            bezier_samples=bezier_samples,
        )
        for profile in normalized_profiles
    ]

    start_enabled = endcap_placement in {"START", "BOTH"} and endcap_mode != "FLAT"
    end_enabled = endcap_placement in {"END", "BOTH"} and endcap_mode != "FLAT"

    new_loops: list[list[np.ndarray]] = []
    new_positions: list[np.ndarray] = []
    cap_indices: list[tuple[int, str]] = []

    start_body_index = 0
    end_body_index = len(normalized_profiles)

    if start_enabled:
        cap_sections = _build_endcap_sections(
            section=normalized_profiles[0],
            base_loops=loops_per_profile[0],
            position=stations[0].origin,
            tangent=stations[0].n,
            direction=-1.0,
            mode=endcap_mode,
            depth=depth,
            radius=radius,
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
        end_body_index = len(normalized_profiles) - 1

    for idx in range(start_body_index, end_body_index):
        new_loops.append(loops_per_profile[idx])
        new_positions.append(positions[idx])

    if end_enabled:
        cap_sections = _build_endcap_sections(
            section=normalized_profiles[-1],
            base_loops=loops_per_profile[-1],
            position=stations[-1].origin,
            tangent=stations[-1].n,
            direction=1.0,
            mode=endcap_mode,
            depth=depth,
            radius=radius,
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

    loops_per_profile = _align_loops_for_loft(new_loops)
    positions = np.asarray(new_positions, dtype=float)
    stations = _build_stations(positions)

    loop_count = len(loops_per_profile[0]) if loops_per_profile else 0
    vertices: list[np.ndarray] = []
    offsets: list[list[int]] = []
    for profile_idx, loops in enumerate(loops_per_profile):
        profile_offsets = []
        station = stations[profile_idx]
        for loop in loops:
            profile_offsets.append(len(vertices))
            pts3 = station.origin + loop[:, 0:1] * station.u + loop[:, 1:2] * station.v
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
    color = getattr(normalized_profiles[0], "color", None)
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


def _normalize_profile_inputs(
    profiles: Sequence[Section | Region | Path2D | object],
    *,
    segments_per_circle: int,
    bezier_samples: int,
) -> list[Section]:
    normalized: list[Section] = []
    for idx, shape in enumerate(profiles):
        section = as_section(
            shape,
            segments_per_circle=segments_per_circle,
            bezier_samples=bezier_samples,
        ).normalized()
        if len(section.regions) != 1:
            raise ValueError(
                "loft currently requires one connected region per profile "
                f"(profile index {idx} has {len(section.regions)} regions)."
            )
        if not section.regions:
            raise ValueError(f"Profile at index {idx} resolved to empty topology.")
        normalized.append(section)
    return normalized


def _validate_profile_topology_input(
    profiles: Sequence[Section],
    *,
    fn_name: str,
    segments_per_circle: int,
    bezier_samples: int,
) -> None:
    if len(profiles) < 2:
        raise ValueError(f"{fn_name} requires at least two profiles.")

    expected_holes = len(profiles[0].regions[0].holes)
    for idx, section in enumerate(profiles):
        region = section.regions[0]
        if len(region.holes) != expected_holes:
            raise ValueError(
                "Unsupported topology transition: hole birth/death (split/merge) "
                f"is not supported in {fn_name} (profile index {idx})."
            )
        loops = _profile_loops(
            section,
            segments_per_circle=segments_per_circle,
            bezier_samples=bezier_samples,
            enforce_winding=True,
        )
        try:
            _classify_loops(loops, expected_holes=expected_holes)
        except ValueError as exc:
            raise ValueError(
                "Unsupported topology transition: region split/merge or invalid "
                f"hole containment in {fn_name} (profile index {idx})."
            ) from exc


def _validate_section_stations(stations: Sequence[Station]) -> None:
    if len(stations) < 2:
        raise ValueError("loft_sections requires at least two stations.")
    prev_t: float | None = None
    for idx, station in enumerate(stations):
        if not np.all(np.isfinite(station.origin)):
            raise ValueError(f"Station at index {idx} has non-finite origin.")
        _validate_station_frame(station, idx)
        if prev_t is not None and station.t <= prev_t:
            raise ValueError("Stations must be strictly ordered by t.")
        prev_t = station.t


def _validate_station_frame(station: Station, idx: int, tol: float = 1e-6) -> None:
    u = station.u
    v = station.v
    n = station.n
    norms = (np.linalg.norm(u), np.linalg.norm(v), np.linalg.norm(n))
    if not all(abs(float(norm) - 1.0) <= tol for norm in norms):
        raise ValueError(f"Station frame at index {idx} must be unit-length.")
    if abs(float(np.dot(u, v))) > tol or abs(float(np.dot(u, n))) > tol or abs(float(np.dot(v, n))) > tol:
        raise ValueError(f"Station frame at index {idx} must be orthogonal.")
    handedness = float(np.dot(np.cross(u, v), n))
    if handedness <= 0.0:
        raise ValueError(f"Station frame at index {idx} must be right-handed.")


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


def _build_stations(positions: np.ndarray) -> list[Station]:
    pts = np.asarray(positions, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("stations require Nx3 positions.")
    normals, binormals, tangents = _compute_frames(pts)
    t_values = _normalized_path_parameters(pts)
    stations: list[Station] = []
    for i in range(pts.shape[0]):
        u, v, n = _orthonormalize_frame(
            _normalized_vector(normals[i]),
            _normalized_vector(binormals[i]),
            _normalized_vector(tangents[i]),
        )
        stations.append(
            Station(
                t=float(t_values[i]),
                origin=pts[i].copy(),
                u=u,
                v=v,
                n=n,
            )
        )
    return stations


def _normalized_path_parameters(positions: np.ndarray) -> np.ndarray:
    if positions.shape[0] <= 1:
        return np.zeros(positions.shape[0], dtype=float)
    seg = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 1e-12:
        return np.linspace(0.0, 1.0, positions.shape[0])
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    return cumulative / total


def _normalized_vector(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return arr / norm


def _orthonormalize_frame(u: np.ndarray, v: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a right-handed orthonormal frame from approximate basis vectors."""

    u = _normalized_vector(u)
    n = _normalized_vector(n)

    # Keep U perpendicular to N.
    u = u - np.dot(u, n) * n
    if np.linalg.norm(u) <= 1e-12:
        aux = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(aux, n)) > 0.9:
            aux = np.array([0.0, 1.0, 0.0], dtype=float)
        u = aux - np.dot(aux, n) * n
    u = _normalized_vector(u)

    # Rebuild V from N and U to guarantee right-handedness.
    v = np.cross(n, u)
    if np.linalg.norm(v) <= 1e-12:
        aux = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(np.dot(aux, n)) > 0.9:
            aux = np.array([0.0, 0.0, 1.0], dtype=float)
        v = aux - np.dot(aux, n) * n - np.dot(aux, u) * u
    v = _normalized_vector(v)

    # Recompute N from UxV to remove residual drift and guarantee orthogonality.
    n = np.cross(u, v)
    n = _normalized_vector(n)

    return u, v, n


def _align_loops_for_loft(loops_per_profile: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
    if not loops_per_profile:
        return loops_per_profile
    hole_count = len(loops_per_profile[0]) - 1
    aligned: list[list[np.ndarray]] = []
    first_outer, first_holes = _validate_profile_loops(loops_per_profile[0], hole_count)
    aligned.append([first_outer, *first_holes])

    for profile_idx in range(1, len(loops_per_profile)):
        outer, holes = _validate_profile_loops(loops_per_profile[profile_idx], hole_count)
        prev_holes = aligned[-1][1:]
        order = _minimum_cost_hole_assignment(prev_holes, holes)
        reordered_holes = [holes[target_idx] for target_idx in order]
        aligned.append([outer, *reordered_holes])
    return aligned


def _section_to_region_loops(section: Section, *, samples: int) -> list[list[np.ndarray]]:
    normalized = section.normalized()
    regions: list[list[np.ndarray]] = []
    ordered_regions = sorted(normalized.regions, key=lambda region: _loop_sort_key(region.outer.points))
    for region in ordered_regions:
        ordered_holes = sorted(region.holes, key=lambda hole: _loop_sort_key(hole.points))
        loops = [region.outer.points, *(hole.points for hole in ordered_holes)]
        regions.append([_resample_loop(_anchor_loop(loop), samples) for loop in loops])
    return regions


def _expand_split_merge_stations(
    *,
    stations: list[Station],
    samples: int,
    split_merge_steps: int,
    split_merge_bias: float,
) -> list[Station]:
    if len(stations) < 2 or split_merge_steps <= 1:
        return stations

    expanded: list[Station] = [stations[0]]
    for idx in range(len(stations) - 1):
        prev = stations[idx]
        curr = stations[idx + 1]
        if prev.section is None or curr.section is None:
            expanded.append(curr)
            continue
        prev_regions = _section_to_region_loops(prev.section, samples=samples)
        curr_regions = _section_to_region_loops(curr.section, samples=samples)
        if not _needs_split_merge_staging(prev_regions, curr_regions):
            expanded.append(curr)
            continue

        transitions = _pair_sections_for_transition(
            prev_regions,
            curr_regions,
            split_merge_mode="resolve",
            split_merge_steps=split_merge_steps,
            split_merge_bias=split_merge_bias,
        )
        u0 = float(np.clip(split_merge_bias - 0.2, 0.0, 1.0))
        u1 = float(np.clip(split_merge_bias + 0.2, 0.0, 1.0))
        if u1 <= u0:
            u0, u1 = 0.25, 0.75
        u_values = np.linspace(u0, u1, split_merge_steps + 2, dtype=float)[1:-1]
        for u in u_values:
            alpha = (float(u) - u0) / (u1 - u0)
            staged_region_loops: list[list[np.ndarray]] = []
            for transition in transitions:
                region_loops: list[np.ndarray] = []
                for prev_loop, curr_loop in zip(
                    transition.region.prev_loops,
                    transition.region.curr_loops,
                    strict=True,
                ):
                    region_loops.append((1.0 - alpha) * prev_loop + alpha * curr_loop)
                staged_region_loops.append(region_loops)
            staged_section = _section_from_region_loops(staged_region_loops, color=curr.section.color)
            expanded.append(_interpolate_station(prev, curr, float(u), staged_section))
        expanded.append(curr)
    return expanded


def _needs_split_merge_staging(
    prev_regions: list[list[np.ndarray]],
    curr_regions: list[list[np.ndarray]],
) -> bool:
    if len(prev_regions) != len(curr_regions):
        return True
    if not prev_regions:
        return False
    order = _minimum_cost_region_assignment(
        [region[0] for region in prev_regions],
        [region[0] for region in curr_regions],
    )
    for prev_idx, curr_idx in enumerate(order):
        if len(prev_regions[prev_idx]) != len(curr_regions[curr_idx]):
            return True
    return False


def _section_from_region_loops(
    regions: list[list[np.ndarray]],
    *,
    color: tuple[float, float, float, float] | None = None,
) -> Section:
    region_objs: list[Region] = []
    for loops in regions:
        if not loops:
            continue
        outer_pts = np.asarray(loops[0], dtype=float)
        if outer_pts.shape[0] < 3:
            continue
        holes: list[Loop] = []
        for hole in loops[1:]:
            hole_pts = np.asarray(hole, dtype=float)
            if hole_pts.shape[0] < 3:
                continue
            holes.append(Loop(hole_pts))
        region_objs.append(Region(outer=Loop(outer_pts), holes=tuple(holes)).normalized())
    return Section(tuple(region_objs), color=color).normalized()


def _interpolate_station(prev: Station, curr: Station, u: float, section: Section) -> Station:
    u = float(np.clip(u, 0.0, 1.0))
    t = (1.0 - u) * prev.t + u * curr.t
    origin = (1.0 - u) * prev.origin + u * curr.origin
    u_vec = (1.0 - u) * prev.u + u * curr.u
    v_vec = (1.0 - u) * prev.v + u * curr.v
    n_vec = (1.0 - u) * prev.n + u * curr.n
    fu, fv, fn = _orthonormalize_frame(u_vec, v_vec, n_vec)
    return Station(t=t, origin=origin, u=fu, v=fv, n=fn, section=section)


def _loop_sort_key(loop: np.ndarray) -> tuple[float, ...]:
    pts = np.asarray(loop, dtype=float)
    centroid = pts.mean(axis=0)
    area = abs(_signed_area(pts))
    perimeter = float(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1).sum())
    anchor = _anchor_loop(pts)
    # Rounded coordinates keep deterministic ordering while avoiding noise-level drift.
    signature = tuple(np.round(anchor.reshape(-1), decimals=9).tolist())
    return (float(centroid[0]), float(centroid[1]), area, perimeter, *signature)


def _pair_sections_for_transition(
    prev_regions: list[list[np.ndarray]],
    curr_regions: list[list[np.ndarray]],
    *,
    split_merge_mode: str,
    split_merge_steps: int,
    split_merge_bias: float,
    ambiguity_mode: str = "auto",
    ambiguity_cost_profile: str = "balanced",
    ambiguity_max_branches: int = 64,
    fairness_mode: str = "local",
    fairness_weight: float = 0.2,
    skeleton_mode: str = "auto",
    fairness_iterations: int = 12,
    region_order_override: tuple[int, ...] | None = None,
) -> list[_PairedSectionTransition]:
    prev_count = len(prev_regions)
    curr_count = len(curr_regions)
    transitions: list[_PairedSectionTransition] = []

    prev_refs = [_RegionRef("actual", i) for i in range(prev_count)]
    curr_refs = [_RegionRef("actual", i) for i in range(curr_count)]

    if prev_count == curr_count:
        if region_order_override is not None:
            if len(region_order_override) != prev_count:
                raise ValueError("Invalid region_order_override length for one_to_one transition.")
            if sorted(region_order_override) != list(range(curr_count)):
                raise ValueError("Invalid region_order_override permutation for one_to_one transition.")
            region_order = region_order_override
        else:
            region_order = _minimum_cost_subset_assignment(
                [region[0] for region in prev_regions],
                [region[0] for region in curr_regions],
                entity="region",
                ambiguity_cost_profile=ambiguity_cost_profile,
                ambiguity_max_branches=ambiguity_max_branches,
                fairness_mode=fairness_mode,
                fairness_weight=fairness_weight,
                fairness_iterations=fairness_iterations,
            )
        for prev_idx, curr_idx in enumerate(region_order):
            paired = _pair_region_loops_for_transition(
                prev_regions[prev_idx],
                curr_regions[curr_idx],
                split_merge_mode=split_merge_mode,
                split_merge_steps=split_merge_steps,
                split_merge_bias=split_merge_bias,
                ambiguity_mode=ambiguity_mode,
                ambiguity_cost_profile=ambiguity_cost_profile,
                ambiguity_max_branches=ambiguity_max_branches,
                fairness_mode=fairness_mode,
                fairness_weight=fairness_weight,
                skeleton_mode=skeleton_mode,
                fairness_iterations=fairness_iterations,
            )
            transitions.append(
                _PairedSectionTransition(
                    prev_region_ref=prev_refs[prev_idx],
                    curr_region_ref=curr_refs[curr_idx],
                    region=paired,
                    region_closures=(),
                    action="stable",
                )
            )
        return transitions

    if prev_count < curr_count:
        matched_targets = _minimum_cost_subset_assignment(
            [region[0] for region in prev_regions],
            [region[0] for region in curr_regions],
            entity="region",
            ambiguity_cost_profile=ambiguity_cost_profile,
            ambiguity_max_branches=ambiguity_max_branches,
            fairness_mode=fairness_mode,
            fairness_weight=fairness_weight,
            fairness_iterations=fairness_iterations,
        )
        matched_target_set = set(matched_targets)
        unmatched_curr = sorted(set(range(curr_count)) - matched_target_set)
        if split_merge_mode == "fail" or _is_many_to_many_region_transition(prev_count, curr_count):
            _assert_unmatched_regions_non_ambiguous(
                unmatched_regions=[curr_regions[idx][0] for idx in unmatched_curr],
                opposite_regions=[region[0] for region in prev_regions],
            )
        for prev_idx, curr_idx in enumerate(matched_targets):
            paired = _pair_region_loops_for_transition(
                prev_regions[prev_idx],
                curr_regions[curr_idx],
                split_merge_mode=split_merge_mode,
                split_merge_steps=split_merge_steps,
                split_merge_bias=split_merge_bias,
                ambiguity_mode=ambiguity_mode,
                ambiguity_cost_profile=ambiguity_cost_profile,
                ambiguity_max_branches=ambiguity_max_branches,
                fairness_mode=fairness_mode,
                fairness_weight=fairness_weight,
                skeleton_mode=skeleton_mode,
                fairness_iterations=fairness_iterations,
            )
            transitions.append(
                _PairedSectionTransition(
                    prev_region_ref=prev_refs[prev_idx],
                    curr_region_ref=curr_refs[curr_idx],
                    region=paired,
                    region_closures=(),
                    action="split_match",
                )
            )
        for curr_idx in unmatched_curr:
            synthetic_prev = _shrunken_region(
                curr_regions[curr_idx],
                scale=_synthetic_seed_scale(split_merge_steps, split_merge_bias),
            )
            paired = _pair_region_loops_for_transition(
                synthetic_prev,
                curr_regions[curr_idx],
                split_merge_mode=split_merge_mode,
                split_merge_steps=split_merge_steps,
                split_merge_bias=split_merge_bias,
                ambiguity_mode=ambiguity_mode,
                ambiguity_cost_profile=ambiguity_cost_profile,
                ambiguity_max_branches=ambiguity_max_branches,
                fairness_mode=fairness_mode,
                fairness_weight=fairness_weight,
                skeleton_mode=skeleton_mode,
                fairness_iterations=fairness_iterations,
            )
            transitions.append(
                _PairedSectionTransition(
                    prev_region_ref=_RegionRef("synthetic", curr_idx),
                    curr_region_ref=curr_refs[curr_idx],
                    region=paired,
                    region_closures=("prev",),
                    action="split_birth",
                )
            )
        return transitions

    matched_prev_for_curr = _minimum_cost_subset_assignment(
        [region[0] for region in curr_regions],
        [region[0] for region in prev_regions],
        entity="region",
        ambiguity_cost_profile=ambiguity_cost_profile,
        ambiguity_max_branches=ambiguity_max_branches,
        fairness_mode=fairness_mode,
        fairness_weight=fairness_weight,
        fairness_iterations=fairness_iterations,
    )
    matched_prev_set = set(matched_prev_for_curr)
    unmatched_prev = sorted(set(range(prev_count)) - matched_prev_set)
    if split_merge_mode == "fail" or _is_many_to_many_region_transition(prev_count, curr_count):
        _assert_unmatched_regions_non_ambiguous(
            unmatched_regions=[prev_regions[idx][0] for idx in unmatched_prev],
            opposite_regions=[region[0] for region in curr_regions],
        )
    inv: dict[int, int] = {prev_idx: curr_idx for curr_idx, prev_idx in enumerate(matched_prev_for_curr)}
    for prev_idx in range(prev_count):
        if prev_idx in inv:
            curr_idx = inv[prev_idx]
            paired = _pair_region_loops_for_transition(
                prev_regions[prev_idx],
                curr_regions[curr_idx],
                split_merge_mode=split_merge_mode,
                split_merge_steps=split_merge_steps,
                split_merge_bias=split_merge_bias,
                ambiguity_mode=ambiguity_mode,
                ambiguity_cost_profile=ambiguity_cost_profile,
                ambiguity_max_branches=ambiguity_max_branches,
                fairness_mode=fairness_mode,
                fairness_weight=fairness_weight,
                skeleton_mode=skeleton_mode,
                fairness_iterations=fairness_iterations,
            )
            transitions.append(
                _PairedSectionTransition(
                    prev_region_ref=prev_refs[prev_idx],
                    curr_region_ref=curr_refs[curr_idx],
                    region=paired,
                    region_closures=(),
                    action="merge_match",
                )
            )
        else:
            synthetic_curr = _shrunken_region(
                prev_regions[prev_idx],
                scale=_synthetic_seed_scale(split_merge_steps, split_merge_bias),
            )
            paired = _pair_region_loops_for_transition(
                prev_regions[prev_idx],
                synthetic_curr,
                split_merge_mode=split_merge_mode,
                split_merge_steps=split_merge_steps,
                split_merge_bias=split_merge_bias,
                ambiguity_mode=ambiguity_mode,
                ambiguity_cost_profile=ambiguity_cost_profile,
                ambiguity_max_branches=ambiguity_max_branches,
                fairness_mode=fairness_mode,
                fairness_weight=fairness_weight,
                skeleton_mode=skeleton_mode,
                fairness_iterations=fairness_iterations,
            )
            transitions.append(
                _PairedSectionTransition(
                    prev_region_ref=prev_refs[prev_idx],
                    curr_region_ref=_RegionRef("synthetic", prev_idx),
                    region=paired,
                    region_closures=("curr",),
                    action="merge_death",
                )
            )
    if len(matched_prev_set) != len(curr_regions):
        raise ValueError("Unsupported topology transition: region split/merge detected.")
    return transitions


def _pair_region_loops_for_transition(
    prev_region_loops: list[np.ndarray],
    curr_region_loops: list[np.ndarray],
    *,
    split_merge_mode: str,
    split_merge_steps: int,
    split_merge_bias: float,
    ambiguity_mode: str = "auto",
    ambiguity_cost_profile: str = "balanced",
    ambiguity_max_branches: int = 64,
    fairness_mode: str = "local",
    fairness_weight: float = 0.2,
    skeleton_mode: str = "auto",
    fairness_iterations: int = 12,
) -> _PairedRegionLoops:
    # Step 1 (Spec 19): fairness/skeleton controls are surfaced and propagated.
    # Optimization behavior is introduced in follow-up steps.
    _ = (fairness_mode, fairness_weight, skeleton_mode, fairness_iterations)

    prev_outer = prev_region_loops[0]
    curr_outer = curr_region_loops[0]
    prev_holes = list(prev_region_loops[1:])
    curr_holes = list(curr_region_loops[1:])

    prev_out: list[np.ndarray] = [prev_outer]
    curr_out: list[np.ndarray] = [curr_outer]
    prev_refs: list[_LoopRef] = [_LoopRef("actual", 0)]
    curr_refs: list[_LoopRef] = [_LoopRef("actual", 0)]
    closures: list[tuple[str, int]] = []

    if len(prev_holes) == len(curr_holes):
        order = _minimum_cost_hole_assignment(prev_holes, curr_holes)
        for src_idx, tgt_idx in enumerate(order):
            prev_out.append(prev_holes[src_idx])
            curr_out.append(curr_holes[tgt_idx])
            prev_refs.append(_LoopRef("actual", src_idx + 1))
            curr_refs.append(_LoopRef("actual", tgt_idx + 1))
        return _PairedRegionLoops(
            prev_loops=tuple(prev_out),
            curr_loops=tuple(curr_out),
            prev_sources=tuple(prev_refs),
            curr_sources=tuple(curr_refs),
            closures=tuple(closures),
        )

    if len(prev_holes) < len(curr_holes):
        matched_targets = _minimum_cost_subset_assignment(
            prev_holes,
            curr_holes,
            entity="hole",
            ambiguity_cost_profile=ambiguity_cost_profile,
            ambiguity_max_branches=ambiguity_max_branches,
            fairness_mode=fairness_mode,
            fairness_weight=fairness_weight,
            fairness_iterations=fairness_iterations,
        )
        matched_target_set = set(matched_targets)
        unmatched_curr = sorted(set(range(len(curr_holes))) - matched_target_set)
        if split_merge_mode == "fail":
            _assert_unmatched_holes_non_ambiguous(
                unmatched_holes=[curr_holes[idx] for idx in unmatched_curr],
                opposite_holes=prev_holes,
            )
        for src_idx, tgt_idx in enumerate(matched_targets):
            prev_out.append(prev_holes[src_idx])
            curr_out.append(curr_holes[tgt_idx])
            prev_refs.append(_LoopRef("actual", src_idx + 1))
            curr_refs.append(_LoopRef("actual", tgt_idx + 1))

        for tgt_idx in unmatched_curr:
            target_loop = curr_holes[tgt_idx]
            seed = _shrunken_loop(
                target_loop,
                scale=_synthetic_seed_scale(split_merge_steps, split_merge_bias),
            )
            loop_slot = len(prev_out)
            prev_out.append(seed)
            curr_out.append(target_loop)
            prev_refs.append(_LoopRef("synthetic", loop_slot))
            curr_refs.append(_LoopRef("actual", tgt_idx + 1))
            closures.append(("prev", loop_slot))
    else:
        matched_prev_for_curr = _minimum_cost_subset_assignment(
            curr_holes,
            prev_holes,
            entity="hole",
            ambiguity_cost_profile=ambiguity_cost_profile,
            ambiguity_max_branches=ambiguity_max_branches,
            fairness_mode=fairness_mode,
            fairness_weight=fairness_weight,
            fairness_iterations=fairness_iterations,
        )
        unmatched_prev = sorted(set(range(len(prev_holes))) - set(matched_prev_for_curr))
        if split_merge_mode == "fail":
            _assert_unmatched_holes_non_ambiguous(
                unmatched_holes=[prev_holes[idx] for idx in unmatched_prev],
                opposite_holes=curr_holes,
            )
        inv_match: dict[int, int] = {
            prev_idx: curr_idx for curr_idx, prev_idx in enumerate(matched_prev_for_curr)
        }
        for prev_idx in range(len(prev_holes)):
            prev_loop = prev_holes[prev_idx]
            if prev_idx in inv_match:
                curr_idx = inv_match[prev_idx]
                prev_out.append(prev_loop)
                curr_out.append(curr_holes[curr_idx])
                prev_refs.append(_LoopRef("actual", prev_idx + 1))
                curr_refs.append(_LoopRef("actual", curr_idx + 1))
            else:
                collapsed = _shrunken_loop(
                    prev_loop,
                    scale=_synthetic_seed_scale(split_merge_steps, split_merge_bias),
                )
                loop_slot = len(prev_out)
                prev_out.append(prev_loop)
                curr_out.append(collapsed)
                prev_refs.append(_LoopRef("actual", prev_idx + 1))
                curr_refs.append(_LoopRef("synthetic", loop_slot))
                closures.append(("curr", loop_slot))

    return _PairedRegionLoops(
        prev_loops=tuple(prev_out),
        curr_loops=tuple(curr_out),
        prev_sources=tuple(prev_refs),
        curr_sources=tuple(curr_refs),
        closures=tuple(closures),
    )


def _minimum_cost_subset_assignment(
    source_loops: list[np.ndarray],
    target_loops: list[np.ndarray],
    *,
    entity: str,
    ambiguity_cost_profile: str = "balanced",
    ambiguity_max_branches: int = 64,
    fairness_mode: str = "local",
    fairness_weight: float = 0.2,
    fairness_iterations: int = 12,
) -> tuple[int, ...]:
    _ = fairness_iterations  # used by global mode selection in planner for now
    if not source_loops:
        return ()
    candidates, _, _ = _enumerate_subset_assignment_candidates(
        source_loops,
        target_loops,
        entity=entity,
        ambiguity_cost_profile=ambiguity_cost_profile,
        ambiguity_max_branches=ambiguity_max_branches,
    )
    if not candidates:
        raise ValueError("Unsupported topology transition: failed to compute subset assignment.")

    if fairness_mode in {"local", "global"} and float(fairness_weight) > 0.0:
        min_base = min(candidate[1][:3] for candidate in candidates)
        pool = [candidate for candidate in candidates if candidate[1][:3] == min_base]
        assignment = min(
            pool,
            key=lambda candidate: (
                float(fairness_weight) * candidate[2],
                candidate[1][3],
                candidate[1][4],
                candidate[1][5],
                candidate[1][6],
            ),
        )[0]
    else:
        assignment = min(candidates, key=lambda candidate: candidate[1])[0]

    for src_idx, dst_idx in enumerate(assignment):
        if not _is_stable_loop_transition(source_loops[src_idx], target_loops[dst_idx]):
            raise ValueError(
                "Unsupported topology transition: "
                f"unsupported_topology_ambiguity ({entity} split/merge ambiguity detected)."
            )
    return assignment


def _enumerate_subset_assignment_candidates(
    source_loops: list[np.ndarray],
    target_loops: list[np.ndarray],
    *,
    entity: str,
    ambiguity_cost_profile: str,
    ambiguity_max_branches: int,
) -> tuple[
    list[
        tuple[
            tuple[int, ...],
            tuple[float, float, float, float, float, tuple[tuple[float, ...], ...], tuple[int, ...]],
            float,
        ]
    ],
    list[np.ndarray],
    list[np.ndarray],
]:
    source = [np.asarray(loop, dtype=float) for loop in source_loops]
    target = [np.asarray(loop, dtype=float) for loop in target_loops]
    if len(source) > len(target):
        raise ValueError("subset assignment expects source count <= target count.")
    if not source:
        return [], [], []

    src_centroids = [loop.mean(axis=0) for loop in source]
    dst_centroids = [loop.mean(axis=0) for loop in target]
    src_areas = [abs(_signed_area(loop)) for loop in source]
    dst_areas = [abs(_signed_area(loop)) for loop in target]
    tgt_signatures = [tuple(np.round(_anchor_loop(loop).reshape(-1), decimals=9).tolist()) for loop in target]

    dist = np.zeros((len(source), len(target)), dtype=float)
    area = np.zeros((len(source), len(target)), dtype=float)
    containment = np.zeros((len(source), len(target)), dtype=float)
    for i in range(len(source)):
        for j in range(len(target)):
            dist[i, j] = float(np.linalg.norm(src_centroids[i] - dst_centroids[j]))
            area[i, j] = abs(src_areas[i] - dst_areas[j])
            containment[i, j] = 1.0 if _is_split_merge_ambiguous(source[i], target[j]) else 0.0

    max_assignments = int(ambiguity_max_branches)
    visited = 0
    candidates: list[
        tuple[
            tuple[int, ...],
            tuple[float, float, float, float, float, tuple[tuple[float, ...], ...], tuple[int, ...]],
            float,
        ]
    ] = []
    d_weight, a_weight = _ambiguity_profile_weights(ambiguity_cost_profile)

    def _recurse(i: int, used: set[int], partial: list[int]) -> None:
        nonlocal visited
        if i == len(source):
            visited += 1
            if visited > max_assignments:
                raise ValueError(
                    "Unsupported topology transition: unsupported_topology_ambiguity "
                    f"({entity} split/merge ambiguity candidate count exceeds ambiguity_max_branches; "
                    "tie_break_stage=candidate_enumeration_limit; "
                    f"candidate_count_after_pruning={visited}; "
                    f"ambiguity_max_branches={max_assignments})"
                )
            assignment = tuple(partial)
            score = _score_subset_assignment(
                assignment=assignment,
                dist=dist,
                area=area,
                containment=containment,
                src_centroids=src_centroids,
                dst_centroids=dst_centroids,
                target_signatures=tgt_signatures,
                distance_weight=d_weight,
                area_weight=a_weight,
            )
            local_fairness = _local_assignment_fairness(assignment, src_centroids, dst_centroids)
            candidates.append((assignment, score, local_fairness))
            return

        candidate_targets = sorted(
            (j for j in range(len(target)) if j not in used),
            key=lambda j: (dist[i, j], area[i, j], containment[i, j], j),
        )
        for j in candidate_targets:
            used.add(j)
            partial.append(j)
            _recurse(i + 1, used, partial)
            partial.pop()
            used.remove(j)

    _recurse(0, set(), [])
    return candidates, src_centroids, dst_centroids


def _local_assignment_fairness(
    assignment: tuple[int, ...],
    src_centroids: list[np.ndarray],
    dst_centroids: list[np.ndarray],
) -> float:
    if not assignment:
        return 0.0
    lengths = [
        float(np.linalg.norm(dst_centroids[dst_idx] - src_centroids[src_idx]))
        for src_idx, dst_idx in enumerate(assignment)
    ]
    if len(lengths) == 1:
        return lengths[0]
    mean_length = sum(lengths) / float(len(lengths))
    variance = sum((length - mean_length) ** 2 for length in lengths) / float(len(lengths))
    span = max(lengths) - min(lengths)
    return float(variance + span)


def _loop_centroid_world(station: PlannedStation, loop: np.ndarray) -> np.ndarray:
    pts = np.asarray(loop, dtype=float)
    centroid = pts.mean(axis=0)
    return station.origin + centroid[0] * station.u + centroid[1] * station.v


def _assignment_vectors_world(
    prev_regions: Sequence[np.ndarray],
    curr_regions: Sequence[np.ndarray],
    *,
    prev_station: PlannedStation,
    curr_station: PlannedStation,
    assignment: tuple[int, ...],
) -> dict[int, np.ndarray]:
    vectors: dict[int, np.ndarray] = {}
    for src_idx, dst_idx in enumerate(assignment):
        prev_point = _loop_centroid_world(prev_station, np.asarray(prev_regions[src_idx], dtype=float))
        curr_point = _loop_centroid_world(curr_station, np.asarray(curr_regions[dst_idx], dtype=float))
        vectors[src_idx] = curr_point - prev_point
    return vectors


def _select_global_region_order(
    *,
    prev_regions: list[np.ndarray],
    curr_regions: list[np.ndarray],
    prev_station: PlannedStation,
    curr_station: PlannedStation,
    previous_interval_vectors: dict[int, np.ndarray] | None,
    ambiguity_cost_profile: str,
    ambiguity_max_branches: int,
    fairness_weight: float,
    fairness_iterations: int,
) -> tuple[tuple[int, ...], bool, bool]:
    candidates, _, _ = _enumerate_subset_assignment_candidates(
        prev_regions,
        curr_regions,
        entity="region",
        ambiguity_cost_profile=ambiguity_cost_profile,
        ambiguity_max_branches=ambiguity_max_branches,
    )
    if not candidates:
        raise ValueError("Unsupported topology transition: failed to compute subset assignment.")

    iteration_budget = max(1, int(fairness_iterations))
    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate[1][0],
            candidate[1][1],
            candidate[1][2],
            candidate[1][3],
            candidate[1][4],
            candidate[1][5],
            candidate[1][6],
        ),
    )
    candidate_pool = ranked_candidates[:iteration_budget]
    hit_iteration_cap = len(ranked_candidates) > len(candidate_pool)
    if not candidate_pool:
        return candidates[0][0], False, False

    if previous_interval_vectors is None or float(fairness_weight) <= 0.0 or len(ranked_candidates) <= 1:
        return candidate_pool[0][0], False, hit_iteration_cap
    if len(candidate_pool) == 1:
        return candidate_pool[0][0], True, hit_iteration_cap

    best_assignment = candidate_pool[0][0]
    best_key: tuple[float, float, float, float, float, tuple[float, ...], tuple[int, ...]] | None = None
    local_weight = max(0.0, 1.0 - float(fairness_weight))
    assert previous_interval_vectors is not None

    for assignment, score, local_fairness in candidate_pool:
        vectors = _assignment_vectors_world(
            prev_regions,
            curr_regions,
            prev_station=prev_station,
            curr_station=curr_station,
            assignment=assignment,
        )
        continuity_penalty = 0.0
        acceleration_penalty = 0.0
        continuity_samples = 0
        for src_idx, prev_vec in previous_interval_vectors.items():
            next_vec = vectors.get(src_idx)
            if next_vec is None:
                continue
            prev_norm = float(np.linalg.norm(prev_vec))
            next_norm = float(np.linalg.norm(next_vec))
            if prev_norm <= 1e-12 or next_norm <= 1e-12:
                continue
            prev_dir = prev_vec / prev_norm
            next_dir = next_vec / next_norm
            dot = float(np.clip(np.dot(prev_dir, next_dir), -1.0, 1.0))
            continuity_penalty += float(1.0 - dot)
            acceleration_penalty += float(np.linalg.norm(next_vec - prev_vec))
            continuity_samples += 1

        if continuity_samples == 0:
            global_penalty = local_fairness
        else:
            global_penalty = float(continuity_penalty + acceleration_penalty)

        base_cost = float(score[0] + score[1] + score[3] + score[4])
        blended_fairness = float(fairness_weight) * global_penalty + local_weight * local_fairness
        key = (
            global_penalty,
            blended_fairness,
            base_cost,
            score[2],
            local_fairness,
            score[3],
            score[4],
            score[5],
            score[6],
        )
        if best_key is None or key < best_key:
            best_key = key
            best_assignment = assignment

    return best_assignment, True, hit_iteration_cap


def _transition_curr_vectors(
    *,
    prev_station: PlannedStation,
    curr_station: PlannedStation,
    transitions: Sequence[_PairedSectionTransition],
) -> dict[int, np.ndarray]:
    vectors: dict[int, np.ndarray] = {}
    for transition in transitions:
        if transition.curr_region_ref.kind != "actual":
            continue
        if not transition.region.prev_loops or not transition.region.curr_loops:
            continue
        prev_point = _loop_centroid_world(prev_station, transition.region.prev_loops[0])
        curr_point = _loop_centroid_world(curr_station, transition.region.curr_loops[0])
        vectors[transition.curr_region_ref.index] = curr_point - prev_point
    return vectors


def _ambiguity_profile_weights(profile: str) -> tuple[float, float]:
    if profile == "distance_first":
        return 1.0, 0.8
    if profile == "area_first":
        return 0.8, 1.0
    return 1.0, 1.0


def _score_subset_assignment(
    *,
    assignment: tuple[int, ...],
    dist: np.ndarray,
    area: np.ndarray,
    containment: np.ndarray,
    src_centroids: list[np.ndarray],
    dst_centroids: list[np.ndarray],
    target_signatures: list[tuple[float, ...]],
    distance_weight: float,
    area_weight: float,
) -> tuple[float, float, float, float, float, tuple[tuple[float, ...], ...], tuple[int, ...]]:
    distance_sum = float(sum(dist[i, j] for i, j in enumerate(assignment))) * float(distance_weight)
    area_sum = float(sum(area[i, j] for i, j in enumerate(assignment))) * float(area_weight)
    containment_penalty = float(sum(containment[i, j] for i, j in enumerate(assignment)))
    crossing_score = float(_assignment_crossing_score(assignment, src_centroids, dst_centroids))
    action_complexity = 0.0
    canonical_key = tuple(target_signatures[j] for j in assignment)
    return (
        distance_sum,
        area_sum,
        containment_penalty,
        crossing_score,
        action_complexity,
        canonical_key,
        assignment,
    )


def _assignment_crossing_score(
    assignment: tuple[int, ...],
    src_centroids: list[np.ndarray],
    dst_centroids: list[np.ndarray],
) -> int:
    score = 0
    for i in range(len(assignment)):
        for k in range(i + 1, len(assignment)):
            a0 = src_centroids[i]
            a1 = dst_centroids[assignment[i]]
            b0 = src_centroids[k]
            b1 = dst_centroids[assignment[k]]
            if _segments_intersect_2d(a0, a1, b0, b1):
                score += 1
    return score


def _compute_fairness_objective_terms(
    stations: Sequence[PlannedStation],
    transitions: Sequence[PlannedTransition],
) -> dict[str, float]:
    terms = _zero_fairness_objective_terms()
    if not transitions:
        return terms

    transition_vectors: list[dict[int, np.ndarray]] = []

    for transition in transitions:
        prev_idx, curr_idx = transition.interval
        prev_station = stations[prev_idx]
        curr_station = stations[curr_idx]
        pairs_by_curr_actual: dict[int, np.ndarray] = {}

        branch_segments_2d: list[tuple[np.ndarray, np.ndarray]] = []
        for region_pair in transition.region_pairs:
            start3, end3 = _region_pair_centerline_world(prev_station, curr_station, region_pair)
            if start3 is None or end3 is None:
                continue
            branch_segments_2d.append((start3[:2], end3[:2]))
            if region_pair.curr_region_ref.kind == "actual":
                pairs_by_curr_actual[region_pair.curr_region_ref.index] = end3 - start3

        terms["branch_crossing"] += float(_count_segment_crossings_2d(branch_segments_2d))
        transition_vectors.append(pairs_by_curr_actual)

        for region_pair in transition.region_pairs:
            if region_pair.action in {"split_birth", "merge_death"} and region_pair.loop_pairs:
                prev_area = abs(_signed_area(region_pair.loop_pairs[0].prev_loop))
                curr_area = abs(_signed_area(region_pair.loop_pairs[0].curr_loop))
                denom = max(prev_area, curr_area, 1e-12)
                terms["synthetic_harshness"] += float(abs(curr_area - prev_area) / denom)

            loop_count = len(region_pair.loop_pairs)
            if loop_count == 0:
                continue
            for closure in region_pair.closures:
                if closure.scope == "loop" and closure.loop_index is not None:
                    loop_idx = closure.loop_index
                    if 0 <= loop_idx < loop_count:
                        loop_pair = region_pair.loop_pairs[loop_idx]
                        terms["closure_stress"] += _normalized_loop_area_delta(loop_pair.prev_loop, loop_pair.curr_loop)
                elif closure.scope == "region":
                    local = 0.0
                    for loop_pair in region_pair.loop_pairs:
                        local += _normalized_loop_area_delta(loop_pair.prev_loop, loop_pair.curr_loop)
                    terms["closure_stress"] += float(local / float(loop_count))

    for i in range(len(transitions) - 1):
        curr_vectors = transition_vectors[i]
        next_transition = transitions[i + 1]
        next_prev_idx = next_transition.interval[0]
        next_curr_idx = next_transition.interval[1]
        next_station_prev = stations[next_prev_idx]
        next_station_curr = stations[next_curr_idx]

        for next_pair in next_transition.region_pairs:
            if next_pair.prev_region_ref.kind != "actual":
                continue
            region_idx = next_pair.prev_region_ref.index
            prev_vec = curr_vectors.get(region_idx)
            if prev_vec is None:
                continue
            start3, end3 = _region_pair_centerline_world(next_station_prev, next_station_curr, next_pair)
            if start3 is None or end3 is None:
                continue
            next_vec = end3 - start3
            prev_norm = float(np.linalg.norm(prev_vec))
            next_norm = float(np.linalg.norm(next_vec))
            if prev_norm <= 1e-12 or next_norm <= 1e-12:
                continue
            prev_dir = prev_vec / prev_norm
            next_dir = next_vec / next_norm
            dot = float(np.clip(np.dot(prev_dir, next_dir), -1.0, 1.0))
            terms["curvature_continuity"] += float(1.0 - dot)
            terms["branch_acceleration"] += float(np.linalg.norm(next_vec - prev_vec))

    return {key: float(value) for key, value in terms.items()}


def _region_pair_centerline_world(
    prev_station: PlannedStation,
    curr_station: PlannedStation,
    region_pair: PlannedRegionPair,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not region_pair.loop_pairs:
        return None, None
    outer_pair = region_pair.loop_pairs[0]
    prev_center = np.asarray(outer_pair.prev_loop, dtype=float).mean(axis=0)
    curr_center = np.asarray(outer_pair.curr_loop, dtype=float).mean(axis=0)
    prev_point = prev_station.origin + prev_center[0] * prev_station.u + prev_center[1] * prev_station.v
    curr_point = curr_station.origin + curr_center[0] * curr_station.u + curr_center[1] * curr_station.v
    return prev_point, curr_point


def _count_segment_crossings_2d(segments: Sequence[tuple[np.ndarray, np.ndarray]]) -> int:
    count = 0
    for i in range(len(segments)):
        a0, a1 = segments[i]
        for j in range(i + 1, len(segments)):
            b0, b1 = segments[j]
            if _segments_intersect_2d(a0, a1, b0, b1):
                count += 1
    return count


def _normalized_loop_area_delta(prev_loop: np.ndarray, curr_loop: np.ndarray) -> float:
    prev_area = abs(_signed_area(prev_loop))
    curr_area = abs(_signed_area(curr_loop))
    denom = max(prev_area, curr_area, 1e-12)
    return float(abs(curr_area - prev_area) / denom)


def _segments_intersect_2d(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> bool:
    def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)

    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


def _assert_unmatched_regions_non_ambiguous(
    *,
    unmatched_regions: list[np.ndarray],
    opposite_regions: list[np.ndarray],
) -> None:
    for candidate in unmatched_regions:
        for other in opposite_regions:
            if _is_split_merge_ambiguous(candidate, other):
                raise ValueError(
                    "Unsupported topology transition: "
                    "unsupported_topology_ambiguity (region split/merge ambiguity detected; "
                    "tie_break_stage=ambiguity_gate; candidate_count_after_pruning=unknown)"
                )


def _assert_unmatched_holes_non_ambiguous(
    *,
    unmatched_holes: list[np.ndarray],
    opposite_holes: list[np.ndarray],
) -> None:
    for candidate in unmatched_holes:
        for other in opposite_holes:
            if _is_split_merge_ambiguous(candidate, other):
                raise ValueError(
                    "Unsupported topology transition: "
                    "unsupported_topology_ambiguity (hole split/merge ambiguity detected; "
                    "tie_break_stage=ambiguity_gate; candidate_count_after_pruning=unknown)"
                )


def _is_many_to_many_region_transition(prev_count: int, curr_count: int) -> bool:
    return prev_count > 1 and curr_count > 1 and prev_count != curr_count


def _classify_region_transition_ambiguity(
    *,
    prev_regions: list[np.ndarray],
    curr_regions: list[np.ndarray],
    ambiguity_max_branches: int,
    ambiguity_cost_profile: str,
) -> str:
    prev_count = len(prev_regions)
    curr_count = len(curr_regions)
    if prev_count <= 1 and curr_count <= 1:
        return "none"
    if _has_containment_ambiguity(prev_regions, curr_regions):
        return "containment"
    if _is_many_to_many_region_transition(prev_count, curr_count):
        if _has_assignment_symmetry_or_permutation(
            prev_regions=prev_regions,
            curr_regions=curr_regions,
            ambiguity_max_branches=ambiguity_max_branches,
            ambiguity_cost_profile=ambiguity_cost_profile,
        ):
            if _is_symmetry_layout(prev_regions, curr_regions):
                return "symmetry"
            return "permutation"
        # Many-to-many decomposition is branch-ambiguous by default even when a
        # deterministic assignment exists; classify for diagnostics/metadata.
        return "permutation"
    if _has_assignment_symmetry_or_permutation(
        prev_regions=prev_regions,
        curr_regions=curr_regions,
        ambiguity_max_branches=ambiguity_max_branches,
        ambiguity_cost_profile=ambiguity_cost_profile,
    ):
        if _is_symmetry_layout(prev_regions, curr_regions):
            return "symmetry"
        return "permutation"
    return "none"


def _has_containment_ambiguity(prev_regions: list[np.ndarray], curr_regions: list[np.ndarray]) -> bool:
    for source in prev_regions:
        c = np.asarray(source, dtype=float).mean(axis=0)
        contained_in = sum(1 for target in curr_regions if _point_in_polygon(c, np.asarray(target, dtype=float)))
        if contained_in > 1:
            return True
    for target in curr_regions:
        c = np.asarray(target, dtype=float).mean(axis=0)
        contained_in = sum(1 for source in prev_regions if _point_in_polygon(c, np.asarray(source, dtype=float)))
        if contained_in > 1:
            return True
    return False


def _has_assignment_symmetry_or_permutation(
    *,
    prev_regions: list[np.ndarray],
    curr_regions: list[np.ndarray],
    ambiguity_max_branches: int,
    ambiguity_cost_profile: str,
) -> bool:
    n = len(prev_regions)
    m = len(curr_regions)
    if n == 0 or m == 0:
        return False
    if n > m:
        prev_regions, curr_regions = curr_regions, prev_regions
        n, m = m, n

    src = [np.asarray(loop, dtype=float) for loop in prev_regions]
    dst = [np.asarray(loop, dtype=float) for loop in curr_regions]
    src_centroids = [loop.mean(axis=0) for loop in src]
    dst_centroids = [loop.mean(axis=0) for loop in dst]
    src_areas = [abs(_signed_area(loop)) for loop in src]
    dst_areas = [abs(_signed_area(loop)) for loop in dst]
    tgt_signatures = [tuple(np.round(_anchor_loop(loop).reshape(-1), decimals=9).tolist()) for loop in dst]
    dist = np.zeros((n, m), dtype=float)
    area = np.zeros((n, m), dtype=float)
    containment = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            dist[i, j] = float(np.linalg.norm(src_centroids[i] - dst_centroids[j]))
            area[i, j] = abs(src_areas[i] - dst_areas[j])
            containment[i, j] = 1.0 if _is_split_merge_ambiguous(src[i], dst[j]) else 0.0

    d_weight, a_weight = _ambiguity_profile_weights(ambiguity_cost_profile)
    visited = 0
    best_prefix: tuple[float, float, float, float, float] | None = None
    tied_best = 0

    def _recurse(i: int, used: set[int], partial: list[int]) -> None:
        nonlocal visited, best_prefix, tied_best
        if i == n:
            visited += 1
            if visited > int(ambiguity_max_branches):
                return
            score = _score_subset_assignment(
                assignment=tuple(partial),
                dist=dist,
                area=area,
                containment=containment,
                src_centroids=src_centroids,
                dst_centroids=dst_centroids,
                target_signatures=tgt_signatures,
                distance_weight=d_weight,
                area_weight=a_weight,
            )
            prefix = score[:5]
            if best_prefix is None or prefix < best_prefix:
                best_prefix = prefix
                tied_best = 1
            elif prefix == best_prefix:
                tied_best += 1
            return
        for j in range(m):
            if j in used:
                continue
            used.add(j)
            partial.append(j)
            _recurse(i + 1, used, partial)
            partial.pop()
            used.remove(j)

    _recurse(0, set(), [])
    return tied_best > 1


def _is_symmetry_layout(prev_regions: list[np.ndarray], curr_regions: list[np.ndarray]) -> bool:
    def _spread_signature(regions: list[np.ndarray]) -> tuple[tuple[float, float], ...]:
        centroids = [np.asarray(loop, dtype=float).mean(axis=0) for loop in regions]
        origin = np.mean(np.vstack(centroids), axis=0)
        offsets = sorted(
            (round(float(c[0] - origin[0]), 6), round(float(c[1] - origin[1]), 6))
            for c in centroids
        )
        return tuple(offsets)

    return _spread_signature(prev_regions) == _spread_signature(curr_regions)


def _shrunken_loop(loop: np.ndarray, scale: float) -> np.ndarray:
    pts = np.asarray(loop, dtype=float)
    centroid = pts.mean(axis=0)
    return centroid + (pts - centroid) * float(scale)


def _shrunken_region(region_loops: list[np.ndarray], scale: float) -> list[np.ndarray]:
    outer = np.asarray(region_loops[0], dtype=float)
    centroid = outer.mean(axis=0)
    scaled: list[np.ndarray] = []
    for loop in region_loops:
        pts = np.asarray(loop, dtype=float)
        scaled.append(centroid + (pts - centroid) * float(scale))
    return scaled


def _resolve_transition_loop_start(
    *,
    station_index: int,
    station_origin: np.ndarray,
    station_u: np.ndarray,
    station_v: np.ndarray,
    region_ref: PlannedRegionRef,
    ref: PlannedLoopRef,
    loop: np.ndarray,
    offsets: list[list[list[int]]],
    vertices: list[np.ndarray],
    loop_start_cache: dict[tuple[int, str, int, str, int], int],
) -> int:
    if region_ref.kind == "actual" and ref.kind == "actual":
        return offsets[station_index][region_ref.index][ref.index]
    cache_key = (station_index, region_ref.kind, region_ref.index, ref.kind, ref.index)
    cached = loop_start_cache.get(cache_key)
    if cached is not None:
        return cached
    if ref.kind == "synthetic" or region_ref.kind == "synthetic":
        start = len(vertices)
        pts3 = station_origin + loop[:, 0:1] * station_u + loop[:, 1:2] * station_v
        vertices.extend(pts3)
        loop_start_cache[cache_key] = start
        return start
    raise ValueError(f"Unknown loop reference kind: {ref.kind}")


def _triangulate_loopset_faces_with_starts(
    loops: Sequence[np.ndarray],
    loop_starts: Sequence[int],
) -> np.ndarray:
    if len(loops) != len(loop_starts):
        raise ValueError("loop_starts length must match loops length.")
    _, local_faces = _triangulate_loops(list(loops))
    if local_faces.size == 0:
        return np.zeros((0, 3), dtype=int)
    total = int(sum(np.asarray(loop).shape[0] for loop in loops))
    index_map = np.empty(total, dtype=int)
    cursor = 0
    for start, loop in zip(loop_starts, loops, strict=True):
        count = int(np.asarray(loop).shape[0])
        index_map[cursor : cursor + count] = np.arange(start, start + count, dtype=int)
        cursor += count
    return index_map[local_faces]


def _validate_profile_loops(
    loops: list[np.ndarray],
    expected_holes: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if len(loops) != expected_holes + 1:
        raise ValueError("Unsupported topology transition: hole split/merge detected.")
    try:
        outer, holes = _classify_loops(loops, expected_holes=expected_holes)
    except ValueError as exc:
        raise ValueError("Unsupported topology transition: region split/merge detected.") from exc
    return outer, holes


def _minimum_cost_hole_assignment(
    source_holes: list[np.ndarray],
    target_holes: list[np.ndarray],
    *,
    area_weight: float = 0.1,
) -> tuple[int, ...]:
    """Return deterministic one-to-one hole correspondence.

    Primary score is centroid distance plus weighted area delta. Tie-breaks are
    resolved lexicographically by:
    1) lower total centroid distance
    2) lower total area delta
    3) lower target-index tuple in source order
    """

    if len(source_holes) != len(target_holes):
        raise ValueError("Unsupported topology transition: hole split/merge detected.")
    assignment = _minimum_cost_loop_assignment_topology(
        source_holes,
        target_holes,
        area_weight=area_weight,
    )
    for src_idx, dst_idx in enumerate(assignment):
        if not _is_stable_loop_transition(source_holes[src_idx], target_holes[dst_idx]):
            raise ValueError("Unsupported topology transition: hole split/merge ambiguity detected.")
    return assignment


def _minimum_cost_region_assignment(
    source_regions: list[np.ndarray],
    target_regions: list[np.ndarray],
    *,
    area_weight: float = 0.1,
) -> tuple[int, ...]:
    if len(source_regions) != len(target_regions):
        raise ValueError("Unsupported topology transition: region split/merge detected.")
    assignment = _minimum_cost_loop_assignment_topology(
        source_regions,
        target_regions,
        area_weight=area_weight,
    )
    for src_idx, dst_idx in enumerate(assignment):
        if not _is_stable_loop_transition(source_regions[src_idx], target_regions[dst_idx]):
            raise ValueError("Unsupported topology transition: region split/merge ambiguity detected.")
    return assignment


def _is_stable_loop_transition(source_loop: np.ndarray, target_loop: np.ndarray) -> bool:
    return _stable_loop_transition(source_loop, target_loop)


def _is_split_merge_ambiguous(a_loop: np.ndarray, b_loop: np.ndarray) -> bool:
    return _split_merge_ambiguous(a_loop, b_loop)


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


def _validate_endcap_parameter_mode(mode: str) -> None:
    allowed = {"independent", "linked"}
    if mode not in allowed:
        raise ValueError(f"endcap_parameter_mode must be one of {sorted(allowed)}.")


def _resolve_endcap_amounts(
    *,
    endcap_mode: str,
    endcap_amount: float | None,
    endcap_depth: float | None,
    endcap_radius: float | None,
    parameter_mode: str,
) -> tuple[float, float]:
    explicit_depth = endcap_depth is not None
    explicit_radius = endcap_radius is not None

    if endcap_mode == "FLAT":
        return 0.0, 0.0

    if endcap_amount is not None:
        if endcap_depth is None:
            endcap_depth = float(endcap_amount)
        if endcap_radius is None:
            endcap_radius = float(endcap_amount)

    if endcap_depth is None and endcap_radius is None:
        raise ValueError("Provide endcap_amount or endcap_depth/endcap_radius for non-flat endcaps.")
    if endcap_depth is None:
        endcap_depth = float(endcap_radius)
    if endcap_radius is None:
        endcap_radius = float(endcap_depth)

    depth = float(endcap_depth)
    radius = float(endcap_radius)

    if parameter_mode == "linked":
        if explicit_depth and explicit_radius and not np.isclose(depth, radius):
            raise ValueError("linked endcap mode requires endcap_depth == endcap_radius.")
        linked = depth if explicit_depth else radius
        depth = linked
        radius = linked

    return depth, radius


def _apply_caps(
    profiles: Sequence[Section],
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
    for section in profiles:
        loops = _loops_resampled(
            section,
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


def _validate_split_merge_mode(mode: str) -> None:
    if mode not in {"fail", "resolve"}:
        raise ValueError("split_merge_mode must be 'fail' or 'resolve'.")


def _validate_split_merge_controls(steps: int, bias: float) -> None:
    if int(steps) < 1:
        raise ValueError("split_merge_steps must be >= 1.")
    if not np.isfinite(float(bias)) or float(bias) < 0.0 or float(bias) > 1.0:
        raise ValueError("split_merge_bias must be within [0.0, 1.0].")


def _resolve_ambiguity_mode(split_merge_mode: str, ambiguity_mode: str | None) -> str:
    if ambiguity_mode is None:
        return "auto" if split_merge_mode == "resolve" else "fail"
    _validate_ambiguity_mode(ambiguity_mode)
    return ambiguity_mode


def _validate_ambiguity_mode(mode: str) -> None:
    if mode not in {"fail", "auto"}:
        raise ValueError("ambiguity_mode must be 'fail' or 'auto'.")


def _validate_ambiguity_cost_profile(profile: str) -> None:
    if profile not in {"balanced", "distance_first", "area_first"}:
        raise ValueError(
            "ambiguity_cost_profile must be one of "
            "['area_first', 'balanced', 'distance_first']."
        )


def _validate_ambiguity_max_branches(max_branches: int) -> None:
    if int(max_branches) < 1:
        raise ValueError("ambiguity_max_branches must be >= 1.")


def _validate_fairness_mode(mode: str) -> None:
    if mode not in {"off", "local", "global"}:
        raise ValueError("fairness_mode must be one of ['global', 'local', 'off'].")


def _validate_fairness_weight(weight: float) -> None:
    if float(weight) < 0.0:
        raise ValueError("fairness_weight must be >= 0.0.")


def _validate_skeleton_mode(mode: str) -> None:
    if mode not in {"off", "auto", "required"}:
        raise ValueError("skeleton_mode must be one of ['auto', 'off', 'required'].")


def _validate_fairness_iterations(iterations: int) -> None:
    if int(iterations) < 1:
        raise ValueError("fairness_iterations must be >= 1.")


def _zero_fairness_objective_terms() -> dict[str, float]:
    return {
        "curvature_continuity": 0.0,
        "branch_crossing": 0.0,
        "branch_acceleration": 0.0,
        "synthetic_harshness": 0.0,
        "closure_stress": 0.0,
    }


def _skeleton_guidance_available(stations: Sequence[PlannedStation]) -> bool:
    _ = stations
    # Skeleton extraction integration is not available yet; auto mode falls back.
    return False


def _count_region_topology_cases(transitions: Sequence[PlannedTransition]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for transition in transitions:
        counts[transition.topology_case] = counts.get(transition.topology_case, 0) + 1
    return counts


def _count_region_actions(transitions: Sequence[PlannedTransition]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for transition in transitions:
        for region_pair in transition.region_pairs:
            counts[region_pair.action] = counts.get(region_pair.action, 0) + 1
    return counts


def _classify_region_topology_case(prev_count: int, curr_count: int) -> str:
    if prev_count == curr_count:
        return "one_to_one"
    if prev_count == 1 and curr_count > 1:
        return "one_to_many"
    if prev_count > 1 and curr_count == 1:
        return "many_to_one"
    if prev_count < curr_count:
        return "many_to_many_expand"
    return "many_to_many_collapse"


def _expected_region_pair_action(
    topology_case: str,
    prev_ref: PlannedRegionRef,
    curr_ref: PlannedRegionRef,
) -> str:
    if topology_case == "one_to_one":
        return "stable"
    if topology_case in {"one_to_many", "many_to_many_expand"}:
        if prev_ref.kind == "actual" and curr_ref.kind == "actual":
            return "split_match"
        if prev_ref.kind == "synthetic" and curr_ref.kind == "actual":
            return "split_birth"
        raise ValueError(
            "Invalid loft plan region refs for expanding transition: "
            f"prev={prev_ref.kind!r} curr={curr_ref.kind!r}."
        )
    if topology_case in {"many_to_one", "many_to_many_collapse"}:
        if prev_ref.kind == "actual" and curr_ref.kind == "actual":
            return "merge_match"
        if prev_ref.kind == "actual" and curr_ref.kind == "synthetic":
            return "merge_death"
        raise ValueError(
            "Invalid loft plan region refs for collapsing transition: "
            f"prev={prev_ref.kind!r} curr={curr_ref.kind!r}."
        )
    raise ValueError(f"Invalid topology_case: {topology_case!r}")


def _synthetic_seed_scale(split_merge_steps: int, split_merge_bias: float) -> float:
    """Return deterministic synthetic seed scale used for birth/death decomposition.

    `split_merge_steps` and `split_merge_bias` are threaded through the v1 resolve
    path so progression stays deterministic and configurable without introducing
    random or ad-hoc seeds.
    """

    steps = max(1, int(split_merge_steps))
    bias = float(np.clip(split_merge_bias, 0.0, 1.0))
    # More steps -> tighter seed. Bias shifts where the synthetic loop appears
    # within the interval; use it as a deterministic seed-size modifier.
    base = 1.0 / (steps + 4.0)
    bias_mod = 0.75 + 0.5 * abs(bias - 0.5)
    return float(np.clip(base * bias_mod, 0.02, 0.2))


def _loops_resampled_anchored(
    section: Section,
    count: int,
    segments_per_circle: int,
    bezier_samples: int,
) -> list[np.ndarray]:
    loops = _profile_loops(
        section,
        segments_per_circle=segments_per_circle,
        bezier_samples=bezier_samples,
        enforce_winding=True,
    )
    return [_resample_loop(_anchor_loop(loop), count) for loop in loops]


def _build_endcap_sections(
    section: Section,
    base_loops: list[np.ndarray],
    position: np.ndarray,
    tangent: np.ndarray,
    direction: float,
    mode: str,
    depth: float,
    radius: float,
    steps: int,
    samples: int,
    segments_per_circle: int,
    bezier_samples: int,
    hole_count: int,
    reverse: bool = False,
) -> list[tuple[list[np.ndarray], np.ndarray]]:
    schedule = _endcap_schedule(mode, depth, radius, steps)
    sections: list[tuple[list[np.ndarray], np.ndarray]] = []
    for t, d in schedule:
        if np.isclose(t, 0.0):
            loops = base_loops
        else:
            try:
                loops = _inset_profile_loops(
                    section,
                    t,
                    join_type=mode,
                    hole_count=hole_count,
                    segments_per_circle=segments_per_circle,
                    bezier_samples=bezier_samples,
                )
            except ValueError as exc:
                msg = str(exc).lower()
                if "hole topology changed" in msg or "hole collapsed" in msg:
                    raise ValueError(
                        "Unsupported topology transition during endcap generation: "
                        "hole birth/death is not supported."
                    ) from exc
                if "inset collapsed" in msg:
                    raise ValueError(
                        "endcap_amount too large for profile features; "
                        "region collapsed during endcap generation."
                    ) from exc
                raise
            loops = [_resample_loop(_anchor_loop(loop), samples) for loop in loops]
        pos = position + tangent * (direction * d)
        sections.append((loops, pos))
    if reverse:
        if len(sections) > 1:
            sections = list(reversed(sections))
    return sections


def _endcap_schedule(mode: str, depth: float, radius: float, steps: int) -> list[tuple[float, float]]:
    if mode == "CHAMFER":
        return [(0.0, 0.0), (radius, depth)]
    if mode == "ROUND":
        count = max(1, steps)
        schedule: list[tuple[float, float]] = []
        for i in range(count + 1):
            theta = (i / count) * (np.pi / 2.0)
            t = radius * np.sin(theta)
            d = depth * (1.0 - np.cos(theta))
            schedule.append((float(t), float(d)))
        return schedule
    if mode == "COVE":
        count = max(1, steps)
        schedule = []
        for i in range(count + 1):
            u = i / count
            t = radius * (u**2)
            d = depth * u
            schedule.append((float(t), float(d)))
        return schedule
    return [(0.0, 0.0)]


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


__all__ = [
    "Station",
    "PlannedStation",
    "PlannedLoopRef",
    "PlannedRegionRef",
    "PlannedLoopPair",
    "PlannedClosure",
    "PlannedRegionPair",
    "PlannedTransition",
    "LoftPlan",
    "loft_profiles",
    "loft",
    "loft_plan_sections",
    "loft_execute_plan",
    "loft_sections",
    "loft_endcaps",
]
