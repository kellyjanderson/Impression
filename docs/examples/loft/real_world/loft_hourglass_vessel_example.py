from __future__ import annotations

import math

from impression.modeling import (
    Loft,
    Section,
    Station,
    as_section,
    export_tessellation_request,
    loft_plan_ambiguities,
    tessellate_surface_body,
)
from impression.modeling.drawing2d import PlanarShape2D, make_circle

TOTAL_HEIGHT = 70.0
SAMPLES = 48
SPLIT_MERGE_STEPS = 12
BASE_COLOR = "#d9b08c"
GLASS_COLOR = "#9ad1d4"
PILLAR_COLOR = "#7a8f6b"
BASE_RADIUS = 29.8
PILLAR_RADIUS = 2.475
PILLAR_OFFSET = 25.0
PILLAR_ANGLES_DEG = (90.0, 210.0, 330.0)
PILLAR_CANDIDATE_ID = "one_to_one:0-1-2-3"
BODY_SECTION_COUNT = 31
BODY_PHASE_GRID = 257
BODY_CURVE_DENSITY_BIAS = 3.0
BODY_MIDDLE_DENSITY_BIAS = 5.0
BODY_MIDDLE_DENSITY_WIDTH = 0.12
BASE_TRANSITION_Z = (0.0, 2.0, 4.5, 7.5)
BASE_TRANSITION_SCALES = (0.78, 0.9, 0.97, 1.0)
BODY_START_Z = 10.5
SHOULDER_OUTER_RADIUS = 22.5
NECK_OUTER_RADIUS = 2.1
SHOULDER_INNER_RADIUS = 16.2
NECK_INNER_RADIUS = 0.6


def _solid_disk(radius: float, color: str) -> PlanarShape2D:
    return make_circle(radius=radius, color=color)


def _ring_profile(outer_radius: float, inner_radius: float, color: str) -> PlanarShape2D:
    outer = make_circle(radius=outer_radius, color=color).outer
    inner = make_circle(radius=inner_radius).outer
    return PlanarShape2D(outer=outer, holes=[inner]).with_color(color)


def _sinusoid_mix(start: float, end: float, phase: float) -> float:
    weight = 0.5 * (1.0 + math.cos(2.0 * math.pi * phase))
    return end + (start - end) * weight


def _body_curve_strength(phase: float) -> float:
    return abs(math.sin(2.0 * math.pi * phase))


def _body_middle_strength(phase: float) -> float:
    distance = (phase - 0.5) / BODY_MIDDLE_DENSITY_WIDTH
    return math.exp(-(distance * distance))


def _body_phases() -> list[float]:
    if BODY_SECTION_COUNT <= 1:
        return [0.0]

    grid = [index / (BODY_PHASE_GRID - 1) for index in range(BODY_PHASE_GRID)]
    weights = [
        1.0
        + BODY_CURVE_DENSITY_BIAS * _body_curve_strength(phase)
        + BODY_MIDDLE_DENSITY_BIAS * _body_middle_strength(phase)
        for phase in grid
    ]

    cumulative = [0.0]
    for left_phase, right_phase, left_weight, right_weight in zip(
        grid[:-1],
        grid[1:],
        weights[:-1],
        weights[1:],
    ):
        span = right_phase - left_phase
        cumulative.append(cumulative[-1] + span * 0.5 * (left_weight + right_weight))

    total_weight = cumulative[-1]
    if total_weight <= 0.0:
        return [index / (BODY_SECTION_COUNT - 1) for index in range(BODY_SECTION_COUNT)]

    phases = [0.0]
    for target_index in range(1, BODY_SECTION_COUNT - 1):
        target_weight = total_weight * target_index / (BODY_SECTION_COUNT - 1)
        upper_index = next(index for index, value in enumerate(cumulative) if value >= target_weight)
        lower_index = upper_index - 1
        lower_weight = cumulative[lower_index]
        upper_weight = cumulative[upper_index]
        if math.isclose(upper_weight, lower_weight):
            phases.append(grid[upper_index])
            continue
        mix = (target_weight - lower_weight) / (upper_weight - lower_weight)
        phase = grid[lower_index] + mix * (grid[upper_index] - grid[lower_index])
        phases.append(phase)
    phases.append(1.0)
    return phases


def _pillar_regions(radius: float) -> list:
    regions = []
    for angle_deg in PILLAR_ANGLES_DEG:
        angle = math.radians(angle_deg)
        center = (
            PILLAR_OFFSET * math.cos(angle),
            PILLAR_OFFSET * math.sin(angle),
        )
        pillar = make_circle(radius=radius, center=center, color=PILLAR_COLOR)
        regions.append(as_section(pillar).regions[0])
    return regions


def _section(
    outer_radius: float,
    *,
    inner_radius: float | None = None,
    pillar_radius: float | None = None,
    color: str = GLASS_COLOR,
) -> Section:
    if inner_radius is None:
        main_region = as_section(_solid_disk(outer_radius, color)).regions[0]
    else:
        main_region = as_section(_ring_profile(outer_radius, inner_radius, color)).regions[0]

    regions = [main_region]
    if pillar_radius is not None:
        regions.extend(_pillar_regions(pillar_radius))
    return Section(tuple(regions))


def build_section_stack() -> tuple[list[float], list[Section]]:
    z_levels = list(BASE_TRANSITION_Z)
    sections = [_section(BASE_RADIUS * scale, color=BASE_COLOR) for scale in BASE_TRANSITION_SCALES]

    body_end_z = TOTAL_HEIGHT - BODY_START_Z
    for phase in _body_phases():
        z_levels.append(BODY_START_Z + (body_end_z - BODY_START_Z) * phase)
        outer_radius = _sinusoid_mix(SHOULDER_OUTER_RADIUS, NECK_OUTER_RADIUS, phase)
        inner_radius = _sinusoid_mix(SHOULDER_INNER_RADIUS, NECK_INNER_RADIUS, phase)
        sections.append(
            _section(
                outer_radius,
                inner_radius=inner_radius,
                pillar_radius=PILLAR_RADIUS,
            )
        )

    top_transition_z = [TOTAL_HEIGHT - z for z in reversed(BASE_TRANSITION_Z)]
    top_transition_scales = list(reversed(BASE_TRANSITION_SCALES))
    z_levels.extend(top_transition_z)
    sections.extend(_section(BASE_RADIUS * scale, color=BASE_COLOR) for scale in top_transition_scales)
    return z_levels, sections


def build_stations() -> list[Station]:
    z_levels, sections = build_section_stack()
    progression = [z / TOTAL_HEIGHT for z in z_levels]
    return [
        Station(
            t=t,
            section=section,
            origin=(0.0, 0.0, z),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 1.0, 0.0),
            n=(0.0, 0.0, 1.0),
        )
        for t, z, section in zip(progression, z_levels, sections)
    ]


def build_ambiguity_selection(stations: list[Station] | None = None) -> dict[tuple[int, int], str]:
    if stations is None:
        stations = build_stations()

    report = loft_plan_ambiguities(
        stations,
        samples=SAMPLES,
        split_merge_mode="resolve",
        split_merge_steps=SPLIT_MERGE_STEPS,
    )

    selection: dict[tuple[int, int], str] = {}
    for interval in report.intervals:
        candidate_ids = [candidate.candidate_id for candidate in interval.candidates]
        if PILLAR_CANDIDATE_ID in candidate_ids:
            selection[interval.interval] = PILLAR_CANDIDATE_ID
    return selection


def build_surface_body():
    stations = build_stations()
    progression = [station.t for station in stations]
    origins = [station.origin for station in stations]
    sections = [station.section for station in stations]
    ambiguity_selection = build_ambiguity_selection(stations)

    return Loft(
        progression=progression,
        stations=origins,
        topology=sections,
        samples=SAMPLES,
        cap_ends=True,
        split_merge_mode="resolve",
        split_merge_steps=SPLIT_MERGE_STEPS,
        ambiguity_mode="interactive",
        ambiguity_selection=ambiguity_selection,
        fairness_mode="local",
    )


def build():
    body = build_surface_body()
    return tessellate_surface_body(body, export_tessellation_request()).mesh
