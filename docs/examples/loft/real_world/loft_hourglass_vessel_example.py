"""Parametric multi-region hourglass vessel example for surfaced loft.

This example keeps the public docs focused on surfaced loft authoring while
exposing top-of-file parameter sets that make the model meaningfully editable.

Use the default `EXAMPLE_*` presets for the richer showcase build. The lighter
`TEST_*` presets exist so the automated suite can exercise the same model family
without paying full showcase cost on every run.
"""

import math
from dataclasses import dataclass
from math import factorial

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

BASE_COLOR = "#d9b08c"
GLASS_COLOR = "#9ad1d4"
PILLAR_COLOR = "#7a8f6b"
BASE_TRANSITION_SCALES = (0.78, 0.9, 0.97, 1.0)


@dataclass(frozen=True)
class HourglassParameters:
    base_diameter: float = 59.6
    base_thickness: float = 10.5
    support_count: int = 3
    total_height: float = 70.0
    support_diameter: float = 4.95
    support_inset: float = 2.325
    hourglass_wall_thickness: float = 1.5
    neck_inner_diameter: float = 1.2
    glass_flair_diameter: float = 45.0


@dataclass(frozen=True)
class HourglassQuality:
    samples: int = 48
    split_merge_steps: int = 12
    body_section_count: int = 31
    body_phase_grid: int = 257
    body_curve_density_bias: float = 3.0
    body_middle_density_bias: float = 5.0
    body_middle_density_width: float = 0.12


EXAMPLE_PARAMETERS = HourglassParameters()
EXAMPLE_QUALITY = HourglassQuality()

TEST_PARAMETERS = EXAMPLE_PARAMETERS
TEST_QUALITY = HourglassQuality(
    samples=36,
    split_merge_steps=10,
    body_section_count=31,
    body_phase_grid=129,
)


def _validate_params(params: HourglassParameters) -> None:
    if params.base_diameter <= 0.0:
        raise ValueError("base_diameter must be > 0.")
    if params.base_thickness <= 0.0:
        raise ValueError("base_thickness must be > 0.")
    if params.support_count < 1:
        raise ValueError("support_count must be >= 1.")
    if params.total_height <= 2.0 * params.base_thickness:
        raise ValueError("total_height must be greater than twice the base_thickness.")
    if params.support_diameter <= 0.0:
        raise ValueError("support_diameter must be > 0.")
    if params.support_inset < 0.0:
        raise ValueError("support_inset must be >= 0.")
    if params.hourglass_wall_thickness <= 0.0:
        raise ValueError("hourglass_wall_thickness must be > 0.")
    if params.neck_inner_diameter <= 0.0:
        raise ValueError("neck_inner_diameter must be > 0.")
    if params.glass_flair_diameter <= 0.0:
        raise ValueError("glass_flair_diameter must be > 0.")

    base_radius = params.base_diameter * 0.5
    support_radius = params.support_diameter * 0.5
    support_offset = base_radius - params.support_inset - support_radius
    if support_offset <= 0.0:
        raise ValueError("support_inset and support_diameter place supports at or past the center.")

    shoulder_outer_radius = params.glass_flair_diameter * 0.5
    neck_inner_radius = params.neck_inner_diameter * 0.5
    neck_outer_radius = neck_inner_radius + params.hourglass_wall_thickness
    shoulder_inner_radius = shoulder_outer_radius - params.hourglass_wall_thickness

    if shoulder_inner_radius <= 0.0:
        raise ValueError("hourglass_wall_thickness is too large for the glass_flair_diameter.")
    if neck_outer_radius >= shoulder_outer_radius:
        raise ValueError("neck_inner_diameter + wall thickness must stay smaller than the glass flair diameter.")
    if support_offset + support_radius > base_radius:
        raise ValueError("support_inset and support_diameter place supports outside the base.")


def _validate_quality(quality: HourglassQuality) -> None:
    if quality.samples < 8:
        raise ValueError("samples must be >= 8.")
    if quality.split_merge_steps < 1:
        raise ValueError("split_merge_steps must be >= 1.")
    if quality.body_section_count < 3:
        raise ValueError("body_section_count must be >= 3.")
    if quality.body_phase_grid <= quality.body_section_count:
        raise ValueError("body_phase_grid must be greater than body_section_count.")
    if quality.body_middle_density_width <= 0.0:
        raise ValueError("body_middle_density_width must be > 0.")


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


def _body_middle_strength(phase: float, quality: HourglassQuality) -> float:
    distance = (phase - 0.5) / quality.body_middle_density_width
    return math.exp(-(distance * distance))


def _resolved_body_section_count(params: HourglassParameters, quality: HourglassQuality) -> int:
    if params.support_count >= 5:
        return max(13, quality.body_section_count - 10)
    return quality.body_section_count


def _resolved_body_middle_density_bias(params: HourglassParameters, quality: HourglassQuality) -> float:
    if params.support_count >= 5:
        return min(quality.body_middle_density_bias, 3.0)
    return quality.body_middle_density_bias


def _body_phases(params: HourglassParameters, quality: HourglassQuality) -> list[float]:
    body_section_count = _resolved_body_section_count(params, quality)
    if body_section_count <= 1:
        return [0.0]

    grid = [index / (quality.body_phase_grid - 1) for index in range(quality.body_phase_grid)]
    weights = [
        1.0
        + quality.body_curve_density_bias * _body_curve_strength(phase)
        + _resolved_body_middle_density_bias(params, quality) * _body_middle_strength(phase, quality)
        for phase in grid
    ]

    cumulative = [0.0]
    for left_phase, right_phase, left_weight, right_weight in zip(grid[:-1], grid[1:], weights[:-1], weights[1:]):
        span = right_phase - left_phase
        cumulative.append(cumulative[-1] + span * 0.5 * (left_weight + right_weight))

    total_weight = cumulative[-1]
    if total_weight <= 0.0:
        return [index / (body_section_count - 1) for index in range(body_section_count)]

    phases = [0.0]
    for target_index in range(1, body_section_count - 1):
        target_weight = total_weight * target_index / (body_section_count - 1)
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


def _support_angles(params: HourglassParameters) -> tuple[float, ...]:
    step = 360.0 / params.support_count
    return tuple(90.0 + step * index for index in range(params.support_count))


def _pillar_candidate_id(params: HourglassParameters) -> str:
    region_indices = "-".join(str(index) for index in range(params.support_count + 1))
    return f"one_to_one:{region_indices}"


def _ambiguity_max_branches(params: HourglassParameters) -> int:
    return max(64, factorial(params.support_count + 1))


def _pillar_offset(params: HourglassParameters) -> float:
    return params.base_diameter * 0.5 - params.support_inset - params.support_diameter * 0.5


def _pillar_regions(params: HourglassParameters, radius: float) -> list:
    regions = []
    support_offset = _pillar_offset(params)
    for angle_deg in _support_angles(params):
        angle = math.radians(angle_deg)
        center = (
            support_offset * math.cos(angle),
            support_offset * math.sin(angle),
        )
        pillar = make_circle(radius=radius, center=center, color=PILLAR_COLOR)
        regions.append(as_section(pillar).regions[0])
    return regions


def _section(
    params: HourglassParameters,
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
        regions.extend(_pillar_regions(params, pillar_radius))
    return Section(tuple(regions))


def _section_stack(params: HourglassParameters, quality: HourglassQuality) -> tuple[list[float], list[Section]]:
    base_radius = params.base_diameter * 0.5
    base_transition_z = tuple(params.base_thickness * scale for scale in (0.0, 2.0 / 10.5, 4.5 / 10.5, 7.5 / 10.5))
    body_start_z = params.base_thickness
    body_end_z = params.total_height - body_start_z
    shoulder_outer_radius = params.glass_flair_diameter * 0.5
    neck_inner_radius = params.neck_inner_diameter * 0.5
    wall_thickness = params.hourglass_wall_thickness
    neck_outer_radius = neck_inner_radius + wall_thickness
    shoulder_inner_radius = shoulder_outer_radius - wall_thickness
    pillar_radius = params.support_diameter * 0.5

    z_levels = list(base_transition_z)
    sections = [_section(params, base_radius * scale, color=BASE_COLOR) for scale in BASE_TRANSITION_SCALES]

    for phase in _body_phases(params, quality):
        z_levels.append(body_start_z + (body_end_z - body_start_z) * phase)
        outer_radius = _sinusoid_mix(shoulder_outer_radius, neck_outer_radius, phase)
        inner_radius = _sinusoid_mix(shoulder_inner_radius, neck_inner_radius, phase)
        sections.append(
            _section(
                params,
                outer_radius,
                inner_radius=inner_radius,
                pillar_radius=pillar_radius,
            )
        )

    top_transition_z = [params.total_height - z for z in reversed(base_transition_z)]
    top_transition_scales = list(reversed(BASE_TRANSITION_SCALES))
    z_levels.extend(top_transition_z)
    sections.extend(_section(params, base_radius * scale, color=BASE_COLOR) for scale in top_transition_scales)
    return z_levels, sections


def build_stations(
    params: HourglassParameters | None = None,
    quality: HourglassQuality | None = None,
) -> list[Station]:
    resolved_params = params or EXAMPLE_PARAMETERS
    resolved_quality = quality or EXAMPLE_QUALITY
    _validate_params(resolved_params)
    _validate_quality(resolved_quality)
    z_levels, sections = _section_stack(resolved_params, resolved_quality)
    progression = [z / resolved_params.total_height for z in z_levels]
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


def build_ambiguity_selection(
    stations: list[Station] | None = None,
    params: HourglassParameters | None = None,
    quality: HourglassQuality | None = None,
) -> dict[tuple[int, int], str]:
    resolved_params = params or EXAMPLE_PARAMETERS
    resolved_quality = quality or EXAMPLE_QUALITY
    _validate_params(resolved_params)
    _validate_quality(resolved_quality)

    if stations is None:
        stations = build_stations(resolved_params, resolved_quality)

    report = loft_plan_ambiguities(
        stations,
        samples=resolved_quality.samples,
        split_merge_mode="resolve",
        split_merge_steps=resolved_quality.split_merge_steps,
        ambiguity_max_branches=_ambiguity_max_branches(resolved_params),
    )

    selection: dict[tuple[int, int], str] = {}
    pillar_candidate_id = _pillar_candidate_id(resolved_params)
    for interval in report.intervals:
        candidate_ids = [candidate.candidate_id for candidate in interval.candidates]
        if pillar_candidate_id in candidate_ids:
            selection[interval.interval] = pillar_candidate_id
    return selection


def build_surface_body(
    params: HourglassParameters | None = None,
    quality: HourglassQuality | None = None,
):
    resolved_params = params or EXAMPLE_PARAMETERS
    resolved_quality = quality or EXAMPLE_QUALITY
    stations = build_stations(resolved_params, resolved_quality)
    progression = [station.t for station in stations]
    origins = [station.origin for station in stations]
    sections = [station.section for station in stations]
    ambiguity_selection = build_ambiguity_selection(stations, resolved_params, resolved_quality)

    return Loft(
        progression=progression,
        stations=origins,
        topology=sections,
        samples=resolved_quality.samples,
        cap_ends=True,
        split_merge_mode="resolve",
        split_merge_steps=resolved_quality.split_merge_steps,
        ambiguity_mode="interactive",
        ambiguity_selection=ambiguity_selection,
        ambiguity_max_branches=_ambiguity_max_branches(resolved_params),
        fairness_mode="local",
    )


def build(
    params: HourglassParameters | None = None,
    quality: HourglassQuality | None = None,
):
    body = build_surface_body(params=params, quality=quality)
    return tessellate_surface_body(body, export_tessellation_request()).mesh
