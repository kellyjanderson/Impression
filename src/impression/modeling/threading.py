from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Callable, Literal, Sequence

import numpy as np

from impression.mesh import Mesh, analyze_mesh
from impression.modeling.csg import boolean_difference
from impression.modeling.primitives import make_cylinder, make_ngon

ThreadHand = Literal["right", "left"]
ThreadKind = Literal["external", "internal"]
ThreadProfileName = Literal[
    "iso",
    "unified",
    "whitworth",
    "trapezoidal",
    "acme",
    "square",
    "buttress",
    "pipe",
    "rounded",
    "custom",
]
EndTreatment = Literal["none", "higbee", "chamfer", "lead_in"]


class ThreadingError(ValueError):
    """Base error for thread generation failures."""


class InvalidThreadSpec(ThreadingError):
    """Raised when a thread specification is not physically meaningful."""


class InvalidFitSpec(ThreadingError):
    """Raised when a fit preset is unknown or malformed."""


class MeshBudgetExceeded(ThreadingError):
    """Raised when generated mesh exceeds the configured budget."""


@dataclass(frozen=True)
class ThreadFitPreset:
    """Manufacturing compensation values in millimeters."""

    name: str
    external_major_delta: float
    internal_major_delta: float
    external_depth_delta: float = 0.0
    internal_depth_delta: float = 0.0


@dataclass(frozen=True)
class MeshQuality:
    """Controls tessellation density and runtime cost."""

    segments_per_turn: int = 48
    circumferential_segments: int | None = None
    max_chord_deviation: float | None = 0.08
    max_triangles: int | None = 400_000
    adaptive_budget: bool = False


@dataclass(frozen=True)
class ThreadMeshEstimate:
    """Predicted mesh size for planning and budget enforcement."""

    predicted_vertices: int
    predicted_faces: int
    z_segments: int
    theta_segments: int


@dataclass(frozen=True)
class ThreadSpec:
    """Canonical thread definition shared by all generation entrypoints."""

    major_diameter: float
    pitch: float
    length: float
    profile: ThreadProfileName = "iso"
    hand: ThreadHand = "right"
    starts: int = 1
    taper_diameter_per_length: float = 0.0
    thread_depth: float | None = None
    crest_flat_ratio: float = 0.125
    root_flat_ratio: float = 0.125
    flank_angle_deg: float | None = None
    kind: ThreadKind = "external"
    thread_length: float | None = None
    thread_offset: float = 0.0
    start_treatment: EndTreatment = "none"
    end_treatment: EndTreatment = "none"
    start_treatment_length: float = 0.0
    end_treatment_length: float = 0.0
    axis_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    custom_profile_points: tuple[tuple[float, float], ...] | None = None


FIT_PRESETS: dict[str, ThreadFitPreset] = {
    "fdm_default": ThreadFitPreset("fdm_default", external_major_delta=-0.15, internal_major_delta=0.20),
    "fdm_tight": ThreadFitPreset("fdm_tight", external_major_delta=-0.08, internal_major_delta=0.10),
    "fdm_loose": ThreadFitPreset("fdm_loose", external_major_delta=-0.20, internal_major_delta=0.30),
    "sla_tight": ThreadFitPreset("sla_tight", external_major_delta=-0.04, internal_major_delta=0.06),
    "cnc_nominal": ThreadFitPreset("cnc_nominal", external_major_delta=-0.01, internal_major_delta=0.01),
}

_METRIC_COARSE: dict[int, float] = {
    1: 0.25,
    2: 0.40,
    3: 0.50,
    4: 0.70,
    5: 0.80,
    6: 1.00,
    8: 1.25,
    10: 1.50,
    12: 1.75,
    16: 2.00,
    20: 2.50,
    24: 3.00,
}


def lookup_standard_thread(
    family: str,
    designation: str | None = None,
    *,
    major_diameter: float | None = None,
    pitch: float | None = None,
    tpi: float | None = None,
    length: float = 10.0,
    starts: int = 1,
    hand: ThreadHand = "right",
    kind: ThreadKind = "external",
) -> ThreadSpec:
    """Build a ThreadSpec from common standards tables and shorthand designations."""

    family_key = family.strip().lower()
    if family_key in {"iso", "metric", "m"}:
        major, resolved_pitch = _resolve_metric_designation(designation, major_diameter, pitch)
        return ThreadSpec(
            major_diameter=major,
            pitch=resolved_pitch,
            length=length,
            profile="iso",
            starts=starts,
            hand=hand,
            kind=kind,
        )

    if family_key in {"unified", "unc", "unf", "unef"}:
        major, resolved_tpi = _resolve_unified_designation(designation, major_diameter, tpi)
        resolved_pitch = 25.4 / resolved_tpi
        return ThreadSpec(
            major_diameter=major,
            pitch=resolved_pitch,
            length=length,
            profile="unified",
            starts=starts,
            hand=hand,
            kind=kind,
        )

    if family_key in {"acme", "trapezoidal", "tr"}:
        if major_diameter is None or pitch is None:
            raise InvalidThreadSpec("Trapezoidal/ACME lookup requires major_diameter and pitch.")
        profile: ThreadProfileName = "acme" if family_key == "acme" else "trapezoidal"
        return ThreadSpec(
            major_diameter=float(major_diameter),
            pitch=float(pitch),
            length=length,
            profile=profile,
            starts=starts,
            hand=hand,
            kind=kind,
        )

    if family_key in {"pipe", "npt", "npt-like"}:
        if major_diameter is None or tpi is None:
            raise InvalidThreadSpec("Pipe lookup requires major_diameter and tpi.")
        return ThreadSpec(
            major_diameter=float(major_diameter),
            pitch=25.4 / float(tpi),
            length=length,
            profile="pipe",
            starts=starts,
            hand=hand,
            kind=kind,
            taper_diameter_per_length=1.0 / 16.0,
        )

    raise InvalidThreadSpec(f"Unsupported thread family '{family}'.")


def apply_fit(spec: ThreadSpec, preset: str | ThreadFitPreset, *, kind: ThreadKind | None = None) -> ThreadSpec:
    """Apply manufacturing clearance compensation to a thread spec."""

    fit = _resolve_fit_preset(preset)
    resolved_kind = kind or spec.kind
    if resolved_kind not in {"external", "internal"}:
        raise InvalidFitSpec("kind must be 'external' or 'internal'.")

    if resolved_kind == "external":
        major = spec.major_diameter + fit.external_major_delta
        depth = _resolve_thread_depth(spec) + fit.external_depth_delta
    else:
        major = spec.major_diameter + fit.internal_major_delta
        depth = _resolve_thread_depth(spec) + fit.internal_depth_delta

    if depth <= 0:
        raise InvalidFitSpec("Fit compensation produced non-positive thread depth.")
    if major <= 0:
        raise InvalidFitSpec("Fit compensation produced non-positive major diameter.")

    return replace(spec, major_diameter=major, thread_depth=depth, kind=resolved_kind)


def paired_fit(spec: ThreadSpec, preset: str | ThreadFitPreset) -> tuple[ThreadSpec, ThreadSpec]:
    """Create a matched external/internal pair for a fit preset."""

    male = apply_fit(replace(spec, kind="external"), preset, kind="external")
    female = apply_fit(replace(spec, kind="internal"), preset, kind="internal")
    return male, female


def validate_thread(spec: ThreadSpec, quality: MeshQuality | None = None) -> ThreadSpec:
    """Validate a ThreadSpec and raise descriptive errors for invalid combinations."""

    if spec.pitch <= 0:
        raise InvalidThreadSpec("pitch must be positive.")
    if spec.length <= 0:
        raise InvalidThreadSpec("length must be positive.")
    if spec.major_diameter <= 0:
        raise InvalidThreadSpec("major_diameter must be positive.")
    if spec.starts < 1:
        raise InvalidThreadSpec("starts must be >= 1.")

    thread_depth = _resolve_thread_depth(spec)
    if thread_depth <= 0:
        raise InvalidThreadSpec("thread depth must be positive.")

    minor = spec.major_diameter - 2.0 * thread_depth
    if minor <= 0:
        raise InvalidThreadSpec(
            f"Thread depth collapses the core (minor diameter <= 0). major={spec.major_diameter:.4f}, depth={thread_depth:.4f}."
        )

    if spec.thread_length is not None and spec.thread_length <= 0:
        raise InvalidThreadSpec("thread_length must be positive when provided.")
    if spec.thread_offset < 0:
        raise InvalidThreadSpec("thread_offset cannot be negative.")

    thread_length = spec.thread_length if spec.thread_length is not None else spec.length
    if spec.thread_offset + thread_length > spec.length + 1e-9:
        raise InvalidThreadSpec("thread_offset + thread_length cannot exceed total length.")

    if spec.start_treatment_length < 0 or spec.end_treatment_length < 0:
        raise InvalidThreadSpec("Treatment lengths must be non-negative.")
    if spec.start_treatment not in {"none", "higbee", "chamfer", "lead_in"}:
        raise InvalidThreadSpec(f"Unsupported start treatment '{spec.start_treatment}'.")
    if spec.end_treatment not in {"none", "higbee", "chamfer", "lead_in"}:
        raise InvalidThreadSpec(f"Unsupported end treatment '{spec.end_treatment}'.")

    axis = np.asarray(spec.axis_direction, dtype=float)
    if np.linalg.norm(axis) == 0:
        raise InvalidThreadSpec("axis_direction must be non-zero.")

    if spec.custom_profile_points is not None and len(spec.custom_profile_points) < 3:
        raise InvalidThreadSpec("custom_profile_points must contain at least 3 points.")

    if quality is not None:
        if quality.segments_per_turn < 8:
            raise InvalidThreadSpec("segments_per_turn must be at least 8.")
        if quality.circumferential_segments is not None and quality.circumferential_segments < 12:
            raise InvalidThreadSpec("circumferential_segments must be at least 12 when provided.")

    return spec


def estimate_mesh_cost(spec: ThreadSpec, quality: MeshQuality = MeshQuality()) -> ThreadMeshEstimate:
    """Predict mesh cost before generation."""

    validate_thread(spec, quality)

    z_segments = max(1, int(np.ceil(spec.length / spec.pitch * quality.segments_per_turn)))
    theta_segments = _resolve_theta_segments(spec, quality)

    side_faces = 2 * z_segments * theta_segments
    cap_faces = 2 * theta_segments
    predicted_faces = side_faces + cap_faces
    predicted_vertices = (z_segments + 1) * theta_segments + 2
    return ThreadMeshEstimate(
        predicted_vertices=predicted_vertices,
        predicted_faces=predicted_faces,
        z_segments=z_segments,
        theta_segments=theta_segments,
    )


def make_external_thread(
    spec: ThreadSpec,
    *,
    quality: MeshQuality = MeshQuality(),
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Generate a manifold external thread solid."""

    resolved = replace(spec, kind="external")
    return _build_thread_mesh(resolved, quality=quality, color=color)


def make_internal_thread(
    spec: ThreadSpec,
    *,
    quality: MeshQuality = MeshQuality(),
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Generate a manifold internal-thread cutter solid."""

    resolved = replace(spec, kind="internal")
    return _build_thread_mesh(resolved, quality=quality, color=color)


def make_threaded_rod(
    spec: ThreadSpec,
    *,
    quality: MeshQuality = MeshQuality(),
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Convenience generator for a threaded rod section."""

    return make_external_thread(replace(spec, kind="external"), quality=quality, color=color)


def make_tapped_hole_cutter(
    spec: ThreadSpec,
    *,
    quality: MeshQuality = MeshQuality(),
    overshoot: float = 0.5,
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Generate negative-volume cutter geometry for tapped holes."""

    if overshoot < 0:
        raise InvalidThreadSpec("overshoot must be non-negative.")
    cutter_spec = replace(
        spec,
        kind="internal",
        length=spec.length + 2.0 * overshoot,
        axis_origin=(spec.axis_origin[0], spec.axis_origin[1], spec.axis_origin[2] - overshoot),
    )
    return make_internal_thread(cutter_spec, quality=quality, color=color)


def make_hex_nut(
    spec: ThreadSpec,
    *,
    thickness: float,
    across_flats: float,
    quality: MeshQuality = MeshQuality(),
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Create a simple hex nut by subtracting an internal thread cutter from a hex prism."""

    if thickness <= 0 or across_flats <= 0:
        raise InvalidThreadSpec("thickness and across_flats must be positive.")

    circ_radius = across_flats / np.sqrt(3.0)
    nut_body = make_ngon(
        sides=6,
        radius=circ_radius,
        height=thickness,
        center=(spec.axis_origin[0], spec.axis_origin[1], spec.axis_origin[2] + thickness / 2.0),
    )

    cutter_spec = replace(spec, kind="internal", length=thickness + 1.0, axis_origin=(spec.axis_origin[0], spec.axis_origin[1], spec.axis_origin[2] - 0.5))
    cutter = make_internal_thread(cutter_spec, quality=quality)
    nut = boolean_difference(nut_body, [cutter])
    if color is not None:
        from impression.modeling._color import set_mesh_color

        set_mesh_color(nut, color)
    return nut


def make_round_nut(
    spec: ThreadSpec,
    *,
    thickness: float,
    outer_diameter: float,
    quality: MeshQuality = MeshQuality(),
    color: Sequence[float] | str | None = None,
) -> Mesh:
    """Create a round nut by subtracting an internal thread cutter from a cylinder."""

    if thickness <= 0 or outer_diameter <= 0:
        raise InvalidThreadSpec("thickness and outer_diameter must be positive.")

    nut_body = make_cylinder(
        radius=outer_diameter / 2.0,
        height=thickness,
        center=(spec.axis_origin[0], spec.axis_origin[1], spec.axis_origin[2] + thickness / 2.0),
    )
    cutter_spec = replace(spec, kind="internal", length=thickness + 1.0, axis_origin=(spec.axis_origin[0], spec.axis_origin[1], spec.axis_origin[2] - 0.5))
    cutter = make_internal_thread(cutter_spec, quality=quality)
    nut = boolean_difference(nut_body, [cutter])
    if color is not None:
        from impression.modeling._color import set_mesh_color

        set_mesh_color(nut, color)
    return nut


def _build_thread_mesh(
    spec: ThreadSpec,
    *,
    quality: MeshQuality,
    color: Sequence[float] | str | None,
) -> Mesh:
    validate_thread(spec, quality)
    estimate = estimate_mesh_cost(spec, quality)
    if quality.max_triangles is not None and estimate.predicted_faces > quality.max_triangles:
        if quality.adaptive_budget:
            reduced = max(8, int(quality.segments_per_turn * quality.max_triangles / max(estimate.predicted_faces, 1)))
            quality = replace(quality, segments_per_turn=reduced)
            estimate = estimate_mesh_cost(spec, quality)
        else:
            raise MeshBudgetExceeded(
                f"Predicted face count {estimate.predicted_faces} exceeds budget {quality.max_triangles}."
            )

    z_count = estimate.z_segments + 1
    z_values = np.linspace(0.0, spec.length, z_count, dtype=float)
    theta_count = estimate.theta_segments
    theta_values = np.linspace(0.0, 2.0 * np.pi, theta_count, endpoint=False, dtype=float)

    profile_fn = _build_profile_function(spec)
    thread_depth = _resolve_thread_depth(spec)
    base_minor = spec.major_diameter - 2.0 * thread_depth
    base_minor_radius = max(base_minor * 0.5, 1e-6)

    axis_origin = np.asarray(spec.axis_origin, dtype=float)
    axis_basis = _build_axis_basis(spec.axis_direction)

    verts = np.zeros((z_count * theta_count + 2, 3), dtype=float)

    for zi, z in enumerate(z_values):
        radius_base = spec.major_diameter * 0.5 + 0.5 * spec.taper_diameter_per_length * z
        minor_radius = radius_base - thread_depth
        amp = _thread_amplitude(spec, z)
        for ti, theta in enumerate(theta_values):
            base_phase = _thread_phase(spec, theta=theta, z=z)
            h = max(profile_fn(base_phase + k / spec.starts) for k in range(spec.starts))
            local_radius = minor_radius + thread_depth * amp * h
            p_local = np.array([local_radius * np.cos(theta), local_radius * np.sin(theta), z], dtype=float)
            verts[zi * theta_count + ti] = axis_origin + axis_basis @ p_local

    bottom_center_idx = z_count * theta_count
    top_center_idx = bottom_center_idx + 1
    p0 = axis_origin + axis_basis @ np.array([0.0, 0.0, z_values[0]], dtype=float)
    p1 = axis_origin + axis_basis @ np.array([0.0, 0.0, z_values[-1]], dtype=float)
    verts[bottom_center_idx] = p0
    verts[top_center_idx] = p1

    faces: list[list[int]] = []

    for zi in range(z_count - 1):
        base = zi * theta_count
        top = (zi + 1) * theta_count
        for ti in range(theta_count):
            tj = (ti + 1) % theta_count
            a = base + ti
            b = base + tj
            c = top + tj
            d = top + ti
            faces.append([a, b, c])
            faces.append([a, c, d])

    for ti in range(theta_count):
        tj = (ti + 1) % theta_count
        faces.append([bottom_center_idx, tj, ti])

    top_offset = (z_count - 1) * theta_count
    for ti in range(theta_count):
        tj = (ti + 1) % theta_count
        faces.append([top_center_idx, top_offset + ti, top_offset + tj])

    mesh = Mesh(vertices=verts, faces=np.asarray(faces, dtype=int))
    if color is not None:
        from impression.modeling._color import set_mesh_color

        set_mesh_color(mesh, color)

    analyze_mesh(mesh)
    return mesh


def _thread_phase(spec: ThreadSpec, *, theta: float, z: float) -> float:
    hand_sign = 1.0 if spec.hand == "right" else -1.0
    base = (hand_sign * theta - (2.0 * np.pi / spec.pitch) * z) / (2.0 * np.pi)
    return _frac(base)


def _thread_amplitude(spec: ThreadSpec, z: float) -> float:
    thread_len = spec.thread_length if spec.thread_length is not None else spec.length
    z0 = spec.thread_offset
    z1 = z0 + thread_len
    if z < z0 or z > z1:
        return 0.0

    amp = 1.0
    amp *= _edge_treatment_factor(z - z0, spec.start_treatment, spec.start_treatment_length)
    amp *= _edge_treatment_factor(z1 - z, spec.end_treatment, spec.end_treatment_length)
    return float(np.clip(amp, 0.0, 1.0))


def _edge_treatment_factor(distance_to_edge: float, mode: EndTreatment, length: float) -> float:
    if mode == "none" or length <= 1e-9:
        return 1.0
    if distance_to_edge >= length:
        return 1.0

    t = np.clip(distance_to_edge / length, 0.0, 1.0)
    if mode == "higbee":
        return t * t * (3.0 - 2.0 * t)
    if mode == "chamfer":
        return t
    if mode == "lead_in":
        return np.sqrt(t)
    return 1.0


def _build_profile_function(spec: ThreadSpec) -> Callable[[float], float]:
    if spec.profile == "custom":
        if not spec.custom_profile_points:
            raise InvalidThreadSpec("custom profile requires custom_profile_points.")
        return _build_custom_profile(spec.custom_profile_points)

    if spec.profile in {"iso", "unified", "pipe"}:
        crest = spec.crest_flat_ratio
        root = spec.root_flat_ratio
        return lambda phase: _trapezoid_wave(phase, crest_ratio=crest, root_ratio=root)

    if spec.profile == "whitworth":
        return lambda phase: _rounded_trapezoid_wave(phase, crest_ratio=0.12, root_ratio=0.12, roundness=0.8)

    if spec.profile in {"acme", "trapezoidal"}:
        return lambda phase: _trapezoid_wave(phase, crest_ratio=0.30, root_ratio=0.30)

    if spec.profile == "square":
        return lambda phase: 1.0 if _frac(phase) < 0.5 else 0.0

    if spec.profile == "buttress":
        return lambda phase: _buttress_wave(phase)

    if spec.profile == "rounded":
        return lambda phase: 0.5 * (1.0 + np.cos(2.0 * np.pi * _frac(phase)))

    raise InvalidThreadSpec(f"Unsupported profile '{spec.profile}'.")


def _build_custom_profile(points: Sequence[tuple[float, float]]) -> Callable[[float], float]:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise InvalidThreadSpec("custom_profile_points must be Nx2 points.")
    phases = arr[:, 0]
    values = arr[:, 1]
    if np.any(np.diff(phases) < 0):
        order = np.argsort(phases)
        phases = phases[order]
        values = values[order]
    phases = np.mod(phases, 1.0)
    order = np.argsort(phases)
    phases = phases[order]
    values = np.clip(values[order], 0.0, 1.0)
    if phases[0] > 0.0:
        phases = np.hstack([[0.0], phases])
        values = np.hstack([[values[-1]], values])
    if phases[-1] < 1.0:
        phases = np.hstack([phases, [1.0]])
        values = np.hstack([values, [values[0]]])

    def fn(phase: float) -> float:
        p = _frac(phase)
        return float(np.interp(p, phases, values))

    return fn


def _trapezoid_wave(phase: float, *, crest_ratio: float, root_ratio: float) -> float:
    p = _frac(phase)
    c = float(np.clip(crest_ratio, 0.0, 0.45))
    r = float(np.clip(root_ratio, 0.0, 0.45))
    flank = max((1.0 - c - r) * 0.5, 1e-9)

    left_crest = 0.5 - c * 0.5
    right_crest = 0.5 + c * 0.5
    root_left = right_crest + flank
    root_right = left_crest - flank

    if left_crest <= p <= right_crest:
        return 1.0

    if p > right_crest:
        if p <= root_left:
            return max(0.0, 1.0 - (p - right_crest) / flank)
        return 0.0

    if p >= root_right:
        return 0.0
    return max(0.0, 1.0 - (left_crest - p) / flank)


def _rounded_trapezoid_wave(phase: float, *, crest_ratio: float, root_ratio: float, roundness: float) -> float:
    raw = _trapezoid_wave(phase, crest_ratio=crest_ratio, root_ratio=root_ratio)
    blend = np.clip(roundness, 0.0, 1.0)
    smooth = raw * raw * (3.0 - 2.0 * raw)
    return float((1.0 - blend) * raw + blend * smooth)


def _buttress_wave(phase: float) -> float:
    p = _frac(phase)
    if p < 0.20:
        return 1.0
    if p < 0.30:
        return 1.0 - (p - 0.20) / 0.10
    if p < 0.80:
        return 0.0
    return (p - 0.80) / 0.20


def _resolve_thread_depth(spec: ThreadSpec) -> float:
    if spec.thread_depth is not None:
        return float(spec.thread_depth)

    pitch = float(spec.pitch)
    if spec.profile in {"iso", "pipe"}:
        return 0.61343 * pitch
    if spec.profile == "unified":
        return 0.64952 * pitch
    if spec.profile == "whitworth":
        return 0.64033 * pitch
    if spec.profile in {"acme", "trapezoidal"}:
        return 0.50 * pitch
    if spec.profile == "square":
        return 0.50 * pitch
    if spec.profile == "buttress":
        return 0.60 * pitch
    if spec.profile == "rounded":
        return 0.45 * pitch
    if spec.profile == "custom":
        return 0.55 * pitch
    raise InvalidThreadSpec(f"Unsupported profile '{spec.profile}'.")


def _resolve_theta_segments(spec: ThreadSpec, quality: MeshQuality) -> int:
    if quality.circumferential_segments is not None:
        return int(max(12, quality.circumferential_segments))

    if quality.max_chord_deviation is None or quality.max_chord_deviation <= 0:
        return 96

    max_radius = spec.major_diameter * 0.5 + abs(spec.taper_diameter_per_length) * spec.length * 0.5
    max_radius = max(max_radius, 1e-6)
    deviation = max(quality.max_chord_deviation, 1e-6)
    angle = 2.0 * np.arccos(np.clip(1.0 - deviation / max_radius, -1.0, 1.0))
    if angle <= 0:
        return 128
    seg = int(np.ceil(2.0 * np.pi / angle))
    return int(max(12, seg))


def _resolve_metric_designation(
    designation: str | None,
    major_diameter: float | None,
    pitch: float | None,
) -> tuple[float, float]:
    if designation:
        token = designation.strip().lower().replace(" ", "")
        if not token.startswith("m"):
            raise InvalidThreadSpec("Metric designation must look like 'M6x1'.")
        payload = token[1:]
        if "x" in payload:
            d_str, p_str = payload.split("x", 1)
            return float(d_str), float(p_str)
        major = float(payload)
        coarse = _METRIC_COARSE.get(int(round(major)))
        if coarse is None:
            raise InvalidThreadSpec(f"No coarse pitch table entry for metric designation '{designation}'.")
        return major, coarse

    if major_diameter is None:
        raise InvalidThreadSpec("Metric lookup requires designation or major_diameter.")
    if pitch is not None:
        return float(major_diameter), float(pitch)

    coarse = _METRIC_COARSE.get(int(round(float(major_diameter))))
    if coarse is None:
        raise InvalidThreadSpec("Metric lookup needs explicit pitch for this major diameter.")
    return float(major_diameter), coarse


def _resolve_unified_designation(
    designation: str | None,
    major_diameter: float | None,
    tpi: float | None,
) -> tuple[float, float]:
    if designation:
        token = designation.strip().lower().replace(" ", "")
        if "-" not in token:
            raise InvalidThreadSpec("Unified designation must look like '1/4-20'.")
        d_token, tpi_token = token.split("-", 1)
        major = _parse_fractional_inch(d_token) * 25.4
        return major, float(tpi_token)

    if major_diameter is None or tpi is None:
        raise InvalidThreadSpec("Unified lookup requires designation or both major_diameter and tpi.")
    return float(major_diameter), float(tpi)


def _parse_fractional_inch(token: str) -> float:
    token = token.strip().lower()
    if "/" in token:
        return float(Fraction(token))
    return float(token)


def _resolve_fit_preset(preset: str | ThreadFitPreset) -> ThreadFitPreset:
    if isinstance(preset, ThreadFitPreset):
        return preset
    key = preset.strip().lower()
    fit = FIT_PRESETS.get(key)
    if fit is None:
        raise InvalidFitSpec(f"Unknown fit preset '{preset}'. Known presets: {', '.join(sorted(FIT_PRESETS))}")
    return fit


def _build_axis_basis(axis_direction: Sequence[float]) -> np.ndarray:
    z_axis = np.asarray(axis_direction, dtype=float)
    z_norm = np.linalg.norm(z_axis)
    if z_norm == 0:
        raise InvalidThreadSpec("axis_direction must be non-zero.")
    z_axis /= z_norm

    helper = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _frac(value: float) -> float:
    return float(value - np.floor(value))


__all__ = [
    "FIT_PRESETS",
    "InvalidFitSpec",
    "InvalidThreadSpec",
    "MeshBudgetExceeded",
    "MeshQuality",
    "ThreadFitPreset",
    "ThreadMeshEstimate",
    "ThreadSpec",
    "ThreadingError",
    "apply_fit",
    "estimate_mesh_cost",
    "lookup_standard_thread",
    "make_external_thread",
    "make_hex_nut",
    "make_internal_thread",
    "make_round_nut",
    "make_tapped_hole_cutter",
    "make_threaded_rod",
    "paired_fit",
    "validate_thread",
]
