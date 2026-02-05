from __future__ import annotations

import pytest

from impression.mesh import analyze_mesh
from impression.modeling import (
    FIT_PRESETS,
    MeshBudgetExceeded,
    MeshQuality,
    ThreadSpec,
    ThreadingError,
    apply_fit,
    estimate_mesh_cost,
    lookup_standard_thread,
    make_external_thread,
    make_hex_nut,
    make_internal_thread,
    make_round_nut,
    make_tapped_hole_cutter,
    make_threaded_rod,
    paired_fit,
    validate_thread,
)


def test_lookup_metric_m6x1() -> None:
    spec = lookup_standard_thread("metric", "M6x1", length=12.0)
    assert spec.major_diameter == pytest.approx(6.0)
    assert spec.pitch == pytest.approx(1.0)
    assert spec.profile == "iso"


def test_lookup_unified_fractional() -> None:
    spec = lookup_standard_thread("unc", "1/4-20", length=12.0)
    assert spec.pitch == pytest.approx(25.4 / 20.0)
    assert spec.major_diameter == pytest.approx(6.35)


def test_validate_rejects_collapsed_minor() -> None:
    spec = ThreadSpec(major_diameter=2.0, pitch=1.0, length=10.0, thread_depth=2.0)
    with pytest.raises(ThreadingError):
        validate_thread(spec)


def test_fit_pair_expands_clearance() -> None:
    base = lookup_standard_thread("metric", "M6x1", length=10.0)
    male, female = paired_fit(base, "fdm_default")
    assert female.major_diameter > male.major_diameter


def test_external_thread_mesh_generation_and_analysis() -> None:
    spec = lookup_standard_thread("metric", "M6x1", length=8.0)
    mesh = make_external_thread(spec, quality=MeshQuality(segments_per_turn=24, circumferential_segments=64))
    analysis = analyze_mesh(mesh)
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0
    assert analysis.nonmanifold_edges == 0


def test_internal_thread_mesh_generation() -> None:
    spec = lookup_standard_thread("metric", "M6x1", length=8.0, kind="internal")
    mesh = make_internal_thread(spec, quality=MeshQuality(segments_per_turn=24, circumferential_segments=64))
    assert mesh.n_vertices > 0
    assert mesh.n_faces > 0


def test_tapped_hole_cutter_extends_length() -> None:
    spec = lookup_standard_thread("metric", "M6x1", length=8.0, kind="internal")
    cutter = make_tapped_hole_cutter(spec, quality=MeshQuality(segments_per_turn=18, circumferential_segments=48), overshoot=0.4)
    zmin = cutter.bounds[4]
    zmax = cutter.bounds[5]
    assert (zmax - zmin) > spec.length


def test_mesh_estimate_close_to_actual() -> None:
    spec = lookup_standard_thread("metric", "M8x1.25", length=10.0)
    quality = MeshQuality(segments_per_turn=20, circumferential_segments=56)
    est = estimate_mesh_cost(spec, quality)
    mesh = make_external_thread(spec, quality=quality)
    assert abs(mesh.n_faces - est.predicted_faces) / est.predicted_faces < 0.20


def test_apply_fit_named_preset() -> None:
    spec = lookup_standard_thread("metric", "M6x1", length=10.0)
    fitted = apply_fit(spec, "sla_tight", kind="external")
    assert fitted.major_diameter < spec.major_diameter


def test_mesh_budget_exceeded_raises() -> None:
    spec = lookup_standard_thread("metric", "M10x1.5", length=20.0)
    with pytest.raises(MeshBudgetExceeded):
        make_external_thread(spec, quality=MeshQuality(segments_per_turn=64, circumferential_segments=256, max_triangles=1000))


def test_mesh_budget_adaptive_reduces_quality() -> None:
    spec = lookup_standard_thread("metric", "M10x1.5", length=20.0)
    mesh = make_external_thread(
        spec,
        quality=MeshQuality(
            segments_per_turn=64,
            circumferential_segments=256,
            max_triangles=5000,
            adaptive_budget=True,
        ),
    )
    assert mesh.n_faces > 0


def test_multistart_thread_generates_mesh() -> None:
    spec = ThreadSpec(
        major_diameter=8.0,
        pitch=2.0,
        starts=4,
        length=16.0,
        profile="trapezoidal",
    )
    mesh = make_external_thread(spec, quality=MeshQuality(segments_per_turn=24, circumferential_segments=96))
    assert mesh.n_faces > 0
    assert mesh.n_vertices > 0


def test_pipe_lookup_sets_taper() -> None:
    spec = lookup_standard_thread("pipe", major_diameter=21.3, tpi=14, length=12.0)
    assert spec.taper_diameter_per_length > 0


def test_threaded_rod_convenience() -> None:
    spec = lookup_standard_thread("metric", "M6x1", length=10.0)
    mesh = make_threaded_rod(spec, quality=MeshQuality(segments_per_turn=20, circumferential_segments=48))
    assert mesh.n_vertices > 0


def test_nut_convenience_generators() -> None:
    spec = lookup_standard_thread("metric", "M6x1", length=8.0)
    hex_nut = make_hex_nut(spec, thickness=5.0, across_flats=10.0, quality=MeshQuality(segments_per_turn=18))
    round_nut = make_round_nut(spec, thickness=5.0, outer_diameter=12.0, quality=MeshQuality(segments_per_turn=18))
    assert hex_nut.n_faces > 0
    assert round_nut.n_faces > 0


def test_treatments_reduce_thread_amplitude_near_start() -> None:
    base = ThreadSpec(major_diameter=6.0, pitch=1.0, length=8.0, starts=1)
    treated = ThreadSpec(
        major_diameter=6.0,
        pitch=1.0,
        length=8.0,
        starts=1,
        start_treatment="higbee",
        start_treatment_length=2.0,
    )
    m0 = make_external_thread(base, quality=MeshQuality(segments_per_turn=20, circumferential_segments=64))
    m1 = make_external_thread(treated, quality=MeshQuality(segments_per_turn=20, circumferential_segments=64))
    assert m1.n_faces == m0.n_faces
    # Treated starts should keep early radii closer to the core than the untreated case.
    start_ring0 = m0.vertices[:64]
    start_ring1 = m1.vertices[:64]
    r0 = (start_ring0[:, 0] ** 2 + start_ring0[:, 1] ** 2) ** 0.5
    r1 = (start_ring1[:, 0] ** 2 + start_ring1[:, 1] ** 2) ** 0.5
    assert r1.max() < r0.max()


def test_fit_presets_registered() -> None:
    assert "fdm_default" in FIT_PRESETS
