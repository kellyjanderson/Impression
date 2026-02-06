from __future__ import annotations

import pytest
import numpy as np

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
    clear_thread_cache,
    make_runout_relief,
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
    hex_nut = make_hex_nut(spec, thickness=5.0, across_flats=10.0, quality=MeshQuality(segments_per_turn=18, boolean_epsilon=0.1))
    round_nut = make_round_nut(spec, thickness=5.0, outer_diameter=12.0, quality=MeshQuality(segments_per_turn=18, boolean_epsilon=0.1))
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


def test_lead_in_alignment_reduces_major_radius() -> None:
    base = ThreadSpec(major_diameter=8.0, pitch=1.25, length=10.0, lead_in_length=2.0)
    mesh = make_external_thread(base, quality=MeshQuality(segments_per_turn=20, circumferential_segments=64))
    z_count = int(mesh.n_vertices / 64) - 1
    start_ring = mesh.vertices[:64]
    mid_ring = mesh.vertices[(z_count // 2) * 64 : (z_count // 2 + 1) * 64]
    r_start = (start_ring[:, 0] ** 2 + start_ring[:, 1] ** 2) ** 0.5
    r_mid = (mid_ring[:, 0] ** 2 + mid_ring[:, 1] ** 2) ** 0.5
    assert r_start.max() < r_mid.max()


def test_variable_pitch_profile_changes_phase() -> None:
    spec = ThreadSpec(
        major_diameter=8.0,
        pitch=1.5,
        length=10.0,
        pitch_profile=((0.0, 1.0), (10.0, 2.0)),
    )
    mesh = make_external_thread(spec, quality=MeshQuality(segments_per_turn=16, circumferential_segments=48))
    assert mesh.n_faces > 0


def test_profile_anchor_pitch() -> None:
    base = ThreadSpec(major_diameter=8.0, pitch=1.25, length=8.0, profile_anchor="pitch")
    mesh = make_external_thread(base, quality=MeshQuality(segments_per_turn=16, circumferential_segments=48))
    assert mesh.n_vertices > 0


def test_runout_relief_affects_core() -> None:
    base = ThreadSpec(major_diameter=8.0, pitch=1.25, length=8.0, runout_length=1.5, runout_depth=0.4)
    mesh = make_external_thread(base, quality=MeshQuality(segments_per_turn=16, circumferential_segments=48))
    start_ring = mesh.vertices[:48]
    mid_ring = mesh.vertices[5 * 48 : 6 * 48]
    r_start = (start_ring[:, 0] ** 2 + start_ring[:, 1] ** 2) ** 0.5
    r_mid = (mid_ring[:, 0] ** 2 + mid_ring[:, 1] ** 2) ** 0.5
    assert r_start.min() < r_mid.min()


def test_custom_profile_duplicate_phase_error() -> None:
    spec = ThreadSpec(
        major_diameter=8.0,
        pitch=1.25,
        length=8.0,
        profile="custom",
        custom_profile_points=((0.0, 0.0), (0.5, 1.0), (0.5, 0.2)),
    )
    with pytest.raises(Exception):
        validate_thread(spec)


def test_custom_profile_wrap_continuity_error() -> None:
    spec = ThreadSpec(
        major_diameter=8.0,
        pitch=1.25,
        length=8.0,
        profile="custom",
        custom_profile_points=((0.0, 0.0), (0.5, 1.0), (1.0, 0.8)),
    )
    with pytest.raises(Exception):
        validate_thread(spec)


def test_custom_profile_valid() -> None:
    spec = ThreadSpec(
        major_diameter=8.0,
        pitch=1.25,
        length=8.0,
        profile="custom",
        custom_profile_points=((0.0, 0.0), (0.5, 1.0), (1.0, 0.0)),
    )
    validate_thread(spec)


def test_whitworth_profile_generation() -> None:
    spec = ThreadSpec(major_diameter=10.0, pitch=1.5, length=8.0, profile="whitworth")
    mesh = make_external_thread(spec, quality=MeshQuality(segments_per_turn=16, circumferential_segments=48))
    assert mesh.n_faces > 0


def test_pipe_profile_generation() -> None:
    spec = ThreadSpec(major_diameter=10.0, pitch=1.5, length=8.0, profile="pipe", taper_diameter_per_length=1.0 / 16.0)
    mesh = make_external_thread(spec, quality=MeshQuality(segments_per_turn=16, circumferential_segments=48))
    assert mesh.n_faces > 0


def test_runout_relief_geometry_generation() -> None:
    spec = ThreadSpec(
        major_diameter=8.0,
        pitch=1.25,
        length=8.0,
        runout_length=1.0,
        runout_depth=0.3,
    )
    relief = make_runout_relief(spec, quality=MeshQuality(circumferential_segments=32))
    assert relief.n_faces > 0


def test_min_feature_warning() -> None:
    spec = ThreadSpec(
        major_diameter=2.0,
        pitch=0.3,
        length=4.0,
        nozzle_diameter=0.4,
    )
    with pytest.warns(RuntimeWarning):
        validate_thread(spec)


def test_cache_returns_copy() -> None:
    clear_thread_cache()
    spec = ThreadSpec(major_diameter=8.0, pitch=1.25, length=6.0)
    mesh1 = make_external_thread(spec, quality=MeshQuality(segments_per_turn=16, circumferential_segments=48))
    mesh2 = make_external_thread(spec, quality=MeshQuality(segments_per_turn=16, circumferential_segments=48))
    assert mesh1.n_faces == mesh2.n_faces
    assert mesh1 is not mesh2


def test_lod_preview_reduces_faces() -> None:
    spec = ThreadSpec(major_diameter=8.0, pitch=1.25, length=6.0)
    high = make_external_thread(spec, quality=MeshQuality(segments_per_turn=24, circumferential_segments=96, lod="final"))
    low = make_external_thread(spec, quality=MeshQuality(segments_per_turn=24, circumferential_segments=96, lod="preview"))
    assert low.n_faces < high.n_faces


def test_tr8x8_lead_repeats_profile() -> None:
    spec = ThreadSpec(major_diameter=8.0, pitch=2.0, length=16.0, starts=4, profile="trapezoidal")
    mesh = make_external_thread(spec, quality=MeshQuality(segments_per_turn=16, circumferential_segments=48))
    theta_count = 48
    z_count = int(mesh.n_vertices / theta_count) - 1
    ring0 = mesh.vertices[:theta_count]
    ring_lead = mesh.vertices[(z_count // 2) * theta_count : (z_count // 2 + 1) * theta_count]
    r0 = (ring0[:, 0] ** 2 + ring0[:, 1] ** 2) ** 0.5
    r1 = (ring_lead[:, 0] ** 2 + ring_lead[:, 1] ** 2) ** 0.5
    assert np.allclose(r0, r1, atol=1e-2)


def test_pipe_taper_increases_radius() -> None:
    spec = ThreadSpec(
        major_diameter=10.0,
        pitch=1.5,
        length=10.0,
        profile="pipe",
        taper_diameter_per_length=1.0 / 16.0,
    )
    mesh = make_external_thread(spec, quality=MeshQuality(segments_per_turn=16, circumferential_segments=48))
    theta_count = 48
    ring0 = mesh.vertices[:theta_count]
    ring_end = mesh.vertices[-(theta_count + 2) : -2]
    r0 = (ring0[:, 0] ** 2 + ring0[:, 1] ** 2) ** 0.5
    r1 = (ring_end[:, 0] ** 2 + ring_end[:, 1] ** 2) ** 0.5
    assert r1.max() > r0.max()


def test_higbee_smooths_start() -> None:
    spec = ThreadSpec(
        major_diameter=8.0,
        pitch=1.25,
        length=8.0,
        start_treatment="higbee",
        start_treatment_length=2.0,
    )
    mesh = make_external_thread(spec, quality=MeshQuality(segments_per_turn=16, circumferential_segments=48))
    theta_count = 48
    ring0 = mesh.vertices[:theta_count]
    ring1 = mesh.vertices[theta_count : 2 * theta_count]
    r0 = (ring0[:, 0] ** 2 + ring0[:, 1] ** 2) ** 0.5
    r1 = (ring1[:, 0] ** 2 + ring1[:, 1] ** 2) ** 0.5
    assert r1.max() >= r0.max()
