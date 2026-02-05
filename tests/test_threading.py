from __future__ import annotations

import pytest

from impression.mesh import analyze_mesh
from impression.modeling import (
    MeshQuality,
    ThreadSpec,
    apply_fit,
    estimate_mesh_cost,
    lookup_standard_thread,
    make_external_thread,
    make_internal_thread,
    make_tapped_hole_cutter,
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
    with pytest.raises(Exception):
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
