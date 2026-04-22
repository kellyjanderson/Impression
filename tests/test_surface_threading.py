from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np
import pytest

from impression.modeling import (
    MeshQuality,
    ThreadSpec,
    ThreadSurfaceAssembly,
    ThreadSurfaceRepresentation,
    apply_fit,
    lookup_standard_thread,
    make_external_thread,
    make_hex_nut,
    make_internal_thread,
    make_round_nut,
    make_runout_relief,
    make_tapped_hole_cutter,
    make_threaded_rod,
    prepare_surface_thread_representation,
)


def test_surface_external_thread_returns_structured_representation_without_deprecation() -> None:
    spec = lookup_standard_thread("metric", "M6x1", length=12.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        representation = make_external_thread(spec, backend="surface")

    assert isinstance(representation, ThreadSurfaceRepresentation)
    assert representation.kind == "external"
    assert representation.profile == "iso"
    assert representation.major_diameter == pytest.approx(6.0)
    assert representation.minor_diameter < representation.major_diameter
    assert len(representation.profile_samples) == 17
    assert representation.profile_samples[0][0] == pytest.approx(0.0)
    assert representation.profile_samples[-1][0] == pytest.approx(1.0)
    basis = np.asarray(representation.axis_basis, dtype=float)
    assert basis.shape == (3, 3)
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-9)
    assert not [item for item in caught if issubclass(item.category, DeprecationWarning)]


def test_surface_internal_thread_representation_tracks_pitch_schedule_turns_and_handedness() -> None:
    spec = ThreadSpec(
        major_diameter=8.0,
        pitch=1.5,
        length=10.0,
        kind="internal",
        hand="left",
        starts=3,
        taper_diameter_per_length=1.0 / 16.0,
        pitch_profile=((0.0, 1.0), (5.0, 1.5), (10.0, 2.0)),
    )

    representation = make_internal_thread(spec, backend="surface")

    assert isinstance(representation, ThreadSurfaceRepresentation)
    assert representation.kind == "internal"
    assert representation.hand == "left"
    assert representation.starts == 3
    assert representation.taper_diameter_per_length == pytest.approx(1.0 / 16.0)
    assert representation.pitch_schedule == ((0.0, 1.0), (5.0, 1.5), (10.0, 2.0))
    assert representation.turn_count > 0.0
    assert representation.stable_identity == prepare_surface_thread_representation(spec).stable_identity


def test_surface_thread_representation_supports_custom_profile_and_axis_normalization() -> None:
    spec = ThreadSpec(
        major_diameter=8.0,
        pitch=2.0,
        length=8.0,
        profile="custom",
        custom_profile_points=((0.0, 0.1), (0.3, 1.0), (0.7, 0.0), (1.0, 0.1)),
        axis_direction=(0.0, 0.0, 5.0),
    )

    representation = prepare_surface_thread_representation(spec, kind="external", profile_sample_count=9)

    assert representation.axis_direction == pytest.approx((0.0, 0.0, 1.0))
    assert len(representation.profile_samples) == 9
    assert all(0.0 <= value <= 1.0 for _phase, value in representation.profile_samples)


def test_surface_thread_representation_rejects_invalid_surface_requests() -> None:
    invalid = ThreadSpec(major_diameter=6.0, pitch=1.0, length=6.0, starts=0)
    with pytest.raises(Exception):
        make_external_thread(invalid, backend="surface")

    spec = lookup_standard_thread("metric", "M6x1", length=6.0)
    with pytest.raises(Exception, match="at least 5"):
        prepare_surface_thread_representation(spec, profile_sample_count=3)


def test_surface_thread_convenience_builders_return_structured_assemblies() -> None:
    spec = lookup_standard_thread("metric", "M6x1", length=8.0)

    rod = make_threaded_rod(spec, backend="surface")
    cutter = make_tapped_hole_cutter(spec, overshoot=0.5, backend="surface")
    relief = make_runout_relief(replace(spec, runout_length=1.0, runout_depth=0.25), backend="surface")
    hex_nut = make_hex_nut(spec, thickness=5.0, across_flats=10.0, backend="surface")
    round_nut = make_round_nut(spec, thickness=5.0, outer_diameter=12.0, backend="surface")

    assert isinstance(rod, ThreadSurfaceAssembly)
    assert rod.assembly_type == "threaded_rod"
    assert rod.operation == "standalone"
    assert rod.operands[0].kind == "thread"

    assert isinstance(cutter, ThreadSurfaceAssembly)
    assert cutter.assembly_type == "tapped_hole_cutter"
    assert cutter.operands[0].payload["kind"] == "internal"

    assert isinstance(relief, ThreadSurfaceAssembly)
    assert relief.assembly_type == "runout_relief"
    assert relief.operation == "union"
    assert len(relief.operands) == 2

    assert isinstance(hex_nut, ThreadSurfaceAssembly)
    assert hex_nut.assembly_type == "hex_nut"
    assert hex_nut.operation == "difference"
    assert [operand.kind for operand in hex_nut.operands] == ["primitive", "thread"]

    assert isinstance(round_nut, ThreadSurfaceAssembly)
    assert round_nut.assembly_type == "round_nut"
    assert round_nut.operation == "difference"
    assert [operand.kind for operand in round_nut.operands] == ["primitive", "thread"]

    assert rod.stable_identity == make_threaded_rod(spec, backend="surface").stable_identity


def test_surface_thread_fit_changes_canonical_geometry_explicitly() -> None:
    base = lookup_standard_thread("metric", "M6x1", length=10.0)
    fitted = apply_fit(base, "fdm_default", kind="external")

    base_rep = make_external_thread(base, backend="surface")
    fitted_rep = make_external_thread(fitted, backend="surface")

    assert isinstance(base_rep, ThreadSurfaceRepresentation)
    assert isinstance(fitted_rep, ThreadSurfaceRepresentation)
    assert fitted_rep.major_diameter < base_rep.major_diameter
    assert fitted_rep.minor_diameter < base_rep.minor_diameter
    assert fitted_rep.stable_identity != base_rep.stable_identity


def test_surface_thread_representation_is_not_changed_by_mesh_quality_knobs() -> None:
    spec = lookup_standard_thread("metric", "M8x1.25", length=12.0)

    coarse = make_external_thread(
        spec,
        backend="surface",
        quality=MeshQuality(segments_per_turn=12, circumferential_segments=24),
    )
    fine = make_external_thread(
        spec,
        backend="surface",
        quality=MeshQuality(segments_per_turn=48, circumferential_segments=128),
    )

    assert isinstance(coarse, ThreadSurfaceRepresentation)
    assert isinstance(fine, ThreadSurfaceRepresentation)
    assert coarse.canonical_payload() == fine.canonical_payload()
    assert coarse.stable_identity == fine.stable_identity
