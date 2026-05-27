from __future__ import annotations

import pytest

from impression.modeling import (
    BSplineSurfacePatch,
    PlanarSurfacePatch,
    RevolutionSurfacePatch,
    SurfaceIntersectionRequest,
    SurfaceIntersectionSolverDispatchRecord,
    SurfaceIntersectionSolverRegistryRecord,
    SurfaceIntersectionSupportDiagnostic,
    SurfaceIntersectionTolerancePolicy,
    assert_surface_intersection_solver_registry_complete,
    build_surface_intersection_solver_registry,
    lookup_surface_intersection_solver,
    make_surface_intersection_request,
)


def test_surface_intersection_request_normalizes_pair_and_policy() -> None:
    policy = SurfaceIntersectionTolerancePolicy(position_tolerance=1e-8, max_iterations=8)

    request = make_surface_intersection_request(
        PlanarSurfacePatch(family="planar"),
        RevolutionSurfacePatch(family="revolution"),
        consumer="csg",
        tolerance_policy=policy,
    )

    assert isinstance(request, SurfaceIntersectionRequest)
    assert request.family_pair == ("planar", "revolution")
    assert request.normalized_family_pair == ("planar", "revolution")
    assert request.consumer == "csg"
    assert request.tolerance_policy is policy
    assert request.canonical_payload()["tolerance_policy"]["max_iterations"] == 8
    with pytest.raises(ValueError, match="positive finite"):
        SurfaceIntersectionTolerancePolicy(parameter_tolerance=0.0)


def test_surface_intersection_solver_registry_covers_promoted_family_pairs() -> None:
    registry = build_surface_intersection_solver_registry()

    assert ("planar", "planar") in registry
    assert ("bspline", "planar") in registry
    assert registry[("planar", "planar")].supported is True
    assert registry[("bspline", "planar")].supported is False
    assert assert_surface_intersection_solver_registry_complete(registry) is registry


def test_surface_intersection_solver_lookup_returns_supported_dispatch() -> None:
    request = make_surface_intersection_request(
        PlanarSurfacePatch(family="planar"),
        PlanarSurfacePatch(family="planar"),
        consumer="seam",
    )

    dispatch = lookup_surface_intersection_solver(request)

    assert isinstance(dispatch, SurfaceIntersectionSolverDispatchRecord)
    assert dispatch.supported is True
    assert dispatch.solver is not None
    assert dispatch.solver.solver_id == "planar-planar-analytic"
    assert dispatch.diagnostics == ()
    assert dispatch.canonical_payload()["supported"] is True


def test_surface_intersection_solver_lookup_reports_unsupported_pair() -> None:
    request = make_surface_intersection_request(
        PlanarSurfacePatch(family="planar"),
        BSplineSurfacePatch(family="bspline"),
        consumer="loft",
    )

    dispatch = lookup_surface_intersection_solver(request)

    assert dispatch.supported is False
    assert dispatch.solver is not None
    assert all(isinstance(diagnostic, SurfaceIntersectionSupportDiagnostic) for diagnostic in dispatch.diagnostics)
    assert dispatch.diagnostics[0].code == "unsupported-family-pair"
    assert dispatch.diagnostics[0].family_pair == ("bspline", "planar")
    assert "bspline/planar" in dispatch.diagnostics[0].message


def test_surface_intersection_registry_assertion_reports_missing_and_unknown_pairs() -> None:
    registry = build_surface_intersection_solver_registry()
    broken = dict(registry)
    broken.pop(("planar", "planar"))
    broken[("unknown", "planar")] = SurfaceIntersectionSolverRegistryRecord(
        first_family="unknown",
        second_family="planar",
        solver_id="unknown-planar",
        support_state="unsupported",
        operations=("csg",),
    )

    with pytest.raises(AssertionError, match="missing surface intersection registry entries"):
        assert_surface_intersection_solver_registry_complete(broken)
