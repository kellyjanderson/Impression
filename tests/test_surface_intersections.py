from __future__ import annotations

import pytest

from impression.modeling import (
    BSplineSurfacePatch,
    NURBSSurfacePatch,
    PlanarSurfacePatch,
    RevolutionSurfacePatch,
    RuledSurfacePatch,
    SurfaceAnalyticSplineResidualReport,
    SurfaceAnalyticSplineSolverIterationRecord,
    SurfaceIntersectionCurveRecord,
    SurfaceIntersectionDegeneracyRecord,
    SurfaceIntersectionOverlapRegionRecord,
    SurfaceIntersectionRequest,
    SurfaceIntersectionResultRecord,
    SurfaceIntersectionSolverDispatchRecord,
    SurfaceIntersectionSolverRegistryRecord,
    SurfaceIntersectionSupportDiagnostic,
    SurfaceIntersectionTolerancePolicy,
    assert_surface_intersection_solver_registry_complete,
    build_surface_intersection_solver_registry,
    classify_surface_intersection_degeneracy,
    lookup_surface_intersection_solver,
    make_surface_intersection_request,
    normalize_surface_intersection_result,
    solve_analytic_spline_surface_intersection,
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
    assert registry[("bspline", "planar")].support_state == "declared-tolerance"
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


def test_surface_intersection_solver_lookup_reports_declared_tolerance_pair() -> None:
    request = make_surface_intersection_request(
        PlanarSurfacePatch(family="planar"),
        BSplineSurfacePatch(family="bspline"),
        consumer="loft",
    )

    dispatch = lookup_surface_intersection_solver(request)

    assert dispatch.supported is True
    assert dispatch.solver is not None
    assert dispatch.solver.support_state == "declared-tolerance"
    assert dispatch.diagnostics == ()


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


def test_surface_intersection_result_normalizes_curves_points_and_quality() -> None:
    request = make_surface_intersection_request(
        PlanarSurfacePatch(family="planar"),
        PlanarSurfacePatch(family="planar"),
    )
    curve = SurfaceIntersectionCurveRecord(
        curve_id="curve-b",
        kind="line",
        points_3d=((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        first_parameters=((1.0, 0.0), (0.0, 0.0)),
        second_parameters=((1.0, 0.0), (0.0, 0.0)),
    )

    result = normalize_surface_intersection_result(
        request,
        curves=(curve,),
        points=((0.25, 0.0, 0.0),),
        max_residual=1e-10,
        quality="within-tolerance",
    )

    assert isinstance(result, SurfaceIntersectionResultRecord)
    assert result.classification == "curves"
    assert result.quality == "within-tolerance"
    assert result.curves[0].length_estimate == pytest.approx(1.0)
    assert result.points == ((0.25, 0.0, 0.0),)
    assert result.degeneracies == ()
    assert result.canonical_payload()["supported"] is True


def test_surface_intersection_result_records_overlap_and_point_contact_degeneracy() -> None:
    request = make_surface_intersection_request(
        PlanarSurfacePatch(family="planar"),
        PlanarSurfacePatch(family="planar"),
    )
    overlap = SurfaceIntersectionOverlapRegionRecord(
        region_id="overlap-a",
        first_loop_uv=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        second_loop_uv=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),
        boundary_curve_ids=("b", "a"),
    )
    point_contact = normalize_surface_intersection_result(request, points=((0.0, 0.0, 0.0),))

    overlap_result = normalize_surface_intersection_result(request, overlap_regions=(overlap,))

    assert overlap_result.classification == "overlap"
    assert overlap_result.quality == "degenerate"
    assert overlap_result.overlap_regions[0].boundary_curve_ids == ("a", "b")
    assert {record.code for record in overlap_result.degeneracies} == {"overlap"}
    assert point_contact.classification == "degenerate"
    assert {record.code for record in point_contact.degeneracies} == {"point-contact"}


def test_surface_intersection_degeneracy_classifier_reports_short_curve_and_high_residual() -> None:
    request = make_surface_intersection_request(
        PlanarSurfacePatch(family="planar"),
        PlanarSurfacePatch(family="planar"),
    )
    curve = SurfaceIntersectionCurveRecord(
        curve_id="short",
        kind="sampled",
        points_3d=((0.0, 0.0, 0.0), (1e-12, 0.0, 0.0)),
    )
    result = normalize_surface_intersection_result(request, curves=(curve,), max_residual=1e-3)

    assert result.classification == "curves"
    assert result.quality == "degenerate"
    assert all(isinstance(record, SurfaceIntersectionDegeneracyRecord) for record in result.degeneracies)
    assert {record.code for record in result.degeneracies} == {"short-curve", "high-residual"}
    assert classify_surface_intersection_degeneracy(result) == result.degeneracies


def test_analytic_spline_solver_intersects_planar_bspline_with_declared_tolerance() -> None:
    plane = PlanarSurfacePatch(
        family="planar",
        origin=(0.5, 0.0, 0.0),
        u_axis=(0.0, 1.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
    )
    spline = BSplineSurfacePatch(family="bspline")
    request = make_surface_intersection_request(plane, spline, consumer="csg")

    result, report = solve_analytic_spline_surface_intersection(request, sample_count=5)

    assert result.supported is True
    assert result.classification == "curves"
    assert result.quality == "within-tolerance"
    assert result.curves[0].kind == "sampled"
    assert result.curves[0].second_parameters
    assert isinstance(report, SurfaceAnalyticSplineResidualReport)
    assert report.converged is True
    assert all(isinstance(iteration, SurfaceAnalyticSplineSolverIterationRecord) for iteration in report.iterations)
    assert report.iterations[0].accepted_point_count >= 2


def test_analytic_spline_solver_intersects_ruled_nurbs_with_declared_tolerance() -> None:
    ruled = RuledSurfacePatch(family="ruled")
    nurbs = NURBSSurfacePatch(family="nurbs")
    request = make_surface_intersection_request(ruled, nurbs, consumer="seam")

    result, report = solve_analytic_spline_surface_intersection(request, sample_count=5)

    assert result.supported is True
    assert result.classification == "curves"
    assert report.converged is True
    assert result.max_residual <= request.tolerance_policy.position_tolerance * 10.0


def test_analytic_spline_solver_intersects_revolution_bspline_with_declared_tolerance() -> None:
    revolution = RevolutionSurfacePatch(family="revolution")
    spline = BSplineSurfacePatch(
        family="bspline",
        control_net=[
            [(1.0, 0.0, 0.0), (1.0, 0.0, 1.0)],
            [(1.0, 0.0, 0.0), (1.0, 0.0, 1.0)],
        ],
    )
    request = make_surface_intersection_request(revolution, spline, consumer="loft")

    result, report = solve_analytic_spline_surface_intersection(request, sample_count=5)

    assert result.supported is True
    assert result.curves
    assert report.converged is True


def test_analytic_spline_solver_refuses_non_analytic_spline_pairs() -> None:
    request = make_surface_intersection_request(
        BSplineSurfacePatch(family="bspline"),
        BSplineSurfacePatch(family="bspline"),
    )

    result, report = solve_analytic_spline_surface_intersection(request)

    assert result.supported is False
    assert result.classification == "unsupported"
    assert report.converged is False
    assert all(isinstance(diagnostic, SurfaceIntersectionSupportDiagnostic) for diagnostic in report.diagnostics)
    assert report.diagnostics[0].code == "unsupported-family-pair"
