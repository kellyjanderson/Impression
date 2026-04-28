from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bspline import BSpline3D
from .fit_records import FitAssessmentReport, FitConfigurationRecord, FitResidualReport
from .inference_descriptors import DenseLoftDescriptorBand


def _open_uniform_knots(control_point_count: int, degree: int) -> tuple[float, ...]:
    internal_count = control_point_count - degree - 1
    if internal_count <= 0:
        internal = ()
    else:
        internal = tuple(
            float(value)
            for value in np.linspace(0.0, 1.0, internal_count + 2, dtype=float)[1:-1]
        )
    return (
        *((0.0,) * (degree + 1)),
        *internal,
        *((1.0,) * (degree + 1)),
    )


def _polyline_residual(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    start = points[0]
    end = points[-1]
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return 0.0
    unit = direction / norm
    max_distance = 0.0
    for point in points:
        offset = point - start
        projection = float(np.dot(offset, unit))
        nearest = start + projection * unit
        max_distance = max(max_distance, float(np.linalg.norm(point - nearest)))
    return max_distance


@dataclass(frozen=True)
class StationDerivedCurveFitCandidate:
    candidate_id: str
    curve: BSpline3D
    residual_report: FitResidualReport
    fit_configuration_identity: tuple[object, ...] | None


def generate_station_derived_curve_fit_candidates(
    descriptor_band: DenseLoftDescriptorBand,
    *,
    fit_configuration: FitConfigurationRecord | None = None,
    acceptance_threshold: float = 0.25,
) -> tuple[StationDerivedCurveFitCandidate, ...]:
    points = np.asarray([descriptor.origin for descriptor in descriptor_band.descriptors], dtype=float)
    if points.shape[0] < 2:
        return ()

    exact_degree = 1
    exact_curve = BSpline3D(
        control_points=points,
        degree=exact_degree,
        knots=_open_uniform_knots(len(points), exact_degree),
        closure="open",
    )
    exact_candidate = StationDerivedCurveFitCandidate(
        candidate_id="station_polyline_exact",
        curve=exact_curve,
        residual_report=FitResidualReport(
            metric_name="max_distance_to_station_polyline",
            residual_value=0.0,
            acceptance_threshold=acceptance_threshold,
            approximation_posture="exact",
            exact_threshold=0.0,
        ),
        fit_configuration_identity=None if fit_configuration is None else fit_configuration.identity,
    )

    approx_points = np.vstack([points[0], points[-1]])
    approx_degree = 1
    approx_curve = BSpline3D(
        control_points=approx_points,
        degree=approx_degree,
        knots=_open_uniform_knots(len(approx_points), approx_degree),
        closure="open",
    )
    approx_candidate = StationDerivedCurveFitCandidate(
        candidate_id="station_endpoint_line",
        curve=approx_curve,
        residual_report=FitResidualReport(
            metric_name="max_distance_to_station_line",
            residual_value=_polyline_residual(points),
            acceptance_threshold=acceptance_threshold,
            approximation_posture="approximate",
            exact_threshold=0.0,
        ),
        fit_configuration_identity=None if fit_configuration is None else fit_configuration.identity,
    )
    return (exact_candidate, approx_candidate)


def compare_station_derived_curve_fit_candidates(
    candidates: tuple[StationDerivedCurveFitCandidate, ...],
) -> tuple[StationDerivedCurveFitCandidate | None, FitAssessmentReport]:
    if not candidates:
        refusal = FitResidualReport(
            metric_name="missing_station_candidates",
            residual_value=1.0,
            acceptance_threshold=0.0,
            approximation_posture="approximate",
        )
        return (
            None,
            FitAssessmentReport(
                residual_report=refusal,
                decision_outcome="refused",
                decision_reason="no_station_derived_candidates",
                fit_configuration_identity=None,
            ),
        )
    best = min(candidates, key=lambda candidate: candidate.residual_report.residual_value)
    assessment = FitAssessmentReport.from_residual(
        best.residual_report,
        fit_configuration_identity=best.fit_configuration_identity,
    )
    if assessment.decision_outcome == "refused":
        return None, assessment
    return best, assessment


__all__ = [
    "StationDerivedCurveFitCandidate",
    "compare_station_derived_curve_fit_candidates",
    "generate_station_derived_curve_fit_candidates",
]
