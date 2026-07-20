from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .bspline import BSpline3D
from .fit_records import FitConfigurationRecord, FitResidualReport
from .inference_candidates import (
    SharedTrajectoryCurveFitCandidate,
    generate_shared_trajectory_curve_fit_candidates,
)
from .inference_descriptors import DenseLoftDescriptorBand

SharedWholeLoftTrajectoryPosture = Literal["accepted", "uncertain", "refused"]


@dataclass(frozen=True)
class SharedWholeLoftTrajectoryCandidate:
    candidate_id: str
    trajectory_curve: BSpline3D
    source_fit_candidate_id: str
    source_residual_report: FitResidualReport
    fit_configuration_identity: tuple[object, ...] | None
    evidence_lane: str = "shared_trajectory_curve_fit"


@dataclass(frozen=True)
class SharedWholeLoftTrajectoryAssessment:
    candidate: SharedWholeLoftTrajectoryCandidate | None
    confidence: float
    posture: SharedWholeLoftTrajectoryPosture
    reason: str
    evidence_lane: str = "shared_trajectory_curve_fit"


def generate_shared_whole_loft_trajectory_candidates(
    descriptor_band: DenseLoftDescriptorBand,
    *,
    fit_configuration: FitConfigurationRecord | None = None,
) -> tuple[SharedWholeLoftTrajectoryCandidate, ...]:
    fit_candidates = generate_shared_trajectory_curve_fit_candidates(
        descriptor_band,
        fit_configuration=fit_configuration,
    )
    return tuple(
        SharedWholeLoftTrajectoryCandidate(
            candidate_id=f"whole_loft_{candidate.candidate_id}",
            trajectory_curve=candidate.curve,
            source_fit_candidate_id=candidate.candidate_id,
            source_residual_report=candidate.residual_report,
            fit_configuration_identity=candidate.fit_configuration_identity,
        )
        for candidate in fit_candidates
    )


def assess_shared_whole_loft_trajectory_candidates(
    candidates: tuple[SharedWholeLoftTrajectoryCandidate, ...],
) -> SharedWholeLoftTrajectoryAssessment:
    if not candidates:
        return SharedWholeLoftTrajectoryAssessment(
            candidate=None,
            confidence=0.0,
            posture="refused",
            reason="missing_shared_whole_loft_trajectory_candidates",
        )

    best = min(candidates, key=lambda candidate: candidate.source_residual_report.residual_value)
    residual_report = best.source_residual_report
    if not residual_report.within_acceptance_threshold:
        return SharedWholeLoftTrajectoryAssessment(
            candidate=None,
            confidence=0.0,
            posture="refused",
            reason="shared_whole_loft_trajectory_residual_above_threshold",
        )
    if residual_report.approximation_posture == "exact":
        return SharedWholeLoftTrajectoryAssessment(
            candidate=best,
            confidence=1.0,
            posture="accepted",
            reason="shared_whole_loft_trajectory_exact_within_threshold",
        )
    return SharedWholeLoftTrajectoryAssessment(
        candidate=best,
        confidence=0.5,
        posture="uncertain",
        reason="shared_whole_loft_trajectory_approximate_within_threshold",
    )


__all__ = [
    "SharedWholeLoftTrajectoryAssessment",
    "SharedWholeLoftTrajectoryCandidate",
    "SharedWholeLoftTrajectoryPosture",
    "assess_shared_whole_loft_trajectory_candidates",
    "generate_shared_whole_loft_trajectory_candidates",
]
