from __future__ import annotations

from dataclasses import dataclass

from .bspline import BSpline3D
from .fit_records import FitConfigurationRecord, FitResidualReport
from .inference_candidates import (
    SharedTrajectoryCurveFitCandidate,
    generate_shared_trajectory_curve_fit_candidates,
)
from .inference_descriptors import DenseLoftDescriptorBand


@dataclass(frozen=True)
class SharedWholeLoftTrajectoryCandidate:
    candidate_id: str
    trajectory_curve: BSpline3D
    source_fit_candidate_id: str
    source_residual_report: FitResidualReport
    fit_configuration_identity: tuple[object, ...] | None
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


__all__ = [
    "SharedWholeLoftTrajectoryCandidate",
    "generate_shared_whole_loft_trajectory_candidates",
]
