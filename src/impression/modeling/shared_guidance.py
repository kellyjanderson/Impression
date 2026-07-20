from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .progression import PathBackedProgression
from .shared_trajectory import SharedWholeLoftTrajectoryCandidate

ExplicitSharedGuidancePlannerStage = Literal["in_between_travel"]


def _normalize_metadata_entries(
    metadata_entries: tuple[tuple[str, str], ...] | list[tuple[str, str]] | None,
) -> tuple[tuple[str, str], ...]:
    if metadata_entries is None:
        return ()
    normalized: list[tuple[str, str]] = []
    for key, value in metadata_entries:
        normalized_key = str(key).strip()
        normalized_value = str(value).strip()
        if not normalized_key:
            raise ValueError("metadata keys must be non-empty.")
        if not normalized_value:
            raise ValueError("metadata values must be non-empty.")
        normalized.append((normalized_key, normalized_value))
    return tuple(normalized)


def _normalize_planner_stage(value: str) -> ExplicitSharedGuidancePlannerStage:
    allowed: set[str] = {"in_between_travel"}
    stage = str(value)
    if stage not in allowed:
        raise ValueError("planner_stage must be one of: in_between_travel.")
    return stage  # type: ignore[return-value]


@dataclass(frozen=True)
class ExplicitSharedGuidanceAttachmentRecord:
    guidance_id: str
    progression_identity: tuple[object, ...]
    trajectory_candidate_id: str
    trajectory_evidence_lane: str
    fit_configuration_identity: tuple[object, ...] | None
    metadata_entries: tuple[tuple[str, str], ...] = ()

    def __init__(
        self,
        *,
        guidance_id: str,
        progression_identity: tuple[object, ...],
        trajectory_candidate_id: str,
        trajectory_evidence_lane: str,
        fit_configuration_identity: tuple[object, ...] | None,
        metadata_entries: tuple[tuple[str, str], ...] | list[tuple[str, str]] | None = None,
    ) -> None:
        normalized_guidance_id = str(guidance_id).strip()
        normalized_candidate_id = str(trajectory_candidate_id).strip()
        normalized_lane = str(trajectory_evidence_lane).strip()
        if not normalized_guidance_id:
            raise ValueError("guidance_id must be non-empty.")
        if not normalized_candidate_id:
            raise ValueError("trajectory_candidate_id must be non-empty.")
        if not normalized_lane:
            raise ValueError("trajectory_evidence_lane must be non-empty.")
        object.__setattr__(self, "guidance_id", normalized_guidance_id)
        object.__setattr__(self, "progression_identity", progression_identity)
        object.__setattr__(self, "trajectory_candidate_id", normalized_candidate_id)
        object.__setattr__(self, "trajectory_evidence_lane", normalized_lane)
        object.__setattr__(self, "fit_configuration_identity", fit_configuration_identity)
        object.__setattr__(self, "metadata_entries", _normalize_metadata_entries(metadata_entries))

    @classmethod
    def from_candidate(
        cls,
        *,
        guidance_id: str,
        progression: PathBackedProgression,
        candidate: SharedWholeLoftTrajectoryCandidate,
        metadata_entries: tuple[tuple[str, str], ...] | list[tuple[str, str]] | None = None,
    ) -> "ExplicitSharedGuidanceAttachmentRecord":
        return cls(
            guidance_id=guidance_id,
            progression_identity=progression.identity,
            trajectory_candidate_id=candidate.candidate_id,
            trajectory_evidence_lane=candidate.evidence_lane,
            fit_configuration_identity=candidate.fit_configuration_identity,
            metadata_entries=metadata_entries,
        )

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            "explicit_shared_guidance_attachment",
            self.guidance_id,
            self.progression_identity,
            self.trajectory_candidate_id,
        )

    @property
    def replay_payload(self) -> tuple[object, ...]:
        return (
            "explicit_shared_guidance_attachment",
            self.guidance_id,
            self.progression_identity,
            self.trajectory_candidate_id,
            self.trajectory_evidence_lane,
            self.fit_configuration_identity,
            self.metadata_entries,
        )


@dataclass(frozen=True)
class ExplicitSharedGuidancePlannerConsumption:
    planner_stage: ExplicitSharedGuidancePlannerStage
    topology_station_ids: tuple[str, ...]
    attachment_identity: tuple[object, ...]
    bounded_to_in_between_travel: bool = True
    overrides_topology_truth: bool = False

    def __init__(
        self,
        *,
        planner_stage: ExplicitSharedGuidancePlannerStage,
        topology_station_ids: tuple[str, ...] | list[str],
        attachment_identity: tuple[object, ...],
        bounded_to_in_between_travel: bool = True,
        overrides_topology_truth: bool = False,
    ) -> None:
        normalized_topology = tuple(str(value).strip() for value in topology_station_ids)
        if any(not value for value in normalized_topology):
            raise ValueError("topology_station_ids must be non-empty strings.")
        object.__setattr__(self, "planner_stage", _normalize_planner_stage(planner_stage))
        object.__setattr__(self, "topology_station_ids", normalized_topology)
        object.__setattr__(self, "attachment_identity", attachment_identity)
        object.__setattr__(self, "bounded_to_in_between_travel", bool(bounded_to_in_between_travel))
        object.__setattr__(self, "overrides_topology_truth", bool(overrides_topology_truth))
        if not self.bounded_to_in_between_travel:
            raise ValueError("explicit shared guidance must remain bounded to in-between travel.")
        if self.overrides_topology_truth:
            raise ValueError("explicit shared guidance must not override topology truth.")

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            "explicit_shared_guidance_planner_consumption",
            self.planner_stage,
            self.topology_station_ids,
            self.attachment_identity,
        )


__all__ = [
    "ExplicitSharedGuidanceAttachmentRecord",
    "ExplicitSharedGuidancePlannerConsumption",
    "ExplicitSharedGuidancePlannerStage",
]
