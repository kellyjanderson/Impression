from __future__ import annotations

from dataclasses import dataclass

from .control_station_inference import StructuralPreservationReport
from .fit_records import FitAssessmentReport


def _normalize_station_entries(
    entries: tuple[tuple[str, str], ...] | list[tuple[str, str]],
    *,
    label: str,
) -> tuple[tuple[str, str], ...]:
    normalized: list[tuple[str, str]] = []
    for station_id, kind in entries:
        normalized_id = str(station_id).strip()
        normalized_kind = str(kind).strip()
        if not normalized_id:
            raise ValueError(f"{label} station ids must be non-empty.")
        if not normalized_kind:
            raise ValueError(f"{label} station kinds must be non-empty.")
        normalized.append((normalized_id, normalized_kind))
    return tuple(normalized)


def _normalize_string_entries(
    values: tuple[str, ...] | list[str],
    *,
    label: str,
) -> tuple[str, ...]:
    normalized = tuple(str(value).strip() for value in values)
    if any(not value for value in normalized):
        raise ValueError(f"{label} entries must be non-empty strings.")
    return normalized


@dataclass(frozen=True)
class InferenceFitDriftSummary:
    metric_name: str
    residual_value: float
    acceptance_threshold: float
    decision_outcome: str
    decision_reason: str

    @classmethod
    def from_fit_assessment(cls, assessment: FitAssessmentReport) -> "InferenceFitDriftSummary":
        return cls(
            metric_name=assessment.residual_report.metric_name,
            residual_value=assessment.residual_report.residual_value,
            acceptance_threshold=assessment.residual_report.acceptance_threshold,
            decision_outcome=assessment.decision_outcome,
            decision_reason=assessment.decision_reason,
        )


@dataclass(frozen=True)
class InferenceStructuralPreservationSummary:
    required_topology_station_ids: tuple[str, ...]
    retained_topology_station_ids: tuple[str, ...]
    dropped_topology_station_ids: tuple[str, ...]

    @classmethod
    def from_report(
        cls,
        report: StructuralPreservationReport,
    ) -> "InferenceStructuralPreservationSummary":
        return cls(
            required_topology_station_ids=report.required_topology_station_ids,
            retained_topology_station_ids=report.retained_topology_station_ids,
            dropped_topology_station_ids=report.dropped_topology_station_ids,
        )


@dataclass(frozen=True)
class SharedInferenceDiagnosticBundle:
    retained_station_entries: tuple[tuple[str, str], ...]
    dropped_station_entries: tuple[tuple[str, str], ...]
    fit_drift: InferenceFitDriftSummary | None
    structural_preservation: InferenceStructuralPreservationSummary | None
    evidence_references: tuple[str, ...]
    provenance_references: tuple[str, ...]
    schema_version: str = "0.1.0.a"

    def __init__(
        self,
        *,
        retained_station_entries: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (),
        dropped_station_entries: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (),
        fit_drift: InferenceFitDriftSummary | None = None,
        structural_preservation: InferenceStructuralPreservationSummary | None = None,
        evidence_references: tuple[str, ...] | list[str] = (),
        provenance_references: tuple[str, ...] | list[str] = (),
        schema_version: str = "0.1.0.a",
    ) -> None:
        normalized_version = str(schema_version).strip()
        if not normalized_version:
            raise ValueError("schema_version must be non-empty.")
        object.__setattr__(
            self,
            "retained_station_entries",
            _normalize_station_entries(retained_station_entries, label="retained"),
        )
        object.__setattr__(
            self,
            "dropped_station_entries",
            _normalize_station_entries(dropped_station_entries, label="dropped"),
        )
        object.__setattr__(self, "fit_drift", fit_drift)
        object.__setattr__(self, "structural_preservation", structural_preservation)
        object.__setattr__(
            self,
            "evidence_references",
            _normalize_string_entries(evidence_references, label="evidence_reference"),
        )
        object.__setattr__(
            self,
            "provenance_references",
            _normalize_string_entries(provenance_references, label="provenance_reference"),
        )
        object.__setattr__(self, "schema_version", normalized_version)

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            "shared_inference_diagnostic_bundle",
            self.schema_version,
            self.retained_station_entries,
            self.dropped_station_entries,
            self.evidence_references,
            self.provenance_references,
        )

    @property
    def replay_payload(self) -> tuple[object, ...]:
        fit_payload = None
        if self.fit_drift is not None:
            fit_payload = (
                self.fit_drift.metric_name,
                self.fit_drift.residual_value,
                self.fit_drift.acceptance_threshold,
                self.fit_drift.decision_outcome,
                self.fit_drift.decision_reason,
            )
        structural_payload = None
        if self.structural_preservation is not None:
            structural_payload = (
                self.structural_preservation.required_topology_station_ids,
                self.structural_preservation.retained_topology_station_ids,
                self.structural_preservation.dropped_topology_station_ids,
            )
        return (
            "shared_inference_diagnostic_bundle",
            self.schema_version,
            self.retained_station_entries,
            self.dropped_station_entries,
            fit_payload,
            structural_payload,
            self.evidence_references,
            self.provenance_references,
        )


__all__ = [
    "InferenceFitDriftSummary",
    "InferenceStructuralPreservationSummary",
    "SharedInferenceDiagnosticBundle",
]
