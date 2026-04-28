from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .inference_diagnostics import SharedInferenceDiagnosticBundle

DeveloperInferenceCertaintyPosture = Literal["certain", "uncertain", "refused"]


def _normalize_developer_certainty_posture(value: str) -> DeveloperInferenceCertaintyPosture:
    allowed: set[str] = {"certain", "uncertain", "refused"}
    posture = str(value)
    if posture not in allowed:
        raise ValueError("certainty_posture must be one of: certain, uncertain, refused.")
    return posture  # type: ignore[return-value]


@dataclass(frozen=True)
class DeveloperInferenceInspection:
    retained_station_entries: tuple[tuple[str, str], ...]
    dropped_station_entries: tuple[tuple[str, str], ...]
    fit_drift: tuple[object, ...] | None
    provenance_references: tuple[str, ...]
    certainty_posture: DeveloperInferenceCertaintyPosture
    summary_reason: str

    def __init__(
        self,
        *,
        retained_station_entries: tuple[tuple[str, str], ...],
        dropped_station_entries: tuple[tuple[str, str], ...],
        fit_drift: tuple[object, ...] | None,
        provenance_references: tuple[str, ...],
        certainty_posture: DeveloperInferenceCertaintyPosture,
        summary_reason: str,
    ) -> None:
        normalized_reason = str(summary_reason).strip()
        if not normalized_reason:
            raise ValueError("summary_reason must be non-empty.")
        object.__setattr__(self, "retained_station_entries", retained_station_entries)
        object.__setattr__(self, "dropped_station_entries", dropped_station_entries)
        object.__setattr__(self, "fit_drift", fit_drift)
        object.__setattr__(self, "provenance_references", provenance_references)
        object.__setattr__(
            self,
            "certainty_posture",
            _normalize_developer_certainty_posture(certainty_posture),
        )
        object.__setattr__(self, "summary_reason", normalized_reason)

    @classmethod
    def from_bundle(
        cls,
        bundle: SharedInferenceDiagnosticBundle,
    ) -> "DeveloperInferenceInspection":
        if bundle.dropped_station_entries:
            posture: DeveloperInferenceCertaintyPosture = "refused"
            reason = "dropped_structure_present"
        elif bundle.fit_drift is not None and bundle.fit_drift.decision_outcome == "refused":
            posture = "refused"
            reason = "fit_refused"
        elif any(reference.startswith("inferred:") for reference in bundle.provenance_references):
            posture = "uncertain"
            reason = "inferred_provenance_requires_inspection"
        else:
            posture = "certain"
            reason = "explicit_structure_without_refusal"

        fit_drift_payload = None
        if bundle.fit_drift is not None:
            fit_drift_payload = (
                bundle.fit_drift.metric_name,
                bundle.fit_drift.residual_value,
                bundle.fit_drift.acceptance_threshold,
                bundle.fit_drift.decision_outcome,
                bundle.fit_drift.decision_reason,
            )
        return cls(
            retained_station_entries=bundle.retained_station_entries,
            dropped_station_entries=bundle.dropped_station_entries,
            fit_drift=fit_drift_payload,
            provenance_references=bundle.provenance_references,
            certainty_posture=posture,
            summary_reason=reason,
        )


__all__ = [
    "DeveloperInferenceCertaintyPosture",
    "DeveloperInferenceInspection",
]
