from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .inference_descriptors import SpanLocalCurveIntentEvidence

CurveIntentPosture = Literal["candidate", "indeterminate"]


@dataclass(frozen=True)
class CurveIntentCandidateReport:
    candidate_kind: str
    confidence: float
    posture: CurveIntentPosture
    evidence_count: int
    reason: str


def classify_curve_intent_candidate(
    evidence: tuple[SpanLocalCurveIntentEvidence, ...],
) -> CurveIntentCandidateReport:
    if not evidence:
        return CurveIntentCandidateReport(
            candidate_kind="none",
            confidence=0.0,
            posture="indeterminate",
            evidence_count=0,
            reason="missing_span_local_evidence",
        )
    stable_regions = all(
        len(set(item.section_region_counts)) == 1 for item in evidence
    )
    stable_tracks = all(
        all(count > 0 for count in item.correspondence_track_counts) for item in evidence
    )
    if stable_regions and stable_tracks:
        return CurveIntentCandidateReport(
            candidate_kind="shared_curve",
            confidence=1.0,
            posture="candidate",
            evidence_count=len(evidence),
            reason="stable_region_and_track_evidence",
        )
    return CurveIntentCandidateReport(
        candidate_kind="none",
        confidence=0.0,
        posture="indeterminate",
        evidence_count=len(evidence),
        reason="weak_or_conflicting_evidence",
    )


__all__ = [
    "CurveIntentCandidateReport",
    "CurveIntentPosture",
    "classify_curve_intent_candidate",
]
