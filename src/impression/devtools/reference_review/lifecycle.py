"""Notes, promotion, provenance, and release-gate helpers."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import time
from typing import Any, Mapping

from .async_core.durable_writes import DurableWriteLane, DurableWriteRequest
from .async_core.qt_handoff import sanitize_error_text
from .source_registry import ReviewContextPayload, ReviewSourceModelRecord


class ReviewState(str, Enum):
    UNREVIEWED = "unreviewed"
    NEEDS_WORK = "needs-work"
    BLOCKED = "blocked"
    APPROVED_SOURCE = "approved-source"
    PROMOTED = "promoted"
    RELEASE_GATE_FAILING = "release-gate-failing"


@dataclass(frozen=True)
class ReviewNoteRecord:
    fixture_id: str
    status: ReviewState
    body: str
    updated_at: float = field(default_factory=time)

    def __post_init__(self) -> None:
        if not self.fixture_id:
            raise ValueError("fixture_id must not be empty")
        object.__setattr__(self, "status", ReviewState(self.status))

    def to_markdown(self) -> str:
        return (
            f"---\nfixture_id: {self.fixture_id}\n"
            f"status: {self.status.value}\nupdated_at: {self.updated_at}\n---\n\n"
            f"{_redact_note_body(self.body).strip()}\n"
        )


@dataclass(frozen=True)
class NoteWriteResult:
    saved: bool
    path: Path
    diagnostic: str | None = None


class ReviewNoteStore:
    """Fixture-scoped durable Markdown note store."""

    def __init__(
        self,
        root: Path,
        *,
        durable_lane: DurableWriteLane | None = None,
        max_note_bytes: int = 32_768,
    ) -> None:
        self.root = root
        self._lane = durable_lane or DurableWriteLane()
        self._max_note_bytes = max_note_bytes

    def note_path(self, fixture_id: str) -> Path:
        safe_id = fixture_id.replace("/", "__")
        return self.root / f"{safe_id}.md"

    def load(self, fixture_id: str) -> ReviewNoteRecord | None:
        path = self.note_path(fixture_id)
        if not path.exists():
            return None
        text = path.read_text()
        status = ReviewState.NEEDS_WORK
        if "status: blocked" in text:
            status = ReviewState.BLOCKED
        elif "status: approved-source" in text:
            status = ReviewState.APPROVED_SOURCE
        body = text.split("---", 2)[-1].strip() if text.startswith("---") else text
        return ReviewNoteRecord(fixture_id=fixture_id, status=status, body=body)

    def save(self, note: ReviewNoteRecord) -> NoteWriteResult:
        text = note.to_markdown()
        path = self.note_path(note.fixture_id)
        if len(text.encode("utf-8")) > self._max_note_bytes:
            return NoteWriteResult(False, path, "note_too_large")

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

        result = self._lane.run(
            DurableWriteRequest(
                name=f"note:{note.fixture_id}",
                root=self.root,
                lock_name=path.stem,
            ),
            write,
        )
        return NoteWriteResult(result.accepted, path, result.diagnostic)


@dataclass(frozen=True)
class PromotionArtifact:
    kind: str
    dirty_path: Path
    gold_path: Path

    @property
    def checksum(self) -> str:
        return _sha256(self.dirty_path)


@dataclass(frozen=True)
class PromotionDiagnostic:
    code: str
    message: str


@dataclass(frozen=True)
class PromotionValidationResult:
    allowed: bool
    diagnostics: tuple[PromotionDiagnostic, ...] = ()


def validate_promotion(
    *,
    source_record: ReviewSourceModelRecord | None,
    artifacts: tuple[PromotionArtifact, ...],
) -> PromotionValidationResult:
    diagnostics: list[PromotionDiagnostic] = []
    if source_record is None:
        diagnostics.append(PromotionDiagnostic("missing-source-record", "source record is required"))
    if not artifacts:
        diagnostics.append(PromotionDiagnostic("missing-artifacts", "at least one dirty artifact is required"))
    for artifact in artifacts:
        if not artifact.dirty_path.exists():
            diagnostics.append(
                PromotionDiagnostic(
                    "missing-dirty-artifact",
                    sanitize_error_text(str(artifact.dirty_path)),
                )
            )
    return PromotionValidationResult(not diagnostics, tuple(diagnostics))


@dataclass(frozen=True)
class PromotionRequest:
    fixture_id: str
    source_record: ReviewSourceModelRecord
    artifacts: tuple[PromotionArtifact, ...]
    root: Path


@dataclass(frozen=True)
class PromotionResult:
    promoted: bool
    diagnostics: tuple[PromotionDiagnostic, ...] = ()
    checksums: Mapping[str, str] = field(default_factory=dict)


class PromotionExecutor:
    """Atomically promote dirty artifacts into gold paths."""

    def __init__(self, durable_lane: DurableWriteLane | None = None) -> None:
        self._lane = durable_lane or DurableWriteLane()

    def promote(self, request: PromotionRequest) -> PromotionResult:
        validation = validate_promotion(
            source_record=request.source_record,
            artifacts=request.artifacts,
        )
        if not validation.allowed:
            return PromotionResult(False, validation.diagnostics)
        checksums = {artifact.kind: artifact.checksum for artifact in request.artifacts}

        def write() -> None:
            for artifact in request.artifacts:
                artifact.gold_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(artifact.dirty_path, artifact.gold_path)
                if _sha256(artifact.gold_path) != checksums[artifact.kind]:
                    raise RuntimeError("checksum_mismatch")

        result = self._lane.run(
            DurableWriteRequest(
                name=f"promotion:{request.fixture_id}",
                root=request.root,
                lock_name=request.fixture_id.replace("/", "__"),
            ),
            write,
        )
        if not result.accepted:
            return PromotionResult(False, (PromotionDiagnostic("promotion-failed", result.diagnostic or "failed"),))
        return PromotionResult(True, checksums=checksums)


@dataclass(frozen=True)
class PromotionProvenanceRecord:
    fixture_id: str
    source_identity: tuple[str, str, str]
    artifact_checksums: Mapping[str, str]
    context: Mapping[str, Any]
    promoted_at: float = field(default_factory=time)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "source_identity": list(self.source_identity),
            "artifact_checksums": dict(self.artifact_checksums),
            "context": dict(self.context),
            "promoted_at": self.promoted_at,
        }


class PromotionProvenanceStore:
    def __init__(self, root: Path, durable_lane: DurableWriteLane | None = None) -> None:
        self.root = root
        self._lane = durable_lane or DurableWriteLane()

    def path_for(self, fixture_id: str) -> Path:
        return self.root / f"{fixture_id.replace('/', '__')}.json"

    def write(self, record: PromotionProvenanceRecord) -> bool:
        path = self.path_for(record.fixture_id)

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record.to_json_dict(), sort_keys=True, indent=2))

        return self._lane.run(
            DurableWriteRequest(
                name=f"provenance:{record.fixture_id}",
                root=self.root,
                lock_name=path.stem,
            ),
            write,
        ).accepted

    def exists(self, fixture_id: str) -> bool:
        return self.path_for(fixture_id).exists()


@dataclass(frozen=True)
class StateReason:
    code: str
    message: str


@dataclass(frozen=True)
class ReviewStateAssessment:
    fixture_id: str
    state: ReviewState
    reasons: tuple[StateReason, ...] = ()


def classify_review_state(
    *,
    fixture_id: str,
    note: ReviewNoteRecord | None,
    promoted: bool,
    source_valid: bool = True,
) -> ReviewStateAssessment:
    if not source_valid:
        return ReviewStateAssessment(
            fixture_id,
            ReviewState.RELEASE_GATE_FAILING,
            (StateReason("invalid-source", "source record is invalid"),),
        )
    if promoted:
        return ReviewStateAssessment(fixture_id, ReviewState.PROMOTED)
    if note is None:
        return ReviewStateAssessment(
            fixture_id,
            ReviewState.UNREVIEWED,
            (StateReason("missing-note", "fixture has not been reviewed"),),
        )
    if note.status is ReviewState.BLOCKED:
        return ReviewStateAssessment(
            fixture_id,
            ReviewState.BLOCKED,
            (StateReason("blocked-note", "review note marks fixture blocked"),),
        )
    if note.status is ReviewState.APPROVED_SOURCE:
        return ReviewStateAssessment(
            fixture_id,
            ReviewState.APPROVED_SOURCE,
            (StateReason("awaiting-promotion", "source approved but artifacts not promoted"),),
        )
    return ReviewStateAssessment(
        fixture_id,
        ReviewState.NEEDS_WORK,
        (StateReason("note-without-promotion", "review note exists without promotion"),),
    )


@dataclass(frozen=True)
class ReleaseGateReport:
    passed: bool
    assessments: tuple[ReviewStateAssessment, ...]

    @property
    def failing(self) -> tuple[ReviewStateAssessment, ...]:
        return tuple(item for item in self.assessments if item.state is not ReviewState.PROMOTED)


def build_release_gate_report(assessments: tuple[ReviewStateAssessment, ...]) -> ReleaseGateReport:
    return ReleaseGateReport(
        passed=all(item.state is ReviewState.PROMOTED for item in assessments),
        assessments=tuple(sorted(assessments, key=lambda item: item.fixture_id)),
    )


def make_provenance_record(
    *,
    fixture_id: str,
    source_record: ReviewSourceModelRecord,
    promotion: PromotionResult,
    context: ReviewContextPayload,
) -> PromotionProvenanceRecord:
    return PromotionProvenanceRecord(
        fixture_id=fixture_id,
        source_identity=source_record.identity.key,
        artifact_checksums=promotion.checksums,
        context=context.to_json_dict(),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65_536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _redact_note_body(body: str) -> str:
    redacted = []
    for line in body.splitlines():
        lower = line.lower()
        if "token" in lower or "password" in lower or "secret" in lower:
            redacted.append("<redacted>")
        else:
            redacted.append(sanitize_error_text(line))
    return "\n".join(redacted)

