"""Capability-bounded Codex sidecar broker for reference review."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import time
from typing import Any, Callable, Mapping
from uuid import uuid4

from .async_core.audit import AuditEmitter, build_audit_event
from .async_core.durable_writes import DurableWriteLane, DurableWriteRequest
from .async_core.messages import ReviewTaskKind, ReviewWorkbenchMessage
from .lifecycle import ReviewNoteRecord
from .source_registry import (
    ReviewContextPayload,
    ReviewSourceModelRecord,
    SourceValidationResult,
    build_review_context_payload,
    resolve_generated_review_module,
)


class SidecarTool(str, Enum):
    READ_FIXTURE_CONTEXT = "read_fixture_context"
    READ_ALLOWED_SOURCE = "read_allowed_source"
    WRITE_CANDIDATE_MODEL = "write_candidate_model"
    WRITE_CANDIDATE_NOTE_PATCH = "write_candidate_note_patch"
    REQUEST_CANDIDATE_REGENERATION = "request_candidate_regeneration"
    LIST_CANDIDATE_OUTPUTS = "list_candidate_outputs"
    EXPLAIN_BLOCKING_DIAGNOSTIC = "explain_blocking_diagnostic"


@dataclass(frozen=True)
class CodexContextPayload:
    fixture_id: str
    source_context: Mapping[str, Any]
    note_summary: str | None = None
    omissions: tuple[str, ...] = ("environment", "chat_history", "unrelated_files")
    max_bytes: int = 32_768

    def __post_init__(self) -> None:
        size = len(json.dumps(self.to_json_dict(), sort_keys=True).encode("utf-8"))
        if size > self.max_bytes:
            raise ValueError("codex_context_too_large")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "source_context": dict(self.source_context),
            "note_summary": self.note_summary,
            "omissions": list(self.omissions),
        }


def build_codex_context_payload(
    source_context: ReviewContextPayload,
    *,
    note: ReviewNoteRecord | None = None,
) -> CodexContextPayload:
    note_summary = note.status.value if note is not None else None
    return CodexContextPayload(
        fixture_id=source_context.fixture_id,
        source_context=source_context.to_json_dict(),
        note_summary=note_summary,
    )


@dataclass(frozen=True)
class ToolPolicyRecord:
    allowed_tools: frozenset[SidecarTool]
    candidate_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_tools",
            frozenset(SidecarTool(tool) for tool in self.allowed_tools),
        )
        object.__setattr__(self, "candidate_root", Path(self.candidate_root))

    def allows(self, tool: SidecarTool) -> bool:
        return SidecarTool(tool) in self.allowed_tools


@dataclass(frozen=True)
class ToolRequestRecord:
    tool: SidecarTool
    fixture_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool", SidecarTool(self.tool))


@dataclass(frozen=True)
class ToolBrokerResult:
    accepted: bool
    result: Any = None
    diagnostic: str | None = None


class ToolPolicyBroker:
    """Deny-by-default sidecar tool router."""

    def __init__(self, policy: ToolPolicyRecord, audit: AuditEmitter | None = None) -> None:
        self.policy = policy
        self.audit = audit or AuditEmitter()
        self._routes: dict[SidecarTool, Callable[[ToolRequestRecord], Any]] = {}

    def register(self, tool: SidecarTool, handler: Callable[[ToolRequestRecord], Any]) -> None:
        self._routes[SidecarTool(tool)] = handler

    def handle_raw(
        self,
        *,
        tool: str,
        fixture_id: str,
        payload: Mapping[str, Any] | None = None,
    ) -> ToolBrokerResult:
        try:
            request = ToolRequestRecord(SidecarTool(tool), fixture_id, payload or {})
        except ValueError:
            audit_request = ReviewWorkbenchMessage(
                owner="codex-sidecar",
                kind=ReviewTaskKind.CODEX_REQUEST,
                request_id=1,
                fixture_id=fixture_id,
            )
            self.audit.emit(
                build_audit_event(
                    "tool_refused",
                    audit_request,
                    details={"tool": tool, "reason": "unknown_tool"},
                )
            )
            return ToolBrokerResult(False, diagnostic="tool_unknown")
        return self.handle(request)

    def handle(self, request: ToolRequestRecord) -> ToolBrokerResult:
        audit_request = ReviewWorkbenchMessage(
            owner="codex-sidecar",
            kind=ReviewTaskKind.CODEX_REQUEST,
            request_id=1,
            fixture_id=request.fixture_id,
        )
        if not self.policy.allows(request.tool):
            self.audit.emit(
                build_audit_event(
                    "tool_refused",
                    audit_request,
                    details={"tool": request.tool.value, "reason": "not_allowed"},
                )
            )
            return ToolBrokerResult(False, diagnostic="tool_not_allowed")
        handler = self._routes.get(request.tool)
        if handler is None:
            self.audit.emit(
                build_audit_event(
                    "tool_refused",
                    audit_request,
                    details={"tool": request.tool.value, "reason": "not_registered"},
                )
            )
            return ToolBrokerResult(False, diagnostic="tool_not_registered")
        try:
            result = handler(request)
        except Exception as exc:
            self.audit.emit(
                build_audit_event(
                    "tool_failed",
                    audit_request,
                    details={"tool": request.tool.value, "error": str(exc)},
                )
            )
            return ToolBrokerResult(False, diagnostic=str(exc))
        self.audit.emit(
            build_audit_event("tool_accepted", audit_request, details={"tool": request.tool.value})
        )
        return ToolBrokerResult(True, result=result)


@dataclass(frozen=True)
class CandidateModelRecord:
    fixture_id: str
    path: Path
    source_validation: SourceValidationResult


@dataclass(frozen=True)
class CandidateWriteResult:
    accepted: bool
    record: CandidateModelRecord | None = None
    diagnostic: str | None = None


class CandidateModelStore:
    def __init__(
        self,
        root: Path,
        *,
        durable_lane: DurableWriteLane | None = None,
        max_bytes: int = 128_000,
    ) -> None:
        self.root = root
        self._lane = durable_lane or DurableWriteLane()
        self._max_bytes = max_bytes

    def write_candidate(
        self,
        *,
        fixture_id: str,
        feature_name: str,
        relative_path: str,
        source_text: str,
    ) -> CandidateWriteResult:
        if len(source_text.encode("utf-8")) > self._max_bytes:
            return CandidateWriteResult(False, diagnostic="candidate_too_large")
        self.root.mkdir(parents=True, exist_ok=True)
        path = (self.root / relative_path).resolve()
        try:
            path.relative_to(self.root.resolve())
        except ValueError:
            return CandidateWriteResult(False, diagnostic="candidate_outside_root")
        if path.exists():
            return CandidateWriteResult(False, diagnostic="candidate_already_exists")

        def write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source_text)

        write_result = self._lane.run(
            DurableWriteRequest(
                name=f"candidate:{fixture_id}",
                root=self.root,
                lock_name=fixture_id.replace("/", "__"),
            ),
            write,
        )
        if not write_result.accepted:
            return CandidateWriteResult(False, diagnostic=write_result.diagnostic)
        validation = resolve_generated_review_module(
            path,
            allowed_root=self.root,
            fixture_id=fixture_id,
            feature_name=feature_name,
        )
        return CandidateWriteResult(
            validation.valid,
            CandidateModelRecord(fixture_id, path, validation),
            None if validation.valid else "candidate_validation_failed",
        )

    def list_candidates(self, fixture_id: str) -> tuple[Path, ...]:
        prefix = fixture_id.replace("/", "__")
        return tuple(sorted(self.root.glob(f"{prefix}*.py")))


@dataclass(frozen=True)
class CandidateNotePatch:
    fixture_id: str
    body: str
    patch_id: str = field(default_factory=lambda: uuid4().hex)


def propose_note_patch(
    *,
    fixture_id: str,
    body: str,
    selected_fixture_id: str | None = None,
    max_bytes: int = 16_384,
) -> ToolBrokerResult:
    if selected_fixture_id is not None and fixture_id != selected_fixture_id:
        return ToolBrokerResult(False, diagnostic="stale_fixture")
    if len(body.encode("utf-8")) > max_bytes:
        return ToolBrokerResult(False, diagnostic="patch_too_large")
    lowered = body.lower()
    if "chat log" in lowered or "password" in lowered or "secret" in lowered:
        return ToolBrokerResult(False, diagnostic="patch_contains_disallowed_detail")
    return ToolBrokerResult(True, result=CandidateNotePatch(fixture_id=fixture_id, body=body))


@dataclass(frozen=True)
class RegenerationRequest:
    fixture_id: str
    source_path: Path
    request_id: str = field(default_factory=lambda: uuid4().hex)


def request_regeneration(
    *,
    fixture_id: str,
    selected_fixture_id: str,
    source_record: ReviewSourceModelRecord,
) -> ToolBrokerResult:
    if fixture_id != selected_fixture_id:
        return ToolBrokerResult(False, diagnostic="stale_fixture")
    return ToolBrokerResult(
        True,
        result=RegenerationRequest(fixture_id=fixture_id, source_path=source_record.source_path),
    )


@dataclass(frozen=True)
class SidecarSessionRecord:
    fixture_id: str
    session_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: float = field(default_factory=time)
    cancelled: bool = False

    def cancel(self) -> "SidecarSessionRecord":
        return SidecarSessionRecord(
            fixture_id=self.fixture_id,
            session_id=self.session_id,
            started_at=self.started_at,
            cancelled=True,
        )


@dataclass(frozen=True)
class SidecarProcessFailure:
    fixture_id: str
    session_id: str
    diagnostic: str


class SidecarProcessLauncher:
    """Records fixture-scoped sidecar sessions without granting write authority."""

    def __init__(self, audit: AuditEmitter | None = None) -> None:
        self.audit = audit or AuditEmitter()

    def start(self, *, fixture_id: str) -> SidecarSessionRecord:
        session = SidecarSessionRecord(fixture_id=fixture_id)
        self.audit.emit(
            build_audit_event(
                "sidecar_session_started",
                ReviewWorkbenchMessage(
                    owner="codex-sidecar",
                    kind=ReviewTaskKind.CODEX_REQUEST,
                    request_id=1,
                    fixture_id=fixture_id,
                ),
                details={"session_id": session.session_id},
            )
        )
        return session

    def fail(self, session: SidecarSessionRecord, diagnostic: str) -> SidecarProcessFailure:
        failure = SidecarProcessFailure(
            fixture_id=session.fixture_id,
            session_id=session.session_id,
            diagnostic=diagnostic,
        )
        self.audit.emit(
            build_audit_event(
                "sidecar_session_failed",
                ReviewWorkbenchMessage(
                    owner="codex-sidecar",
                    kind=ReviewTaskKind.CODEX_REQUEST,
                    request_id=1,
                    fixture_id=session.fixture_id,
                ),
                details={"session_id": session.session_id, "diagnostic": diagnostic},
            )
        )
        return failure


def default_context_for_source(record: ReviewSourceModelRecord) -> CodexContextPayload:
    return build_codex_context_payload(build_review_context_payload(record))
