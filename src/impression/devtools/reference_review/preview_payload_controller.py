"""Async controller route for reference review preview payload builds."""

from __future__ import annotations

import contextlib
import io
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable

from .async_core import (
    DispatchResult,
    RequestIdAllocator,
    ReviewTaskKind,
    ReviewWorkbenchMessage,
    TaskDispatcher,
    WorkerResultEnvelope,
)
from .async_core.qt_handoff import sanitize_error_text
from .preview_payload import PreviewPayload, PreviewPayloadRequest
from .preview_payload_builder import build_serialized_preview_payload
from .source_registry import ReviewSourceModelRecord


@dataclass(frozen=True)
class PreviewPayloadCleanupDiagnostic:
    """Result of a controller-owned temporary payload cleanup attempt."""

    code: str
    reason: str
    owner: str
    request_id: int
    fixture_id: str | None
    deleted: bool = False
    message: str = ""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "reason": self.reason,
            "owner": self.owner,
            "request_id": self.request_id,
            "fixture_id": self.fixture_id,
            "deleted": self.deleted,
            "message": self.message,
        }


@dataclass(frozen=True)
class PreviewPayloadControllerDiagnostic:
    """Sanitized controller-level diagnostic for payload launch and worker output."""

    code: str
    message: str
    owner: str
    request_id: int
    fixture_id: str | None
    stdout: str = ""
    stderr: str = ""

    def to_json_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "owner": self.owner,
            "request_id": self.request_id,
            "fixture_id": self.fixture_id,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True)
class PreviewPayloadProcessResult:
    """Worker completion payload plus captured process streams."""

    payload: PreviewPayload
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.payload.ok


@dataclass(frozen=True)
class ActivePreviewRequestState:
    """Identity of the preview request that is allowed to mutate pane state."""

    owner: str
    request_id: int
    fixture_id: str
    generation: int

    @property
    def identity(self) -> tuple[str, int, str, int]:
        return (self.owner, self.request_id, self.fixture_id, self.generation)


@dataclass(frozen=True)
class PreviewPayloadReadyEvent:
    """Current payload event safe to hand to the preview pane."""

    payload: PreviewPayload
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class PreviewPayloadFailedEvent:
    """Current payload failure event safe to hand to the preview pane."""

    diagnostic: PreviewPayloadControllerDiagnostic
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class PreviewPayloadHandoffDecision:
    """Decision for whether a payload completion reached the preview pane."""

    accepted: bool
    reason: str
    event: PreviewPayloadReadyEvent | None = None
    failure_event: PreviewPayloadFailedEvent | None = None
    cleanup: PreviewPayloadCleanupDiagnostic | None = None
    diagnostic: PreviewPayloadControllerDiagnostic | None = None


class PreviewPayloadProcessController:
    """Launch preview payload work through the shared async envelope pattern."""

    def __init__(
        self,
        *,
        owner: str = "preview-payload",
        dispatcher: TaskDispatcher | None = None,
        owns_dispatcher: bool | None = None,
        request_ids: RequestIdAllocator | None = None,
        payload_dir: Path | None = None,
        cwd: Path | None = None,
        home: Path | None = None,
    ) -> None:
        self._owner = owner
        self._dispatcher = dispatcher
        self._owns_dispatcher = dispatcher is None if owns_dispatcher is None else owns_dispatcher
        self._process_executor = None if dispatcher is not None else ProcessPoolExecutor(max_workers=1)
        self._launch_executor = None if dispatcher is not None else ThreadPoolExecutor(max_workers=1)
        self._process_futures: list[Future[WorkerResultEnvelope]] = []
        self._process_lock = Lock()
        self._request_ids = request_ids or RequestIdAllocator()
        self._payload_dir = payload_dir
        self._cwd = cwd
        self._home = home
        self._generation = 0
        self._active_identity: tuple[str, int, str, int] | None = None
        self._diagnostics: list[PreviewPayloadControllerDiagnostic] = []
        self._owned_payload_paths: dict[tuple[str, int, str, int], Path] = {}

    @property
    def active_identity(self) -> tuple[str, int, str, int] | None:
        return self._active_identity

    @property
    def active_state(self) -> ActivePreviewRequestState | None:
        if self._active_identity is None:
            return None
        owner, request_id, fixture_id, generation = self._active_identity
        return ActivePreviewRequestState(owner, request_id, fixture_id, generation)

    @property
    def diagnostics(self) -> tuple[PreviewPayloadControllerDiagnostic, ...]:
        return tuple(self._diagnostics)

    def close(self) -> None:
        for future in self._process_futures:
            future.cancel()
        self._process_futures = []
        if self._launch_executor is not None:
            self._launch_executor.shutdown(wait=False, cancel_futures=True)
            self._launch_executor = None
        if self._process_executor is not None:
            with self._process_lock:
                process_executor = self._process_executor
                self._process_executor = None
            process_executor.shutdown(wait=False, cancel_futures=True)
        if self._dispatcher is not None and self._owns_dispatcher:
            self._dispatcher.close()

    def adopt_payload(self, payload: PreviewPayload) -> None:
        if payload.payload_path is None:
            return
        self._owned_payload_paths[payload.identity] = payload.payload_path

    def cleanup_payload(
        self,
        payload: PreviewPayload,
        *,
        reason: str,
    ) -> PreviewPayloadCleanupDiagnostic:
        return self.cleanup_identity(payload.identity, reason=reason)

    def cleanup_cancelled(
        self,
        identity: tuple[str, int, str, int],
    ) -> PreviewPayloadCleanupDiagnostic:
        return self.cleanup_identity(identity, reason="cancelled")

    def cancel_active_request(self) -> PreviewPayloadCleanupDiagnostic | None:
        identity = self._active_identity
        self._active_identity = None
        if identity is None:
            return None
        return self.cleanup_identity(identity, reason="cancelled")

    def cleanup_stale_payload(self, payload: PreviewPayload) -> PreviewPayloadCleanupDiagnostic:
        if payload.identity == self._active_identity:
            return PreviewPayloadCleanupDiagnostic(
                code="preview-payload-cleanup-skipped-current",
                reason="stale",
                owner=payload.request.owner,
                request_id=payload.request.request_id,
                fixture_id=payload.request.fixture_id,
                message="current payload ownership retained",
            )
        return self.cleanup_payload(payload, reason="stale")

    def cleanup_identity(
        self,
        identity: tuple[str, int, str, int],
        *,
        reason: str,
    ) -> PreviewPayloadCleanupDiagnostic:
        owner, request_id, fixture_id, _generation = identity
        path = self._owned_payload_paths.get(identity)
        if path is None:
            return PreviewPayloadCleanupDiagnostic(
                code="preview-payload-cleanup-unowned",
                reason=reason,
                owner=owner,
                request_id=request_id,
                fixture_id=fixture_id,
                message="payload is not owned by this controller",
            )
        try:
            path.unlink()
        except FileNotFoundError:
            self._owned_payload_paths.pop(identity, None)
            return PreviewPayloadCleanupDiagnostic(
                code="preview-payload-cleanup-missing",
                reason=reason,
                owner=owner,
                request_id=request_id,
                fixture_id=fixture_id,
                message="payload file was already absent",
            )
        except Exception as exc:
            return PreviewPayloadCleanupDiagnostic(
                code="preview-payload-cleanup-failed",
                reason=reason,
                owner=owner,
                request_id=request_id,
                fixture_id=fixture_id,
                message=sanitize_error_text(str(exc), cwd=self._cwd, home=self._home),
            )
        self._owned_payload_paths.pop(identity, None)
        return PreviewPayloadCleanupDiagnostic(
            code="preview-payload-cleanup-deleted",
            reason=reason,
            owner=owner,
            request_id=request_id,
            fixture_id=fixture_id,
            deleted=True,
            message="payload file deleted",
        )

    def handle_completion(
        self,
        envelope,
        handoff: Callable[[PreviewPayloadReadyEvent], None],
        diagnostic_handoff: Callable[[PreviewPayloadFailedEvent], None] | None = None,
    ) -> PreviewPayloadHandoffDecision:
        if not envelope.ok or not isinstance(envelope.result, PreviewPayloadProcessResult):
            diagnostic = PreviewPayloadControllerDiagnostic(
                code="preview-payload-completion-invalid",
                message=sanitize_error_text(envelope.error or "invalid preview completion", cwd=self._cwd, home=self._home),
                owner=envelope.owner,
                request_id=envelope.request_id,
                fixture_id=envelope.fixture_id,
            )
            return PreviewPayloadHandoffDecision(False, "invalid_completion", diagnostic=diagnostic)
        return self.handle_process_result(envelope.result, handoff, diagnostic_handoff)

    def handle_process_result(
        self,
        result: PreviewPayloadProcessResult,
        handoff: Callable[[PreviewPayloadReadyEvent], None],
        diagnostic_handoff: Callable[[PreviewPayloadFailedEvent], None] | None = None,
    ) -> PreviewPayloadHandoffDecision:
        payload = result.payload
        self.adopt_payload(payload)
        if payload.identity != self._active_identity:
            cleanup = self.cleanup_stale_payload(payload)
            return PreviewPayloadHandoffDecision(False, "stale_payload_rejected", cleanup=cleanup)
        if not payload.ok:
            diagnostic = PreviewPayloadControllerDiagnostic(
                code=payload.diagnostic.code if payload.diagnostic is not None else "preview-payload-failed",
                message=payload.diagnostic.message if payload.diagnostic is not None else "payload build failed",
                owner=payload.request.owner,
                request_id=payload.request.request_id,
                fixture_id=payload.request.fixture_id,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            failure_event = PreviewPayloadFailedEvent(
                diagnostic=diagnostic,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            if diagnostic_handoff is not None:
                diagnostic_handoff(failure_event)
            return PreviewPayloadHandoffDecision(
                False,
                "payload_failure",
                failure_event=failure_event,
                diagnostic=diagnostic,
            )
        event = PreviewPayloadReadyEvent(payload=payload, stdout=result.stdout, stderr=result.stderr)
        handoff(event)
        return PreviewPayloadHandoffDecision(True, "current_payload_ready", event=event)

    def launch(self, record: ReviewSourceModelRecord) -> DispatchResult:
        self._generation += 1
        generation = self._generation
        request_id = self._request_ids.next(self._owner)
        message = ReviewWorkbenchMessage(
            owner=self._owner,
            kind=ReviewTaskKind.PREVIEW_BUILD,
            request_id=request_id,
            fixture_id=record.fixture_id,
            payload={"generation": generation},
        )
        self._active_identity = (
            self._owner,
            request_id,
            record.fixture_id,
            generation,
        )
        if self._dispatcher is not None:
            result = self._dispatcher.dispatch(
                message,
                lambda worker_message: _run_payload_worker(
                    worker_message,
                    record,
                    generation=generation,
                    payload_dir=self._payload_dir,
                    cwd=self._cwd,
                    home=self._home,
                ),
            )
        else:
            result = self._launch_process(message, record, generation=generation)
        if not result.accepted:
            self._diagnostics.append(
                PreviewPayloadControllerDiagnostic(
                    code="preview-payload-launch-rejected",
                    message=result.diagnostic or "dispatch rejected",
                    owner=message.owner,
                    request_id=message.request_id,
                    fixture_id=message.fixture_id,
                )
            )
        return result

    def _launch_process(
        self,
        message: ReviewWorkbenchMessage,
        record: ReviewSourceModelRecord,
        *,
        generation: int,
    ) -> DispatchResult:
        if self._process_executor is None or self._launch_executor is None:
            return DispatchResult(False, message, diagnostic="preview_executor_closed")
        retained: list[Future[WorkerResultEnvelope]] = []
        for future in self._process_futures:
            if future.done():
                continue
            future.cancel()
            if not future.cancelled():
                retained.append(future)
        future = self._launch_executor.submit(
            self._submit_process_payload,
            message,
            record,
            generation=generation,
        )
        retained.append(future)
        self._process_futures = retained
        return DispatchResult(True, message, future=future)

    def _submit_process_payload(
        self,
        message: ReviewWorkbenchMessage,
        record: ReviewSourceModelRecord,
        *,
        generation: int,
    ) -> WorkerResultEnvelope:
        with self._process_lock:
            process_executor = self._process_executor
            if process_executor is None:
                return WorkerResultEnvelope(
                    request=message,
                    ok=False,
                    error="preview_executor_closed",
                )
            future = process_executor.submit(
                _run_payload_worker_envelope,
                message,
                record,
                generation=generation,
                payload_dir=self._payload_dir,
                cwd=self._cwd,
                home=self._home,
            )
        try:
            return future.result()
        except Exception as exc:
            return WorkerResultEnvelope(
                request=message,
                ok=False,
                error=sanitize_error_text(
                    str(exc) or exc.__class__.__name__,
                    cwd=self._cwd,
                    home=self._home,
                ),
            )


def _run_payload_worker_envelope(
    message: ReviewWorkbenchMessage,
    record: ReviewSourceModelRecord,
    *,
    generation: int,
    payload_dir: Path | None,
    cwd: Path | None,
    home: Path | None,
) -> WorkerResultEnvelope:
    try:
        result = _run_payload_worker(
            message,
            record,
            generation=generation,
            payload_dir=payload_dir,
            cwd=cwd,
            home=home,
        )
    except Exception as exc:
        return WorkerResultEnvelope(
            request=message,
            ok=False,
            error=sanitize_error_text(str(exc) or exc.__class__.__name__, cwd=cwd, home=home),
        )
    return WorkerResultEnvelope(request=message, ok=True, result=result)


def _run_payload_worker(
    message: ReviewWorkbenchMessage,
    record: ReviewSourceModelRecord,
    *,
    generation: int,
    payload_dir: Path | None,
    cwd: Path | None,
    home: Path | None,
) -> PreviewPayloadProcessResult:
    stdout = io.StringIO()
    stderr = io.StringIO()
    request = PreviewPayloadRequest.from_workbench_message(
        message,
        record,
        generation=generation,
    )
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        payload = build_serialized_preview_payload(
            request,
            payload_dir=payload_dir,
            cwd=cwd,
            home=home,
        )
    stdout_text = sanitize_error_text(stdout.getvalue(), cwd=cwd, home=home)
    stderr_text = sanitize_error_text(stderr.getvalue(), cwd=cwd, home=home)
    return PreviewPayloadProcessResult(
        payload=payload,
        stdout=stdout_text,
        stderr=stderr_text,
    )
