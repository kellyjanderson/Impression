"""Bounded task dispatcher for reference review workers."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

from .audit import AuditEmitter, build_audit_event
from .messages import ReviewTaskKind, ReviewWorkbenchMessage, WorkerResultEnvelope
from .staleness import LatestRequestTracker

TaskCallable = Callable[[ReviewWorkbenchMessage], Any]


@dataclass(frozen=True)
class WorkerPolicy:
    """Queue and coalescing policy for a task kind."""

    max_pending: int = 1
    coalesce: bool = False

    def __post_init__(self) -> None:
        if self.max_pending < 1:
            raise ValueError("max_pending must be positive")


@dataclass(frozen=True)
class DispatchResult:
    """Result of accepting or rejecting a dispatch request."""

    accepted: bool
    request: ReviewWorkbenchMessage
    future: Future[WorkerResultEnvelope] | None = None
    diagnostic: str | None = None


class TaskDispatcher:
    """Submit bounded worker tasks and return typed result envelopes."""

    def __init__(
        self,
        *,
        max_workers: int = 2,
        policies: dict[ReviewTaskKind, WorkerPolicy] | None = None,
        tracker: LatestRequestTracker | None = None,
        audit: AuditEmitter | None = None,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._policies = policies or {}
        self._tracker = tracker or LatestRequestTracker()
        self._audit = audit or AuditEmitter()
        self._lock = Lock()
        self._pending: dict[tuple[str, ReviewTaskKind], int] = {}

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def dispatch(
        self,
        request: ReviewWorkbenchMessage,
        worker: TaskCallable,
    ) -> DispatchResult:
        policy = self._policies.get(request.kind, WorkerPolicy())
        pending_key = (request.owner, request.kind)
        with self._lock:
            pending = self._pending.get(pending_key, 0)
            if pending >= policy.max_pending and not policy.coalesce:
                self._audit.emit(
                    build_audit_event(
                        "dispatch_rejected",
                        request,
                        details={"reason": "queue_full"},
                    )
                )
                return DispatchResult(False, request, diagnostic="queue_full")
            if policy.coalesce and pending >= policy.max_pending:
                self._tracker.cancel(request)
            self._pending[pending_key] = pending + 1

        self._tracker.register(request)
        self._audit.emit(build_audit_event("dispatch_accepted", request))
        future = self._executor.submit(self._run, request, worker, pending_key)
        return DispatchResult(True, request, future=future)

    def _run(
        self,
        request: ReviewWorkbenchMessage,
        worker: TaskCallable,
        pending_key: tuple[str, ReviewTaskKind],
    ) -> WorkerResultEnvelope:
        try:
            result = worker(request)
        except Exception as exc:
            envelope = WorkerResultEnvelope(request=request, ok=False, error=str(exc))
            self._audit.emit(
                build_audit_event(
                    "task_failed",
                    request,
                    details={"error": str(exc)},
                )
            )
        else:
            envelope = WorkerResultEnvelope(request=request, ok=True, result=result)
            self._audit.emit(build_audit_event("task_completed", request))
        finally:
            with self._lock:
                remaining = self._pending[pending_key] - 1
                if remaining:
                    self._pending[pending_key] = remaining
                else:
                    self._pending.pop(pending_key, None)
        return envelope

