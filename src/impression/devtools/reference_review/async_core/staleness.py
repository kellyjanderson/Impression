"""Latest-request and cancellation guards for worker completions."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event, Lock
from uuid import uuid4

from .messages import ReviewWorkbenchMessage, WorkerResultEnvelope


@dataclass(frozen=True)
class CancellationToken:
    """Thread-safe cancellation marker passed to worker callables."""

    cancellation_id: str = field(default_factory=lambda: uuid4().hex)
    _event: Event = field(default_factory=Event, repr=False, compare=False)

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True)
class CompletionDecision:
    """Decision for whether a worker completion may mutate owner state."""

    accepted: bool
    reason: str
    request: ReviewWorkbenchMessage


class LatestRequestTracker:
    """Track latest requests by owner/kind/fixture in constant time."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._latest: dict[tuple[str, object, str | None], int] = {}
        self._tokens: dict[tuple[str, object, str | None], CancellationToken] = {}

    def register(self, request: ReviewWorkbenchMessage) -> CancellationToken:
        token = CancellationToken(request.cancellation_id or uuid4().hex)
        key = request.owner_key
        with self._lock:
            old = self._tokens.get(key)
            if old is not None and not request.kind.is_durable_write:
                old.cancel()
            self._latest[key] = request.request_id
            self._tokens[key] = token
        return token

    def is_latest(self, request: ReviewWorkbenchMessage) -> bool:
        with self._lock:
            return self._latest.get(request.owner_key) == request.request_id

    def decide(self, envelope: WorkerResultEnvelope) -> CompletionDecision:
        if self.is_latest(envelope.request):
            return CompletionDecision(True, "latest", envelope.request)
        return CompletionDecision(False, "stale_completion_rejected", envelope.request)

    def cancel(self, request: ReviewWorkbenchMessage) -> bool:
        with self._lock:
            token = self._tokens.get(request.owner_key)
        if token is None:
            return False
        token.cancel()
        return True

