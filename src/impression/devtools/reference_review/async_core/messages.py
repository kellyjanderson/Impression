"""Typed task envelopes for reference review background work."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from threading import Lock
from time import monotonic
from types import MappingProxyType
from typing import Any, Mapping


class ReviewTaskKind(str, Enum):
    """Known workbench task lanes."""

    SOURCE_SCAN = "source_scan"
    SOURCE_LOAD = "source_load"
    PREVIEW_BUILD = "preview_build"
    ARTIFACT_GENERATION = "artifact_generation"
    NOTE_WRITE = "note_write"
    PROMOTION_WRITE = "promotion_write"
    CODEX_REQUEST = "codex_request"
    AUDIT_WRITE = "audit_write"

    @property
    def is_durable_write(self) -> bool:
        return self in {
            ReviewTaskKind.NOTE_WRITE,
            ReviewTaskKind.PROMOTION_WRITE,
        }


class RequestIdAllocator:
    """Allocate per-owner monotonic request ids."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, count[int]] = {}

    def next(self, owner: str) -> int:
        if not owner:
            raise ValueError("owner must not be empty")
        with self._lock:
            counter = self._counters.setdefault(owner, count(1))
            return next(counter)


@dataclass(frozen=True)
class ReviewWorkbenchMessage:
    """Immutable task request envelope shared by workbench workers."""

    owner: str
    kind: ReviewTaskKind
    request_id: int
    fixture_id: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None
    cancellation_id: str | None = None
    created_at_monotonic: float = field(default_factory=monotonic)

    def __post_init__(self) -> None:
        if not self.owner:
            raise ValueError("owner must not be empty")
        if self.request_id < 1:
            raise ValueError("request_id must be positive")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        object.__setattr__(self, "kind", ReviewTaskKind(self.kind))
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    @property
    def owner_key(self) -> tuple[str, ReviewTaskKind, str | None]:
        return (self.owner, self.kind, self.fixture_id)

    def to_audit_payload(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "kind": self.kind.value,
            "request_id": self.request_id,
            "fixture_id": self.fixture_id,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class WorkerResultEnvelope:
    """Immutable completion envelope produced by workers."""

    request: ReviewWorkbenchMessage
    ok: bool
    result: Any = None
    error: str | None = None
    completed_at_monotonic: float = field(default_factory=monotonic)

    def __post_init__(self) -> None:
        if self.ok and self.error is not None:
            raise ValueError("successful results must not include error text")
        if not self.ok and not self.error:
            raise ValueError("failed results require error text")

    @property
    def owner(self) -> str:
        return self.request.owner

    @property
    def kind(self) -> ReviewTaskKind:
        return self.request.kind

    @property
    def request_id(self) -> int:
        return self.request.request_id

    @property
    def fixture_id(self) -> str | None:
        return self.request.fixture_id

