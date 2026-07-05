"""Structured audit events for reference review async tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any, Callable, Mapping

from .messages import ReviewWorkbenchMessage

_SENSITIVE_KEY_PARTS = ("secret", "token", "password", "credential")


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(part in key_text.lower() for part in _SENSITIVE_KEY_PARTS):
                redacted[key_text] = "<redacted>"
            else:
                redacted[key_text] = _redact(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value[:25]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(frozen=True)
class AuditEvent:
    """JSON-compatible, fixture-scoped task event."""

    event: str
    task_kind: str
    request_id: int
    fixture_id: str | None
    owner: str
    details: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "task_kind": self.task_kind,
            "request_id": self.request_id,
            "fixture_id": self.fixture_id,
            "owner": self.owner,
            "details": _redact(dict(self.details)),
            "timestamp": self.timestamp,
        }


def build_audit_event(
    event: str,
    request: ReviewWorkbenchMessage,
    *,
    details: Mapping[str, Any] | None = None,
) -> AuditEvent:
    return AuditEvent(
        event=event,
        task_kind=request.kind.value,
        request_id=request.request_id,
        fixture_id=request.fixture_id,
        owner=request.owner,
        details=details or {},
    )


class AuditEmitter:
    """Non-blocking audit hook with a pluggable sink."""

    def __init__(self, sink: Callable[[AuditEvent], None] | None = None) -> None:
        self._sink = sink
        self.events: list[AuditEvent] = []

    def emit(self, event: AuditEvent) -> bool:
        self.events.append(event)
        if self._sink is None:
            return True
        try:
            self._sink(event)
        except Exception:
            return False
        return True

