"""UI handoff adapter and task error sanitization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .messages import WorkerResultEnvelope

_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9_])/(?:[^\\s:]+/)*[^\\s:]+")


def sanitize_error_text(text: str, *, cwd: Path | None = None, home: Path | None = None) -> str:
    sanitized = text
    if cwd is not None:
        sanitized = sanitized.replace(str(cwd), "<workspace>")
    if home is not None:
        sanitized = sanitized.replace(str(home), "<home>")
    return _ABSOLUTE_PATH.sub("<path>", sanitized)


@dataclass(frozen=True)
class SanitizedDiagnostic:
    owner: str
    request_id: int
    fixture_id: str | None
    message: str


class UICompletionBridge:
    """Forward worker completions to an injected UI-thread handoff callback."""

    def __init__(
        self,
        handoff: Callable[[WorkerResultEnvelope], None],
        *,
        cwd: Path | None = None,
        home: Path | None = None,
    ) -> None:
        self._handoff = handoff
        self._cwd = cwd
        self._home = home

    def post(self, envelope: WorkerResultEnvelope) -> WorkerResultEnvelope:
        if not envelope.ok and envelope.error is not None:
            envelope = WorkerResultEnvelope(
                request=envelope.request,
                ok=False,
                error=sanitize_error_text(envelope.error, cwd=self._cwd, home=self._home),
                completed_at_monotonic=envelope.completed_at_monotonic,
            )
        self._handoff(envelope)
        return envelope

    def diagnostic_for(self, envelope: WorkerResultEnvelope) -> SanitizedDiagnostic | None:
        if envelope.ok or envelope.error is None:
            return None
        return SanitizedDiagnostic(
            owner=envelope.owner,
            request_id=envelope.request_id,
            fixture_id=envelope.fixture_id,
            message=envelope.error,
        )

