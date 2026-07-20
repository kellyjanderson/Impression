"""Preview bridge decision and selected-fixture load binding contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..async_core import LatestRequestTracker, ReviewTaskKind, ReviewWorkbenchMessage
from ..source_registry import ReviewSourceModelRecord


class PreviewAdapterMode(str, Enum):
    SUPERVISED_EXTERNAL = "supervised-external"
    EMBEDDED_RENDERED = "embedded-rendered"


@dataclass(frozen=True)
class PreviewAdapterDecision:
    selected: PreviewAdapterMode
    rejected: tuple[str, ...] = ()


def choose_preview_adapter(
    *,
    embedded_available: bool = True,
    supervised_external_available: bool = True,
) -> PreviewAdapterDecision:
    rejected: list[str] = []
    if embedded_available:
        if supervised_external_available:
            rejected.append("supervised-external:not-in-review-surface")
        return PreviewAdapterDecision(PreviewAdapterMode.EMBEDDED_RENDERED, tuple(rejected))
    if supervised_external_available:
        return PreviewAdapterDecision(
            PreviewAdapterMode.SUPERVISED_EXTERNAL,
            ("embedded-rendered:unavailable",),
        )
    return PreviewAdapterDecision(
        PreviewAdapterMode.SUPERVISED_EXTERNAL,
        ("supervised-external:unavailable", "embedded-rendered:unavailable"),
    )


@dataclass(frozen=True)
class PreviewBridgeState:
    fixture_id: str | None = None
    ready: bool = False
    diagnostic: str | None = None
    camera_commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreviewLoadBinding:
    request: ReviewWorkbenchMessage
    source: ReviewSourceModelRecord


class PreviewBridgeController:
    def __init__(self) -> None:
        self._tracker = LatestRequestTracker()
        self.state = PreviewBridgeState()
        self._next_request_id = 1
        self._current_request_id: int | None = None

    def bind_selected_fixture(self, source: ReviewSourceModelRecord) -> PreviewLoadBinding:
        request = ReviewWorkbenchMessage(
            owner="preview-bridge",
            kind=ReviewTaskKind.PREVIEW_BUILD,
            request_id=self._next_request_id,
            fixture_id=source.fixture_id,
        )
        self._next_request_id += 1
        self._current_request_id = request.request_id
        self._tracker.register(request)
        self.state = PreviewBridgeState(fixture_id=source.fixture_id, ready=False)
        return PreviewLoadBinding(request=request, source=source)

    def complete(self, request: ReviewWorkbenchMessage, *, diagnostic: str | None = None) -> PreviewBridgeState:
        if request.request_id != self._current_request_id or not self._tracker.is_latest(request):
            return self.state
        self.state = PreviewBridgeState(
            fixture_id=request.fixture_id,
            ready=diagnostic is None,
            diagnostic=diagnostic,
            camera_commands=self.state.camera_commands,
        )
        return self.state

    def camera_command(self, command: str) -> PreviewBridgeState:
        if command not in {"orbit", "pan", "zoom", "reset"}:
            raise ValueError("unsupported_camera_command")
        self.state = PreviewBridgeState(
            fixture_id=self.state.fixture_id,
            ready=self.state.ready,
            diagnostic=self.state.diagnostic,
            camera_commands=(*self.state.camera_commands, command),
        )
        return self.state
