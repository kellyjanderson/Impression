"""Coalesced render-command records for the Reference Review preview pane."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from ..preview_payload import PreviewPayload
from .preview_controls import PreviewDisplayOptions


PreviewRenderIdentity = tuple[str, int, str, int]


class PreviewRenderCommandKind(str, Enum):
    """Renderer command categories accepted by the preview widget."""

    LOADING = "loading"
    CLEAR = "clear"
    FAILURE = "failure"
    PAYLOAD = "payload"
    DISPLAY = "display"
    RESET_CAMERA = "reset_camera"

    @property
    def lane(self) -> str:
        if self in {
            PreviewRenderCommandKind.LOADING,
            PreviewRenderCommandKind.CLEAR,
            PreviewRenderCommandKind.FAILURE,
        }:
            return "lifecycle"
        if self is PreviewRenderCommandKind.PAYLOAD:
            return "payload"
        if self is PreviewRenderCommandKind.DISPLAY:
            return "display"
        return "camera"


@dataclass(frozen=True)
class PreviewRenderCommand:
    """Immutable request to mutate the preview renderer on the Qt thread."""

    kind: PreviewRenderCommandKind
    owner: str | None = None
    request_id: int | None = None
    fixture_id: str | None = None
    generation: int | None = None
    payload: PreviewPayload | None = None
    display_options: PreviewDisplayOptions | None = None
    message: str = ""
    diagnostic: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))

    @classmethod
    def loading(
        cls,
        message: str = "Loading preview...",
        *,
        identity: PreviewRenderIdentity | None = None,
    ) -> "PreviewRenderCommand":
        return cls._from_identity(
            PreviewRenderCommandKind.LOADING,
            identity,
            message=message,
        )

    @classmethod
    def clear(
        cls,
        message: str = "No fixture selected.",
        *,
        identity: PreviewRenderIdentity | None = None,
    ) -> "PreviewRenderCommand":
        return cls._from_identity(
            PreviewRenderCommandKind.CLEAR,
            identity,
            message=message,
        )

    @classmethod
    def failure(
        cls,
        message: str,
        *,
        diagnostic: str | None = None,
        identity: PreviewRenderIdentity | None = None,
    ) -> "PreviewRenderCommand":
        return cls._from_identity(
            PreviewRenderCommandKind.FAILURE,
            identity,
            message=message,
            diagnostic=diagnostic,
        )

    @classmethod
    def payload_ready(
        cls,
        payload: PreviewPayload,
        *,
        display_options: PreviewDisplayOptions | None = None,
    ) -> "PreviewRenderCommand":
        return cls(
            kind=PreviewRenderCommandKind.PAYLOAD,
            owner=payload.request.owner,
            request_id=payload.request.request_id,
            fixture_id=payload.request.fixture_id,
            generation=payload.request.generation,
            payload=payload,
            display_options=display_options,
        )

    @classmethod
    def display(
        cls,
        options: PreviewDisplayOptions,
        *,
        identity: PreviewRenderIdentity | None = None,
    ) -> "PreviewRenderCommand":
        return cls._from_identity(
            PreviewRenderCommandKind.DISPLAY,
            identity,
            display_options=options,
        )

    @classmethod
    def reset_camera(
        cls,
        *,
        identity: PreviewRenderIdentity | None = None,
    ) -> "PreviewRenderCommand":
        return cls._from_identity(PreviewRenderCommandKind.RESET_CAMERA, identity)

    @classmethod
    def _from_identity(
        cls,
        kind: PreviewRenderCommandKind,
        identity: PreviewRenderIdentity | None,
        **kwargs: Any,
    ) -> "PreviewRenderCommand":
        if identity is None:
            return cls(kind=kind, **kwargs)
        owner, request_id, fixture_id, generation = identity
        return cls(
            kind=kind,
            owner=owner,
            request_id=request_id,
            fixture_id=fixture_id,
            generation=generation,
            **kwargs,
        )

    @property
    def identity(self) -> PreviewRenderIdentity | None:
        if (
            self.owner is None
            or self.request_id is None
            or self.fixture_id is None
            or self.generation is None
        ):
            return None
        return (self.owner, self.request_id, self.fixture_id, self.generation)

    @property
    def lane(self) -> str:
        return self.kind.lane


@dataclass(frozen=True)
class PreviewRenderCommandResult:
    """Outcome of enqueuing or applying a preview render command."""

    command: PreviewRenderCommand
    accepted: bool
    status: str
    replaced: bool = False
    ready: bool = False
    diagnostic: str | None = None


@dataclass(frozen=True)
class PreviewRenderQueueState:
    """Inspectable state for the coalescing render queue."""

    pending_count: int
    pending_lanes: tuple[str, ...]


class PreviewRenderCommandQueue:
    """Tiny latest-wins queue drained only by the preview widget."""

    _LANE_ORDER = ("lifecycle", "payload", "display", "camera")

    def __init__(self) -> None:
        self._pending: dict[str, PreviewRenderCommand] = {}

    @property
    def state(self) -> PreviewRenderQueueState:
        return PreviewRenderQueueState(
            pending_count=len(self._pending),
            pending_lanes=tuple(
                lane for lane in self._LANE_ORDER if lane in self._pending
            ),
        )

    def enqueue(self, command: PreviewRenderCommand) -> PreviewRenderCommandResult:
        lane = command.lane
        replaced = lane in self._pending
        self._pending[lane] = command
        return PreviewRenderCommandResult(
            command=command,
            accepted=True,
            status="queued",
            replaced=replaced,
        )

    def drain(self) -> tuple[PreviewRenderCommand, ...]:
        commands: list[PreviewRenderCommand] = []
        for lane in self._LANE_ORDER:
            command = self._pending.pop(lane, None)
            if command is not None:
                commands.append(command)
        return tuple(commands)

    def clear(self) -> PreviewRenderQueueState:
        self._pending.clear()
        return self.state
