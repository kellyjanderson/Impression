"""Preview bridge decision and selected-fixture load binding contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..async_core import LatestRequestTracker, ReviewTaskKind, ReviewWorkbenchMessage
from ..source_registry import ReviewSourceModelRecord

if TYPE_CHECKING:
    from .interactive_preview import InteractivePreviewLaunch


class PreviewAdapterMode(str, Enum):
    SUPERVISED_EXTERNAL = "supervised-external"
    EMBEDDED_PYVISTAQT = "embedded-pyvistaqt"


@dataclass(frozen=True)
class PreviewAdapterDecision:
    selected: PreviewAdapterMode
    rejected: tuple[str, ...] = ()


def choose_preview_adapter(
    *,
    embedded_available: bool = False,
    supervised_external_available: bool = True,
) -> PreviewAdapterDecision:
    rejected: list[str] = []
    if supervised_external_available:
        if not embedded_available:
            rejected.append("embedded-pyvistaqt:not-available-or-too-coupled")
        return PreviewAdapterDecision(PreviewAdapterMode.SUPERVISED_EXTERNAL, tuple(rejected))
    if embedded_available:
        return PreviewAdapterDecision(PreviewAdapterMode.EMBEDDED_PYVISTAQT, ("supervised-external:unavailable",))
    return PreviewAdapterDecision(
        PreviewAdapterMode.SUPERVISED_EXTERNAL,
        ("supervised-external:unavailable", "embedded-pyvistaqt:unavailable"),
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


class InteractivePreviewBridge:
    """QML bridge for launching the supervised external PyVista preview."""

    def __init__(
        self,
        records: tuple[ReviewSourceModelRecord, ...],
        *,
        launcher: Callable[[Path, str], "InteractivePreviewLaunch"] | None = None,
    ) -> None:
        from PySide6.QtCore import QObject, Slot

        class _Bridge(QObject):
            def __init__(self, outer: "InteractivePreviewBridge") -> None:
                super().__init__()
                self._outer = outer

            @Slot(str, result=str)
            def openPreview(self, fixture_id: str) -> str:
                return self._outer.open_preview(fixture_id)

        self._artifact_paths = {
            record.fixture_id: record.artifact_paths[0] for record in records if record.artifact_paths
        }
        self._launcher = launcher or _launch_interactive_stl_preview
        self.qt_object = _Bridge(self)
        self.launches: list[InteractivePreviewLaunch] = []

    def open_preview(self, fixture_id: str) -> str:
        artifact_path = self._artifact_paths.get(fixture_id)
        if artifact_path is None:
            return "missing-artifact"
        launch = self._launcher(artifact_path, f"Impression Preview - {fixture_id}")
        self.launches.append(launch)
        return "launched" if launch.accepted else launch.diagnostic or "preview-launch-failed"


def _launch_interactive_stl_preview(path: Path, title: str) -> "InteractivePreviewLaunch":
    from .interactive_preview import launch_interactive_stl_preview

    return launch_interactive_stl_preview(path, title=title)
