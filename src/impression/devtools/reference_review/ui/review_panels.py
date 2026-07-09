"""Artifact, notes, Codex, candidate, and refusal panel view models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Iterable

from ..async_core.qt_handoff import sanitize_error_text
from ..codex_sidecar import SidecarTool, ToolPolicyBroker, ToolRequestRecord
from ..lifecycle import NoteWriteResult, ReviewNoteRecord, ReviewNoteStore, ReviewState
from .markdown_context import BlockedLinkDiagnostic, MarkdownContextRenderer, RenderedMarkdownContext


@dataclass(frozen=True)
class MarkdownPanelState:
    fixture_id: str
    rendered: RenderedMarkdownContext
    local_link_targets: tuple[str, ...] = ()

    @property
    def blocked_message(self) -> str | None:
        if not self.rendered.blocked_links:
            return None
        return f"{len(self.rendered.blocked_links)} external link blocked"


def build_markdown_panel_state(
    *,
    fixture_id: str,
    source_digest: str,
    markdown_text: str,
    renderer: MarkdownContextRenderer | None = None,
) -> MarkdownPanelState:
    renderer = renderer or MarkdownContextRenderer()
    rendered = renderer.render(
        fixture_id=fixture_id,
        source_digest=source_digest,
        text=markdown_text,
    )
    local_links = tuple(
        target
        for target in _extract_markdown_targets(markdown_text)
        if not target.startswith(("http://", "https://"))
    )
    return MarkdownPanelState(fixture_id, rendered, local_links)


@dataclass(frozen=True)
class ArtifactRecord:
    kind: str
    path: Path
    thumbnail_path: Path | None = None
    diff_status: str = "unknown"


@dataclass(frozen=True)
class ArtifactTileViewModel:
    kind: str
    display_name: str
    thumbnail_status: str
    diff_status: str
    action_enabled: bool = True


class ArtifactPanelViewModel:
    def __init__(self, artifacts: Iterable[ArtifactRecord] = ()) -> None:
        self.artifacts = tuple(artifacts)

    @property
    def tiles(self) -> tuple[ArtifactTileViewModel, ...]:
        return tuple(self._tile_for(artifact) for artifact in self.artifacts)

    @property
    def empty(self) -> bool:
        return not self.artifacts

    def request_action(self, *, fixture_id: str, kind: str) -> dict[str, str]:
        return {"fixture_id": fixture_id, "artifact_kind": kind, "route": "service-owned"}

    def _tile_for(self, artifact: ArtifactRecord) -> ArtifactTileViewModel:
        thumbnail_status = "loading"
        if artifact.thumbnail_path is not None:
            thumbnail_status = "ready" if artifact.thumbnail_path.exists() else "missing"
        return ArtifactTileViewModel(
            kind=artifact.kind,
            display_name=artifact.path.name,
            thumbnail_status=thumbnail_status,
            diff_status=artifact.diff_status,
        )


@dataclass(frozen=True)
class NotesPanelState:
    fixture_id: str
    body: str = ""
    status: ReviewState = ReviewState.NEEDS_WORK
    dirty: bool = False
    save_failure: str | None = None


class NotesPanelViewModel:
    def __init__(self, fixture_id: str, store: ReviewNoteStore) -> None:
        self.fixture_id = fixture_id
        self.store = store
        note = store.load(fixture_id)
        self.state = NotesPanelState(
            fixture_id=fixture_id,
            body=note.body if note is not None else "",
            status=note.status if note is not None else ReviewState.NEEDS_WORK,
        )

    def edit(self, body: str) -> NotesPanelState:
        self.state = NotesPanelState(
            fixture_id=self.fixture_id,
            body=body,
            status=ReviewState.NEEDS_WORK,
            dirty=True,
        )
        return self.state

    def flag_blocked(self) -> NotesPanelState:
        self.state = NotesPanelState(
            fixture_id=self.fixture_id,
            body=self.state.body,
            status=ReviewState.BLOCKED,
            dirty=True,
        )
        return self.state

    def save(self) -> NoteWriteResult:
        if "full chat log" in self.state.body.lower():
            self.state = NotesPanelState(
                fixture_id=self.fixture_id,
                body=self.state.body,
                status=self.state.status,
                dirty=True,
                save_failure="note_contains_disallowed_chat_log",
            )
            return NoteWriteResult(False, self.store.note_path(self.fixture_id), self.state.save_failure)
        result = self.store.save(
            ReviewNoteRecord(
                fixture_id=self.fixture_id,
                status=self.state.status,
                body=self.state.body,
            )
        )
        self.state = NotesPanelState(
            fixture_id=self.fixture_id,
            body=self.state.body,
            status=self.state.status,
            dirty=not result.saved,
            save_failure=result.diagnostic,
        )
        return result


@dataclass(frozen=True)
class ChatStreamState:
    fixture_id: str
    request_id: int
    visible_text: str = ""
    buffered_text: str = ""
    cancelled: bool = False
    refusal: str | None = None
    last_flush: float = field(default_factory=monotonic)


class ChatStreamPanelViewModel:
    def __init__(self, *, throttle_seconds: float = 0.05) -> None:
        self._next_request_id = 1
        self._throttle_seconds = throttle_seconds
        self.state: ChatStreamState | None = None

    def start(self, *, fixture_id: str, broker: ToolPolicyBroker) -> ChatStreamState:
        request_id = self._next_request_id
        self._next_request_id += 1
        result = broker.handle(
            ToolRequestRecord(SidecarTool.READ_FIXTURE_CONTEXT, fixture_id=fixture_id)
        )
        refusal = None if result.accepted else result.diagnostic
        self.state = ChatStreamState(fixture_id=fixture_id, request_id=request_id, refusal=refusal)
        return self.state

    def append_chunk(self, text: str, *, now: float | None = None) -> ChatStreamState:
        if self.state is None:
            raise ValueError("stream_not_started")
        now = monotonic() if now is None else now
        buffered = self.state.buffered_text + text
        visible = self.state.visible_text
        last_flush = self.state.last_flush
        if now - self.state.last_flush >= self._throttle_seconds:
            visible += buffered
            buffered = ""
            last_flush = now
        self.state = ChatStreamState(
            fixture_id=self.state.fixture_id,
            request_id=self.state.request_id,
            visible_text=visible,
            buffered_text=buffered,
            cancelled=self.state.cancelled,
            refusal=self.state.refusal,
            last_flush=last_flush,
        )
        return self.state

    def cancel_on_fixture_change(self, fixture_id: str) -> ChatStreamState | None:
        if self.state is None or self.state.fixture_id == fixture_id:
            return self.state
        self.state = ChatStreamState(
            fixture_id=self.state.fixture_id,
            request_id=self.state.request_id,
            visible_text=self.state.visible_text,
            buffered_text=self.state.buffered_text,
            cancelled=True,
            refusal=self.state.refusal,
            last_flush=self.state.last_flush,
        )
        return self.state


@dataclass(frozen=True)
class CandidateListItem:
    fixture_id: str
    display_path: str
    candidate_path: Path
    selected: bool = False


@dataclass(frozen=True)
class CandidateAdoptionRequest:
    fixture_id: str
    candidate_path: Path
    human_confirmed: bool
    accepted: bool
    diagnostic: str | None = None


class CandidateListViewModel:
    def __init__(self, *, fixture_id: str, broker: ToolPolicyBroker) -> None:
        self.fixture_id = fixture_id
        self.broker = broker
        self.items: tuple[CandidateListItem, ...] = ()
        self.selected_path: Path | None = None

    def refresh(self) -> tuple[CandidateListItem, ...]:
        result = self.broker.handle(
            ToolRequestRecord(SidecarTool.LIST_CANDIDATE_OUTPUTS, fixture_id=self.fixture_id)
        )
        if not result.accepted:
            self.items = ()
            return self.items
        paths = tuple(Path(path) for path in result.result)
        self.items = tuple(
            CandidateListItem(
                fixture_id=self.fixture_id,
                display_path=path.name,
                candidate_path=path,
                selected=path == self.selected_path,
            )
            for path in paths
        )
        return self.items

    def select(self, candidate_path: Path) -> tuple[CandidateListItem, ...]:
        approved_paths = {item.candidate_path for item in self.items}
        if candidate_path not in approved_paths:
            raise ValueError("candidate_not_broker_approved")
        self.selected_path = candidate_path
        self.items = tuple(
            CandidateListItem(
                fixture_id=item.fixture_id,
                display_path=item.display_path,
                candidate_path=item.candidate_path,
                selected=item.candidate_path == candidate_path,
            )
            for item in self.items
        )
        return self.items

    def request_adoption(self, *, human_confirmed: bool) -> CandidateAdoptionRequest:
        if self.selected_path is None:
            return CandidateAdoptionRequest(
                self.fixture_id,
                Path(),
                human_confirmed,
                False,
                "candidate_not_selected",
            )
        if not human_confirmed:
            return CandidateAdoptionRequest(
                self.fixture_id,
                self.selected_path,
                False,
                False,
                "human_confirmation_required",
            )
        result = self.broker.handle(
            ToolRequestRecord(
                SidecarTool.REQUEST_CANDIDATE_ADOPTION,
                fixture_id=self.fixture_id,
                payload={
                    "candidate_path": self.selected_path.as_posix(),
                    "human_confirmed": human_confirmed,
                },
            )
        )
        return CandidateAdoptionRequest(
            self.fixture_id,
            self.selected_path,
            True,
            result.accepted,
            result.diagnostic,
        )


@dataclass(frozen=True)
class RefusalDisplayState:
    visible: bool
    message: str | None = None
    action_label: str | None = None


def build_refusal_display(diagnostic: str | None) -> RefusalDisplayState:
    if not diagnostic:
        return RefusalDisplayState(False)
    message = sanitize_error_text(diagnostic).replace("_", " ")
    for marker in ("password=", "token=", "secret="):
        if marker in message.lower():
            head = message.lower().split(marker, 1)[0].strip()
            message = f"{head} <redacted>".strip()
            break
    return RefusalDisplayState(True, message=message, action_label="Review policy")


def _extract_markdown_targets(markdown_text: str) -> tuple[str, ...]:
    targets: list[str] = []
    for part in markdown_text.split("](")[1:]:
        targets.append(part.split(")", 1)[0])
    return tuple(targets)
