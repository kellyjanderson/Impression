from __future__ import annotations

from pathlib import Path

from impression.devtools.reference_review import ReviewNoteStore
from impression.devtools.reference_review.codex_sidecar import (
    SidecarTool,
    ToolPolicyBroker,
    ToolPolicyRecord,
)
from impression.devtools.reference_review.ui import (
    ArtifactPanelViewModel,
    ArtifactRecord,
    CandidateListViewModel,
    ChatStreamPanelViewModel,
    NotesPanelViewModel,
    build_markdown_panel_state,
    build_refusal_display,
)


def _broker(tmp_path: Path, allowed: set[SidecarTool]) -> ToolPolicyBroker:
    return ToolPolicyBroker(
        ToolPolicyRecord(allowed_tools=frozenset(allowed), candidate_root=tmp_path)
    )


def test_markdown_panel_blocks_external_links_and_keeps_local_targets() -> None:
    state = build_markdown_panel_state(
        fixture_id="demo/fixture",
        source_digest="abc",
        markdown_text="Read [local](fixture.md) not [remote](https://example.com).",
    )

    assert state.blocked_message == "1 external link blocked"
    assert state.local_link_targets == ("fixture.md",)
    assert "https://example.com" not in state.rendered.html


def test_artifact_panel_tiles_track_thumbnail_and_request_service_action(tmp_path: Path) -> None:
    thumbnail = tmp_path / "thumb.png"
    thumbnail.write_bytes(b"png")
    panel = ArtifactPanelViewModel(
        (
            ArtifactRecord("png", tmp_path / "dirty.png", thumbnail, diff_status="changed"),
            ArtifactRecord("stl", tmp_path / "dirty.stl", tmp_path / "missing.png"),
        )
    )

    tiles = panel.tiles
    action = panel.request_action(fixture_id="demo/fixture", kind="png")

    assert tiles[0].thumbnail_status == "ready"
    assert tiles[1].thumbnail_status == "missing"
    assert tiles[0].diff_status == "changed"
    assert action == {
        "fixture_id": "demo/fixture",
        "artifact_kind": "png",
        "route": "service-owned",
    }


def test_notes_panel_saves_through_store_and_rejects_full_chat_logs(tmp_path: Path) -> None:
    store = ReviewNoteStore(tmp_path / "notes")
    panel = NotesPanelViewModel("demo/fixture", store)

    panel.edit("Needs a wider lip.")
    saved = panel.save()
    panel.edit("full chat log: no")
    refused = panel.save()

    assert saved.saved
    assert store.load("demo/fixture").body == "Needs a wider lip."
    assert not refused.saved
    assert refused.diagnostic == "note_contains_disallowed_chat_log"
    assert panel.state.save_failure == "note_contains_disallowed_chat_log"


def test_chat_stream_panel_routes_through_broker_throttles_and_cancels(
    tmp_path: Path,
) -> None:
    broker = _broker(tmp_path, {SidecarTool.READ_FIXTURE_CONTEXT})
    broker.register(SidecarTool.READ_FIXTURE_CONTEXT, lambda request: {"fixture": request.fixture_id})
    panel = ChatStreamPanelViewModel(throttle_seconds=1.0)

    started = panel.start(fixture_id="demo/fixture", broker=broker)
    buffered = panel.append_chunk("hello", now=started.last_flush + 0.5)
    flushed = panel.append_chunk(" world", now=started.last_flush + 1.5)
    cancelled = panel.cancel_on_fixture_change("demo/other")

    assert started.refusal is None
    assert buffered.visible_text == ""
    assert buffered.buffered_text == "hello"
    assert flushed.visible_text == "hello world"
    assert cancelled is not None
    assert cancelled.cancelled


def test_chat_stream_panel_surfaces_broker_refusals(tmp_path: Path) -> None:
    panel = ChatStreamPanelViewModel()

    state = panel.start(fixture_id="demo/fixture", broker=_broker(tmp_path, set()))

    assert state.refusal == "tool_not_allowed"


def test_candidate_list_uses_only_broker_approved_paths_and_adopts_with_confirmation(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "demo__fixture_candidate.py"
    candidate.write_text("def build():\n    return None\n")
    broker = _broker(
        tmp_path,
        {SidecarTool.LIST_CANDIDATE_OUTPUTS, SidecarTool.REQUEST_CANDIDATE_ADOPTION},
    )
    broker.register(SidecarTool.LIST_CANDIDATE_OUTPUTS, lambda request: (candidate,))
    broker.register(
        SidecarTool.REQUEST_CANDIDATE_ADOPTION,
        lambda request: {"accepted": request.payload["human_confirmed"]},
    )
    panel = CandidateListViewModel(fixture_id="demo/fixture", broker=broker)

    items = panel.refresh()
    panel.select(candidate)
    unconfirmed = panel.request_adoption(human_confirmed=False)
    confirmed = panel.request_adoption(human_confirmed=True)

    assert items[0].display_path == "demo__fixture_candidate.py"
    assert not unconfirmed.accepted
    assert unconfirmed.diagnostic == "human_confirmation_required"
    assert confirmed.accepted


def test_refusal_display_sanitizes_policy_diagnostic() -> None:
    hidden = build_refusal_display(None)
    visible = build_refusal_display("tool_not_allowed password=/tmp/secret")

    assert not hidden.visible
    assert visible.visible
    assert visible.message == "tool not allowed <redacted>"
    assert visible.action_label == "Review policy"
