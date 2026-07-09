from __future__ import annotations

from pathlib import Path

from impression.devtools.reference_review import ReviewSourceModelRecord
from impression.devtools.reference_review.ui import (
    FixtureQueueViewModel,
    MarkdownContextRenderer,
    PreviewAdapterMode,
    PreviewBridgeController,
    choose_preview_adapter,
)


def _record(tmp_path: Path, fixture_id: str, expected_output: str | None = None) -> ReviewSourceModelRecord:
    path = tmp_path / f"{fixture_id.replace('/', '_')}.py"
    path.write_text("def build():\n    return None\n")
    return ReviewSourceModelRecord(
        fixture_id=fixture_id,
        feature_name="demo",
        source_path=path,
        expected_output=expected_output,
    )


def test_queue_selects_first_dirty_fixture_and_uses_display_paths(tmp_path: Path) -> None:
    clean = _record(tmp_path, "demo/clean")
    dirty = _record(tmp_path, "demo/dirty", expected_output="dirty.png")
    queue = FixtureQueueViewModel(
        (clean, dirty),
        statuses={"demo/clean": "approved", "demo/dirty": "dirty"},
    )

    assert queue.selected_context.fixture_id == "demo/dirty"
    assert queue.selected_context.source_display_path == "demo_dirty.py"
    assert str(tmp_path) not in queue.selected_context.source_display_path
    assert queue.items[0].status == "approved"


def test_queue_navigation_and_empty_state(tmp_path: Path) -> None:
    first = _record(tmp_path, "demo/first")
    second = _record(tmp_path, "demo/second")
    queue = FixtureQueueViewModel((first, second), statuses={"demo/first": "approved"})
    empty = FixtureQueueViewModel(())

    assert queue.selected_context.fixture_id == "demo/second"
    assert queue.previous().fixture_id == "demo/first"
    assert queue.next().fixture_id == "demo/second"
    assert queue.select_fixture("missing") is False
    assert empty.selected_context.empty


def test_preview_adapter_decision_prefers_embedded_review_surface() -> None:
    decision = choose_preview_adapter(embedded_available=True, supervised_external_available=True)

    assert decision.selected is PreviewAdapterMode.EMBEDDED_RENDERED
    assert decision.rejected == ("supervised-external:not-in-review-surface",)


def test_preview_load_binding_rejects_stale_completions_and_routes_camera(
    tmp_path: Path,
) -> None:
    first = _record(tmp_path, "demo/first")
    second = _record(tmp_path, "demo/second")
    bridge = PreviewBridgeController()

    stale = bridge.bind_selected_fixture(first)
    current = bridge.bind_selected_fixture(second)
    stale_state = bridge.complete(stale.request)
    current_state = bridge.complete(current.request)
    camera_state = bridge.camera_command("reset")

    assert stale_state.fixture_id == "demo/second"
    assert not stale_state.ready
    assert current_state.ready
    assert camera_state.camera_commands == ("reset",)


def test_markdown_renderer_blocks_external_links_and_caches_render() -> None:
    renderer = MarkdownContextRenderer()

    first = renderer.render(
        fixture_id="demo/fixture",
        source_digest="abc",
        text="See [local](fixture.md) and [external](https://example.com).",
    )
    second = renderer.render(
        fixture_id="demo/fixture",
        source_digest="abc",
        text="See [local](fixture.md) and [external](https://example.com).",
    )

    assert first is second
    assert len(first.blocked_links) == 1
    assert first.blocked_links[0].reason == "external-link-blocked"
    assert "https://example.com" not in first.html
