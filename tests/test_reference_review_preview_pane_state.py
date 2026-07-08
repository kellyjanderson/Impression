from __future__ import annotations

from pathlib import Path

from impression.devtools.reference_review.ui import (
    PreviewWidgetPayloadState,
    build_preview_wrapper_smoke_record,
    preview_pane_empty_state,
    preview_pane_failure_state,
    preview_pane_loading_state,
    preview_pane_ready_state,
    resolve_preview_toolbar_enabled,
    route_preview_toolbar_command,
)


def test_preview_pane_state_transitions_empty_loading_ready() -> None:
    empty = preview_pane_empty_state()
    loading = preview_pane_loading_state("fixture/box")
    ready = preview_pane_ready_state(
        PreviewWidgetPayloadState(
            generation=3,
            fixture_id="fixture/box",
            ready=True,
        )
    )

    assert empty.mode == "empty"
    assert empty.message == "No fixture selected."
    assert not empty.toolbar_enabled
    assert loading.mode == "loading"
    assert loading.fixture_id == "fixture/box"
    assert loading.message == "Loading preview..."
    assert not loading.toolbar_enabled
    assert ready.mode == "ready"
    assert ready.fixture_id == "fixture/box"
    assert ready.toolbar_enabled


def test_preview_pane_failure_state_redacts_diagnostic_paths(tmp_path: Path) -> None:
    state = preview_pane_failure_state(
        "fixture/box",
        f"failed under {tmp_path}/private.py and /var/tmp/token",
        cwd=tmp_path,
    )

    assert state.mode == "failure"
    assert state.fixture_id == "fixture/box"
    assert state.message == "Preview unavailable."
    assert not state.toolbar_enabled
    assert state.diagnostic is not None
    assert str(tmp_path) not in state.diagnostic
    assert "/var/tmp" not in state.diagnostic


def test_preview_toolbar_commands_are_disabled_until_ready() -> None:
    class FakeWidget:
        def reset_view(self) -> None:
            raise AssertionError("disabled command should not reach widget")

    state = preview_pane_loading_state("fixture/box")
    routed = route_preview_toolbar_command(FakeWidget(), state, "reset")

    assert not resolve_preview_toolbar_enabled(state)
    assert not routed.executed
    assert routed.diagnostic == "preview-toolbar-disabled"


def test_preview_toolbar_routes_reset_and_camera_presets_to_widget() -> None:
    class FakeWidget:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def reset_view(self) -> None:
            self.commands.append("reset")

        def apply_camera_preset(self, preset: str) -> None:
            self.commands.append(preset)

    widget = FakeWidget()
    state = preview_pane_ready_state(
        PreviewWidgetPayloadState(
            generation=1,
            fixture_id="fixture/box",
            ready=True,
        )
    )

    reset = route_preview_toolbar_command(widget, state, "reset")
    top = route_preview_toolbar_command(widget, state, "top")
    unsupported = route_preview_toolbar_command(widget, state, "spin")

    assert resolve_preview_toolbar_enabled(state)
    assert reset.executed
    assert top.executed
    assert widget.commands == ["reset", "top"]
    assert not unsupported.executed
    assert unsupported.diagnostic == "unsupported-preview-toolbar-command"


def test_preview_wrapper_smoke_record_uses_dirty_impress_fixture_file() -> None:
    record = build_preview_wrapper_smoke_record()

    assert record.expected_fixture_type == ".impress"
    assert record.verifies_lifecycle
    assert record.fixture_file.as_posix() == "tests/reference_review_fixtures/dirty-impress-fixtures.json"
    assert record.command == (
        ".venv/bin/impression-reference-review",
        "--fixture-file",
        "tests/reference_review_fixtures/dirty-impress-fixtures.json",
    )
