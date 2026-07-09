# Reference Review Test Spec 75: Preview Render Command Queue (v1.0)

## Overview

Verify the ad hoc preview render command queue remediation for Reference
Review. The tests must prove that preview worker completions and UI actions are
decoupled from direct renderer mutation, stale completions are harmless, and
rapid interaction is bounded.

## Paired Specification

- [Reference Review Spec 75: Preview Render Command Queue](../specifications/reference-review-75-preview-render-command-queue-v1_0.md)

## Test Scope

This test specification covers:

- command record validation
- queue coalescing
- stale success/failure rejection
- current failure presentation
- display-option coalescing
- renderer lifecycle preservation
- shell integration for rapid selection and rapid display-control changes

## Automated Tests

Add focused tests for:

- `PreviewRenderCommand` requires a known kind and carries identity fields.
- `PreviewRenderCommandQueue` keeps only the latest display command.
- `PreviewRenderCommandQueue` replaces stale payload commands with the newest
  current payload command.
- draining the queue applies at most one renderer scene update for repeated
  display toggles.
- display-option commands update an existing decoded dataset without re-reading
  the payload JSON.
- a stale successful payload completion does not call `set_preview_payload`.
- a stale failed payload completion does not call `clear_preview`.
- an exception from an old future does not clear the current preview.
- current payload failure disables display controls and shows sanitized
  unavailable text.
- closing the window clears pending commands before renderer disposal.

## Integration Tests

Extend existing Reference Review UI shell tests to cover:

- rapid selection of two fixtures where the first completion arrives last
- rapid toggling of preview display controls while a payload is loading
- rapid toggling of preview display controls after a payload is ready
- renderer surface is created once and reused across queued payload/display
  commands
- fake renderer receives one coherent final command after coalesced toggles

## Manual Smoke

Run:

```bash
.venv/bin/impression-reference-review --fixture-file tests/reference_review_fixtures/dirty-impress-fixtures.json
```

Then verify:

- rapidly select several fixtures
- rapidly toggle authored/inspection color
- rapidly toggle object edges and triangle wireframe
- rapidly toggle bounds grid and axis triad
- preview remains responsive
- no old completion clears or overwrites the selected fixture
- renderer does not beachball during interaction

## Required Validation Commands

Run:

```bash
.venv/bin/python -m pytest tests/test_reference_review_async_core.py tests/test_reference_review_ui_shell.py tests/test_reference_review_preview_payload_controller.py -q
.venv/bin/python -m pytest tests/test_preview_controller.py -q
git diff --check
```

## Acceptance

This test specification is complete when:

- all automated tests above are implemented or explicitly mapped to equivalent
  existing coverage
- the required validation commands pass
- the manual smoke launch is documented with observed behavior
