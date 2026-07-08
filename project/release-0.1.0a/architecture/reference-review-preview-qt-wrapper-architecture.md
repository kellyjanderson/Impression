# Reference Review Preview Qt Wrapper Architecture

## Overview

This supplemental architecture document defines the embedded Qt wrapper around
the actual Impression preview engine.

The wrapper is a host, not a renderer rewrite. Its job is to place the real
preview inside the Reference Review Workbench and keep renderer lifetime stable
while fixture selection changes.

## Parent And Related Architecture

- [Reference Review Preview Remediation Plan](reference-review-preview-remediation-plan.md)
- [Reference Review Preview Engine Sharing Architecture](reference-review-preview-engine-sharing-architecture.md)
- [Reference Review Preview Payload Boundary Architecture](reference-review-preview-payload-boundary-architecture.md)
- [Reference Review Qt Workbench UI](reference-review-qt-workbench-ui.md)
- [Reference Review Async Concurrency](reference-review-async-concurrency.md)

## Target Components

- `ImpressionPreviewWidget`: Qt widget that embeds one live preview surface.
- `ReferenceReviewPreviewPane`: workbench pane that owns visible preview state,
  diagnostics, and toolbar routing.
- `PreviewWidgetState`: selected fixture id, loading state, failure diagnostic,
  payload generation id, and interactive-ready state.
- `PreviewToolbarAdapter`: routes reset and camera preset commands to the
  widget without owning camera semantics.

## Ownership

`ImpressionPreviewWidget` owns:

- one long-lived render surface
- renderer initialization and disposal
- applying prepared payloads to the existing renderer
- forwarding reset and camera commands to the shared preview controller
- widget-level busy, empty, failed, and disposed states

`ReferenceReviewPreviewPane` owns:

- visible placeholder, loading, failed, and interactive states
- fixture label and diagnostics
- toolbar button enablement
- selected fixture generation id
- status messages surfaced from async payload building

The shared preview controller owns:

- scene application
- camera and interaction semantics
- edge and style policy

## Renderer Lifetime Contract

- Create the renderer once when the widget initializes the preview surface.
- Replace scenes inside the existing renderer when fixtures change.
- Never create a new renderer for every mouse event, frame, or fixture refresh.
- Never use PNG snapshot rendering for interactive review.
- Dispose the renderer only when the widget closes or the application shuts
  down.
- Handle stale payload results before calling into the widget.

## Public Widget Contract

```python
class ImpressionPreviewWidget(QWidget):
    previewReady = Signal(str)
    previewFailed = Signal(str, str)

    def set_preview_payload(self, payload: PreviewPayload) -> None: ...
    def clear_preview(self) -> None: ...
    def reset_view(self) -> None: ...
    def set_busy(self, busy: bool) -> None: ...
    def dispose(self) -> None: ...
```

The exact signal names may change, but the boundary must remain stable:
fixture identity and diagnostics flow outward; prepared payloads and camera
commands flow inward.

## Required Code Changes

- Add a dedicated preview widget module instead of keeping preview behavior in
  the workbench shell.
- Move renderer creation and disposal into the widget lifecycle.
- Make the widget call the shared `impression.preview` controller.
- Replace shell-level PyVista scene construction with widget calls.
- Keep preview toolbar state in the pane, not in the render controller.
- Add tests for widget lifecycle, payload replacement, stale-result rejection,
  and close/dispose behavior.

## Non-Goals

- Do not make the widget import fixture discovery or promotion code.
- Do not make the widget own async worker processes.
- Do not make the widget implement mesh conversion or edge extraction.
- Do not introduce a separate preview application window for normal review.

## Specification Manifest For Discovery

### Candidate Spec: Embedded Impression Preview Widget

Discovery purpose:
- Add the Qt widget host that embeds the shared Impression preview engine in
  the Reference Review Workbench.

Responsibilities:
- Functions/methods:
  - widget initialization
  - payload handoff
  - clear/reset/dispose methods
  - preview state signal emission
- Data structures/models:
  - widget state record
  - payload generation id
- Dependencies/services:
  - PySide6 widgets
  - shared preview controller
  - PyVistaQt render surface
- Returns/outputs/signals:
  - preview ready signal
  - preview failed signal
  - disposed state
- UI surfaces/components:
  - preview widget
  - preview pane
- UI fields/elements:
  - preview viewport
  - reset/camera toolbar routing
- Reusable code plan:
  - Existing code reused as-is: shared preview controller from
    `impression.preview`
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: workbench preview widget module
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - renderer mutation happens only on the Qt UI thread
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - widget displays already-selected fixture payloads only
- Performance-sensitive behavior:
  - renderer is long-lived and scene replacement does not recreate the surface
- Cross-screen reusable behavior:
  - preview pane feeds artifact comparison and promotion readiness

Project readiness fields:
- Implementation owner/module:
  - future `src/impression/devtools/reference_review/ui/preview_widget.py`
- Chosen defaults / parameters:
  - dark blue background and light orange object color supplied through shared
    preview style
- Test strategy:
  - Qt widget lifecycle tests, mocked controller tests, and manual real-render
    smoke with `.impress` fixtures
- Data ownership:
  - widget owns renderer lifecycle; pane owns visible workbench state
- Routes:
  - preview pane to widget to shared preview controller
- Open questions / nuance discovered:
  - offscreen Qt tests may need a fake render surface because VTK interactor
    is unstable on offscreen platforms
- Readiness blockers:
  - shared preview controller extraction

Score:
- Functions/methods: 4 x 2 = 8
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 2 x 2 = 4
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 37.5

Split decision:
- Split required before final specification. Split into widget lifecycle,
  payload handoff, and pane toolbar/state leaves.

## Change History

- 2026-07-07: Created supplemental architecture for the embedded Qt wrapper
  around the shared preview engine.
