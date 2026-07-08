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

## Manifest Review History

- 2026-07-07 loop 1: Initial review found the embedded widget candidate scored
  above the split threshold because it bundled renderer lifetime, payload
  application, pane state, toolbar routing, and real-render verification.
- 2026-07-07 loop 2: Split renderer lifecycle from payload handoff because
  lifecycle must be stable before scene replacement is wired.
- 2026-07-07 loop 3: Split preview-pane visible state from the widget because
  pane diagnostics and toolbar enablement are workbench UI behavior.
- 2026-07-07 loop 4: Final review confirmed no remaining candidate scores at
  or above `25`; `16-24` candidates include cohesion explanations.

### Candidate Spec: Preview Widget Renderer Lifecycle

Discovery purpose:
- Add the Qt widget host that owns one long-lived embedded render surface and
  disposes it only with the widget lifecycle.

Responsibilities:
- Functions/methods:
  - widget initialization
  - render surface creation
  - clear/dispose methods
- Data structures/models:
  - renderer lifecycle state
- Dependencies/services:
  - PySide6 widgets
  - PyVistaQt render surface
- Returns/outputs/signals:
  - lifecycle diagnostic
- UI surfaces/components:
  - preview widget
- UI fields/elements:
  - preview viewport
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
  - none
- Performance-sensitive behavior:
  - renderer is long-lived and not recreated per frame or fixture
- Cross-screen reusable behavior:
  - none

Project readiness fields:
- Implementation owner/module:
  - future `src/impression/devtools/reference_review/ui/preview_widget.py`
- Chosen defaults / parameters:
  - dark blue background and light orange object color supplied through shared
    preview style
- Test strategy:
  - Qt widget lifecycle tests and mocked render-surface tests
- Data ownership:
  - widget owns renderer lifecycle only
- Routes:
  - preview pane creates widget; widget creates renderer
- Open questions / nuance discovered:
  - offscreen Qt tests may need a fake render surface because VTK interactor
    is unstable on offscreen platforms
- Readiness blockers:
  - shared preview controller extraction

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Split decision:
- No split needed. Cohesion reason: renderer creation, stable ownership, and
  disposal are one lifecycle boundary.

### Candidate Spec: Preview Widget Payload Application

Discovery purpose:
- Let the preview widget accept a prepared payload and apply it through the
  shared preview controller without recreating the renderer.

Responsibilities:
- Functions/methods:
  - `set_preview_payload`
  - `clear_preview`
  - payload generation check
- Data structures/models:
  - payload generation id
  - widget payload state
- Dependencies/services:
  - shared preview controller
  - preview payload record
- Returns/outputs/signals:
  - preview ready signal
  - preview failed signal
- UI surfaces/components:
  - preview widget
- UI fields/elements:
  - preview viewport
- Reusable code plan:
  - Existing code reused as-is: shared preview controller
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - payload application happens on Qt UI thread
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - scene replacement reuses the existing renderer
- Cross-screen reusable behavior:
  - preview payload state feeds preview readiness

Project readiness fields:
- Implementation owner/module:
  - future `ui/preview_widget.py`
- Chosen defaults / parameters:
  - stale payloads are rejected before widget application
- Test strategy:
  - mocked payload handoff and scene replacement tests
- Data ownership:
  - widget owns current payload generation after pane validation
- Routes:
  - preview pane to widget to shared preview controller
- Open questions / nuance discovered:
  - payload shape depends on preview payload boundary spec
- Readiness blockers:
  - shared preview controller extraction

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- No split needed. Cohesion reason: widget payload acceptance, clear, ready,
  and failed signals are one scene-replacement boundary; stale-result ownership
  belongs to the payload-boundary architecture.

### Candidate Spec: Preview Pane Visible State

Discovery purpose:
- Keep workbench-visible preview state outside the render widget while still
  embedding the widget in the preview pane.

Responsibilities:
- Functions/methods:
  - preview pane state reducer
  - diagnostic display update
- Data structures/models:
  - preview pane state record
- Dependencies/services:
  - `ImpressionPreviewWidget`
  - workbench selection model
- Returns/outputs/signals:
  - toolbar enabled state
  - pane diagnostic state
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - placeholder
  - loading indicator
  - diagnostic banner
- Reusable code plan:
  - Existing code reused as-is: workbench panel patterns
  - Additions to existing reusable library/module: preview pane state model
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - pane state mutates only on Qt UI thread after owner/request checks
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics avoid unsafe environment dumps
- Performance-sensitive behavior:
  - pane state changes do not recreate the renderer
- Cross-screen reusable behavior:
  - preview state feeds review readiness, artifact panels, and promotion state

Project readiness fields:
- Implementation owner/module:
  - future preview pane module or shell preview section
- Chosen defaults / parameters:
  - toolbar disabled until widget is interactive
- Test strategy:
  - pane state transition tests
- Data ownership:
  - pane owns visible state; widget owns renderer state
- Routes:
  - selection and payload controller events to pane state
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - preview widget lifecycle

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 3 x 1 = 3
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- No split needed. Cohesion reason: placeholder, loading, and diagnostic states
  are one visible preview-pane state boundary; toolbar routing is separate.

### Candidate Spec: Preview Toolbar Command Routing

Discovery purpose:
- Route preview toolbar commands to the widget without letting the toolbar own
  camera or interaction semantics.

Responsibilities:
- Functions/methods:
  - toolbar command router
  - command enablement resolver
- Data structures/models:
  - camera command record
- Dependencies/services:
  - `ImpressionPreviewWidget`
  - preview pane state
- Returns/outputs/signals:
  - toolbar enabled state
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - reset control
  - camera preset controls
- Reusable code plan:
  - Existing code reused as-is: workbench toolbar patterns
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - commands execute on Qt UI thread
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - commands do not recreate renderer
- Cross-screen reusable behavior:
  - toolbar command state supports preview and promotion readiness

Project readiness fields:
- Implementation owner/module:
  - future preview pane module
- Chosen defaults / parameters:
  - toolbar disabled until widget is interactive
- Test strategy:
  - command enablement and routing tests with a fake widget
- Data ownership:
  - pane owns command state; shared preview controller owns command semantics
- Routes:
  - toolbar action to pane router to widget method
- Open questions / nuance discovered:
  - exact preset list inherits current UI definition
- Readiness blockers:
  - preview pane visible state

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 21.5

Split decision:
- No split needed. Cohesive command-routing leaf.

### Candidate Spec: Preview Wrapper Real-Render Smoke And Lifecycle Evidence

Discovery purpose:
- Define the verification evidence needed to prove the embedded wrapper uses a
  stable live renderer with real `.impress` fixtures.

Responsibilities:
- Functions/methods:
  - manual smoke command
  - lifecycle evidence capture
- Data structures/models:
  - smoke result record
- Dependencies/services:
  - test fixtures
  - workbench launcher
- Returns/outputs/signals:
  - smoke pass/fail note
- UI surfaces/components:
  - preview pane
- UI fields/elements:
  - preview viewport
- Reusable code plan:
  - Existing code reused as-is: workbench launcher and fixtures
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - verifies launch, interaction, fixture switch, and shutdown ordering
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - verifies renderer is not recreated per interaction
- Cross-screen reusable behavior:
  - smoke evidence supports review readiness

Project readiness fields:
- Implementation owner/module:
  - future paired test specification
- Chosen defaults / parameters:
  - use dirty `.impress` fixtures, not demo PNG/STL snapshot smoke
- Test strategy:
  - manual real-render smoke plus focused lifecycle tests
- Data ownership:
  - evidence belongs to test/review artifacts
- Routes:
  - launcher to fixture to preview pane to widget
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - preview widget lifecycle

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 20.5

Split decision:
- No split needed. Cohesive verification leaf for the wrapper's renderer
  lifetime and real-fixture behavior.

## Change History

- 2026-07-07: Ran four manifest review loops, split the high-scoring embedded
  widget candidate, and rescored resulting implementation leaves.
- 2026-07-07: Created supplemental architecture for the embedded Qt wrapper
  around the shared preview engine.
