# Reference Review Qt Workbench UI

## Overview

This document defines the visible application structure for the Reference
Review Workbench.

The workbench should use a PySide 6 desktop application shell with in-window
review panels, controls, state presentation, and keyboard interaction. The 3D
model preview is hosted inside the review window through an embedded
PyVista-rendered preview surface so reviewers can orbit, pan, zoom, and reset without
leaving the workbench.

## Parent Architecture

- [Reference Review Workbench Architecture](reference-review-workbench-architecture.md)
- [Reference Review Fixture Source Contract](reference-review-fixture-source-contract.md)
- [Reference Review Async Concurrency](reference-review-async-concurrency.md)

## Technology Choice

- Python/PySide 6 owns application startup, service wiring, task submission,
  model loading, reference lifecycle calls, and visible review panels.
- Qt widgets own the primary workbench shell and embedded preview surface.
- Qt Quick/QML remains available for component gallery and QML component assets
  but is not the primary STL review surface.
- The preview bridge owns embedded 3D preview integration. The workbench must
  not launch a separate preview application for normal STL review.

The architecture does not let visible controls call PyVista or filesystem
operations directly; selected, validated fixture records are routed through the
preview shell.

## Primary Surfaces

- Queue panel: dirty fixtures, failed-note fixtures, promoted fixtures, filters.
- Preview panel: interactive model viewport with orbit, pan, zoom, reset view,
  fit selected, axes, bounds, and camera preset controls.
- Context panel: feature description, expected output, source model path,
  owning test/spec links, generation command, fixture diagnostics.
- Markdown context panel: rendered architecture/spec/test-spec/research/note
  excerpts for the selected fixture.
- Artifact panel: dirty/gold/diff images, STL metadata, slice diagnostics, and
  generated artifact paths.
- Notes panel: review status, note editor, unresolved-state marker, provenance.
- Codex panel: constrained assistant chat, candidate list, sandbox refusals.
- Action bar: previous, next, regenerate, flag, promote, open source, reveal
  artifact folder.

## Layout And Overflow Policy

The first shell uses a dense desktop workbench layout:

- left queue panel with fixed minimum width and resizable maximum width
- center preview panel as the dominant visual region
- right inspector stack for context, artifacts, notes, and Codex
- bottom action/status strip

Qt Quick rules:

- Layout-managed children use `Layout.*` properties instead of conflicting
  anchors.
- Queue rows have fixed height and elide long fixture ids.
- Context and notes panels wrap text inside bounded scroll surfaces.
- Artifact thumbnails use fixed cell geometry and never resize queue rows.
- Markdown context uses a bounded scroll surface; headings and code blocks must
  not resize the inspector stack unpredictably.
- Buttons use icon controls where the command is familiar, with tooltips and
  accessible names.
- Long paths elide in the middle and expose full paths through copy/reveal
  actions.
- The preview panel keeps stable geometry while model loads, errors, or camera
  changes.

## Component Reuse Plan

The workbench should use Qt Quick Controls 2 as the base component framework
and an Impression-owned QML component library as the product framework.

Do not begin panel implementation by writing raw rectangles, labels, and mouse
areas in each panel. Build the reusable workbench components first, test their
states, then compose panels from those components.

Reusable QML component families:

- `WorkbenchIconButton`
- `WorkbenchSegmentedControl`
- `WorkbenchPanelHeader`
- `WorkbenchStatusPill`
- `WorkbenchQueueRow`
- `WorkbenchSplitDivider`
- `ArtifactThumbnailTile`
- `PathField`
- `MarkdownContextView`
- `DiagnosticBanner`
- `ReviewActionBar`
- `CodexMessageBubble`

Reuse rules imported from ViewDown:

- Build primitives first: icon, label, button, divider, row, tile, panel.
- Compose primitives into reusable controls before making specialized panels.
- Put behavior at the lowest reusable level that preserves meaning.
- Screenshot every reusable component in idle, hover, focus, pressed, disabled,
  selected, loading, warning, and error states where applicable.
- Keep component geometry stable across state changes.

Framework choice:

- Foundation: Qt Quick Controls 2.
- Initial style: Fusion or macOS style during the spike; final style chosen by
  screenshot review.
- Product layer: `src/impression/devtools/reference_review/ui/components/`.
- Not v1 foundation: Kirigami, FluentPySide, or a broad third-party visual
  framework.

Reason:

- Qt Quick Controls 2 gives standard control behavior and state machinery.
- The Impression workbench component library gives the actual product speed:
  known panel headers, queue rows, icon buttons, diagnostic banners, markdown
  views, artifact tiles, action bars, and Codex messages.
- Third-party visual frameworks do not own our fixture review semantics,
  promotion states, source model diagnostics, or reference artifact workflow.

## State Coverage

Required UI states:

- empty queue
- loading queue
- source missing
- source loading
- source loaded
- preview loading
- preview interactive
- preview failed
- artifact missing
- artifact dirty-only
- artifact promoted
- markdown context loading
- markdown context rendered
- markdown context link blocked
- markdown context render failed
- notes unsaved
- notes saved-failed
- notes promoted
- Codex idle
- Codex streaming
- Codex refused tool call
- promotion blocked
- promotion confirmed
- narrow window
- dark mode

## Consultant Review Lenses

The workbench should reuse the ViewDown consultant-profile method for design
review:

- Mara review: workflow clarity, focus order, keyboard access, disabled states,
  recovery states, and whether the reviewer always knows what is selected.
- Ilya review: preview dominance, split-panel proportions, icon grammar,
  spacing, dense inspector readability, screenshot state matrix, and component
  geometry.
- Sabine review: command names, promotion/rejection copy, diagnostic wording,
  "fixture", "model", "artifact", "gold", and "dirty" terminology.

## Screenshot And Design Asset Plan

The workbench needs design assets before implementation freeze:

- component gallery for all reusable controls
- normal dirty fixture review screen
- missing source diagnostic
- preview failure diagnostic
- failed-note fixture state
- promotion confirmation
- Codex refused tool call
- Codex candidate generated
- artifact comparison state
- markdown spec/context reading state
- narrow window state
- dark mode state

## Markdown Rendering

Much of the workbench context is Markdown: architecture docs, specifications,
test specifications, review notes, expected-output prose, and research. The UI
therefore needs an explicit Markdown rendering boundary.

The v1 renderer should prefer Qt-native Markdown through `QTextDocument` for
short local context and note display. A `MarkdownContextRenderer` service should
own rendering, local-link resolution, external-link blocking, and error
diagnostics.

`markdown-it-py` is the preferred fallback when the workbench needs stronger
CommonMark/GFM-style parsing, excerpt extraction, or controlled HTML output.
Qt WebEngine is reserved for a later richer documentation browser if native Qt
Markdown surfaces are too limited.

Markdown display rules:

- read-only
- local project links only by default
- external links blocked with a visible diagnostic
- raw HTML/script execution disabled in the default renderer
- code blocks scroll horizontally inside the Markdown panel
- long headings and paths wrap or elide without pushing action controls away

Screenshots must include surrounding shell context for modal or confirmation
states, mirroring the lesson from ViewDown settings review.

## Specification Manifest For Discovery

## Manifest Review History

- 2026-05-31 loop 1: Critical review found the original shell candidate mixed
  application bootstrap, panel implementation, preview, Markdown, Codex UI,
  component library, packaging, and screenshots.
- 2026-05-31 loop 2: Rescored after moving service responsibilities back to
  child service architecture documents.
- 2026-05-31 loop 3: Split component framework and component gallery because
  the gallery is test evidence, not the runtime shell.
- 2026-05-31 loop 4: Added packaging/dependency policy as a UI-owned leaf
  because QML layout and optional Qt modules affect deployability.
- 2026-05-31 loop 5: Final review confirmed no final UI split remains at or
  above the split threshold; broad pre-split groupings now carry final split
  scores.

### Candidate Spec: PySide QML Shell Bootstrap

Discovery purpose:
- Create the PySide application shell and register QML bridge objects without
  implementing individual panels.

Responsibilities:
- Functions/methods:
  - shell bootstrap
  - QML bridge registration
- Data structures/models:
  - shell configuration
  - registered bridge map
- Dependencies/services:
  - PySide6
  - async dispatcher protocol
- Returns/outputs/signals:
  - shell ready signal
  - shell diagnostic
- UI surfaces/components:
  - main workbench window
- UI fields/elements:
  - top-level split layout container
- Reusable code plan:
  - Existing code reused as-is: none
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: workbench UI package
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - shell submits work through dispatcher only
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - no unrestricted file controls in shell
- Performance-sensitive behavior:
  - startup avoids model loading on UI thread
- Cross-screen reusable behavior:
  - shell hosts every panel

Project readiness fields:
- Implementation owner/module:
  - future `src/impression/devtools/reference_review/ui/shell`
- Chosen defaults / parameters:
  - PySide 6 shell with Qt Quick/QML chrome
- Test strategy:
  - launch smoke test and QML bridge registration test
- Data ownership:
  - UI shell owns visible application state only
- Routes:
  - launcher to shell to QML engine
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 0 x 0.5 = 0
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 26

Split decision:
- Split required by score. Implement as launcher/bootstrap (`13`) and bridge
  registration (`13`) final specs.
  - Split score: QML launcher/bootstrap total `13`
  - Split score: bridge registration total `13`

### Candidate Spec: Qt Quick Controls Style And Component Framework

Discovery purpose:
- Establish Qt Quick Controls 2 styling, tokens, and Impression-owned wrapper
  component rules.

Responsibilities:
- Functions/methods:
  - style configuration loader
  - component registration
- Data structures/models:
  - style token record
  - component contract record
- Dependencies/services:
  - Qt Quick Controls 2
- Returns/outputs/signals:
  - style loaded diagnostic
- UI surfaces/components:
  - shared component foundation
- UI fields/elements:
  - icon button, text field, status badge, split pane
- Reusable code plan:
  - Existing code reused as-is: Qt Quick Controls 2
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: workbench component library
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - stable component sizing and no layout thrash
- Cross-screen reusable behavior:
  - all panels consume shared controls

Project readiness fields:
- Implementation owner/module:
  - future `ui/components`
- Chosen defaults / parameters:
  - Qt Quick Controls 2 base with Impression wrappers; no Kirigami or
    FluentPySide requirement
- Test strategy:
  - component instantiation and visual state smoke tests
- Data ownership:
  - component library owns shared visual primitives
- Routes:
  - QML import to component use
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 4 x 1 = 4
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 21.5

Split decision:
- No split needed. Component-specific implementation specs can derive from the
  final component contract if later needed.

### Candidate Spec: Queue And Context Panels

Discovery purpose:
- Implement fixture navigation and source/context display for selecting the
  model under review.

Responsibilities:
- Functions/methods:
  - selection binding
  - navigation action dispatch
- Data structures/models:
  - queue view model
  - selected fixture view model
- Dependencies/services:
  - source registry
  - async dispatcher
- Returns/outputs/signals:
  - selected fixture changed
  - navigation requested
- UI surfaces/components:
  - queue panel
  - context panel
- UI fields/elements:
  - previous, next, fixture id, status badge, source path, expected output
- Reusable code plan:
  - Existing code reused as-is: shared workbench components
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - selection changes submit source-load tasks through dispatcher
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - redacted paths use source-registry display fields
- Performance-sensitive behavior:
  - large queues virtualize or reuse delegates
- Cross-screen reusable behavior:
  - selection drives preview, notes, artifacts, and Codex

Project readiness fields:
- Implementation owner/module:
  - future `ui/panels/queue_context`
- Chosen defaults / parameters:
  - first dirty fixture selected by default
- Test strategy:
  - navigation, selection, redacted display, and empty-state tests
- Data ownership:
  - UI owns selected fixture; source registry owns fixture data
- Routes:
  - queue selection to selected fixture model to dependent panels
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - source registry protocol must exist

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 2 x 2 = 4
- UI fields/elements: 6 x 1 = 6
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 32.5

Split decision:
- Split required by score. Implement as queue navigation panel (`17.5`) and
  selected fixture context panel (`15`) final specs.
  - Split score: queue navigation panel total `17.5`
  - Split score: selected fixture context panel total `15`

### Candidate Spec: Interactive Preview Bridge Panel

Discovery purpose:
- Attach the model preview bridge to the selected fixture without tying the QML
  shell to PyVista internals.

Responsibilities:
- Functions/methods:
  - preview bridge attachment
  - camera action dispatch
- Data structures/models:
  - preview state record
  - camera command record
- Dependencies/services:
  - preview bridge
  - async dispatcher
- Returns/outputs/signals:
  - preview ready
  - preview diagnostic
- UI surfaces/components:
  - preview panel
- UI fields/elements:
  - orbit, pan, zoom, reset
- Reusable code plan:
  - Existing code reused as-is: Impression preview model-loading semantics
  - Additions to existing reusable library/module: preview bridge adapter
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - preview loads run off UI thread; camera controls stay responsive
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - preview only loads selected source records
- Performance-sensitive behavior:
  - model load and tessellation never block QML event loop
- Cross-screen reusable behavior:
  - preview state feeds artifact comparison and promotion readiness

Project readiness fields:
- Implementation owner/module:
  - `ui/shell` for the embedded workbench host
  - `ui/preview_bridge` for preview adapter state policy
- Chosen defaults / parameters:
  - embedded PyVista-rendered preview required for in-app STL review
  - supervised external preview rejected for normal workbench interaction
- Test strategy:
  - preview load, source change, camera action, and failure-state tests
- Data ownership:
  - preview bridge owns render state; UI owns controls
- Routes:
  - selected fixture to preview task to bridge state
- Open questions / nuance discovered:
  - Qt offscreen tests use a placeholder because VTK's interactor is not stable
    on the offscreen platform
- Readiness blockers:
  - none for the embedded preview route

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 4 x 1 = 4
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 31.5

Split decision:
- Split required by score. Implement as preview adapter decision spike (`9`),
  preview load binding (`13.5`), and camera controls (`9`) final specs.
  - Split score: preview adapter decision spike total `9`
  - Split score: preview load binding total `13.5`
  - Split score: camera controls total `9`

### Candidate Spec: Markdown Context Renderer

Discovery purpose:
- Render fixture Markdown/context safely and consistently in the workbench.

Responsibilities:
- Functions/methods:
  - Markdown render binding
  - blocked-link handler
- Data structures/models:
  - Markdown context render state
  - blocked link diagnostic
- Dependencies/services:
  - Qt text renderer
  - optional markdown-it-py renderer
- Returns/outputs/signals:
  - rendered context
  - blocked link diagnostic
- UI surfaces/components:
  - Markdown context panel
- UI fields/elements:
  - rendered Markdown, blocked-link message, local-link action
- Reusable code plan:
  - Existing code reused as-is: fixture context payload
  - Additions to existing reusable library/module: Markdown context renderer
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - expensive render may run through dispatcher
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - external links are blocked or explicitly confirmed
- Performance-sensitive behavior:
  - rendered context is cached per fixture/source digest
- Cross-screen reusable behavior:
  - renderer can be reused by notes preview and diagnostic panels

Project readiness fields:
- Implementation owner/module:
  - future `ui/markdown_context`
- Chosen defaults / parameters:
  - Qt-native Markdown first; markdown-it-py/WebEngine only behind boundary
- Test strategy:
  - Markdown rendering, blocked link, long text, and cache invalidation tests
- Data ownership:
  - context payload owns content; renderer owns presentation
- Routes:
  - context payload to renderer to QML panel
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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
- Readiness blockers: 0 x 2 = 0
- Total: 26.5

Split decision:
- Split required by score. Implement as renderer backend (`13.5`) and QML
  panel/link policy (`13`) final specs.
  - Split score: renderer backend total `13.5`
  - Split score: QML panel/link policy total `13`

### Candidate Spec: Artifact And Notes Review Panels

Discovery purpose:
- Present derived artifacts and notes controls while delegating writes to the
  promotion/notes service.

Responsibilities:
- Functions/methods:
  - artifact selection binding
  - note edit action dispatch
- Data structures/models:
  - artifact tile view model
  - note editor state
- Dependencies/services:
  - promotion/notes service
  - async dispatcher
- Returns/outputs/signals:
  - artifact action requested
  - note save requested
- UI surfaces/components:
  - artifact panel
  - notes panel
- UI fields/elements:
  - artifact tiles, diff badge, status, notes editor, flag
- Reusable code plan:
  - Existing code reused as-is: shared components
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - note saves route through dispatcher and durable write lane
- Destructive/write behavior:
  - none directly from UI; service owns writes
- Security/privacy-sensitive behavior:
  - notes editor does not include full chat logs by default
- Performance-sensitive behavior:
  - thumbnails load asynchronously
- Cross-screen reusable behavior:
  - review state updates queue and action bar

Project readiness fields:
- Implementation owner/module:
  - future `ui/panels/artifacts_notes`
- Chosen defaults / parameters:
  - notes without promotion remain a failed review state
- Test strategy:
  - artifact list, thumbnail loading, note edit, save failure, and flag tests
- Data ownership:
  - UI owns edit state; service owns durable notes
- Routes:
  - selected fixture to artifact/note view models to service action
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - promotion/notes service protocol must exist

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 2 x 2 = 4
- UI fields/elements: 5 x 1 = 5
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 31.5

Split decision:
- Split required by score. Implement as artifact panel (`15.5`) and notes
  panel (`16`) final specs.
  - Split score: artifact panel total `15.5`
  - Split score: notes panel total `16`

### Candidate Spec: Codex Sidecar Panel

Discovery purpose:
- Provide chat, candidate list, refusal display, and adoption actions for the
  sandboxed Codex sidecar.

Responsibilities:
- Functions/methods:
  - chat action dispatch
  - candidate adoption action dispatch
- Data structures/models:
  - sidecar stream state
  - candidate list view model
- Dependencies/services:
  - Codex sidecar broker
  - async dispatcher
- Returns/outputs/signals:
  - sidecar request
  - candidate adoption request
  - refusal display request
- UI surfaces/components:
  - Codex panel
  - candidate list
  - refusal banner
- UI fields/elements:
  - chat input, response stream, candidate path, regenerate, adopt, refusal
- Reusable code plan:
  - Existing code reused as-is: shared components
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - streams are cancellable and stale-guarded per fixture
- Destructive/write behavior:
  - UI never writes candidate files directly
- Security/privacy-sensitive behavior:
  - UI displays broker refusals and cannot bypass policy
- Performance-sensitive behavior:
  - streamed text updates are throttled
- Cross-screen reusable behavior:
  - sidecar uses selected fixture and updates candidate/adoption state

Project readiness fields:
- Implementation owner/module:
  - future `ui/panels/codex_sidecar`
- Chosen defaults / parameters:
  - human adoption required before candidates affect review source
- Test strategy:
  - stream, cancellation, refusal, candidate selection, and adopt action tests
- Data ownership:
  - broker owns authority; UI owns visible stream and actions
- Routes:
  - selected fixture to sidecar panel to broker request
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - Codex broker protocol must exist

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 3 x 1 = 3
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 3 x 2 = 6
- UI fields/elements: 6 x 1 = 6
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 38.5

Split decision:
- Split required by score. Implement as chat stream panel (`13.5`), candidate
  list/adoption UI (`14`), and refusal display (`11`) final specs.
  - Split score: chat stream panel total `13.5`
  - Split score: candidate list/adoption UI total `14`
  - Split score: refusal display total `11`

### Candidate Spec: Component Gallery And Screenshot State Suite

Discovery purpose:
- Create durable UI evidence for component states, overflow, focus, and panel
  behavior.

Responsibilities:
- Functions/methods:
  - screenshot scenario runner
  - component state enumerator
- Data structures/models:
  - component state contract record
  - screenshot scenario record
- Dependencies/services:
  - QML test harness
  - component library
- Returns/outputs/signals:
  - screenshot artifact
  - state coverage report
- UI surfaces/components:
  - reusable component gallery
- UI fields/elements:
  - hover, focus, disabled, loading, error, empty, overflow states
- Reusable code plan:
  - Existing code reused as-is: shared components
  - Additions to existing reusable library/module: screenshot test harness
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - screenshot generation runs outside interactive UI session
- Destructive/write behavior:
  - writes test screenshots only
- Security/privacy-sensitive behavior:
  - fixtures use synthetic non-secret data
- Performance-sensitive behavior:
  - screenshot suite is bounded for CI/dev runs
- Cross-screen reusable behavior:
  - gallery validates all panel components

Project readiness fields:
- Implementation owner/module:
  - future `ui/tests/component_gallery`
- Chosen defaults / parameters:
  - screenshots include surrounding shell context for dialogs and confirmations
- Test strategy:
  - automated screenshot/state matrix plus manual UI review checklist
- Data ownership:
  - UI test suite owns visual state evidence
- Routes:
  - component scenarios to rendered screenshots to review artifacts
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - component library must exist

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 7 x 1 = 7
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 35.5

Split decision:
- Split required by score. Implement as component gallery (`14.5`),
  screenshot runner (`13`), and accessibility/overflow state matrix (`8`) final
  specs.
  - Split score: component gallery total `14.5`
  - Split score: screenshot runner total `13`
  - Split score: accessibility/overflow state matrix total `8`

### Candidate Spec: Workbench UI Dependency And Packaging Policy

Discovery purpose:
- Keep UI dependencies optional and packageable while avoiding accidental core
  Impression dependency leaks.

Responsibilities:
- Functions/methods:
  - optional extra declaration checker
  - QML resource layout verifier
- Data structures/models:
  - dependency policy record
  - package resource manifest
- Dependencies/services:
  - PySide6 deployment tooling
  - import boundary checks
- Returns/outputs/signals:
  - dependency policy report
  - packaging smoke result
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: packaging metadata
  - Additions to existing reusable library/module: devtool dependency checks
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - packaging smoke writes build artifacts only
- Security/privacy-sensitive behavior:
  - optional extras must not expose sandbox bypasses
- Performance-sensitive behavior:
  - package smoke remains bounded
- Cross-screen reusable behavior:
  - dependency policy protects all workbench UI modules

Project readiness fields:
- Implementation owner/module:
  - future packaging/devtool checks
- Chosen defaults / parameters:
  - workbench lives behind optional extra; WebEngine remains optional
- Test strategy:
  - import-boundary check and package smoke test
- Data ownership:
  - packaging policy owns optional dependency boundaries
- Routes:
  - package metadata to import checks to smoke result
- Open questions / nuance discovered:
  - exact deployment tool can be selected later
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 21.5

Split decision:
- No split needed. This is a packaging and dependency-boundary leaf.

## Change History

- 2026-05-31: Ran five critical review, rescore, and split passes over the
  specification manifest. Split the UI work into shell, component framework,
  queue/context, preview, Markdown, artifact/notes, Codex panel, screenshot,
  and dependency/packaging leaves with final split scores.
- 2026-05-30: Created UI architecture split with PySide 6, Qt Quick/QML,
  preview bridge, component reuse, state coverage, concurrency expectations,
  and ViewDown-derived screenshot/design review plan.
