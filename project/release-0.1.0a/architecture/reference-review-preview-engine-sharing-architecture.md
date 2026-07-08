# Reference Review Preview Engine Sharing Architecture

## Overview

This supplemental architecture document defines what must change in
`impression.preview` so the CLI preview and Reference Review Workbench use the
same preview engine.

The target is not to move the workbench into the CLI application. The target is
to extract reusable preview behavior from the CLI preview implementation so the
CLI and the embedded workbench widget both call the same controller.

## Parent And Related Architecture

- [Reference Review Preview Remediation Plan](reference-review-preview-remediation-plan.md)
- [Reference Review Preview Qt Wrapper Architecture](reference-review-preview-qt-wrapper-architecture.md)
- [Reference Review Preview Payload Boundary Architecture](reference-review-preview-payload-boundary-architecture.md)
- [Reference Review Qt Workbench UI](reference-review-qt-workbench-ui.md)

## Current Problem

The workbench has duplicated preview behavior that belongs to
`impression.preview`: scene application, feature-edge policy, camera reset,
plotter configuration, and interaction semantics.

Duplicating that behavior creates two preview implementations. Any difference
between them can make the workbench behave differently from the CLI preview,
which is the opposite of the intended review tool.

## Target Components

- `PreviewSceneController`: reusable, UI-host-neutral controller that applies
  Impression preview datasets to a provided plotter.
- `PreviewStyle`: configuration for background color, object color, edge
  visibility, axes, bounds, and other preview defaults.
- `PreviewInteractionPolicy`: shared defaults for camera and mouse behavior.
- CLI preview host: command-line application surface that owns its existing
  top-level application lifecycle but delegates scene behavior to the shared
  controller.
- Workbench preview host: embedded Qt widget that owns its widget lifecycle but
  delegates scene behavior to the shared controller.

## Responsibility Boundary

`impression.preview` owns:

- dataset-to-scene application
- mesh conversion policy
- feature-edge or object-edge policy
- camera reset and fit behavior
- interaction defaults
- preview styling defaults
- plotter configuration that is independent of the host window

CLI and workbench hosts own:

- application or widget lifecycle
- command-line parsing or fixture selection
- loading state and diagnostics
- host-specific toolbar actions
- renderer creation and disposal timing

## Required Code Changes

- Extract scene application from CLI-only methods into a reusable controller.
- Move camera reset behavior into that controller or a directly shared helper.
- Move edge-rendering policy into a shared helper called by the controller.
- Make CLI preview call the controller instead of private CLI-only methods.
- Make the workbench wrapper call the same controller.
- Add import-boundary tests that prevent workbench UI modules from duplicating
  preview scene application.

## Non-Goals

- Do not make `impression.preview` import the Reference Review Workbench.
- Do not make the CLI preview depend on PySide workbench modules.
- Do not introduce a second mouse-control policy for the workbench.
- Do not preserve workbench-only mesh conversion code after the shared
  controller exists.

## Specification Manifest For Discovery

## Manifest Review History

- 2026-07-07 loop 1: Initial review found the original shared controller
  candidate scored above the split threshold because it bundled controller API,
  scene application, CLI migration, and regression guards.
- 2026-07-07 loop 2: Split scene semantics from CLI host migration so renderer
  behavior can be extracted before either host is rewired.
- 2026-07-07 loop 3: Split import-boundary and parity guards from the runtime
  extraction because those are verification constraints, not scene behavior.
- 2026-07-07 loop 4: Final review confirmed no remaining candidate scores at
  or above `25`; `16-24` candidates include cohesion explanations.

### Candidate Spec: Shared Preview Controller API And Style Records

Discovery purpose:
- Define the reusable preview-controller API and style/interaction records that
  CLI preview and the workbench wrapper will both call.

Responsibilities:
- Functions/methods:
  - preview controller constructor
  - plotter configuration method
  - style/default resolver
- Data structures/models:
  - preview style record
  - interaction policy record
  - controller options record
- Dependencies/services:
  - `impression.preview`
  - PyVista plotter protocol
- Returns/outputs/signals:
  - configured controller
  - style resolution diagnostic
- UI surfaces/components:
  - CLI preview host
  - workbench preview widget host
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: CLI preview defaults as source behavior
  - Additions to existing reusable library/module: controller API in
    `impression.preview`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none; caller owns thread affinity when using the controller
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - controller initialization is lightweight and does not create renderers
- Cross-screen reusable behavior:
  - shared by CLI preview and workbench preview

Project readiness fields:
- Implementation owner/module:
  - `src/impression/preview.py`
- Chosen defaults / parameters:
  - workbench style uses dark blue background and light orange object color via
    style configuration
- Test strategy:
  - controller API construction tests and style default tests
- Data ownership:
  - preview controller owns preview semantic configuration, not renderer
    lifecycle
- Routes:
  - CLI host or workbench widget to shared preview controller
- Open questions / nuance discovered:
  - exact controller class names may change during implementation
- Readiness blockers:
  - none

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 2 x 2 = 4
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 22.5

Split decision:
- No split needed. Cohesive API/configuration leaf; scene mutation, CLI
  migration, and guard tests are separate candidates.

### Candidate Spec: Shared Scene Application And Camera Policy

Discovery purpose:
- Move scene application, edge policy, and camera reset behavior into the
  shared preview controller without changing host lifecycle ownership.

Responsibilities:
- Functions/methods:
  - scene application method
  - edge policy helper
  - camera reset method
- Data structures/models:
  - scene application options
  - camera reset diagnostic
- Dependencies/services:
  - `impression.preview`
  - PyVista plotter protocol
- Returns/outputs/signals:
  - applied scene
  - camera reset diagnostic
- UI surfaces/components:
  - CLI preview host
  - workbench preview widget host
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: CLI private scene, edge, and camera behavior
  - Additions to existing reusable library/module: shared controller methods
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - caller must invoke scene mutation on the caller-owned render thread
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - avoids duplicate mesh conversion and edge extraction paths
- Cross-screen reusable behavior:
  - shared by CLI preview and workbench preview

Project readiness fields:
- Implementation owner/module:
  - `src/impression/preview.py`
- Chosen defaults / parameters:
  - object edge policy matches CLI preview unless overridden by style
- Test strategy:
  - controller scene-application tests with a mock plotter and camera reset
    tests
- Data ownership:
  - controller owns scene semantics; hosts own renderer lifecycle
- Routes:
  - host render surface to shared controller scene methods
- Open questions / nuance discovered:
  - mock plotter coverage may need a small plotter protocol shim
- Readiness blockers:
  - shared preview controller API

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 2 x 2 = 4
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- No split needed. Cohesion reason: scene application, edge policy, and camera
  reset are one render-semantic boundary; host migration is separate.

### Candidate Spec: CLI Preview Host Delegation

Discovery purpose:
- Rewire the existing CLI preview host to use the shared preview controller
  without changing CLI launch behavior.

Responsibilities:
- Functions/methods:
  - CLI host delegation call
  - compatibility adapter for existing CLI options
- Data structures/models:
  - CLI-to-controller option mapping
- Dependencies/services:
  - `impression.preview`
  - shared preview controller
- Returns/outputs/signals:
  - CLI preview scene result
- UI surfaces/components:
  - CLI preview window
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: CLI application lifecycle
  - Additions to existing reusable library/module: CLI controller delegation
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - unchanged from CLI preview host
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - no additional renderer creation during delegation
- Cross-screen reusable behavior:
  - keeps CLI and workbench preview behavior aligned

Project readiness fields:
- Implementation owner/module:
  - `src/impression/preview.py`
- Chosen defaults / parameters:
  - CLI defaults are preserved unless the shared controller already exposes the
    same behavior
- Test strategy:
  - CLI delegation smoke and option mapping tests
- Data ownership:
  - CLI host owns launch and renderer lifecycle; controller owns scene behavior
- Routes:
  - CLI options to controller options to scene application
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - shared scene application and camera policy

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 20.5

Split decision:
- No split needed. Cohesive host-migration leaf; controller extraction remains
  separate.

### Candidate Spec: Preview Parity And Import-Boundary Guards

Discovery purpose:
- Add regression guards that prevent the workbench from regrowing a separate
  preview renderer or importing UI modules into shared preview code.

Responsibilities:
- Functions/methods:
  - import-boundary check
  - duplicate-renderer scan
- Data structures/models:
  - guard report
- Dependencies/services:
  - test suite
  - shared preview controller
- Returns/outputs/signals:
  - test pass/fail diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: current test harness
  - Additions to existing reusable library/module: preview import-boundary tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - guards against duplicate heavy render paths
- Cross-screen reusable behavior:
  - protects CLI and workbench parity

Project readiness fields:
- Implementation owner/module:
  - future focused preview regression tests
- Chosen defaults / parameters:
  - workbench UI must call shared preview controller for scene application
- Test strategy:
  - import-boundary and duplicate-renderer guard tests
- Data ownership:
  - tests own boundary enforcement only
- Routes:
  - test suite to preview modules
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - shared controller extraction

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 15.5

Split decision:
- No split needed. Below split-review threshold and cohesive as a verification
  guard leaf.

## Change History

- 2026-07-07: Ran four manifest review loops, split the high-scoring shared
  controller candidate, and rescored all resulting candidates below `25`.
- 2026-07-07: Created supplemental architecture for extracting shared preview
  engine behavior from `impression.preview`.
