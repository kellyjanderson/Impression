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

### Candidate Spec: Shared Preview Scene Controller Extraction

Discovery purpose:
- Extract reusable scene, camera, style, and edge policy from CLI preview so
  CLI preview and Reference Review Workbench share the same preview behavior.

Responsibilities:
- Functions/methods:
  - preview scene controller
  - scene application method
  - camera reset method
  - edge policy helper
- Data structures/models:
  - preview style record
  - interaction policy record
- Dependencies/services:
  - `impression.preview`
  - PyVista plotter protocol
- Returns/outputs/signals:
  - configured plotter scene
  - camera reset result or diagnostic
- UI surfaces/components:
  - CLI preview host
  - workbench preview widget host
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: CLI preview private scene and camera behavior
  - Additions to existing reusable library/module: public controller API in
    `impression.preview`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - controller mutates only the caller-owned render surface on the caller's UI
    thread
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - avoids duplicate mesh conversion paths and repeated renderer creation
- Cross-screen reusable behavior:
  - shared by CLI preview and workbench preview

Project readiness fields:
- Implementation owner/module:
  - `src/impression/preview.py`
- Chosen defaults / parameters:
  - workbench style uses dark blue background and light orange object color via
    style configuration
- Test strategy:
  - controller import tests, CLI delegation tests, and workbench import-boundary
    tests
- Data ownership:
  - preview controller owns preview semantics but not renderer lifecycle
- Routes:
  - CLI host or workbench widget to shared preview controller
- Open questions / nuance discovered:
  - exact controller class names may change during implementation
- Readiness blockers:
  - none

Score:
- Functions/methods: 4 x 2 = 8
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
- Readiness blockers: 0 x 2 = 0
- Total: 26.5

Split decision:
- No split needed. This is one coherent extraction leaf around shared preview
  semantics.

## Change History

- 2026-07-07: Created supplemental architecture for extracting shared preview
  engine behavior from `impression.preview`.
