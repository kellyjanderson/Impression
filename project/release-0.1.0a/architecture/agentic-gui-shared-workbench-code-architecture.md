# Agentic GUI Shared Workbench Code Architecture

## Overview

This document records the initial code-sharing plan for a future agentic
Impression GUI and the existing Reference Review Workbench.

The agentic GUI is expected to begin as a close copy of the review workbench,
but its product workflow differs: it uses a workspace/file browser, code view,
model preview, and Codex chat interface instead of a fixture-review queue. The
copy must therefore be treated as a fork point for extracting reusable app
infrastructure, not as a permanent duplication.

## Relationship To Existing Architecture

- [Reference Review Qt Workbench UI](reference-review-qt-workbench-ui.md)
- [Reference Review Preview Engine Sharing Architecture](reference-review-preview-engine-sharing-architecture.md)
- [Reference Review Preview Payload Boundary Architecture](reference-review-preview-payload-boundary-architecture.md)
- [Reference Review Async Concurrency](reference-review-async-concurrency.md)
- [Reference Review Codex Sandbox](reference-review-codex-sandbox.md)

The existing review workbench remains the first concrete consumer. The agentic
GUI should be allowed to live in a separate repository, but shared app code
must not remain named after reference review once a second consumer exists.

## Repository And Package Boundary

Preferred direction:

- Create the agentic GUI as a separate application repository.
- Create a neutral shared workbench package rather than making the new app
  import `impression.devtools.reference_review`.
- Keep Impression core focused on modeling, file IO, preview semantics, and
  reusable rendering behavior.
- Let application-shell code live outside Impression core so the agentic GUI
  can evolve without turning Impression into the owner of every GUI workflow.

Recommended package shape:

- `impression-workbench-kit` or equivalent neutral package name.
- Python import namespace: `impression_workbench`.
- Optional Qt dependency group for UI widgets and QML assets.
- No hard dependency from the shared kit back to the `impression` package core
  unless that dependency is isolated in an adapter module.

Reasoning:

- If shared code stays under `impression.devtools.reference_review`, the
  agentic GUI inherits review-specific names and responsibilities.
- If shared code moves directly into the new agentic GUI repository, the
  existing review app must depend on the new app to reuse base components.
- If shared code moves into a neutral package, both apps can depend on it and
  keep their domain workflows separate.

## Initial Shared Code Candidates

Shared immediately:

- button bar and icon button primitives;
- status badge/pill components;
- panel header and diagnostic banner components;
- split-view sizing helpers and stable panel geometry rules;
- markdown/code text panel foundations;
- Codex/chat panel base components;
- preview display options and display-control command routing;
- preview toolbar/action-bar layout;
- QML resource packaging and resource smoke checks;
- Qt app bootstrap helpers that are not review-domain specific.

Shared after one more consumer proves the shape:

- queue/list row primitives generalized from fixture rows to file/workspace
  rows;
- async dispatcher and stale-result helpers;
- durable write helpers;
- preview render queue command envelopes;
- Codex sidecar/session abstractions.

Remain app-specific:

- fixture discovery and review status;
- reference artifact promotion;
- review notes lifecycle;
- workspace file discovery and file tree filtering;
- source code parsing/indexing;
- agent task/thread state;
- app-specific command names and approval/refusal policy.

Remain in Impression core:

- model construction APIs;
- `.impress` IO;
- preview scene semantics;
- dataset-to-renderer application policy;
- camera and interaction semantics shared by CLI preview and embedded preview
  hosts.

## Staged Extraction Plan

### Stage 1: Neutralize Before Copying

Before or alongside the first agentic GUI copy, identify review workbench files
that are already reusable and rename their conceptual ownership:

- `preview_controls.py` becomes shared preview display control logic.
- QML components such as `StatusBadge.qml`, `MarkdownPanel.qml`, and
  `CodexPanel.qml` become neutral workbench components.
- icon assets move under neutral workbench resource packaging.
- tests assert behavior through neutral names where possible.

This can happen inside the Impression repository first if needed, but the
namespace should already avoid `reference_review` terminology for reusable
parts.

### Stage 2: Create The Shared Package

Create a small shared package with only the code needed by both apps. Keep the
package narrow:

- component primitives;
- shared Qt packaging helpers;
- preview display controls;
- app-shell utility records;
- common tests and component gallery assets.

The package should not own fixture review, file browsing, or Codex task
workflow. Those belong to consuming apps.

### Stage 3: Rewire Reference Review

Update the Reference Review Workbench to import shared components from the
shared package while keeping domain panels under
`impression.devtools.reference_review`.

The review app should remain runnable through `impression-reference-review`.
The optional `reference-review-ui` extra may depend on the shared package once
packaging is ready.

### Stage 4: Build The Agentic GUI App

Create the agentic GUI repository as an app that depends on:

- `impression`;
- the shared workbench package;
- Qt/PySide dependencies;
- Codex integration dependencies or local process adapters.

The first screen can copy the review layout, but the left source model becomes
a file/workspace browser and the right panel becomes an agent chat/task panel.

## Import Boundary Rules

- Shared workbench components must not import
  `impression.devtools.reference_review`.
- Shared workbench components must not assume fixture ids, dirty/gold artifact
  status, or review approval state.
- Reference review code may import shared workbench components.
- The agentic GUI may import shared workbench components and Impression APIs.
- Impression core must not import either application shell.
- Preview rendering behavior should be shared through Impression preview
  controllers, not duplicated inside app UI packages.

## Open Questions

- Should the shared package be created as its own repository immediately, or
  incubated in a neutral namespace inside Impression until the agentic GUI repo
  exists?
- Should QML assets be the primary shared component format, or should early
  sharing prioritize Python Qt widgets because the live review shell is widget
  based?
- What is the first common Codex abstraction: chat transcript UI, task
  invocation protocol, sidecar process lifecycle, or context payload builder?
- Should the package name carry the Impression brand, or stay more general so
  the appkit can support non-Impression agentic workbenches later?

## Change History

- 2026-07-09: Initial architecture note created while deciding whether the
  agentic GUI should be a separate project and how its base code should be
  shared with the Reference Review Workbench.
