# Reference Review Workbench Architecture

## Overview

This document is the parent architecture for the Impression Reference Review
Workbench.

The workbench is a development-facing tool for reviewing dirty reference
fixtures by loading the actual model under test, inspecting it interactively,
recording durable review notes, and explicitly promoting accepted outputs to
gold references.

The original single architecture note was too broad. This parent now owns only
the system map, cross-document commitments, and lessons imported from the
ViewDown desktop-app work.

## Child Architecture Documents

- [Reference Review Fixture Source Contract](reference-review-fixture-source-contract.md)
- [Reference Review Qt Workbench UI](reference-review-qt-workbench-ui.md)
- [Reference Review Async Concurrency](reference-review-async-concurrency.md)
- [Reference Review Promotion And Notes Lifecycle](reference-review-promotion-and-notes-lifecycle.md)
- [Reference Review Codex Sandbox](reference-review-codex-sandbox.md)

## Related Impression Architecture

This workbench extends the active release reference evidence architecture:

- [Model Output Reference Verification](model-output-reference-verification.md)
- [Reference Artifact Promotion Architecture](reference-artifact-promotion-architecture.md)

It preserves the current rule: dirty artifacts are useful for bootstrap and
review, but they are not promoted evidence until a human explicitly promotes
them.

## ViewDown Lessons Applied

The split architecture follows lessons from the sibling ViewDown project:

- ViewDown split application shell, concurrency, components, rendering,
  navigation, and filesystem concerns into separate architecture documents.
  Impression should use the same separation here instead of letting the
  workbench become one "mega feature."
- ViewDown's `async-concurrency.md` made UI responsiveness explicit with typed
  envelopes, owner routing, request ids, stale-result guards, and UI-thread-only
  state mutation. The workbench adopts that shape for model loading, preview
  rendering, artifact generation, note writes, promotion, and Codex tool calls.
- ViewDown's UI work improved only after reusable controls and screenshot
  review were treated as durable assets. The workbench therefore needs a Qt
  component plan, screenshot/state matrix, and visible states before detailed
  implementation.
- ViewDown's PRs around sidebar affordances, settings polish, dark mode, and
  visual screenshot coverage show that "quiet" desktop UI fails when state,
  focus, and overflow are underdefined. The workbench UI docs must define those
  states up front.
- ViewDown's consultant profiles are useful review lenses:
  - Mara Vale: interaction state, workflow clarity, accessibility, and recovery.
  - Ilya Chen: visual hierarchy, spacing, component geometry, and screenshot QA.
  - Sabine Roth: command wording, diagnostic language, terminology, and notes.
- ViewDown's captured chat context reinforces that Qt Quick/QML, async-first
  architecture, and composable primitives were intentional from project
  inception, not cleanup decisions made after implementation drift.

## Architectural Commitments

- The model under test is the primary review subject.
- PNG, STL, slice, and diagnostic artifacts are derived evidence, not the thing
  being authored.
- Every reviewable fixture must expose a loadable source model or deterministic
  preview-compatible fixture entrypoint.
- The first implementation should use a PySide 6 application shell with Qt
  Quick/QML for the workbench chrome.
- The interactive 3D preview is isolated behind a preview bridge so the UI can
  evolve without tying every panel to PyVista internals.
- Blocking model load, tessellation, artifact generation, filesystem scan,
  note write, promotion, and Codex tool work must run off the UI thread.
- QML-visible state mutates only on the UI thread through typed completion
  envelopes.
- Codex may suggest and write candidate model files through an allowlisted
  broker, but it may not promote references, mutate gold artifacts, or write
  outside approved candidate and note roots.
- Promotion requires human review of both the live interactive model and its
  derived artifacts.

## System Components

- Fixture source contract: maps fixture ids to loadable model sources,
  entrypoints, parameters, generation commands, and feature context.
- Qt workbench UI: owns navigation, panels, state presentation, preview
  embedding, review actions, component reuse, and screenshot evidence.
- Async concurrency layer: owns task submission, worker isolation, result
  routing, stale-result rejection, cancellation, backpressure, and UI-thread
  handoff.
- Promotion and notes lifecycle: owns durable notes, unresolved review states,
  promotion provenance, gold artifact writes, and release-gate reports.
- Codex sandbox: owns constrained assistant context, tool allowlist, candidate
  model writes, regeneration requests, and refusal diagnostics.

## System Flow

```text
Active release reference roots
-> fixture source/context resolver
-> Qt workbench queue model
-> selected fixture source load
-> interactive preview bridge
-> optional derived artifact comparison
-> notes or Codex candidate iteration
-> regeneration from source
-> human promotion gate
-> gold reference artifacts and promotion provenance
```

## Specification Manifest For Discovery

### Candidate Spec: Reference Review Architecture Index And Domain Split

Discovery purpose:
- Keep the parent architecture, child architecture list, cross-document
  commitments, and ViewDown-derived lessons consistent.

Responsibilities:
- Functions/methods:
  - not applicable
- Data structures/models:
  - architecture index
  - cross-document commitment list
- Dependencies/services:
  - child architecture documents
  - active release reference evidence docs
- Returns/outputs/signals:
  - updated architecture navigation
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: child architecture manifests
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - not applicable
- Destructive/write behavior:
  - not applicable
- Security/privacy-sensitive behavior:
  - not applicable
- Performance-sensitive behavior:
  - not applicable
- Cross-screen reusable behavior:
  - parent commitments constrain all workbench UI and service specs

Project readiness fields:
- Implementation owner/module:
  - documentation-only architecture index
- Chosen defaults / parameters:
  - child documents own implementation candidates by domain
- Test strategy:
  - link check and architecture review
- Data ownership:
  - parent doc owns domain boundaries; child docs own design details
- Routes:
  - parent architecture to child architecture to manifest candidates
- Open questions / nuance discovered:
  - none
- Readiness blockers:
  - none

Score:
- Functions/methods: 0 x 2 = 0
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 0 x 2 = 0
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 7.5

Split decision:
- No split needed. This candidate is the parent index/governance leaf.

## Change History

- 2026-05-30: Split the original broad workbench architecture into a parent
  index and child architecture documents for fixture source contracts, Qt UI,
  concurrency, promotion/notes, and Codex sandboxing. Context: the original
  note mixed too many implementation domains and underdefined UI/concurrency.
- 2026-05-30: Created initial architecture for a preview-derived reference
  review workbench with fixture navigation, durable review notes, explicit
  promotion, and a sandboxed Codex sidecar.
