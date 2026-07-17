# Reference Review Spec 77: Non-Blocking Shell Bootstrap And Task Handoff (v1.0)

Status: Split parent - superseded by child specs 77a and 77b

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own
verification surface.
Draft count: 1 IWU.
Basis: one shell-bootstrap and task-handoff stabilization leaf. This count is a
draft creation estimate and must be verified by `review specs`.

## Overview

Make Reference Review launch initialize only the shell, UI hierarchy,
lightweight models, and deferred task lanes. Expensive work must start after
the event loop is alive and must return through typed UI-thread handoff.

## Backlink

- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)
- [Reference Review Hybrid Stabilization Plan](../planning/reference-review-hybrid-stabilization-plan.md)

## Source Manifest

- Source candidate: `Non-Blocking Shell Bootstrap And Task Handoff`
- Source artifact: `project/release-0.1.0a/architecture/acd-reference-review-hybrid-stabilization.md`

## Scope

This specification covers:

- launch-time deferral of expensive work
- UI-thread ownership of shell/QML/model mutation
- typed handoff for worker completions
- stale-completion rejection before UI mutation
- focused launch smoke proving startup does not select/build/render a fixture

## Responsibilities

- Functions/methods:
  - shell startup orchestration
  - deferred fixture refresh trigger
  - worker completion handoff handler
  - stale-result check before UI mutation
- Data structures/models:
  - existing Review Workbench message/result records or compatible kit records
- Dependencies/services:
  - Qt event loop
  - task dispatcher or equivalent worker lane
  - latest-request/staleness helper
- Returns/outputs/signals:
  - UI-safe fixture refresh request
  - UI-safe worker completion
  - visible shell-ready state
- UI surfaces/components:
  - Reference Review shell
  - fixture list model
  - preview pane placeholder state
- UI fields/elements:
  - initial selected fixture state
  - visible loading/diagnostic state where present
- Reusable code plan:
  - Existing code reused as-is:
    - current shell and dispatcher scaffolding
  - Additions to existing reusable library/module:
    - narrow use of kit staleness or Qt handoff helpers if import-safe
  - New reusable library/module to create:
    - none expected
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - no blocking UI-thread fixture import, model build, tessellation,
    renderer scene build, filesystem scan, or durable write during launch
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - sanitized failure handoff for background exceptions
- Performance-sensitive behavior:
  - startup must remain responsive before fixture data is loaded
- Cross-screen reusable behavior:
  - shell handoff behavior protects fixture list, preview, notes, and status

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/shell.py`
- `src/impression/devtools/reference_review/async_core/*` or compatible kit
  imports if accepted by Spec 76

Routes:

- app startup
- fixture list refresh
- worker completion to UI shell

Reuse/extraction decision:

- prefer existing local async core during stabilization unless kit imports are
  proven safe by Spec 76
- do not introduce a broad owner-route rewrite in this leaf

## Data And Defaults

Chosen defaults / parameters:

- startup selected fixture is none until fixture list data is safely available
- initial preview state is placeholder or non-rendering shell state

Data ownership:

- shell owns UI model mutation
- workers own expensive background work only

Open questions and resolved assumptions:

- resolved: no background completion may directly mutate UI state
- open for implementation: exact probe used to prove no fixture build occurs
  during shell bootstrap

Implementation prerequisites:

- Spec 76 import and launch baseline decisions

## Behavior

The implementation must:

- start the Qt shell without synchronous fixture source import;
- avoid renderer scene construction during visible bootstrap;
- defer fixture refresh until after the event loop is alive;
- route worker results through UI-safe handoff;
- reject stale completions before changing selected fixture, preview, notes, or
  status state;
- surface launch/bootstrap failures as diagnostics rather than a silent
  beachball where practical.

## Verification

Test strategy:

- focused shell startup test proving no fixture is selected/built during
  bootstrap;
- focused handoff test proving stale worker completion cannot mutate current
  UI-visible state;
- manual launch smoke before and after first fixture selection.

Additional verification requirements:

- run relevant UI shell tests
- run `git diff --check`

## Readiness Fields

App type:

- mixed GUI and console entrypoint

User/caller surface:

- Reference Review desktop app window launched through the console entrypoint

Invocation route:

- shell startup, deferred refresh trigger, worker completion handoff

Wiring owner/module:

- `src/impression/devtools/reference_review/ui/shell.py`

Observable result:

- the app opens responsively and fixture/background work cannot freeze the UI at
  launch

Integration validation:

- shell tests plus manual or automated launch smoke

Readiness blockers:

- depends on Spec 76 for import-boundary decisions

## Review Score

- Fresh review score: 42.5
- IWU recount: 2 IWU
- Split decision: split required. Non-blocking startup and completion handoff
  are independently failing async routes.

## Refinement Status

Split parent. Do not implement directly.

## Child Specifications

- [Reference Review Spec 77a: Non-Blocking Shell Startup](reference-review-77a-non-blocking-shell-startup-v1_0.md)
- [Reference Review Spec 77b: UI Handoff And Stale Completion Guard](reference-review-77b-ui-handoff-and-stale-completion-guard-v1_0.md)

## Split Coverage

| Parent responsibility | Child owner | Status |
| --- | --- | --- |
| launch-time deferral of expensive work | Spec 77a | Covered |
| UI-thread shell/QML/model ownership during startup | Spec 77a | Covered |
| typed handoff for worker completions | Spec 77b | Covered |
| stale-completion rejection before UI mutation | Spec 77b | Covered |
| startup smoke proving no selected/built/rendered fixture | Spec 77a | Covered |

## Acceptance

This specification is complete when shell bootstrap is non-blocking and worker
handoff cannot mutate UI state without current owner/request validation.
