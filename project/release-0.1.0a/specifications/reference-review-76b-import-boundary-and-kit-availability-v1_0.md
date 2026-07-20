# Reference Review Spec 76b: Import Boundary And Kit Availability (v1.0)

Status: Final leaf after review pass 1

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one deterministic import-boundary and kit-availability decision.

## Overview

Make the Reference Review import boundary deterministic: the review app must
not import `impression_gui`, and any `impression_workbench` helper adopted for
stabilization must be importable from the same `.venv` entrypoint used by the
app.

## Backlink

- [Parent Spec 76](reference-review-76-launch-baseline-and-import-boundary-stabilization-v1_0.md)
- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)

## Scope

This specification covers:

- clean-process import tests for Reference Review UI modules
- `impression_gui` import guard
- `.venv` `impression_workbench` availability check when kit helpers are used
- keep-local versus import-from-kit stabilization decision

This specification excludes launch failure classification, owned by Spec 76a.

## Responsibilities

- Functions/methods:
  - import-boundary probe
  - forbidden-import guard
  - kit availability probe
- Data structures/models:
  - import result or diagnostic record
- Dependencies/services:
  - `.venv` Python executable
  - Reference Review UI modules
  - `impression_workbench` package when helpers are used
- Returns/outputs/signals:
  - import pass/fail diagnostic
  - kit adoption decision
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: Reference Review package imports
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - import probes do not start app workers
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostic output avoids unrelated environment dumps
- Performance-sensitive behavior:
  - probes fail quickly
- Cross-screen reusable behavior:
  - import boundary protects all Reference Review app surfaces

## Implementation Boundary

Owner/module:

- Reference Review UI package imports and focused tests

Routes:

- clean-process import route
- `.venv` package availability route

Reuse/extraction decision:

- use kit helpers only when import-safe and API-compatible
- keep local helpers for this stabilization PR when kit adoption would create
  launch risk

## Behavior

The implementation must:

- prove Reference Review does not import `impression_gui`;
- prove any adopted `impression_workbench` helper is available from the app
  `.venv`;
- preserve local helpers when kit availability is not deterministic;
- avoid model, fixture, and renderer work during import checks.

## Verification

Test strategy:

- isolated subprocess import test;
- forbidden import assertion for `impression_gui`;
- kit availability assertion when kit imports are present.

Additional verification requirements:

- run `git diff --check`

## Review Score

- Functions/methods: 3 x 2 = 6
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 22.5

Split decision:

- No split. Cohesion reason: all responsibilities serve one import-boundary
  decision for the stabilization branch.

## Readiness Fields

App type: mixed GUI and console entrypoint.
User/caller surface: import side of `.venv/bin/impression-reference-review`.
Invocation route: package import and app bootstrap import path.
Wiring owner/module: Reference Review UI package and tests.
Observable result: app import path is deterministic and does not depend on
`impression_gui`.
Integration validation: clean-process import tests.
Readiness blockers: none.

## Refinement Status

Final leaf.

## Child Specifications

None.

## Acceptance

This specification is complete when the import boundary is deterministic and
kit adoption decisions are explicit for the stabilization branch.
