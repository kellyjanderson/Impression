# Reference Review Spec 76: Launch Baseline And Import Boundary Stabilization (v1.0)

Status: Split parent - superseded by child specs 76a and 76b

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own
verification surface.
Draft count: 1 IWU.
Basis: one launch-baseline and import-boundary stabilization leaf. This count
is a draft creation estimate and must be verified by `review specs`.

## Overview

Establish the Reference Review app's real launch failure mode and make the
`.venv` entrypoint import boundary deterministic before deeper stabilization
work proceeds.

## Backlink

- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)
- [Reference Review Hybrid Stabilization Plan](../planning/reference-review-hybrid-stabilization-plan.md)

## Source Manifest

- Source candidate: `Launch Baseline And Import Boundary Stabilization`
- Source artifact: `project/release-0.1.0a/architecture/acd-reference-review-hybrid-stabilization.md`

## Scope

This specification covers:

- launch command and fixture-file baseline capture
- crash/hang classification
- `.venv` importability for `impression_workbench`
- guard against `impression_gui` imports from Reference Review
- narrow keep-local vs import-from-kit decision for currently duplicated
  helpers

## Responsibilities

- Functions/methods:
  - entrypoint launch probe or shell bootstrap probe
  - import-boundary probe
  - optional import guard test
- Data structures/models:
  - launch baseline record or diagnostic text used by tests
- Dependencies/services:
  - `.venv` console entrypoint
  - `impression_workbench` package when available
  - Reference Review UI package imports
- Returns/outputs/signals:
  - deterministic pass/fail launch baseline result
  - deterministic pass/fail import-boundary result
- UI surfaces/components:
  - Reference Review shell startup route
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is:
    - Reference Review console entrypoint
    - current shell bootstrap code
  - Additions to existing reusable library/module:
    - none unless a narrow import probe helper is already present
  - New reusable library/module to create:
    - none expected
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - launch baseline must not start background work as part of the import probe
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics must not expose unrelated local paths beyond the failing module
    or expected workspace paths
- Performance-sensitive behavior:
  - launch/import probes must fail quickly and avoid model build/tessellation
- Cross-screen reusable behavior:
  - import boundary protects the full review app

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/shell.py`
- `src/impression/devtools/reference_review/ui/__init__.py`
- existing entrypoint configuration for `impression-reference-review`
- focused test modules under `tests/`

Routes:

- console route: `.venv/bin/impression-reference-review`
- import route: importing Reference Review UI modules in a clean process

Reuse/extraction decision:

- Import from `impression_workbench` only when the package is available through
  the same `.venv` route used by the app.
- Do not import from `impression_gui`.
- If a kit helper is not reliably importable, keep the local Reference Review
  helper for this stabilization pass and leave the full-kit migration as
  follow-up work.

## Data And Defaults

Chosen defaults / parameters:

- baseline fixture file defaults to the current real Reference Review fixture
  file used for manual smoke, not demo-only data
- import probes run without selecting or rendering a fixture

Data ownership:

- launch diagnostic ownership stays with the shell/entrypoint route
- kit availability is a packaging/import concern, not a fixture concern

Open questions and resolved assumptions:

- resolved: Reference Review must not depend on `impression_gui`
- open for implementation: whether `impression_workbench` is already installed
  into the active `.venv`

Implementation prerequisites:

- existing Reference Review entrypoint
- existing ACD and stabilization plan

## Behavior

The implementation must:

- record or encode the real launch command used for stabilization;
- classify the current failure as import/dependency, QML/bootstrap,
  UI-thread blocking/handoff, preview renderer construction, or
  fixture/payload selection failure;
- verify `impression_workbench` importability from the app `.venv` before
  replacing local helpers with kit imports;
- prevent Reference Review from importing `impression_gui`;
- avoid fixture source import, model build, tessellation, or renderer scene
  construction during import-boundary tests.

## Verification

Test strategy:

- focused import-boundary test for Reference Review UI modules;
- focused check that `impression_gui` is not imported by Reference Review;
- focused or manual launch smoke through the `.venv` entrypoint.

Additional verification requirements:

- run `git diff --check`
- record any manual launch command used for the stabilization evidence

## Readiness Fields

App type:

- mixed GUI and console entrypoint

User/caller surface:

- `.venv/bin/impression-reference-review`

Invocation route:

- console command imports and starts the Reference Review Qt shell

Wiring owner/module:

- Reference Review entrypoint and UI shell modules

Observable result:

- the app reaches shell startup or fails with a classified diagnostic instead
  of an unclassified crash/hang

Integration validation:

- import-boundary tests and manual or automated entrypoint smoke

Readiness blockers:

- none known from the ACD; `review specs` must verify this remains true

## Review Score

- Fresh review score: 33.5
- IWU recount: 2 IWU
- Split decision: split required. Launch failure classification and import
  boundary/kit availability are independently failing stabilization routes.

## Refinement Status

Split parent. Do not implement directly.

## Child Specifications

- [Reference Review Spec 76a: Launch Failure Baseline](reference-review-76a-launch-failure-baseline-v1_0.md)
- [Reference Review Spec 76b: Import Boundary And Kit Availability](reference-review-76b-import-boundary-and-kit-availability-v1_0.md)

## Split Coverage

| Parent responsibility | Child owner | Status |
| --- | --- | --- |
| launch command and fixture-file baseline capture | Spec 76a | Covered |
| crash/hang classification | Spec 76a | Covered |
| `.venv` importability for `impression_workbench` | Spec 76b | Covered |
| guard against `impression_gui` imports | Spec 76b | Covered |
| keep-local vs import-from-kit decision | Spec 76b | Covered |

## Acceptance

This specification is complete when the launch/import boundary is deterministic,
Reference Review has no `impression_gui` dependency, and kit import decisions
are explicit for the stabilization branch.
