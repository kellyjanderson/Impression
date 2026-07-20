# Reference Review Spec 76a: Launch Failure Baseline (v1.0)

Status: Final leaf after review pass 1

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one launch-baseline probe and failure-classification route.

## Overview

Capture the real Reference Review launch command and classify the current
launch failure without importing fixtures, building models, tessellating, or
constructing preview scene content during the probe.

## Backlink

- [Parent Spec 76](reference-review-76-launch-baseline-and-import-boundary-stabilization-v1_0.md)
- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)

## Scope

This specification covers:

- launch command and fixture-file baseline capture
- failure classification
- controlled shell or entrypoint probe
- diagnostic capture sufficient to choose the next stabilization route

This specification excludes:

- kit import decisions, owned by Spec 76b
- shell deferral implementation, owned by Spec 77a
- preview renderer fixes, owned by Specs 78a and 78b

## Responsibilities

- Functions/methods:
  - launch probe
  - failure classifier
- Data structures/models:
  - launch diagnostic record or equivalent captured result
- Dependencies/services:
  - `.venv` console entrypoint
  - Reference Review shell import/start route
- Returns/outputs/signals:
  - classified launch result
- UI surfaces/components:
  - shell startup route only
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: console entrypoint and shell bootstrap
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - probe must not trigger background fixture/model/preview work
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics avoid unrelated local data
- Performance-sensitive behavior:
  - probe fails quickly rather than hanging indefinitely
- Cross-screen reusable behavior:
  - none

## Implementation Boundary

Owner/module:

- Reference Review entrypoint and `src/impression/devtools/reference_review/ui/shell.py`

Routes:

- `.venv/bin/impression-reference-review`
- shell/bootstrap probe path used by tests

Reuse/extraction decision:

- no shared-kit adoption in this leaf

## Data And Defaults

Chosen defaults / parameters:

- baseline uses real fixture data chosen for stabilization smoke
- failure classes are import/dependency, QML/bootstrap, UI-thread
  blocking/handoff, preview renderer construction, and fixture/payload
  selection

Data ownership:

- launch diagnostics belong to the shell/entrypoint stabilization route

Open questions and resolved assumptions:

- resolved: the probe must not do render work

Implementation prerequisites:

- active hybrid stabilization ACD

## Behavior

The implementation must:

- record the launch command used for stabilization;
- expose or record the selected fixture file when one is used;
- classify launch failure into one of the ACD-defined categories;
- fail in a controlled way in test routes instead of beachballing indefinitely.

## Verification

Test strategy:

- focused launch probe test or controlled shell probe test;
- manual entrypoint smoke when automation cannot observe the GUI event loop.

Additional verification requirements:

- run `git diff --check`

## Review Score

- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19

Split decision:

- No split. Cohesion reason: this leaf only classifies launch failure and
  provides the baseline for later implementation.

## Readiness Fields

App type: mixed GUI and console entrypoint.
User/caller surface: `.venv/bin/impression-reference-review`.
Invocation route: console command to shell startup probe.
Wiring owner/module: Reference Review entrypoint and shell.
Observable result: launch failure is classified or shell reaches startup.
Integration validation: controlled shell/entrypoint probe plus manual smoke.
Readiness blockers: none.

## Refinement Status

Final leaf.

## Child Specifications

None.

## Acceptance

This specification is complete when the launch failure mode is reproducible,
classified, and recorded without performing fixture/model/render work in the
probe.
