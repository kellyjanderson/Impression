# Reference Review Spec 80: Stabilization Merge Gate (v1.0)

Status: Final leaf after review pass 1

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own
verification surface.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one stabilization validation and merge-gate leaf.

## Overview

Define the proof required before the hybrid stabilization branch can be pushed,
opened as a pull request, and merged into the release working branch.

## Backlink

- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)
- [Reference Review Hybrid Stabilization Plan](../planning/reference-review-hybrid-stabilization-plan.md)

## Source Manifest

- Source candidate: `Stabilization Merge Gate`
- Source artifact: `project/release-0.1.0a/architecture/acd-reference-review-hybrid-stabilization.md`

## Scope

This specification covers:

- focused stabilization test command set
- real-entrypoint manual smoke
- merge readiness evidence
- PR handoff boundary
- post-merge full-kit migration handoff

## Responsibilities

- Functions/methods:
  - no product code functions required unless a test runner helper is added
- Data structures/models:
  - none expected
- Dependencies/services:
  - pytest or existing test runner
  - git
  - GitHub PR route
- Returns/outputs/signals:
  - passing focused tests
  - clean `git diff --check`
  - manual smoke evidence
  - pushed branch and PR
- UI surfaces/components:
  - full Reference Review app through manual smoke
- UI fields/elements:
  - fixture list, preview, notes, status, approve/decline, show-approved filter
- Reusable code plan:
  - Existing code reused as-is:
    - existing tests and entrypoint
  - Additions to existing reusable library/module:
    - none expected
  - New reusable library/module to create:
    - none expected
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - validation must include route-level coverage for async preview and UI
    handoff behavior from the preceding specs
- Destructive/write behavior:
  - manual smoke should use fixture data or temporary copies that do not
    accidentally promote unintended artifacts
- Security/privacy-sensitive behavior:
  - PR and smoke notes must not include unrelated local secrets or paths beyond
    workspace evidence
- Performance-sensitive behavior:
  - launch smoke must confirm the app is responsive enough to interact with
- Cross-screen reusable behavior:
  - merge gate validates fixture list, preview, notes, and review action routes

## Implementation Boundary

Owner/module:

- release validation commands
- Git/GitHub workflow for the stabilization branch
- focused tests under `tests/`

Routes:

- test command route
- `.venv/bin/impression-reference-review` manual smoke route
- branch push and PR route

Reuse/extraction decision:

- this leaf should not introduce new product abstractions
- full Workbench Kit migration is deferred until after merge

## Data And Defaults

Chosen defaults / parameters:

- test set includes preview payload, UI shell routing, notes/status, and
  display-control routing tests
- manual smoke uses real fixture data

Data ownership:

- validation evidence belongs to the branch/PR

Open questions and resolved assumptions:

- open for implementation: exact release working branch name to merge into
- resolved: no full-kit migration work should be added to satisfy this gate

Implementation prerequisites:

- Specs 76a, 76b, 77a, 77b, 78a, 78b, 79a, 79b, and 79c

## Behavior

The implementation must:

- run focused tests for preview payload behavior;
- run focused tests for UI shell launch/routing behavior;
- run focused tests for notes/status/promotion behavior;
- run focused tests for display-control state and command routing;
- run `git diff --check`;
- manually smoke the real app through the `.venv` entrypoint;
- commit the stabilization unit after validation;
- push the branch;
- create a PR;
- merge only after stabilization criteria are satisfied;
- leave full Workbench Kit migration as follow-up planning.

## Verification

Test strategy:

- this specification is itself the validation gate; it is verified by executing
  the listed tests and manual smoke

Additional verification requirements:

- record any commands that failed and the fixes applied before merge

## Readiness Fields

App type:

- mixed GUI and console entrypoint validation workflow

User/caller surface:

- tests, console entrypoint, GitHub PR

Invocation route:

- test commands, manual app launch, git push, PR creation, PR merge

Wiring owner/module:

- stabilization branch owner

Observable result:

- the stabilized app is validated and merged, and full kit migration remains
  separate follow-up work

Integration validation:

- focused tests, manual app smoke, and PR merge evidence

Prerequisites:

- Specs 76a, 76b, 77a, 77b, 78a, 78b, 79a, 79b, and 79c

Readiness blockers:

- none.

## Review Score

- Functions/methods: 1 x 2 = 2
- Data structures/models: 0 x 1 = 0
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 17

Split decision:

- No split. Cohesion reason: this is one validation gate, not product
  implementation; splitting test commands from PR/merge evidence would weaken
  the merge criterion.

## Refinement Status

Final leaf.

## Child Specifications

None at draft creation time.

## Acceptance

This specification is complete when the stabilization branch has passed the
defined validation gate, has been pushed, has a PR, and is merged into the
release working branch.
