# Reference Review Spec 75a1: Preview Render Command Record Contract (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one immutable command-record contract with validation and focused unit tests.

## Overview

Define immutable preview render command records that can carry payload,
display, lifecycle, failure, and camera commands without mutating Qt widgets.

## Backlink

- [Reference Review Spec 75: Preview Render Command Queue](reference-review-75-preview-render-command-queue-v1_0.md)
- [Reference Review Spec 75a: Preview Render Command Records And Coalescing Queue](reference-review-75a-preview-render-command-records-and-coalescing-queue-v1_0.md)

## Source Manifest

- This leaf is split from ad hoc remediation spec 75a.
- Manifest score: 14.5

## Scope

This specification covers:

- `PreviewRenderCommandKind`
- `PreviewRenderCommand`
- `PreviewRenderCommandResult`
- command identity fields
- command validation

## Responsibilities

- Functions/methods:
  - command construction validation
- Data structures/models:
  - render command kind
  - render command
  - command result
- Dependencies/services:
  - existing payload identity shape
- Returns/outputs/signals:
  - command result record
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - New reusable library/module to create:
    - `src/impression/devtools/reference_review/ui/preview_render_queue.py`
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - commands are frozen and safe to pass between producer paths
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - no unsanitized raw diagnostic text is introduced by command records
- Performance-sensitive behavior:
  - none
- Cross-screen reusable behavior:
  - none

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/preview_render_queue.py`

Routes:

- imported by shell and preview widget/controller after this leaf

Reuse/extraction decision:

- create a local typed command module; do not add external queue dependencies

## Data And Defaults

Chosen defaults / parameters:

- command kinds are explicit enum values
- identity fields are optional only for neutral lifecycle commands

Data ownership:

- records own immutable command data only

Open questions and resolved assumptions:

- no external queue library is required

Implementation prerequisites:

- existing payload identity fields

## Behavior

The implementation must:

- reject unknown command kinds
- preserve identity fields for stale-result checks
- support command results suitable for queue diagnostics

## Verification

Test strategy:

- unit tests for enum validation, immutable records, identity fields, and
  command result values

Additional verification requirements:

- run queue-related tests and `git diff --check`

## Manifest Assessment

Score:

- Functions/methods: 1 x 2 = 2
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 0 x 2 = 0
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 14.5

Split decision:

- No split needed. Cohesion reason: this leaf defines only the immutable
  command contract.

## Refinement Status

Final leaf.

## Child Specifications

None.

## Acceptance

This specification is complete when command records exist and are covered by
unit tests.
