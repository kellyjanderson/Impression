# Surface Spec 242: Surface Body Completion Capability And Evidence Gate (v1.0)

## Overview

Define the release-level capability matrix and evidence gate required before
Impression can claim complete authored surface-body support.

## Backlink

- [Architecture: Surface Body Completion Architecture](../architecture/surface-body-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `Surface Body Completion
Capability And Evidence Gate` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - capability matrix audit command or maintained checker
  - completion gate evaluator
- Data structures/models:
  - completion capability record
  - evidence status record
- Dependencies/services:
  - patch-family manifests
  - CSG manifests
  - loft manifests
  - `.impress` manifests
- Returns/outputs/signals:
  - pass/fail completion report
  - missing evidence diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: architecture work tracker and existing
    capability/spec matrices
  - Additions to existing reusable library/module: release verification helper
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
  - bounded repository scan
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `project/release-0.1.0a/planning/` and release verification tooling

Routes:

- architecture work tracker to progression and verification evidence

Reuse/extraction decision:

- extend existing planning/checker patterns; no new runtime geometry module

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- incomplete evidence blocks completion claims

Data ownership:

- completion matrix owns release-level support truth

## Behavior

The implementation must:

- collect surface-body capability status from patch-family, CSG, loft,
  `.impress`, primitive, feature, and verification tracks
- distinguish specified, implemented, verified, unsupported, and retired states
- fail completion when evidence is missing, stale, or only architectural
- produce diagnostics that name the missing spec, implementation owner, and
  required evidence type
- never infer completion from checked progression alone

## Verification

Test strategy:

- documentation checker tests
- unit tests for matrix classification
- negative tests for missing evidence and stale checked items

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 4 x 1 = 4
- Returns/outputs/signals: 2 x 1 = 2
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
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 15.5

Split decision:

- Small. This remains one release-gate spec because it coordinates evidence
  rather than implementing geometry.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- completion claims are gated by explicit evidence
- missing evidence diagnostics are actionable
- no surface-body completion status can pass from documentation alone
