# Surface Spec 254: Surface Body Completion Reference Evidence Matrix (v1.0)

## Overview

Define the required reference images, STL/tessellation artifacts, round-trip
fixtures, and negative diagnostics needed for every promoted model-outputting
capability.

## Backlink

- [Architecture: Surface Body Completion Architecture](../architecture/surface-body-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `Surface Body Completion
Reference Evidence Matrix` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - evidence matrix checker
  - reference artifact completeness assertion
- Data structures/models:
  - completion evidence record
  - reference fixture requirement record
- Dependencies/services:
  - reference artifact lifecycle
  - no-hidden-mesh-fallback tests
  - `.impress` round-trip fixtures
  - CSG/loft/family fixtures
- Returns/outputs/signals:
  - missing evidence report
  - promotion-blocking failure
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: reference artifact lifecycle tooling
  - Additions to existing reusable library/module: completion evidence matrix
    checker
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - may write dirty reference artifacts during verification runs
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - fixture generation bounded by named matrix
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- reference artifact tests and release verification tooling

Routes:

- capability matrix to required fixtures to reference/test gates

Reuse/extraction decision:

- extend existing reference artifact lifecycle instead of adding a separate
  artifact system

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- promoted model-outputting capabilities require positive and negative evidence

Data ownership:

- evidence matrix owns completion evidence requirements

## Behavior

The implementation must:

- map every promoted model-outputting capability to required reference evidence
- require positive model-output evidence and negative diagnostic evidence where
  unsupported states exist
- distinguish dirty generated artifacts from promoted baselines
- fail promotion when required images, STL/tessellation artifacts, round-trip
  fixtures, or refusal diagnostics are missing

## Verification

Test strategy:

- matrix completeness tests
- promotion-gate failure tests
- reference artifact lifecycle tests for dirty versus promoted artifacts

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 4 x 1 = 4
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 18.5

Split decision:

- Review for split. Cohesion reason: this is one evidence matrix and promotion
  gate; individual fixtures remain owned by feature specs.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- every promoted model-outputting capability has required evidence listed
- missing evidence blocks completion
- dirty artifacts are not mistaken for promoted baselines
