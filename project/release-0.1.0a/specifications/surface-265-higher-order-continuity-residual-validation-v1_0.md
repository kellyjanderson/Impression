# Surface Spec 265: Higher-Order Continuity Residual Validation (v1.0)

## Overview

Validate C1/C2/G1/G2 requests by computing residuals from boundary derivative
samples without downgrading requested continuity.

## Backlink

- [Architecture: Higher-Order Seam Continuity Architecture](../architecture/higher-order-seam-continuity-architecture.md)

## Scope

This specification promotes the manifest candidate `Higher-Order Continuity Residual Validation` into a final implementation leaf.

This specification covers:

- Validate C1/C2/G1/G2 requests by computing residuals from boundary
  derivative samples without downgrading requested continuity.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - continuity validator
  - residual classifier
- Data structures/models:
  - continuity validation report
  - residual metrics
  - observed continuity class record
- Dependencies/services:
  - constraint records
  - boundary derivative evaluator
- Returns/outputs/signals:
  - pass/fail report
  - residual summary
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: seam validation result records
  - Additions to existing reusable library/module: higher-order report helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes higher-continuity request behavior from blanket refusal to
    validation where supported
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by seam sample count and derivative cost
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface.py`

Routes:

- constraint and derivative samples to validation report

Reuse/extraction decision:

- Existing code reused as-is: seam validation result records
- Additions to existing reusable library/module: higher-order report helpers
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- failing validation never downgrades requested continuity

Data ownership:

- validator owns observed continuity; seam owns requested continuity

Open questions and resolved assumptions:

- G2 residual thresholds need family-specific numerical stability notes

Implementation prerequisites:

- boundary derivative evaluator must exist

## Behavior

The implementation must:

- Validate C1/C2/G1/G2 requests by computing residuals from boundary
  derivative samples without downgrading requested continuity.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- positive C1/G1 fixtures and negative residual threshold fixtures

Additional verification requirements:

- add focused unit coverage for the records and helpers introduced by this specification
- add regression coverage for deterministic diagnostic text or diagnostic keys
- ensure reference artifact behavior is updated when this leaf changes model-output evidence

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
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
- Readiness blockers: 1 x 2 = 2
- Total: 19.5

Split decision:

- Review for split. Cohesion reason: residual validation is now separated from
  user-facing violation locator diagnostics.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Higher-Order Continuity Residual Validation` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
