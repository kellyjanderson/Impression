# Surface Spec 266: Higher-Order Continuity Violation Locators (v1.0)

## Overview

Convert failed higher-order continuity validation into exact seam, boundary
use, parameter, and residual diagnostics.

## Backlink

- [Architecture: Higher-Order Seam Continuity Architecture](../architecture/higher-order-seam-continuity-architecture.md)

## Scope

This specification promotes the manifest candidate `Higher-Order Continuity Violation Locators` into a final implementation leaf.

This specification covers:

- Convert failed higher-order continuity validation into exact seam, boundary
  use, parameter, and residual diagnostics.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - violation locator builder
  - residual hot-spot selector
  - diagnostic formatter
- Data structures/models:
  - violation record
  - seam parameter locator
  - boundary-use diagnostic
- Dependencies/services:
  - continuity validation report
  - seam boundary-use records
- Returns/outputs/signals:
  - localized violation diagnostics
  - suggested authored fix hints
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: seam validation result records
  - Additions to existing reusable library/module: higher-order locator helpers
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
  - bounded by failed residual sample count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface.py`

Routes:

- validation report to localized diagnostics

Reuse/extraction decision:

- Existing code reused as-is: seam validation result records
- Additions to existing reusable library/module: higher-order locator helpers
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- diagnostics name the requested class and observed residual failure

Data ownership:

- validator owns observed residuals; locator owns user-facing diagnostic path

Open questions and resolved assumptions:

- fix hints should remain advice and never mutate source geometry

Implementation prerequisites:

- residual validation report must exist

## Behavior

The implementation must:

- Convert failed higher-order continuity validation into exact seam, boundary
  use, parameter, and residual diagnostics.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- exact locator tests for tangent and curvature failures

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
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 18.5

Split decision:

- Review for split. Cohesion reason: locator diagnostics are one user-facing
  report contract and are now separated from residual math.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Higher-Order Continuity Violation Locators` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
