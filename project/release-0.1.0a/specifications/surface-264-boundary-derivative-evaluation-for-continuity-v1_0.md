# Surface Spec 264: Boundary Derivative Evaluation For Continuity (v1.0)

## Overview

Provide derivative and normal evaluation along patch boundaries for promoted
families.

## Backlink

- [Architecture: Higher-Order Seam Continuity Architecture](../architecture/higher-order-seam-continuity-architecture.md)

## Scope

This specification promotes the manifest candidate `Boundary Derivative Evaluation For Continuity` into a final implementation leaf.

This specification covers:

- Provide derivative and normal evaluation along patch boundaries for promoted
  families.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - boundary evaluator
  - first derivative evaluator
  - second derivative evaluator
- Data structures/models:
  - boundary derivative sample
  - residual summary
- Dependencies/services:
  - patch family evaluators
  - seam boundary-use records
- Returns/outputs/signals:
  - derivative samples
  - unsupported-family diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: patch evaluation APIs
  - Additions to existing reusable library/module: boundary derivative helpers
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
  - bounded by seam sample count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface.py`

Routes:

- seam boundary use to patch evaluator to continuity validator

Reuse/extraction decision:

- Existing code reused as-is: patch evaluation APIs
- Additions to existing reusable library/module: boundary derivative helpers
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- exact derivative preferred; numeric derivative allowed only with residual
    metadata

Data ownership:

- patch families own evaluation; seam layer owns boundary sampling

Open questions and resolved assumptions:

- subdivision and implicit derivative support may need declared-tolerance mode

Implementation prerequisites:

- promoted family derivative API coverage

## Behavior

The implementation must:

- Provide derivative and normal evaluation along patch boundaries for promoted
  families.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- analytic planar/revolution/ruled derivative tests plus unsupported-family
    diagnostics

Additional verification requirements:

- add focused unit coverage for the records and helpers introduced by this specification
- add regression coverage for deterministic diagnostic text or diagnostic keys
- ensure reference artifact behavior is updated when this leaf changes model-output evidence

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
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
- Total: 17.5

Split decision:

- Review for split. Cohesion reason: derivative evaluation is one shared seam
  service; family-specific evaluator work may split after this contract lands.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Boundary Derivative Evaluation For Continuity` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
