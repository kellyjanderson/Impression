# Surface Spec 274: Bounded Implicit Contour Extraction Adapter (v1.0)

## Overview

Extract surface-native declared-tolerance contour records for safe implicit
field surface intersections under the explicit safety and budget policy.

## Backlink

- [Architecture: Exact Surface Intersection Kernel Architecture](../architecture/exact-surface-intersection-kernel-architecture.md)

## Scope

This specification promotes the manifest candidate `Bounded Implicit Contour Extraction Adapter` into a final implementation leaf.

This specification covers:

- Extract surface-native declared-tolerance contour records for safe implicit
  field surface intersections under the explicit safety and budget policy.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - implicit intersection adapter
  - implicit contour extractor
- Data structures/models:
  - implicit contour record
  - contour extraction trace
- Dependencies/services:
  - implicit safety and budget policy
  - implicit field evaluator
- Returns/outputs/signals:
  - implicit contour records
  - extraction trace
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: implicit field evaluator
  - Additions to existing reusable library/module: bounded implicit
    intersection adapter
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - adapter only executes after safety policy approval
- Performance-sensitive behavior:
  - strict budgets for cells, depth, and iterations
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface_intersections.py`

Routes:

- approved implicit request to bounded contour extractor

Reuse/extraction decision:

- Existing code reused as-is: implicit field evaluator
- Additions to existing reusable library/module: bounded implicit
    intersection adapter
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- sampled contour records remain surface-native, not mesh truth

Data ownership:

- implicit family owns evaluation; contour record owns extracted curve truth

Open questions and resolved assumptions:

- contour extraction must not create a hidden tessellation fallback

Implementation prerequisites:

- implicit safety and budget policy must exist

## Behavior

The implementation must:

- Extract surface-native declared-tolerance contour records for safe implicit
  field surface intersections under the explicit safety and budget policy.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- safe implicit positive fixture and contour extraction trace tests

Additional verification requirements:

- add focused unit coverage for the records and helpers introduced by this specification
- add regression coverage for deterministic diagnostic text or diagnostic keys
- ensure reference artifact behavior is updated when this leaf changes model-output evidence

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 18.5

Split decision:

- Review for split. Cohesion reason: the adapter owns contour extraction only;
  residual/result classification is split separately.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Bounded Implicit Contour Extraction Adapter` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
