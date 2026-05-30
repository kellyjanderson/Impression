# Surface Spec 269: Intersection Curve Result And Degeneracy Records (v1.0)

## Overview

Define a common intersection result record that can express curves, points,
overlap regions, quality, residuals, and degeneracy.

## Backlink

- [Architecture: Exact Surface Intersection Kernel Architecture](../architecture/exact-surface-intersection-kernel-architecture.md)

## Scope

This specification promotes the manifest candidate `Intersection Curve Result And Degeneracy Records` into a final implementation leaf.

This specification covers:

- Define a common intersection result record that can express curves, points,
  overlap regions, quality, residuals, and degeneracy.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - result normalizer
  - degeneracy classifier
- Data structures/models:
  - intersection curve record
  - overlap region record
  - degeneracy record
- Dependencies/services:
  - tolerance policy
  - patch-local curve mapper
- Returns/outputs/signals:
  - normalized intersection result
  - degeneracy diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current CSG curve primitive records
  - Additions to existing reusable library/module: shared result records
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
  - normalization bounded by curve count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface_intersections.py`

Routes:

- solver output to CSG trim/seam consumers

Reuse/extraction decision:

- Existing code reused as-is: current CSG curve primitive records
- Additions to existing reusable library/module: shared result records
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- tangent and overlap cases are explicit result classes

Data ownership:

- intersection module owns transient intersection truth

Open questions and resolved assumptions:

- overlap region representation must support both trim loops and shell splits

Implementation prerequisites:

- none

## Behavior

The implementation must:

- Define a common intersection result record that can express curves, points,
  overlap regions, quality, residuals, and degeneracy.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- result normalization and degeneracy classification tests

Additional verification requirements:

- add focused unit coverage for the records and helpers introduced by this specification
- add regression coverage for deterministic diagnostic text or diagnostic keys
- ensure reference artifact behavior is updated when this leaf changes model-output evidence

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
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
- Readiness blockers: 0 x 2 = 0
- Total: 14.5

Split decision:

- No split needed. The candidate is a cohesive result-record contract.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Intersection Curve Result And Degeneracy Records` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
