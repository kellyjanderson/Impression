# Surface Spec 225: Loft Reference QA Role Boundary (v1.0)

## Overview

Make loft reference/example tests distinguish modeled solid bodies from
annotations, labels, debug meshes, and tessellated views.

## Backlink

- [Architecture: Mesh Execution Tessellation Boundary Architecture](../architecture/mesh-execution-tessellation-boundary-architecture.md)

## Scope

This specification promotes the split manifest candidate `Loft Reference QA
Role Boundary` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - loft scene role helper
  - reference mesh-quality assertion
- Data structures/models:
  - scene item role record
  - expected QA mode record
- Dependencies/services:
  - `tests/test_loft.py`
  - loft real-world examples
- Returns/outputs/signals:
  - watertight assertion for model bodies
  - role-specific assertion for annotations
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: existing example outputs
  - Additions to existing reusable library/module: role-aware test helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes tests/examples only
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded tessellation only for model body QA
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `tests/test_loft.py` and loft real-world example metadata

Routes:

- example scene output to role-aware verification

Reuse/extraction decision:

- reuse tessellation helpers only at verification boundary

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- labels/annotations are not required to be watertight solids unless declared
  as model bodies

Data ownership:

- examples own scene item roles; tests own QA interpretation

## Behavior

The implementation must:

- add or infer scene item roles for loft reference examples
- apply watertight mesh checks only to modeled solid bodies
- apply annotation-appropriate checks to labels and drafting aids
- keep tessellation in QA/export checks only, not as modeled loft truth

## Constraints

- The current splitter-manifold issue must not be solved by treating the label
  as a mesh fallback body.
- Role-aware QA cannot hide a broken modeled manifold by reclassifying it as an
  annotation.

## Verification

Test strategy:

- splitter-manifold example passes with role-aware checks and no hidden loft
  mesh fallback

Automated or review verification must prove the modeled loft body remains
surface-native and any mesh analysis occurs only at the test/export boundary.

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Split decision:

- Review for split. Cohesion reason: this is one reference QA contract.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when loft reference tests verify solids,
annotations, and tessellated views according to role, and the splitter-manifold
test no longer confuses a surfaced label with a watertight model body.
