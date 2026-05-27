# Surface Spec 258: Surface CSG Cap Patch Family Selection (v1.0)

## Overview

Select and construct generated cap patch payloads for closed CSG cut regions
without deriving cap truth from a tessellated mesh.

## Backlink

- [Architecture: Higher-Order Surface CSG Solver Architecture](../architecture/higher-order-surface-csg-solver-architecture.md)

## Scope

This specification promotes the manifest candidate `Surface CSG Cap Patch Family Selection` into a final implementation leaf.

This specification covers:

- Select and construct generated cap patch payloads for closed CSG cut regions
  without deriving cap truth from a tessellated mesh.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - cap family selector
  - cap patch payload builder
  - unsupported cap diagnostic builder
- Data structures/models:
  - cap construction record
  - generated cap payload record
  - unsupported cap diagnostic
- Dependencies/services:
  - fragment graph
  - patch family registry
  - trim loops
- Returns/outputs/signals:
  - cap patch payloads
  - unsupported cap diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current trim and CSG curve records
  - Additions to existing reusable library/module: cap/cut-boundary helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes boolean result construction
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by cut-boundary count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/csg.py`

Routes:

- fragment graph to cap payload records to result shell assembly

Reuse/extraction decision:

- Existing code reused as-is: current trim and CSG curve records
- Additions to existing reusable library/module: cap/cut-boundary helpers
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- cap family must be explicit; unsupported cap cases refuse

Data ownership:

- cap construction owns generated patches; source fragments remain immutable

Open questions and resolved assumptions:

- non-planar cap generation may require blend/loft producer coordination

Implementation prerequisites:

- fragment graph builder must exist

## Behavior

The implementation must:

- Select and construct generated cap patch payloads for closed CSG cut regions
  without deriving cap truth from a tessellated mesh.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- planar cap, non-planar unsupported cap, and generated cap payload tests

Additional verification requirements:

- add focused unit coverage for the records and helpers introduced by this specification
- add regression coverage for deterministic diagnostic text or diagnostic keys
- ensure reference artifact behavior is updated when this leaf changes model-output evidence

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
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
- Total: 22.5

Split decision:

- Review for split. Cohesion reason: cap family selection and payload
  construction are one generated-patch contract; trim boundary construction is
  split separately.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Surface CSG Cap Patch Family Selection` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
