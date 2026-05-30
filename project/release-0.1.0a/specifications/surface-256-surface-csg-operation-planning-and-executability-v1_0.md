# Surface Spec 256: Surface CSG Operation Planning And Executability (v1.0)

## Overview

Produce boolean plans that accumulate unsupported pair and invalid topology
diagnostics before execution.

## Backlink

- [Architecture: Higher-Order Surface CSG Solver Architecture](../architecture/higher-order-surface-csg-solver-architecture.md)

## Scope

This specification promotes the manifest candidate `Surface CSG Operation Planning And Executability` into a final implementation leaf.

This specification covers:

- Produce boolean plans that accumulate unsupported pair and invalid topology
  diagnostics before execution.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - operation planner
  - executability gate
  - diagnostic accumulator
- Data structures/models:
  - CSG operation plan
  - pair dispatch record
  - plan diagnostic record
- Dependencies/services:
  - solver registry
  - operand preparation
  - seam validation
- Returns/outputs/signals:
  - executable plan
  - non-executable diagnostic bundle
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: boolean operand records
  - Additions to existing reusable library/module: planner records and gate
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes boolean execution refusal behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by patch-pair count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/csg.py`

Routes:

- public boolean API to planner to executor

Reuse/extraction decision:

- Existing code reused as-is: boolean operand records
- Additions to existing reusable library/module: planner records and gate
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- plans with unresolved unsupported diagnostics cannot execute

Data ownership:

- operation plan owns executability state

Open questions and resolved assumptions:

- coincident cases may need separate exact-overlap records

Implementation prerequisites:

- none

## Behavior

The implementation must:

- Produce boolean plans that accumulate unsupported pair and invalid topology
  diagnostics before execution.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- multi-diagnostic plan tests and executor refusal tests

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
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Split decision:

- Review for split. Cohesion reason: planning and executability are one
  lifecycle invariant; intersection execution is split separately.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Surface CSG Operation Planning And Executability` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
