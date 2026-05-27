# Surface Spec 255: Surface CSG Family Pair Solver Registry (v1.0)

## Overview

Define the auditable registry that classifies every promoted family pair for
every boolean operation.

## Backlink

- [Architecture: Higher-Order Surface CSG Solver Architecture](../architecture/higher-order-surface-csg-solver-architecture.md)

## Scope

This specification promotes the manifest candidate `Surface CSG Family Pair Solver Registry` into a final implementation leaf.

This specification covers:

- Define the auditable registry that classifies every promoted family pair for
  every boolean operation.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - family-pair registry builder
  - coverage assertion
  - support-state lookup
- Data structures/models:
  - solver registry record
  - family-pair support record
  - unsupported-pair diagnostic
- Dependencies/services:
  - patch family capability matrix
  - CSG operation set
- Returns/outputs/signals:
  - support classification
  - missing-pair coverage failure
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current CSG family support records
  - Additions to existing reusable library/module: CSG solver registry helpers
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
  - constant-time pair lookup
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/csg.py`

Routes:

- boolean entrypoints to registry to planner

Reuse/extraction decision:

- Existing code reused as-is: current CSG family support records
- Additions to existing reusable library/module: CSG solver registry helpers
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- unknown pairs fail coverage; unsupported pairs must be explicit

Data ownership:

- CSG owns solver support truth

Open questions and resolved assumptions:

- sampled-family pairs need support-state names that do not imply mesh truth

Implementation prerequisites:

- none

## Behavior

The implementation must:

- Define the auditable registry that classifies every promoted family pair for
  every boolean operation.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- matrix completeness tests for all promoted family pairs

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
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Split decision:

- Review for split. Cohesion reason: registry, lookup, and coverage assertion
  are one durable matrix contract.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Surface CSG Family Pair Solver Registry` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
