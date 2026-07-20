# Surface Spec 273: Implicit Intersection Safety And Budget Policy (v1.0)

## Overview

Define the safety, budget, and refusal policy for implicit field surface
intersections before any implicit solver executes.

## Backlink

- [Architecture: Exact Surface Intersection Kernel Architecture](../architecture/exact-surface-intersection-kernel-architecture.md)

## Scope

This specification promotes the manifest candidate `Implicit Intersection Safety And Budget Policy` into a final implementation leaf.

This specification covers:

- Define the safety, budget, and refusal policy for implicit field surface
  intersections before any implicit solver executes.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - implicit safety checker
  - budget/refusal checker
- Data structures/models:
  - implicit intersection budget
  - refusal diagnostic
- Dependencies/services:
  - implicit field safety policy
- Returns/outputs/signals:
  - safety decision
  - budget refusal diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: implicit safety policy
  - Additions to existing reusable library/module: implicit intersection
    budget policy
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - unsafe implicit fields must refuse before solver execution
- Performance-sensitive behavior:
  - strict budgets for cells, depth, and iterations
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface_intersections.py`

Routes:

- registry dispatch to safety policy to bounded adapter

Reuse/extraction decision:

- Existing code reused as-is: implicit safety policy
- Additions to existing reusable library/module: implicit intersection
    budget policy
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- unsafe field and budget exhaustion are deterministic refusal

Data ownership:

- policy owns executability; implicit family owns evaluation

Open questions and resolved assumptions:

- safety decisions must be serializable for diagnostic references

Implementation prerequisites:

- implicit evaluation and safety contracts

## Behavior

The implementation must:

- Define the safety, budget, and refusal policy for implicit field surface
  intersections before any implicit solver executes.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- unsafe refusal, budget refusal, and safe-decision tests

Additional verification requirements:

- add focused unit coverage for the records and helpers introduced by this specification
- add regression coverage for deterministic diagnostic text or diagnostic keys
- ensure reference artifact behavior is updated when this leaf changes model-output evidence

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 1 x 1 = 1
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
- Total: 17.5

Split decision:

- Review for split. Cohesion reason: implicit safety and budget policy are one
  executability gate; bounded solver execution is split separately.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Implicit Intersection Safety And Budget Policy` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
