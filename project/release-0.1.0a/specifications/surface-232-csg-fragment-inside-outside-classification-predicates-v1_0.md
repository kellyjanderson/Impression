# Surface Spec 232: CSG Fragment Inside/Outside Classification Predicates (v1.0)

## Overview

Classify split surface fragments as inside, outside, or on the opposing operand
using surface-native predicates.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification extracts fragment classification hidden inside Surface Spec
220.

## Responsibilities

- Functions/methods:
  - fragment sample selector
  - surface containment predicate
  - on-boundary classifier
- Data structures/models:
  - fragment classification record
  - classification diagnostic record
- Dependencies/services:
  - fragment arrangement records
  - shell containment helpers
- Returns/outputs/signals:
  - inside/outside/on classification
  - ambiguous classification diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: shell/body relation helpers where available
  - Additions to existing reusable library/module: CSG fragment predicates
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG classification behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by fragment count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- CSG classification helpers

Routes:

- trim fragments to operation selection

Reuse/extraction decision:

- add to existing CSG module/private helpers

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- ambiguous classification refuses rather than guessing or falling back to mesh

Data ownership:

- CSG owns classification records

## Behavior

The implementation must:

- classify fragments with deterministic surface-native predicates
- distinguish inside, outside, and on-boundary cases
- retain source patch and curve provenance for each classification
- reject ambiguous or numerically unstable classifications explicitly

## Verification

Test strategy:

- closed body, open body, touching, containment, and near-boundary fragment
  classification fixtures

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
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
- Total: 18.5

Split decision:

- Review for split. Cohesion reason: this is one classification predicate layer.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

