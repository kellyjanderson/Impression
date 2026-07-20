# Surface Spec 248: Loft Ambiguity Locator Diagnostics (v1.0)

## Overview

Make every unresolved loft ambiguity report exact topology, station, interval,
entity, relationship group, candidate lifecycle, and suggested rail locators.

## Backlink

- [Architecture: Surface Body Completion Architecture](../architecture/surface-body-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `Loft Ambiguity Locator
Diagnostics` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - ambiguity locator builder
  - suggested rail formatter
- Data structures/models:
  - ambiguity locator payload
  - suggested authored rail record
  - relationship group reference
- Dependencies/services:
  - topology path records
  - lifecycle records
  - loft diagnostics
- Returns/outputs/signals:
  - exact ambiguity diagnostic
  - missing-locator assertion failure
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: topology path and lifecycle identity records
  - Additions to existing reusable library/module: locator payload helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes diagnostic payload shape
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - locator creation bounded by ambiguity count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/loft.py`

Routes:

- planner ambiguity detection to locator builder to diagnostic aggregate

Reuse/extraction decision:

- use shared topology identity records rather than local string paths

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- ambiguity records without exact locators are invalid

Data ownership:

- topology owns entity identities
- loft diagnostics own ambiguity payloads

## Behavior

The implementation must:

- attach topology, station, interval, entity, relationship-group, and
  candidate-lifecycle locators to unresolved ambiguity records
- include suggested authored rails as non-mutating advice
- fail tests when any ambiguity diagnostic lacks a required locator field
- support split/merge, point birth/death, containment, and missing rail
  ambiguity classes

## Verification

Test strategy:

- split/merge ambiguity locator tests
- point birth/death locator tests
- containment ambiguity locator tests
- missing rail suggestion tests
- missing-locator assertion tests

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
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
- Total: 18.5

Split decision:

- Review for split. Cohesion reason: all locator fields serve the same
  diagnostic contract and must be complete together.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- ambiguity diagnostics always include exact locators
- suggested rails are advice only
- missing locator fields fail automated tests
