# Surface Spec 284: Seam Continuity Negative Diagnostic Fixtures (v1.0)

## Overview

Create negative diagnostic fixtures for unsupported or failed higher-order
seam continuity requests.

## Backlink

- [Architecture: Reference Artifact Promotion Architecture](../architecture/reference-artifact-promotion-architecture.md)

## Scope

This specification promotes the manifest candidate `Seam Continuity Negative Diagnostic Fixtures` into a final implementation leaf.

This specification covers:

- Create negative diagnostic fixtures for unsupported or failed higher-order
  seam continuity requests.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - seam continuity fixture runner
  - snapshot comparator integration
- Data structures/models:
  - seam negative fixture record
  - expected diagnostic key record
  - expected continuity residual record
- Dependencies/services:
  - seam continuity refusal
  - diagnostic snapshot normalizer
  - continuity violation locators
- Returns/outputs/signals:
  - diagnostic snapshot
  - diagnostic drift failure
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current seam refusal diagnostics
  - Additions to existing reusable library/module: seam fixture records
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes dirty diagnostic snapshots during bootstrap
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded fixture count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- reference test helpers under `tests/`

Routes:

- failing seam validation to snapshot to matrix

Reuse/extraction decision:

- Existing code reused as-is: current seam refusal diagnostics
- Additions to existing reusable library/module: seam fixture records
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- seam fixtures pin requested continuity, observed residual, and locator path

Data ownership:

- seam fixtures own expected refusal contracts

Open questions and resolved assumptions:

- fixture snapshots must distinguish unsupported class from supported class
    with failed residual

Implementation prerequisites:

- diagnostic snapshot normalizer and continuity locator specs must exist

## Behavior

The implementation must:

- Create negative diagnostic fixtures for unsupported or failed higher-order
  seam continuity requests.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- unsupported continuity, failed C1/G1, and failed C2/G2 snapshot tests

Additional verification requirements:

- add focused unit coverage for the records and helpers introduced by this specification
- add regression coverage for deterministic diagnostic text or diagnostic keys
- ensure reference artifact behavior is updated when this leaf changes model-output evidence

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
- Readiness blockers: 1 x 2 = 2
- Total: 20.5

Split decision:

- Review for split. Cohesion reason: seam continuity negative fixtures are one
  diagnostic family and no longer bundle loft ambiguity.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Seam Continuity Negative Diagnostic Fixtures` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
