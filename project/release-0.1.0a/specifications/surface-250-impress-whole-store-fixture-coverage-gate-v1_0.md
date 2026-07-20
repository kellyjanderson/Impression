# Surface Spec 250: .impress Whole-Store Fixture Coverage Gate (v1.0)

## Overview

Prove the `.impress` whole-store fixture covers all promoted families, topology
rails, lifecycle records, trims, seams, identities, metadata, and operation
provenance.

## Backlink

- [Architecture: Surface Body Completion Architecture](../architecture/surface-body-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `.impress Whole-Store
Fixture Coverage Gate` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - whole-store fixture builder
  - all-family coverage assertion
- Data structures/models:
  - whole-store fixture record
  - topology rail payload
  - operation provenance payload
- Dependencies/services:
  - `.impress` root/store codecs
  - patch payload codecs
  - topology path records
- Returns/outputs/signals:
  - fixture coverage report
  - missing payload diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `.impress` codec specs and surface records
  - Additions to existing reusable library/module: whole-store fixture suite
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes temporary `.impress` fixture files during tests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - fixture size and deterministic traversal bounds
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `.impress` serialization modules and test fixtures

Routes:

- SurfaceBodyStore to writer to reader to validated SurfaceBodyStore

Reuse/extraction decision:

- extend existing `.impress` codec tests with whole-store fixture coverage

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- missing promoted-family payloads fail fixture coverage

Data ownership:

- `.impress` owns persistence
- surface store owns runtime object identity

## Behavior

The implementation must:

- create a whole-store fixture containing all promoted patch families and
  topology-relevant payloads
- assert identity, metadata, trims, seams, adjacency, rails, lifecycle records,
  and operation provenance survive round-trip
- report missing payload coverage by family and store area
- exclude mesh as canonical surface truth in whole-store fixtures

## Verification

Test strategy:

- all-family round-trip tests
- identity and provenance preservation tests
- missing-family coverage failure tests
- topology rail and lifecycle payload coverage tests

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

- Review for split. Cohesion reason: this is one fixture coverage gate after
  separating refusal/security behavior.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the whole-store fixture round-trips promoted families and topology payloads
- missing coverage fails deterministically
- mesh is not stored as canonical surface truth
