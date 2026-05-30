# Surface Spec 276: Surface Reference Requirement Matrix (v1.0)

## Overview

Define the durable matrix connecting model-outputting capabilities to required
promoted evidence.

## Backlink

- [Architecture: Reference Artifact Promotion Architecture](../architecture/reference-artifact-promotion-architecture.md)

## Scope

This specification promotes the manifest candidate `Surface Reference Requirement Matrix` into a final implementation leaf.

This specification covers:

- Define the durable matrix connecting model-outputting capabilities to
  required promoted evidence.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - requirement matrix loader
  - capability coverage assertion
- Data structures/models:
  - reference requirement record
  - artifact class record
  - fixture contract record
- Dependencies/services:
  - surface completion evidence gate
  - reference artifact lifecycle
- Returns/outputs/signals:
  - missing requirement diagnostic
  - matrix coverage report
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current completion evidence matrix
  - Additions to existing reusable library/module: reference requirement records
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
  - bounded repository scan
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface.py` and reference test helpers

Routes:

- capability matrix to reference requirement matrix to progression

Reuse/extraction decision:

- Existing code reused as-is: current completion evidence matrix
- Additions to existing reusable library/module: reference requirement records
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- promoted model-outputting capabilities require at least one positive and
    applicable negative evidence path

Data ownership:

- release verification owns evidence requirements

Open questions and resolved assumptions:

- fixture contract version may live in metadata or sidecar manifest

Implementation prerequisites:

- none

## Behavior

The implementation must:

- Define the durable matrix connecting model-outputting capabilities to
  required promoted evidence.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- matrix completeness and missing requirement tests

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

- No split needed. The candidate is one evidence-matrix record contract.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Surface Reference Requirement Matrix` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
