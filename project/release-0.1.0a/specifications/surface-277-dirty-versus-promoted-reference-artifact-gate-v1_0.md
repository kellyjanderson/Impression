# Surface Spec 277: Dirty Versus Promoted Reference Artifact Gate (v1.0)

## Overview

Ensure dirty generated artifacts cannot satisfy promoted completion evidence.

## Backlink

- [Architecture: Reference Artifact Promotion Architecture](../architecture/reference-artifact-promotion-architecture.md)

## Scope

This specification promotes the manifest candidate `Dirty Versus Promoted Reference Artifact Gate` into a final implementation leaf.

This specification covers:

- Ensure dirty generated artifacts cannot satisfy promoted completion
  evidence.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - artifact state classifier
  - promotion gate
  - contract invalidation checker
- Data structures/models:
  - artifact state record
  - fixture contract version record
  - promotion diagnostic
- Dependencies/services:
  - reference artifact lifecycle tooling
  - filesystem paths
- Returns/outputs/signals:
  - promoted/missing/dirty report
  - invalidation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: reference artifact lifecycle skill/process
  - Additions to existing reusable library/module: promotion gate helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - may create dirty artifacts during verification; must not promote
    automatically
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded file existence and checksum checks
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- reference test helpers under `tests/`

Routes:

- test output to dirty artifact store to promotion gate

Reuse/extraction decision:

- Existing code reused as-is: reference artifact lifecycle skill/process
- Additions to existing reusable library/module: promotion gate helpers
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- dirty artifacts fail promotion evidence

Data ownership:

- reference artifact lifecycle owns file state

Open questions and resolved assumptions:

- checksum versus metadata versioning should be decided by implementation

Implementation prerequisites:

- none

## Behavior

The implementation must:

- Ensure dirty generated artifacts cannot satisfy promoted completion
  evidence.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- dirty-only, clean-present, partial-missing, and invalidated-contract tests

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
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Split decision:

- Review for split. Cohesion reason: dirty/promoted classification and
  promotion refusal are one artifact lifecycle gate.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Dirty Versus Promoted Reference Artifact Gate` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
