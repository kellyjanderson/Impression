# Surface Spec 262: Surface CSG Result Provenance Mapping (v1.0)

## Overview

Preserve stable operand, fragment, cap, and generated-boundary provenance for
CSG result shells and diagnostics.

## Backlink

- [Architecture: Higher-Order Surface CSG Solver Architecture](../architecture/higher-order-surface-csg-solver-architecture.md)

## Scope

This specification promotes the manifest candidate `Surface CSG Result Provenance Mapping` into a final implementation leaf.

This specification covers:

- Preserve stable operand, fragment, cap, and generated-boundary provenance
  for CSG result shells and diagnostics.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - provenance mapper
  - provenance diagnostic builder
  - equivalent operand ordering normalizer
- Data structures/models:
  - result provenance map
  - provenance diagnostic
  - operand ordering normalization record
- Dependencies/services:
  - assembled SurfaceBody candidate
  - fragment graph
  - cap/cut-boundary records
- Returns/outputs/signals:
  - result provenance map
  - provenance diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: provenance records
  - Additions to existing reusable library/module: CSG provenance mapping
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes boolean provenance behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by face and provenance-map count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/csg.py`

Routes:

- assembled result and fragment records to provenance map to diagnostics

Reuse/extraction decision:

- Existing code reused as-is: provenance records
- Additions to existing reusable library/module: CSG provenance mapping
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- provenance keys must be deterministic for equivalent operand ordering

Data ownership:

- provenance map owns source traceability; result body owns geometry truth

Open questions and resolved assumptions:

- generated cap provenance must remain inspectable without exposing mesh ids

Implementation prerequisites:

- result shell assembly and cut-boundary records must exist

## Behavior

The implementation must:

- Preserve stable operand, fragment, cap, and generated-boundary provenance
  for CSG result shells and diagnostics.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- source face, generated cap, cut-boundary, and ordering-normalization tests

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

- Review for split. Cohesion reason: provenance mapping is one traceability
  contract and no longer bundles validity handoff.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Surface CSG Result Provenance Mapping` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
