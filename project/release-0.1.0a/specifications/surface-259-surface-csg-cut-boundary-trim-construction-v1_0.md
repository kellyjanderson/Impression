# Surface Spec 259: Surface CSG Cut Boundary Trim Construction (v1.0)

## Overview

Construct cut-boundary trim loops and exposed/open boundary diagnostics from
classified fragment graph cuts and generated cap records.

## Backlink

- [Architecture: Higher-Order Surface CSG Solver Architecture](../architecture/higher-order-surface-csg-solver-architecture.md)

## Scope

This specification promotes the manifest candidate `Surface CSG Cut Boundary Trim Construction` into a final implementation leaf.

This specification covers:

- Construct cut-boundary trim loops and exposed/open boundary diagnostics from
  classified fragment graph cuts and generated cap records.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - cut-boundary trim builder
  - open/shared boundary classifier
- Data structures/models:
  - cut-boundary record
  - boundary exposure diagnostic
  - trim attachment record
- Dependencies/services:
  - fragment graph
  - cap construction records
  - trim loops
- Returns/outputs/signals:
  - cut-boundary trim loops
  - boundary exposure diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current trim and CSG curve records
  - Additions to existing reusable library/module: cut-boundary trim helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes boolean result construction
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by cut-boundary count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/csg.py`

Routes:

- fragment graph and cap records to cut-boundary trim records

Reuse/extraction decision:

- Existing code reused as-is: current trim and CSG curve records
- Additions to existing reusable library/module: cut-boundary trim helpers
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- trim attachment must preserve source and generated-cap provenance

Data ownership:

- cut-boundary records own trim attachment truth for reconstruction

Open questions and resolved assumptions:

- exposed boundaries must identify whether they are legal open shells or
    invalid closed-body breaks

Implementation prerequisites:

- fragment graph and cap patch records must exist

## Behavior

The implementation must:

- Construct cut-boundary trim loops and exposed/open boundary diagnostics from
  classified fragment graph cuts and generated cap records.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- curved cut boundary, open boundary refusal, shared boundary stability tests

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

- Review for split. Cohesion reason: cut-boundary trim construction is one
  trim-attachment contract and no longer bundles cap payload generation.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Surface CSG Cut Boundary Trim Construction` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
