# Surface Spec 260: Surface CSG Result Shell Assembly (v1.0)

## Overview

Assemble surviving fragments and generated caps into durable result shells,
without running final seam or validity policy.

## Backlink

- [Architecture: Higher-Order Surface CSG Solver Architecture](../architecture/higher-order-surface-csg-solver-architecture.md)

## Scope

This specification promotes the manifest candidate `Surface CSG Result Shell Assembly` into a final implementation leaf.

This specification covers:

- Assemble surviving fragments and generated caps into durable result shells,
  without running final seam or validity policy.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - result shell assembler
  - reconstruction diagnostic builder
- Data structures/models:
  - result shell assembly record
  - reconstruction diagnostic
  - shell ordering record
- Dependencies/services:
  - fragment graph
  - cap/cut-boundary records
- Returns/outputs/signals:
  - reconstructed SurfaceBody candidate
  - invalid reconstruction diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: shell assembly and validity records
  - Additions to existing reusable library/module: result reconstruction
    helpers
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
  - bounded by fragment and shell count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/csg.py`

Routes:

- fragment/cap records to SurfaceBody candidate

Reuse/extraction decision:

- Existing code reused as-is: shell assembly and validity records
- Additions to existing reusable library/module: result reconstruction
    helpers
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- invalid reconstruction refuses; no geometric invention

Data ownership:

- result body owns durable truth; fragment graph remains transient

Open questions and resolved assumptions:

- multi-shell ordering must be stable across equivalent operand order

Implementation prerequisites:

- fragment graph and cap/cut-boundary records must exist

## Behavior

The implementation must:

- Assemble surviving fragments and generated caps into durable result shells,
  without running final seam or validity policy.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- empty, single-shell, multi-shell, nested-shell, and stable ordering tests

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
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 19.5

Split decision:

- Review for split. Cohesion reason: shell assembly is one durable-body
  construction contract; validity and provenance handoff are split separately.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Surface CSG Result Shell Assembly` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
