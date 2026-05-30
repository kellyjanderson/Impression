# Surface Spec 311: Advanced Family CSG Refusal And No-Fallback Diagnostics (v1.0)

## Overview

Convert unsupported advanced-family CSG classifications into deterministic non-executable operation plans and no-hidden-mesh-fallback diagnostics.

## Backlink

- [Architecture: Advanced Patch Family Implementation Completion Architecture](../architecture/advanced-patch-family-implementation-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `Advanced Family CSG Refusal And No-Fallback Diagnostics` into a final
implementation leaf.

This specification covers:

- the records, helpers, diagnostics, operation rows, and evidence named by the
  owning manifest candidate
- the exact implementation boundary required by the owning architecture
- deterministic refusal behavior for unsupported, unsafe, unavailable, or
  under-evidenced states

## Responsibilities

- Functions/methods:
  - operation plan refusal builder
  - no-fallback assertion helper
- Data structures/models:
  - non-executable CSG plan diagnostic
  - fallback violation diagnostic
- Dependencies/services:
  - advanced CSG support matrix
  - no-hidden-mesh-fallback tests
- Returns/outputs/signals:
  - non-executable plan diagnostics
  - fallback regression failures
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current CSG diagnostic records
  - Additions to existing reusable library/module: advanced-family refusal rows
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - implicit operations preserve safety refusal reason
- Performance-sensitive behavior:
  - none
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/csg.py`, `tests/test_no_hidden_mesh_fallback.py`

Routes:

- support matrix to operation planner to refusal diagnostic

Reuse/extraction decision:

- extend current operation planning diagnostics

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- every unsupported advanced-family pair refuses before solver execution

Data ownership:

- CSG operation planner owns refusal diagnostics

Open questions and resolved assumptions:

- none

Implementation prerequisites:

- advanced CSG support classification matrix

## Behavior

The implementation must:

- satisfy every responsibility listed above with explicit records, helpers,
  diagnostics, or operation-matrix entries
- preserve authored surface truth and never use mesh as a hidden fallback
- keep unavailable, unsupported, unsafe, or non-applicable states explicit and
  inspectable
- make readiness and availability evidence deterministic enough for release
  progression and future completion reports

## Verification

Test strategy:

- refusal diagnostics and no-hidden-mesh-fallback tests

Additional verification requirements:

- add focused unit coverage for each new record, helper, diagnostic, and matrix
  row introduced by this leaf
- add negative coverage for malformed, unsupported, unsafe, missing-evidence, or
  non-applicable states named by this leaf
- include no-hidden-mesh-fallback assertions where the leaf touches authoring,
  operation selection, CSG, seams, tessellation, or reference evidence
- update reference or diagnostic fixtures when this leaf changes visible model
  output or durable refusal behavior

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 0 x 2 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 16.5

Split decision:

- Review for split. Cohesion reason: this candidate owns only refusal and
  fallback diagnostics.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Advanced Family CSG Refusal And No-Fallback Diagnostics` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers,
  diagnostics, operation rows, or evidence gates
- unsupported, unsafe, unavailable, or non-applicable cases fail with
  deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired
  test specification created from this leaf
