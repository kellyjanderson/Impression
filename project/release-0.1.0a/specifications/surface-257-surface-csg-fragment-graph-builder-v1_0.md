# Surface Spec 257: Surface CSG Fragment Graph Builder (v1.0)

## Overview

Build the transient classified fragment graph from intersection records,
existing trims, and operation selection rules.

## Backlink

- [Architecture: Higher-Order Surface CSG Solver Architecture](../architecture/higher-order-surface-csg-solver-architecture.md)

## Scope

This specification promotes the manifest candidate `Surface CSG Fragment Graph Builder` into a final implementation leaf.

This specification covers:

- Build the transient classified fragment graph from intersection records,
  existing trims, and operation selection rules.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - fragment graph builder
  - operation fragment selector
- Data structures/models:
  - fragment graph record
  - fragment provenance record
  - fragment classification edge record
- Dependencies/services:
  - intersection records
  - trim loops
  - operation plan
- Returns/outputs/signals:
  - classified fragment graph
  - graph construction diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current fragment/provenance records
  - Additions to existing reusable library/module: graph assembly helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes boolean internal execution state
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded graph traversal by fragment count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/csg.py`

Routes:

- intersection records to graph to reconstruction specs

Reuse/extraction decision:

- Existing code reused as-is: current fragment/provenance records
- Additions to existing reusable library/module: graph assembly helpers
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- invalid graph construction refuses; no mesh-derived graph substitute

Data ownership:

- fragment graph owns transient execution truth

Open questions and resolved assumptions:

- overlap-region graph edges must preserve both source operand references

Implementation prerequisites:

- exact intersection record contract from the intersection architecture

## Behavior

The implementation must:

- Build the transient classified fragment graph from intersection records,
  existing trims, and operation selection rules.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- graph construction, provenance, empty fragment, and operation-selection
    tests

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

- Review for split. Cohesion reason: this candidate now owns only the transient
  fragment graph; cap construction and durable shell reconstruction are split.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Surface CSG Fragment Graph Builder` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
