# Surface Spec 285: Legacy Primitive Mesh Assumption Inventory (v1.0)

## Overview

Find and classify tests/tools/docs that still assume public primitives return
mesh objects.

## Backlink

- [Architecture: Legacy Primitive Mesh Assumption Migration Architecture](../architecture/legacy-primitive-mesh-assumption-migration-architecture.md)

## Scope

This specification promotes the manifest candidate `Legacy Primitive Mesh Assumption Inventory` into a final implementation leaf.

This specification covers:

- Find and classify tests/tools/docs that still assume public primitives
  return mesh objects.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - repository scan helper
  - call-site classifier
- Data structures/models:
  - migration finding record
  - call-site classification record
- Dependencies/services:
  - primitive API names
  - tessellation boundary policy
- Returns/outputs/signals:
  - inventory report
  - stale assumption diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current tests and mesh boundary architecture
  - Additions to existing reusable library/module: optional scan helper
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
  - bounded repository text scan
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- tests and developer tooling

Routes:

- repository scan to rewrite plan

Reuse/extraction decision:

- Existing code reused as-is: current tests and mesh boundary architecture
- Additions to existing reusable library/module: optional scan helper
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- direct primitive-to-mesh-helper call sites are stale unless API is
    mesh-named

Data ownership:

- migration report owns call-site truth until rewritten

Open questions and resolved assumptions:

- some legacy examples may intentionally remain mesh-specific and should move
    to explicit mesh docs

Implementation prerequisites:

- none

## Behavior

The implementation must:

- Find and classify tests/tools/docs that still assume public primitives
  return mesh objects.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- inventory fixture with known stale and accepted call sites

Additional verification requirements:

- add focused unit coverage for the records and helpers introduced by this specification
- add regression coverage for deterministic diagnostic text or diagnostic keys
- ensure reference artifact behavior is updated when this leaf changes model-output evidence

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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
- Total: 13.5

Split decision:

- No split needed. The candidate is a bounded inventory/report task.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Legacy Primitive Mesh Assumption Inventory` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
