# Surface Spec 287: Mesh Compatibility API Documentation And Example Rewrite (v1.0)

## Overview

Update docs and examples so mesh use is visibly mesh-specific and public
primitives are taught as surface-body constructors.

## Backlink

- [Architecture: Legacy Primitive Mesh Assumption Migration Architecture](../architecture/legacy-primitive-mesh-assumption-migration-architecture.md)

## Scope

This specification promotes the manifest candidate `Mesh Compatibility API Documentation And Example Rewrite` into a final implementation leaf.

This specification covers:

- Update docs and examples so mesh use is visibly mesh-specific and public
  primitives are taught as surface-body constructors.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - documentation rewrite
  - example smoke checks
- Data structures/models:
  - none
- Dependencies/services:
  - public primitive API
  - tessellation API
  - mesh compatibility API
- Returns/outputs/signals:
  - updated docs
  - example smoke pass
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: docs/examples
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - documentation edits
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - none
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- project docs and examples

Routes:

- docs/examples to public APIs

Reuse/extraction decision:

- Existing code reused as-is: docs/examples
- Additions to existing reusable library/module: none
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- default examples return `SurfaceBody`

Data ownership:

- docs own user-facing teaching contract

Open questions and resolved assumptions:

- old mesh-primary examples may move under explicit compatibility docs

Implementation prerequisites:

- none

## Behavior

The implementation must:

- Update docs and examples so mesh use is visibly mesh-specific and public
  primitives are taught as surface-body constructors.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- doc/example smoke tests where present

Additional verification requirements:

- add focused unit coverage for the records and helpers introduced by this specification
- add regression coverage for deterministic diagnostic text or diagnostic keys
- ensure reference artifact behavior is updated when this leaf changes model-output evidence

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 0 x 1 = 0
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 0 x 2 = 0
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 12.5

Split decision:

- No split needed. The candidate is a bounded documentation and example
  migration.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Mesh Compatibility API Documentation And Example Rewrite` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
