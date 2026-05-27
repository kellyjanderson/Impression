# Surface Spec 286: Preview And Reference Tests Explicit Tessellation Rewrite (v1.0)

## Overview

Rewrite preview/reference tests that need mesh so they tessellate surface
bodies explicitly.

## Backlink

- [Architecture: Legacy Primitive Mesh Assumption Migration Architecture](../architecture/legacy-primitive-mesh-assumption-migration-architecture.md)

## Scope

This specification promotes the manifest candidate `Preview And Reference Tests Explicit Tessellation Rewrite` into a final implementation leaf.

This specification covers:

- Rewrite preview/reference tests that need mesh so they tessellate surface
  bodies explicitly.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - test fixture rewrite
  - tessellation boundary helper
- Data structures/models:
  - none beyond existing tessellation records
- Dependencies/services:
  - `tessellate_surface_body`
  - reference artifact helpers
  - preview helpers
- Returns/outputs/signals:
  - passing preview/reference tests
  - no hidden mesh assumption diagnostics
- UI surfaces/components:
  - preview windows only as test consumers
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: tessellation API
  - Additions to existing reusable library/module: shared test helper if
    duplication appears
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - may write dirty reference artifacts during tests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - reference fixtures should remain bounded and deterministic
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `tests/test_preview_isolation.py`, `tests/test_reference_images.py`, and
    related helpers

Routes:

- public primitive to SurfaceBody to tessellation to preview/reference

Reuse/extraction decision:

- Existing code reused as-is: tessellation API
- Additions to existing reusable library/module: shared test helper if
    duplication appears
- New reusable library/module to create: none

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- surface-output fixtures call public primitives then tessellate explicitly

Data ownership:

- test fixture owns whether it proves surface output or mesh tool behavior

Open questions and resolved assumptions:

- some reference baselines may need invalidation after surface-first rewrite

Implementation prerequisites:

- reference artifact promotion policy

## Behavior

The implementation must:

- Rewrite preview/reference tests that need mesh so they tessellate surface
  bodies explicitly.
- preserve surface-native truth and avoid mesh fallback unless the route explicitly crosses the tessellation boundary
- emit deterministic diagnostics for invalid, unsupported, unsafe, or underconstrained states
- keep generated records stable enough for regression tests and reference artifacts

## Verification

Test strategy:

- full preview/reference tests after rewrite

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
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 19.5

Split decision:

- Review for split. Cohesion reason: preview and reference failures share the
  same explicit tessellation rewrite. Reference baseline promotion can split if
  artifact churn becomes large.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Preview And Reference Tests Explicit Tessellation Rewrite` is implemented in the owner/module named above
- manifest responsibilities are represented by explicit records, helpers, or diagnostics
- unsupported or unsafe cases fail with deterministic diagnostics rather than hidden fallback behavior
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
