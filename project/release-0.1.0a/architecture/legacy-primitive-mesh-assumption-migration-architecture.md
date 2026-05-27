# Legacy Primitive Mesh Assumption Migration Architecture

## Overview

This document defines the migration for old tests, tools, examples, and
preview helpers that still assume public primitive constructors return `Mesh`
objects by default.

The surface-first contract is now:

```python
body = make_box(...)
mesh = tessellate_surface_body(body).mesh
```

Legacy mesh use remains valid only when the API names the mesh boundary:

```python
mesh = make_box_mesh(...)
mesh = mesh_from_surface_body(make_box(...))
```

The migration must update callers, not weaken the primitive API back into a
mesh-default path.

## Related Architecture

This document extends:

- [Mesh Execution To Tessellation Boundary Architecture](mesh-execution-tessellation-boundary-architecture.md)
- [Surface-First Internal Model Architecture](surface-first-internal-model.md)
- [Surface Body Completion Architecture](surface-body-completion-architecture.md)
- [Testing Architecture](testing-architecture.md)

## Migration Scope

The migration covers:

- tests that pass `SurfaceBody` into mesh-only helpers
- preview/isolation tests that call `mesh_to_pyvista(make_box())`
- reference-image fixtures that combine primitive outputs as meshes
- command-line or example code that expects primitive mesh attributes
- internal tools that access `.vertices`, `.faces`, or `.n_faces` on primitive
  outputs
- docs that still teach primitive constructors as mesh constructors

It does not remove explicit mesh compatibility APIs.

## Components

### Call-Site Inventory

The inventory finds primitive calls in contexts requiring mesh:

- `mesh_to_pyvista(...)`
- `combine_meshes(...)`
- direct `.vertices`, `.faces`, `.n_faces`
- STL export helpers
- pyvista preview setup
- mesh repair/analysis utilities

Each call site is classified as:

- surface-native consumer
- tessellation-boundary consumer
- explicit mesh compatibility consumer
- obsolete mesh-primary test

### Migration Adapter Policy

When a call site needs mesh, it must cross the boundary visibly:

- use `make_*_mesh(...)` when testing legacy mesh compatibility
- use `tessellate_surface_body(...)` when testing surface truth through mesh
  output
- use `mesh_from_surface_body(...)` only when the compatibility adapter itself
  is under test

### Test Rewrite Rules

Old tests should be rewritten according to what they prove:

- surface API tests assert `SurfaceBody`, patch families, seams, metadata, and
  `.impress` behavior
- preview/export tests tessellate explicitly
- mesh utility tests call mesh-named constructors
- reference artifact tests decide whether the fixture is surface-output evidence
  or mesh-tool evidence

### Tooling And Documentation Update

Tooling and docs should name the boundary:

- "surface primitive" for public constructors
- "mesh compatibility primitive" for `make_*_mesh`
- "tessellation output" for preview/export/STL

This keeps examples from teaching hidden fallback by accident.

## Data Flow

```text
Legacy primitive call site
-> inventory classification
-> rewrite rule
-> surface assertion or explicit tessellation/mesh call
-> regression test proving no hidden mesh assumption remains
```

## Cross-Domain Decisions

### Do Not Reintroduce Backend Defaults

The migration should not solve test failures by making public primitives mesh by
default again.

### Mesh Tools Stay Mesh Tools

Mesh analysis, repair, STL, and preview can still use meshes. They must receive
mesh through named mesh constructors or tessellation.

### Reference Fixtures Need Intent

If a reference fixture is meant to prove surface output, it should start from
`SurfaceBody` and tessellate explicitly. If it is meant to prove mesh utility
behavior, it should call mesh-specific APIs.

## Specification Manifest for Discovery

### Candidate Spec: Legacy Primitive Mesh Assumption Inventory

Discovery purpose:
- Find and classify tests/tools/docs that still assume public primitives return
  mesh objects.

Responsibilities:
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

Project readiness fields:
- Implementation owner/module:
  - tests and developer tooling
- Chosen defaults / parameters:
  - direct primitive-to-mesh-helper call sites are stale unless API is
    mesh-named
- Test strategy:
  - inventory fixture with known stale and accepted call sites
- Data ownership:
  - migration report owns call-site truth until rewritten
- Routes:
  - repository scan to rewrite plan
- Open questions / nuance discovered:
  - some legacy examples may intentionally remain mesh-specific and should move
    to explicit mesh docs
- Readiness blockers:
  - none

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

### Candidate Spec: Preview And Reference Tests Explicit Tessellation Rewrite

Discovery purpose:
- Rewrite preview/reference tests that need mesh so they tessellate surface
  bodies explicitly.

Responsibilities:
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

Project readiness fields:
- Implementation owner/module:
  - `tests/test_preview_isolation.py`, `tests/test_reference_images.py`, and
    related helpers
- Chosen defaults / parameters:
  - surface-output fixtures call public primitives then tessellate explicitly
- Test strategy:
  - full preview/reference tests after rewrite
- Data ownership:
  - test fixture owns whether it proves surface output or mesh tool behavior
- Routes:
  - public primitive to SurfaceBody to tessellation to preview/reference
- Open questions / nuance discovered:
  - some reference baselines may need invalidation after surface-first rewrite
- Readiness blockers:
  - reference artifact promotion policy

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

### Candidate Spec: Mesh Compatibility API Documentation And Example Rewrite

Discovery purpose:
- Update docs and examples so mesh use is visibly mesh-specific and public
  primitives are taught as surface-body constructors.

Responsibilities:
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

Project readiness fields:
- Implementation owner/module:
  - project docs and examples
- Chosen defaults / parameters:
  - default examples return `SurfaceBody`
- Test strategy:
  - doc/example smoke tests where present
- Data ownership:
  - docs own user-facing teaching contract
- Routes:
  - docs/examples to public APIs
- Open questions / nuance discovered:
  - old mesh-primary examples may move under explicit compatibility docs
- Readiness blockers:
  - none

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

## Change History

- 2026-05-27: Critically reviewed and rescored the specification manifest.
  Context: the inventory, test rewrite, and documentation/example migration
  candidates remain bounded after review and did not need additional splitting.
- 2026-05-27: Added legacy primitive mesh assumption migration architecture and
  manifest. Context: broad tests still exposed old assumptions that public
  primitives return meshes by default.
