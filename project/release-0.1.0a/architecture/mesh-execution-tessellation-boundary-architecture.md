# Mesh Execution To Tessellation Boundary Architecture

## Overview

Impression now treats `SurfaceBody` as the canonical modeled result.

That means mesh may remain a strong downstream representation, but mesh must no
longer be a peer executor for authored modeling operations.

This architecture records the current mesh-execution audit and defines the
target boundary:

```text
authored inputs
-> topology / surface planning
-> SurfaceBody
-> tessellation
-> Mesh consumers
```

Mesh execution is acceptable only after the tessellation boundary, or in
explicit mesh-tool workflows whose inputs and outputs are honestly mesh.

## Backlinks

- [Surface-First Internal Model Architecture](surface-first-internal-model.md)
- [Surface-Native Capability Replacement Architecture](surface-native-capability-replacement-architecture.md)
- [Mesh Analysis and Repair Architecture](surface-mesh-decommission-architecture.md)
- [Full Surface Patch Family Architecture](full-surface-patch-family-architecture.md)
- [.impress Surface-Native File Format Architecture](impress-surface-native-file-format-architecture.md)

## Boundary Rule

The rule is:

- modeling APIs produce `SurfaceBody`, surfaced assemblies, or topology/surface
  planning artifacts
- tessellation APIs produce `Mesh`
- preview, export, STL, repair, and analysis consume mesh explicitly
- no modeling API silently falls back to mesh when surfaced execution is
  incomplete

The anti-pattern is:

```text
authored inputs
-> mesh executor
-> mesh result
-> surface wrapper or file persistence
```

That path loses authored topology, stable identity, patch family information,
seams, trims, correspondence decisions, and `.impress` fidelity.

## Audit Summary

The audit found two categories of mesh code.

The first category is legitimate boundary code. These modules either tessellate
surface truth or consume mesh as an explicit downstream product:

- `src/impression/modeling/tessellation.py`
- `src/impression/cli.py` export and preview merge paths
- mesh analysis, repair, and explicit mesh utility paths when presented as
  mesh tools

The second category is mesh execution inside modeling capability. These paths
must move behind the tessellation boundary or become legacy mesh tools:

- `src/impression/modeling/primitives.py`
- `src/impression/modeling/loft.py`
- `src/impression/modeling/csg.py`
- `src/impression/modeling/text.py`
- `src/impression/modeling/drafting.py`
- `src/impression/modeling/heightmap.py`
- `src/impression/modeling/hinges.py`
- `src/impression/modeling/group.py`
- `src/impression/modeling/transform.py`
- `src/impression/modeling/ops.py`
- `src/impression/modeling/_ops_mesh.py`
- mesh color helpers that only understand `Mesh`

This does not mean every line in those modules is wrong. It means their
modeling-facing mesh paths need to be reclassified and migrated.

## Detailed Findings

### Primitives

`primitives.py` still defaults primitive factories to `backend="mesh"` and
returns `Mesh | SurfaceBody`.

Examples include box, cylinder, ngon, polyhedron, sphere, torus, cone, and
related helpers. Surface alternatives exist, which is good, but mesh remains
the default modeled result.

Target architecture:

- public authored primitive constructors return `SurfaceBody` by default
- old mesh constructors become explicit tessellation or legacy mesh helpers
- internal mesh generation helpers are not used as surfaced modeling truth

### Primitive Mesh Path Excision

Primitive excision is the first concrete migration target because the surface
replacement functions already exist in `_surface_primitives.py`.

The primitive migration has four mechanical goals:

- remove `backend="mesh"` from authored primitive defaults
- make public primitive constructors return `SurfaceBody`
- move retained mesh constructors behind tessellation or an explicit legacy
  mesh namespace
- remove primitive mesh helper use from surfaced modeling modules

The current public constructors and their required replacements are:

| Current API | Current mesh path | Surface replacement | Required excision |
| --- | --- | --- | --- |
| `make_box` | `_box_mesh(...)` | `make_surface_box(...)` | Make `SurfaceBody` the default return; keep mesh only as `make_box_mesh(...)` or tessellated output. |
| `make_cylinder` | `_circular_frustum_mesh(radius, radius, ...)` | `make_surface_cylinder(...)` | Preserve authored axis, caps, and color metadata in surface form; move mesh cylinder to tessellation/legacy. |
| `make_ngon` | `_circular_frustum_mesh(..., sides, ...)` | `make_surface_ngon(...)` | Preserve `side_length` resolution in the surface constructor; mesh ngon becomes explicit mesh helper only. |
| `make_polyhedron` | `_regular_polyhedron_data(...)` plus triangulation | `make_surface_polyhedron(...)` | Surface polyhedron owns planar patch faces; mesh triangulation is a tessellation detail. |
| `make_nhedron` | wrapper around mesh-default `make_polyhedron` | `make_surface_nhedron(...)` or surface `make_polyhedron(...)` | Keep compatibility alias, but do not route through mesh defaults. |
| `make_sphere` | `_sphere_mesh(...)` | `make_surface_sphere(...)` | Surface sphere owns analytic/spherical patch data; resolution becomes tessellation guidance, not modeled mesh density. |
| `make_torus` | `_torus_mesh(...)` | `make_surface_torus(...)` | Surface torus owns analytic/periodic patch data; `n_theta` and `n_phi` become tessellation guidance. |
| `make_cone` | `_circular_frustum_mesh(bottom, top, ...)` | `make_surface_cone(...)` | Surface cone/frustum owns axis and radius metadata; mesh frustum helper moves out of authored path. |
| `make_prism` | `_rectangular_frustum_mesh(...)` | `make_surface_prism(...)` | Surface prism owns planar side and cap patches; mesh helper is retained only for legacy or tessellation. |

The private mesh helper functions in `primitives.py` are not allowed to remain
as casual dependencies of authored modeling:

- `_orient_mesh(...)`
- `_box_mesh(...)`
- `_sphere_mesh(...)`
- `_torus_mesh(...)`
- `_circular_frustum_mesh(...)`
- `_rectangular_frustum_mesh(...)`

Each helper must be handled in one of three ways:

- deleted after all callers use surface constructors plus tessellation
- moved to a legacy mesh module such as `modeling/legacy_mesh_primitives.py`
- moved behind the tessellation boundary if it becomes the implementation of a
  surface patch tessellator

The last option is allowed only when the helper consumes a `SurfaceBody`,
`SurfacePatch`, or tessellation request. It must not consume authored primitive
arguments directly as the modeled truth.

#### Primitive API Shape

The target authored API shape is:

```python
body = make_box(size=(1, 2, 3), center=(0, 0, 0))
mesh = tessellate_surface_body(body).mesh
```

Compatibility may exist, but should be visibly mesh-specific:

```python
mesh = make_box_mesh(size=(1, 2, 3), center=(0, 0, 0))
```

or:

```python
mesh = mesh_from_surface_body(make_box(size=(1, 2, 3)))
```

The target API must not require users to remember `backend="surface"` to stay on
the canonical path.

#### Primitive Dependency Cleanup

Downstream modeling modules currently depend on primitive mesh defaults.

The primitive excision must update those callers at the same time:

- drafting calls to `make_line`, `make_arrow`, and primitive composition must
  pass through surfaced drafting bodies
- hinge mesh solids that use `make_box`, `make_cylinder`, or mesh booleans must
  move to surfaced hinge assemblies or explicit mesh compatibility
- CSG and hull code must not receive primitive mesh defaults from authored
  constructors
- tests and examples must tessellate explicitly when they need `Mesh`

#### Primitive Acceptance Criteria

Primitive mesh excision is complete when:

- every public primitive constructor returns `SurfaceBody` by default
- no public primitive constructor has `backend="mesh"` as its default
- no surfaced modeling module imports private primitive mesh helpers
- mesh primitive helpers live only in tessellation or explicit legacy mesh
  modules
- `.impress` serialization of primitives records surface patch semantics, not
  primitive-generated triangle soup
- tests prove that preview/export still receive mesh through
  `tessellate_surface_body(...)`

### Loft

`loft.py` contains both a mesh executor and a surface executor.

The most important audit hits are:

- `loft_execute_plan(...) -> Mesh`
- `emit_mesh_faces_from_sample_correspondence(...) -> Mesh`
- public loft paths that still return mesh for plan execution
- `make_surface_consumer_collection(...)` handoff from loft surface execution

This is the highest-risk separation issue because loft correspondence is now
topology-owned. The correspondence system should feed the surface executor.
Mesh face emission should not be the canonical executor of that plan.

Target architecture:

- `LoftPlan` execution returns `SurfaceBody`
- topology-owned correspondence is consumed by surface patch generation
- mesh emission from correspondence moves to tessellation or explicit debug
  tooling
- any mesh executor spec is renamed or reworked as a legacy compatibility or
  tessellation-boundary spec
- hidden fallback from surface loft failure to mesh loft is forbidden

### CSG

`csg.py` still exposes `BooleanBackend = Literal["manifold", "surface"]` with
`backend="manifold"` defaults and mesh/manifold execution for public boolean
operations.

Surface boolean operands and structured results exist, but canonical execution
is incomplete in places, and mesh/manifold remains the default route for
`boolean_union`, `boolean_difference`, and `boolean_intersection`.

Target architecture:

- public authored boolean operations are surface-native
- manifold/mesh CSG remains only an explicit mesh tool, repair tool, analysis
  tool, or temporary compatibility path
- unsupported surface booleans fail with diagnostics instead of silently
  returning mesh truth

### Text And Drafting

`text.py` and `drafting.py` both support surface paths, but still default to
`backend="mesh"` and contain direct mesh extrusion or direct mesh construction.

Text has a strong surfaced route through profiles and surface linear extrude.
Drafting has partial surface support for lines, planes, arrows, dimensions, and
text handoff.

Target architecture:

- authored text returns surfaced extrusion by default
- drafting geometry returns surfaced annotation bodies or consumer records by
  default
- mesh text extrusion and mesh drafting shapes become tessellation products or
  legacy debug helpers

### Heightmap And Displacement

`heightmap.py` builds mesh heightfields first. Its `backend="surface"` path wraps
triangle mesh faces into planar `SurfaceBody` patches, and displacement
tessellates a surface input before producing a displaced mesh and wrapping it
back into triangle patches.

That is a mesh-derived surfaced wrapper, not native authored surface modeling.

Target architecture:

- heightmaps define native sampled surface patches or explicit subdivision /
  displacement surface patches
- displacement modifies surfaced geometry or creates a surfaced displacement
  representation
- mesh sampling remains a tessellation or analysis strategy, not the modeled
  result

### Hinges

`hinges.py` remains an Impression-owned feature-builder area. It contains
surface assembly descriptions, but mesh paths must remain explicit compatibility
or mesh-tool APIs.

Target architecture:

- hinge authorship produces surface assemblies or `SurfaceBody`
  output
- printable mesh solids are generated by tessellating those assemblies
- any retained mesh-only generators are explicitly legacy or mesh-tool APIs

### Grouping And Transforms

`group.py` defines `MeshGroup`, and `transform.py` mutates `Mesh | MeshGroup`.

This is useful for old mesh workflows, but it cannot be the canonical
composition and transform layer for authored models.

Target architecture:

- surfaced composition is represented by `SurfaceBody`, `SurfaceShell`, surfaced
  assemblies, or scene/consumer collections
- transforms operate on surfaced bodies without tessellating
- `MeshGroup` remains a legacy mesh utility or downstream consumer helper

### Ops And Mesh Utilities

`ops.py` and `_ops_mesh.py` contain mesh hull and manifold conversion helpers.

Target architecture:

- explicit mesh hull remains a mesh tool
- authored hull or convex-envelope modeling is surface-native or explicitly
  unsupported until surfaced support exists
- mesh utility modules are physically separated from surface modeling modules

## Components

### Surface Modeling Kernel

Owns authored modeling truth:

- topology
- sections, paths, and regions
- patch families
- trims, seams, adjacency, and stable identity
- surfaced CSG and loft execution
- surfaced assemblies for specialized generators

This layer may estimate, validate, and plan. It may not return mesh as the
canonical modeled result.

### Tessellation Boundary

Owns conversion from surfaced truth to mesh:

- tessellation requests and tolerances
- patch sampling
- seam-aware vertex reconciliation
- mesh metadata derived from surface identity
- consumer records and preview/export payloads

The existing `tessellation.py` module is the current home for this boundary.
As the system matures, mesh-producing helpers from modeling modules should move
into this boundary or into explicit mesh-tool modules.

### Mesh Consumer Layer

Owns downstream mesh use:

- preview
- render payloads
- STL and mesh export
- mesh QA
- mesh repair
- slicing and section analysis
- imported mesh workflows

This layer may create and mutate mesh because mesh is its declared working
type.

### Legacy Mesh Compatibility Layer

Owns old public APIs during migration.

Compatibility APIs may continue returning mesh temporarily, but they must:

- be named, documented, or warned as mesh-primary
- not be used by new surfaced modeling paths
- not persist as `.impress` canonical geometry
- have removal or replacement specs

### Diagnostics Layer

Owns surfacing incomplete capability honestly.

If a surface executor cannot complete a model, the result should be:

- a structured unsupported result
- a diagnostic error
- a planning artifact explaining the missing capability

It should not be an automatic mesh fallback presented as successful surfaced
modeling.

## Data Flow

### Canonical Authored Flow

```text
user-authored operation
-> topology and surface planning
-> SurfaceBody / surfaced assembly
-> optional .impress persistence
-> tessellation request
-> Mesh
-> preview / export / analysis
```

### Explicit Mesh Tool Flow

```text
imported mesh or tessellated SurfaceBody
-> explicit mesh tool
-> mesh report, repaired mesh, or exported mesh
```

### Forbidden Fallback Flow

```text
surface operation fails
-> mesh operation succeeds
-> result is treated as canonical surface body
```

## Public API Policy

New public modeling APIs should not use `backend="mesh"` defaults.

Allowed API shapes are:

- `make_box(...) -> SurfaceBody`
- `tessellate_surface_body(body, request=...) -> TessellationResult`
- `mesh_from_surface_body(body, ...) -> Mesh` only as an explicit compatibility
  or consumer helper
- `make_box_mesh(...) -> Mesh` only if documented as a mesh utility or legacy
  compatibility API

Disallowed API shapes are:

- `make_box(..., backend="mesh") -> Mesh | SurfaceBody` as the main authored
  API
- `loft(..., backend="mesh")` as a normal modeling option
- surface wrappers around mesh-generated triangle soups as final authored
  truth
- automatic mesh fallbacks when a surface operation is unsupported

## `.impress` Persistence Policy

`.impress` files persist surface-native truth.

They may reference cached tessellation artifacts for convenience, but those
artifacts are rebuildable consumer data. They are not the canonical model.

The file format must reject or quarantine:

- mesh-only modeled bodies masquerading as surface bodies
- triangle-patch wrappers created only because surface modeling was missing
- mesh fallback results without original authored topology and surface
  semantics

Imported mesh may be supported in `.impress`, but it must be represented as an
explicit imported mesh object, not as native surface modeling truth.

## Migration Plan

1. Inventory every public `Mesh | SurfaceBody` return path and every
   `backend="mesh"` default.
2. Classify each mesh-producing path as tessellation boundary, mesh consumer,
   legacy compatibility, explicit mesh tool, or invalid modeling fallback.
3. Rework loft so the topology-owned correspondence plan feeds only the surface
   executor for canonical modeling.
4. Move mesh face emission from loft into tessellation/debug tooling.
5. Change primitive, text, drafting, hinge, and heightmap authored
   defaults to surface output once their surface paths are complete.
6. Split mesh utility code into explicit mesh-tool modules so modeling modules
   do not import mesh execution casually.
7. Add tests that assert modeling APIs do not return `Mesh` unless the API name
   or module explicitly declares mesh tooling.
8. Add tests that unsupported surface operations fail with diagnostics rather
   than falling back to mesh.
9. Add `.impress` persistence guards that prevent mesh fallback results from
   being serialized as canonical surface bodies.

## Specification Manifest for Discovery

The following manifest uses the shared `specification-manifest-entry` template
from `/Users/k/Documents/Projects/.agents/process/templates/manifest-entry-template.md`.

Scores follow the shared policy:

- `25+`: split required before implementation
- `16-24`: explicit split review required
- `0-15`: small/cohesive if readiness fields are complete

Spec promotion status: final specification documents have been created for every candidate in this manifest.

### Candidate Spec: Mesh Execution Inventory And Classification

Discovery purpose:
- Create the authoritative inventory of mesh-producing code paths before
  writing migration specs that might otherwise miss a hidden executor.

Responsibilities:
- Functions/methods:
  - mesh-producing public API scanner
  - mesh-producing private helper scanner
  - classification report generator
- Data structures/models:
  - mesh path classification record
  - owner spec reference record
- Dependencies/services:
  - `rg`/static source search
  - existing modeling modules
- Returns/outputs/signals:
  - durable inventory table
  - per-symbol classification
  - owner/migration target
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: source tree and architecture docs
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes documentation only
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded repository source scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `project/release-0.1.0a/specifications/`
- Chosen defaults / parameters:
  - classify paths as `tessellation-boundary`, `mesh-consumer`,
    `legacy-compatibility`, `explicit-mesh-tool`, or `invalid-modeling-fallback`
- Test strategy:
  - documentation review plus static source-search reproducibility notes
- Data ownership:
  - architecture tracker owns the source list until final specs are written
- Routes:
  - architecture document to implementation specs
- Reuse/extraction decision:
  - reuse existing architecture tracker and source inventory; no code extraction
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Inventory must identify symbols, not only modules, or later specs will stay
  too vague.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: this is one discovery/inventory artifact; splitting before
  the first inventory would create disconnected partial truth.

### Candidate Spec: Primitive Surface Defaults: Box, Prism, Polyhedron

Discovery purpose:
- Promote planar-faced primitive authorship to surface-default output for the primitive families that already lower cleanly to planar patch sets.

Responsibilities:
- Functions/methods:
  - `make_box`
  - `make_prism`
  - `make_polyhedron`
  - `make_nhedron`
- Data structures/models:
  - `SurfaceBody`
  - planar primitive metadata payload
- Dependencies/services:
  - `primitives.py`
  - `_surface_primitives.py`
  - `tessellation.py`
- Returns/outputs/signals:
  - surface-default primitive return
  - explicit tessellated mesh output
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `make_surface_box`, `make_surface_prism`, `make_surface_polyhedron`
  - Additions to existing reusable library/module: `primitives.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes public defaults for listed primitives
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - tessellation should preserve existing preview/export performance
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/primitives.py`
- Chosen defaults / parameters:
  - listed constructors return `SurfaceBody` by default
- Test strategy:
  - unit tests for default return type, tessellation, color metadata, and compatibility mesh route
- Data ownership:
  - authored truth lives in `SurfaceBody`; mesh is derived
- Routes:
  - primitive API to surface primitive constructor to tessellation
- Reuse/extraction decision:
  - add to existing primitive modules; no new reusable module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- `make_nhedron` should stay an alias, but it must not route through mesh-default `make_polyhedron`.

Score:
- Functions/methods: 4 x 2 = 8
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
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
- Total: 22.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: these are planar-faced primitive defaults with the same surface and tessellation route.

### Candidate Spec: Primitive Surface Defaults: Cylinder, Cone, Ngon

Discovery purpose:
- Promote axial/frustum primitive authorship to surface-default output while preserving direction, cap, side-count, and radius semantics.

Responsibilities:
- Functions/methods:
  - `make_cylinder`
  - `make_cone`
  - `make_ngon`
- Data structures/models:
  - `SurfaceBody`
  - revolution/frustum primitive metadata payload
- Dependencies/services:
  - `primitives.py`
  - `_surface_primitives.py`
  - `tessellation.py`
- Returns/outputs/signals:
  - surface-default primitive return
  - explicit tessellated mesh output
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `make_surface_cylinder`, `make_surface_cone`, `make_surface_ngon`
  - Additions to existing reusable library/module: `primitives.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes public defaults for listed primitives
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - tessellation should preserve resolution behavior as boundary policy
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/primitives.py`
- Chosen defaults / parameters:
  - listed constructors return `SurfaceBody` by default; resolution is tessellation guidance
- Test strategy:
  - unit tests for caps, direction, side length, radius alias, tessellation, and mesh compatibility
- Data ownership:
  - authored truth lives in `SurfaceBody`; mesh is derived
- Routes:
  - primitive API to surface primitive constructor to tessellation
- Reuse/extraction decision:
  - add to existing primitive modules; no new reusable module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Existing `resolution` parameters need a compatibility interpretation without making mesh density modeled truth.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
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
- Total: 20.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: these share one axial/frustum surface-default migration pattern.

### Candidate Spec: Primitive Surface Defaults: Sphere And Torus

Discovery purpose:
- Promote curved analytic primitives to surface-default output and make sampling density a tessellation concern.

Responsibilities:
- Functions/methods:
  - `make_sphere`
  - `make_torus`
- Data structures/models:
  - `SurfaceBody`
  - analytic curved primitive metadata payload
- Dependencies/services:
  - `primitives.py`
  - `_surface_primitives.py`
  - `tessellation.py`
- Returns/outputs/signals:
  - surface-default primitive return
  - explicit tessellated mesh output
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `make_surface_sphere`, `make_surface_torus`
  - Additions to existing reusable library/module: `primitives.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes public defaults for listed primitives
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - tessellation should preserve quality and periodic sampling bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/primitives.py`
- Chosen defaults / parameters:
  - listed constructors return `SurfaceBody` by default; angular sample counts become tessellation guidance
- Test strategy:
  - unit tests for default return type, periodic seam tessellation, color metadata, and mesh compatibility
- Data ownership:
  - authored truth lives in analytic surface bodies
- Routes:
  - primitive API to surface primitive constructor to tessellation
- Reuse/extraction decision:
  - add to existing primitive modules; no new reusable module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Periodic seam behavior should be tested explicitly so surface-default output does not regress watertight tessellation.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 2 x 0.5 = 1
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
- Total: 18

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: these are the two current analytic curved primitives with the same tessellation-density issue.

### Candidate Spec: Primitive Mesh Compatibility API Names

Discovery purpose:
- Define explicit mesh primitive compatibility API names so old mesh behavior is available only through visibly mesh-specific calls.

Responsibilities:
- Functions/methods:
  - explicit primitive mesh helper names
  - compatibility warning hooks
- Data structures/models:
  - mesh compatibility API record
  - deprecation diagnostic
- Dependencies/services:
  - `primitives.py`
  - legacy mesh deprecation helper
- Returns/outputs/signals:
  - explicit mesh primitive result
  - compatibility warning
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current primitive mesh implementations during migration
  - Additions to existing reusable library/module: `primitives.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds/renames compatibility APIs
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - compatibility wrappers should not add extra mesh generation work
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/primitives.py`
- Chosen defaults / parameters:
  - compatibility calls use `*_mesh` or another chosen visibly mesh-specific convention
- Test strategy:
  - unit tests for names, warnings, and old mesh behavior access
- Data ownership:
  - compatibility API owns explicit mesh outputs only
- Routes:
  - compatibility API to mesh helper route
- Reuse/extraction decision:
  - add to existing `primitives.py`
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Name convention should be chosen once and reused across all compatibility primitive APIs.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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
- Total: 16.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this is one naming and public compatibility contract.

### Candidate Spec: Primitive Planar And Frustum Helper Quarantine

Discovery purpose:
- Delete or quarantine mesh helpers for box, rectangular frustum, and circular frustum construction.

Responsibilities:
- Functions/methods:
  - `_box_mesh`
  - `_circular_frustum_mesh`
  - `_rectangular_frustum_mesh`
- Data structures/models:
  - helper quarantine classification
  - legacy helper owner record
- Dependencies/services:
  - `primitives.py`
  - optional legacy mesh primitive module
- Returns/outputs/signals:
  - removed helper import
  - quarantined helper route
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current planar/frustum helpers during migration
  - Additions to existing reusable library/module: `primitives.py`
  - New reusable library/module to create: optional legacy mesh primitive module
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - deletes or moves private helper functions
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - retained helpers preserve current mesh quality
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/primitives.py` and optional legacy mesh module
- Chosen defaults / parameters:
  - helpers are not imported by authored surface modeling paths
- Test strategy:
  - static import tests and retained helper behavior tests
- Data ownership:
  - helper owns mesh-only construction
- Routes:
  - explicit compatibility or legacy route
- Reuse/extraction decision:
  - create legacy mesh module only if deletion is not possible
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Planar/frustum helpers are grouped because they feed box, prism, cylinder, cone, and ngon compatibility.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 21.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: these helpers share planar/frustum compatibility geometry.

### Candidate Spec: Primitive Curved Helper Quarantine

Discovery purpose:
- Delete or quarantine mesh helpers for sphere and torus construction.

Responsibilities:
- Functions/methods:
  - `_sphere_mesh`
  - `_torus_mesh`
- Data structures/models:
  - helper quarantine classification
  - legacy helper owner record
- Dependencies/services:
  - `primitives.py`
  - optional legacy mesh primitive module
- Returns/outputs/signals:
  - removed helper import
  - quarantined helper route
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current curved helpers during migration
  - Additions to existing reusable library/module: `primitives.py`
  - New reusable library/module to create: optional legacy mesh primitive module
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - deletes or moves private helper functions
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - retained helpers preserve current mesh quality
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/primitives.py` and optional legacy mesh module
- Chosen defaults / parameters:
  - helpers are not imported by authored surface modeling paths
- Test strategy:
  - static import tests and retained helper behavior tests
- Data ownership:
  - helper owns mesh-only construction
- Routes:
  - explicit compatibility or legacy route
- Reuse/extraction decision:
  - create legacy mesh module only if deletion is not possible
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Curved helpers are separate because periodic seam behavior must be preserved for compatibility.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 1 x 3 = 3
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

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: these helpers share curved periodic compatibility geometry.

### Candidate Spec: Primitive Tessellation Helper Relocation

Discovery purpose:
- Define the narrow case where primitive mesh helpers may move behind tessellation because they consume surface/tessellation inputs.

Responsibilities:
- Functions/methods:
  - tessellation helper adapter
  - `_orient_mesh` relocation or deletion
- Data structures/models:
  - tessellation helper contract
  - surface-to-mesh adapter record
- Dependencies/services:
  - `tessellation.py`
  - `primitives.py`
- Returns/outputs/signals:
  - tessellated mesh result
  - boundary violation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current helper math if applicable
  - Additions to existing reusable library/module: `tessellation.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - relocates helper code behind tessellation
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - relocated helpers must obey tessellation request bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/tessellation.py`
- Chosen defaults / parameters:
  - relocated helpers consume `SurfaceBody`, `SurfacePatch`, or tessellation requests, never authored primitive arguments
- Test strategy:
  - tests for boundary inputs and no direct authored primitive route
- Data ownership:
  - tessellation owns mesh output
- Routes:
  - surface body to tessellation helper route
- Reuse/extraction decision:
  - add to existing `tessellation.py` only when helper is truly boundary-owned
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- If a helper still consumes primitive constructor arguments, it belongs in compatibility, not tessellation.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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
- Total: 16.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this is one tessellation-boundary relocation rule.

### Candidate Spec: Loft Spec 60 Revision Or Retirement

Discovery purpose:
- Correct the specification posture so mesh executor correspondence consumption is not treated as canonical modeled execution.

Responsibilities:
- Functions/methods:
  - Spec 60 revision
  - canonical executor wording update
- Data structures/models:
  - loft spec status record
  - executor boundary policy
- Dependencies/services:
  - Loft Spec 60
  - mesh-boundary architecture
- Returns/outputs/signals:
  - revised or retired spec posture
  - migration note
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: existing loft specs
  - Additions to existing reusable library/module: specification docs
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes only the scoped API, spec, or migration boundary described here
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded to the scoped operation and existing quality/request controls
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `project/release-0.1.0a/specifications/`
- Chosen defaults / parameters:
  - mesh executor language becomes legacy/debug/tessellation-boundary only
- Test strategy:
  - spec review and no-canonical-mesh wording checks
- Data ownership:
  - specifications own documented contract
- Routes:
  - architecture to revised spec
- Reuse/extraction decision:
  - revise existing spec
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This is specification reconciliation, not code relocation.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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
- Total: 17.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this candidate now owns one discovered boundary.

### Candidate Spec: Loft Mesh Emission Relocation

Discovery purpose:
- Move loft mesh face emission into tessellation/debug/compatibility boundaries instead of canonical plan execution.

Responsibilities:
- Functions/methods:
  - `loft_execute_plan`
  - `emit_mesh_faces_from_sample_correspondence`
- Data structures/models:
  - `LoftPlan`
  - debug mesh result
- Dependencies/services:
  - `loft.py`
  - `tessellation.py`
- Returns/outputs/signals:
  - canonical surface result
  - explicit debug/tessellated mesh
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: loft plan validation and correspondence records
  - Additions to existing reusable library/module: `loft.py`, `tessellation.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes only the scoped API, spec, or migration boundary described here
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded to the scoped operation and existing quality/request controls
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py` and `tessellation.py`
- Chosen defaults / parameters:
  - `LoftPlan -> SurfaceBody` is canonical; mesh emission is explicit only
- Test strategy:
  - tests proving no surface loft fallback to mesh and explicit mesh route behavior
- Data ownership:
  - `LoftPlan` owns correspondence; tessellation/debug owns mesh
- Routes:
  - planner to surface executor to tessellation/debug route
- Reuse/extraction decision:
  - add to existing loft/tessellation modules
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This should be implemented after or alongside surface executor correspondence consumption.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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
- Total: 21.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this candidate now owns one discovered boundary.

### Candidate Spec: Text Surface Default Public API

Discovery purpose:
- Make text authored APIs surface-default while preserving profile generation and orientation semantics.

Responsibilities:
- Functions/methods:
  - `make_text`
  - `text`
  - `_surface_text_extrude`
- Data structures/models:
  - `Section`
  - `SurfaceBody`
- Dependencies/services:
  - `text.py`
  - `_surface_ops.py`
  - font loading
- Returns/outputs/signals:
  - surfaced text body
  - deterministic placeholder for empty text
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `text_profiles`, `sections_from_paths`
  - Additions to existing reusable library/module: `text.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes public text default return type
- Security/privacy-sensitive behavior:
  - local font path handling only
- Performance-sensitive behavior:
  - outline processing bounded by glyph count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/text.py`
- Chosen defaults / parameters:
  - `make_text` and `text` return `SurfaceBody` by default
- Test strategy:
  - unit tests for default return type, orientation, color metadata, and empty text
- Data ownership:
  - text profiles feed surface extrusion
- Routes:
  - text API to profiles to surface extrusion to tessellation
- Reuse/extraction decision:
  - add to existing `text.py`; reuse profile generation
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Empty text should avoid serializing a visible fake body.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 23

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: public text default, profile reuse, and surface extrusion are one behavior change.

### Candidate Spec: Text Mesh Compatibility And Empty Text Behavior

Discovery purpose:
- Quarantine mesh text extrusion and define compatibility behavior without making mesh the authored default.

Responsibilities:
- Functions/methods:
  - `_mesh_text_extrude`
  - explicit text mesh compatibility helper
- Data structures/models:
  - mesh compatibility result
  - empty text placeholder policy
- Dependencies/services:
  - `text.py`
  - tessellation boundary
- Returns/outputs/signals:
  - explicit compatibility mesh
  - no-hidden-fallback diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current mesh extrusion during migration
  - Additions to existing reusable library/module: `text.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes compatibility routing
- Security/privacy-sensitive behavior:
  - local font path handling only
- Performance-sensitive behavior:
  - compatibility extrusion bounded by glyph count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/text.py`
- Chosen defaults / parameters:
  - mesh text is explicit compatibility or tessellation output only
- Test strategy:
  - unit tests for compatibility helper and empty text non-visibility
- Data ownership:
  - mesh compatibility owns mesh-only output
- Routes:
  - explicit helper or tessellation route
- Reuse/extraction decision:
  - add to existing `text.py`; no new module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Compatibility behavior must be named clearly enough that new authored code does not choose it accidentally.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: mesh compatibility and empty text policy are the only remaining non-default text edge behaviors.

### Candidate Spec: Drafting Surface Defaults

Discovery purpose:
- Move drafting geometry from mesh-default annotations to surfaced annotation
  bodies or surface consumer records.

Responsibilities:
- Functions/methods:
  - `make_line`
  - `make_plane`
  - `make_arrow`
  - `make_dimension`
- Data structures/models:
  - `SurfaceBody`
  - `SurfaceConsumerCollection`
- Dependencies/services:
  - `drafting.py`
  - `_surface_primitives.py`
  - `text.py`
- Returns/outputs/signals:
  - surfaced drafting body
  - consumer collection
  - explicit mesh compatibility
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface primitive constructors
  - Additions to existing reusable library/module: `drafting.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes public drafting default return type
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - dimension text generation bounded by label content
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/drafting.py`
- Chosen defaults / parameters:
  - drafting APIs return surface bodies or consumer collections by default
- Test strategy:
  - unit tests for each drafting helper, transforms, color metadata, and
    explicit tessellation
- Data ownership:
  - drafting authored truth lives in surface annotations
- Routes:
  - drafting API to surface primitive/text helpers to tessellation
- Reuse/extraction decision:
  - add to existing `drafting.py`; reuse surface primitives and text surface
    extrusion
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- `make_dimension` may remain a collection instead of a single `SurfaceBody`
  because it is naturally a composed consumer annotation.

Score:
- Functions/methods: 4 x 2 = 8
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 22.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: all listed helpers are one annotation subsystem and share the
  same surface default and consumer-collection decision.

### Candidate Spec: Heightmap Sampled Surface Payload

Discovery purpose:
- Define the native sampled heightfield surface payload independent of alpha/cache policy.

Responsibilities:
- Functions/methods:
  - sampled surface constructor
  - heightmap surface payload builder
- Data structures/models:
  - sampled height surface patch
  - height sample grid
- Dependencies/services:
  - `heightmap.py`
  - patch family modules
- Returns/outputs/signals:
  - native surface heightfield
  - payload validation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: image normalization
  - Additions to existing reusable library/module: `heightmap.py`, patch family modules
  - New reusable library/module to create: sampled patch support if absent
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes only the scoped API, spec, or migration boundary described here
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded to the scoped operation and existing quality/request controls
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/heightmap.py` and patch family module
- Chosen defaults / parameters:
  - heightmap surface mode creates native sampled payload, not triangle wrapper
- Test strategy:
  - tests for payload shape and tessellation
- Data ownership:
  - source image samples define surface payload
- Routes:
  - heightmap API to sampled surface patch
- Reuse/extraction decision:
  - add sampled patch support or reuse full patch-family representation
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Alpha/cache behavior is split because it affects data inclusion and invalidation.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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
- Total: 20.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this candidate now owns one discovered boundary.

### Candidate Spec: Heightmap Alpha Mask And Cache Policy

Discovery purpose:
- Define alpha/mask semantics and cache invalidation for native heightmap surfaces.

Responsibilities:
- Functions/methods:
  - alpha mask policy resolver
  - heightmap cache key policy
- Data structures/models:
  - alpha/mask policy
  - cache key record
- Dependencies/services:
  - `heightmap.py`
  - image loading/cache
- Returns/outputs/signals:
  - alpha-aware surface payload
  - cache invalidation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: alpha normalization and cache key logic
  - Additions to existing reusable library/module: `heightmap.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes only the scoped API, spec, or migration boundary described here
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded to the scoped operation and existing quality/request controls
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/heightmap.py`
- Chosen defaults / parameters:
  - `mask` and `ignore` semantics are preserved without mesh wrapping
- Test strategy:
  - tests for alpha modes, cache hits/misses, and no mesh wrapper serialization
- Data ownership:
  - heightmap module owns sample/cache policy
- Routes:
  - image loader to native heightmap surface builder
- Reuse/extraction decision:
  - add to existing `heightmap.py`
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Mask semantics affect topology and must be explicit before implementation specs.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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
- Total: 18.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this candidate now owns one discovered boundary.

### Candidate Spec: Heightmap Displacement Surface Contract

Discovery purpose:
- Define the surfaced displacement operation without projection sampling details.

Responsibilities:
- Functions/methods:
  - `displace_heightmap`
  - displacement surface operation
- Data structures/models:
  - displacement surface payload
  - displaced surface result
- Dependencies/services:
  - `heightmap.py`
  - surface patch evaluators
- Returns/outputs/signals:
  - displaced surface representation
  - unsupported operation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: image sampling helpers
  - Additions to existing reusable library/module: `heightmap.py`, patch family modules
  - New reusable library/module to create: displacement patch support if absent
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes only the scoped API, spec, or migration boundary described here
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded to the scoped operation and existing quality/request controls
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/heightmap.py` and patch family module
- Chosen defaults / parameters:
  - displacement cannot tessellate source surface and wrap result as canonical truth
- Test strategy:
  - tests for no mesh wrapping and surfaced output
- Data ownership:
  - source surface plus image samples own displacement truth
- Routes:
  - surface input to displacement representation
- Reuse/extraction decision:
  - add displacement support to existing heightmap and patch modules
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Projection sampling is split because it has its own bounds and coordinate policy.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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
- Total: 20.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this candidate now owns one discovered boundary.

### Candidate Spec: Heightmap Projection Sampling Policy

Discovery purpose:
- Define planar projection sampling, bounds, and refusal behavior for heightmap displacement.

Responsibilities:
- Functions/methods:
  - projection resolver
  - sample bounds validator
- Data structures/models:
  - projection bounds policy
  - sample coordinate record
- Dependencies/services:
  - `heightmap.py`
  - image sampler
- Returns/outputs/signals:
  - sampled displacement values
  - projection diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current planar sampling helpers
  - Additions to existing reusable library/module: `heightmap.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes only the scoped API, spec, or migration boundary described here
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded to the scoped operation and existing quality/request controls
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/heightmap.py`
- Chosen defaults / parameters:
  - only planar projection is required in V1; unsupported projections refuse explicitly
- Test strategy:
  - tests for xy/xz/yz bounds, degenerate bounds, alpha modes, and unsupported projections
- Data ownership:
  - heightmap module owns sampling policy
- Routes:
  - displacement API to projection sampler
- Reuse/extraction decision:
  - add to existing `heightmap.py`
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Projection policy should not be hidden inside the surface displacement object.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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
- Total: 18.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this candidate now owns one discovered boundary.

### Candidate Spec: Heightmap Mesh Compatibility And Serialization Guard

Discovery purpose:
- Preserve explicit mesh compatibility while preventing mesh-derived triangle wrappers from being serialized as native surface truth.

Responsibilities:
- Functions/methods:
  - mesh compatibility heightmap helper
  - serialization guard
- Data structures/models:
  - mesh compatibility result
  - invalid surface-wrapper diagnostic
- Dependencies/services:
  - `heightmap.py`
  - `.impress` codec
  - tests
- Returns/outputs/signals:
  - explicit mesh result
  - refusal diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current mesh implementation
  - Additions to existing reusable library/module: `heightmap.py`, `.impress` guard tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes compatibility and persistence behavior
- Security/privacy-sensitive behavior:
  - local image path input
- Performance-sensitive behavior:
  - compatibility mesh generation keeps existing LOD bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/heightmap.py` and future `.impress` codec
- Chosen defaults / parameters:
  - mesh heightfields are explicit mesh data only; triangle wrappers are rejected as native surface truth
- Test strategy:
  - tests for explicit mesh route and serialization refusal
- Data ownership:
  - mesh compatibility owns mesh-only data; `.impress` owns persisted surface truth
- Routes:
  - explicit mesh helper or serializer guard route
- Reuse/extraction decision:
  - add guards to existing modules
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This spec should land after the inventory names all heightmap mesh wrapper paths.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 21.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: compatibility and persistence guard are one anti-wrapper policy.

### Candidate Spec: Traditional Hinge Surface Lowering

Discovery purpose:
- Define surface assembly lowering for traditional barrel hinge leaves and related parts.

Responsibilities:
- Functions/methods:
  - traditional hinge leaf lowering
  - barrel/leaf component lowering
- Data structures/models:
  - `HingeSurfaceAssembly`
  - traditional hinge component
- Dependencies/services:
  - `hinges.py`
  - surface primitives
- Returns/outputs/signals:
  - surfaced traditional hinge body
  - unsupported dependency diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: hinge validation and surface component records
  - Additions to existing reusable library/module: `hinges.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes only the scoped API, spec, or migration boundary described here
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded to the scoped operation and existing quality/request controls
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/hinges.py`
- Chosen defaults / parameters:
  - traditional hinge helpers lower to surface assemblies/bodies by default
- Test strategy:
  - tests for leaf/barrel/pin clearances and tessellation output
- Data ownership:
  - hinge assembly owns authored structure
- Routes:
  - hinge API to surface assembly to tessellation/boolean
- Reuse/extraction decision:
  - add to existing `hinges.py`
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Traditional hinges have different primitive and boolean dependencies than flexure hinges.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this candidate now owns one discovered boundary.

### Candidate Spec: Living And Bistable Hinge Surface Lowering

Discovery purpose:
- Define surface assembly lowering for living and bistable hinge helpers.

Responsibilities:
- Functions/methods:
  - living hinge lowering
  - bistable hinge lowering
- Data structures/models:
  - `HingeSurfaceAssembly`
  - flexure component
- Dependencies/services:
  - `hinges.py`
  - surface primitives
- Returns/outputs/signals:
  - surfaced flexure hinge body
  - unsupported dependency diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: hinge validation and surface component records
  - Additions to existing reusable library/module: `hinges.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes only the scoped API, spec, or migration boundary described here
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded to the scoped operation and existing quality/request controls
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/hinges.py`
- Chosen defaults / parameters:
  - living and bistable hinge helpers lower to surface assemblies/bodies by default
- Test strategy:
  - tests for slit/ligament topology and tessellation output
- Data ownership:
  - hinge assembly owns authored structure
- Routes:
  - hinge API to surface assembly to tessellation/boolean
- Reuse/extraction decision:
  - add to existing `hinges.py`
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Flexure hinges share slot/ligament topology concerns distinct from barrel hinges.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this candidate now owns one discovered boundary.

### Candidate Spec: Surface Transform API Defaults

Discovery purpose:
- Make authored transform helpers operate on `SurfaceBody` and surfaced assemblies without tessellating.

Responsibilities:
- Functions/methods:
  - `translate`
  - `rotate`
  - `scale`
  - `mirror`
  - `multmatrix`
- Data structures/models:
  - `SurfaceBody` transform
  - transformed surface result
- Dependencies/services:
  - `transform.py`
  - surface model
- Returns/outputs/signals:
  - transformed surface body
  - explicit mesh compatibility result
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface transform methods
  - Additions to existing reusable library/module: `transform.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes authored transform target semantics
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - transforms should avoid tessellating or copying mesh buffers
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/transform.py`
- Chosen defaults / parameters:
  - surface transforms are canonical; mesh transforms are compatibility only
- Test strategy:
  - tests for transform preservation, tessellation after transform, and mesh compatibility
- Data ownership:
  - surface body owns authored transform state
- Routes:
  - transform API to surface transform or compatibility mesh route
- Reuse/extraction decision:
  - add to existing `transform.py`
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Need preserve current mesh behavior for explicit mesh inputs without letting it set authored defaults.

Score:
- Functions/methods: 5 x 2 = 10
- Data structures/models: 2 x 1 = 2
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
- Total: 22.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: these helpers are one transform API family.

### Candidate Spec: Surface Composition Public Type

Discovery purpose:
- Decide and define the public surfaced composition type that replaces `MeshGroup` for authored grouping.

Responsibilities:
- Functions/methods:
  - surface group constructor
  - public composition API
- Data structures/models:
  - surfaced group or assembly
  - `SurfaceBody` composition
- Dependencies/services:
  - `surface_scene.py`
  - surface model
- Returns/outputs/signals:
  - composed surface object
  - explicit unsupported diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `surface_scene.py` collection concepts
  - Additions to existing reusable library/module: `surface_scene.py`
  - New reusable library/module to create: optional surface group module
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds public composition boundary
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - composition should avoid premature tessellation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface_scene.py` or new surface group module
- Chosen defaults / parameters:
  - composition is a surface object/assembly, not `MeshGroup`
- Test strategy:
  - tests for construction, identity, transform attachment, and invalid composition
- Data ownership:
  - surfaced composition owns authored grouping
- Routes:
  - modeling group API to surface composition
- Reuse/extraction decision:
  - create new module only if `surface_scene.py` is not the right public boundary
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Implementation owner/module must be chosen before final implementation spec promotion.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 21.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: public type and construction are one API boundary.

### Candidate Spec: Surface Composition Traversal And Tessellation Handoff

Discovery purpose:
- Define traversal order and tessellation handoff for surfaced composition objects.

Responsibilities:
- Functions/methods:
  - composition traversal
  - surface collection handoff
  - tessellation handoff
- Data structures/models:
  - traversal order record
  - consumer collection
- Dependencies/services:
  - `surface_scene.py`
  - `tessellation.py`
- Returns/outputs/signals:
  - tessellation-ready collection
  - traversal diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface consumer collection concepts
  - Additions to existing reusable library/module: `surface_scene.py`, `tessellation.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds traversal and handoff behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - traversal should avoid premature mesh generation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface_scene.py` and `src/impression/modeling/tessellation.py`
- Chosen defaults / parameters:
  - traversal is deterministic and hands off to tessellation only at the boundary
- Test strategy:
  - tests for traversal order, transform composition, and tessellation handoff
- Data ownership:
  - composition owns authored grouping; tessellation owns mesh output
- Routes:
  - surface composition to consumer collection to tessellation
- Reuse/extraction decision:
  - add to existing modules
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This should follow the public type spec or be implemented in the same PR only if still small.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 2 x 1 = 2
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

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: traversal and tessellation handoff are one downstream behavior boundary.

### Candidate Spec: MeshGroup Explicit Compatibility API

Discovery purpose:
- Keep `MeshGroup` available only through explicit mesh compatibility APIs.

Responsibilities:
- Functions/methods:
  - `MeshGroup`
  - `group`
- Data structures/models:
  - mesh group compatibility object
  - quarantine classification
- Dependencies/services:
  - `group.py`
  - mesh utility namespace
- Returns/outputs/signals:
  - explicit mesh group output
  - compatibility diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `MeshGroup`
  - Additions to existing reusable library/module: `group.py`
  - New reusable library/module to create: optional mesh tools namespace
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes only the scoped API, spec, or migration boundary described here
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded to the scoped operation and existing quality/request controls
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/group.py`
- Chosen defaults / parameters:
  - `MeshGroup` is compatibility/consumer only
- Test strategy:
  - tests for explicit mesh group behavior and surface API separation
- Data ownership:
  - mesh group owns only mesh inputs/outputs
- Routes:
  - explicit mesh helper route
- Reuse/extraction decision:
  - quarantine current `MeshGroup`; extract only if needed
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Import-boundary enforcement is split out so API compatibility does not blur module ownership.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this candidate now owns one discovered boundary.

### Candidate Spec: MeshGroup Import Boundary Enforcement

Discovery purpose:
- Prevent surface-authored modules from depending on `MeshGroup` as composition truth.

Responsibilities:
- Functions/methods:
  - mesh group import-boundary check
  - transform compatibility route check
- Data structures/models:
  - import-boundary rule
  - violation diagnostic
- Dependencies/services:
  - `group.py`
  - `transform.py`
- Returns/outputs/signals:
  - static/import test result
  - boundary violation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current module layout
  - Additions to existing reusable library/module: tests/static checks
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes only the scoped API, spec, or migration boundary described here
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded to the scoped operation and existing quality/request controls
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - test suite
- Chosen defaults / parameters:
  - surface modeling modules cannot import `MeshGroup` as authored composition truth
- Test strategy:
  - static import tests plus transform route tests
- Data ownership:
  - tests own enforcement; mesh group owns compatibility only
- Routes:
  - static check to modeling modules
- Reuse/extraction decision:
  - add reusable test/static helper if useful
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This should coordinate with Mesh Utility Quarantine to avoid duplicate static rules.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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
- Total: 18.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this candidate now owns one discovered boundary.

### Candidate Spec: Mesh Utility Quarantine

Discovery purpose:
- Move retained mesh utilities into explicit mesh-tool modules and prevent
  surfaced modeling modules from depending on mesh execution casually.

Responsibilities:
- Functions/methods:
  - mesh utility import boundary checker
  - mesh tool module routing
- Data structures/models:
  - mesh utility classification
- Dependencies/services:
  - `_ops_mesh.py`
  - `group.py`
  - `transform.py`
  - mesh analysis/repair modules
- Returns/outputs/signals:
  - quarantined module layout
  - import-boundary diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: retained mesh utility code
  - Additions to existing reusable library/module: mesh utility modules
  - New reusable library/module to create: optional `modeling/mesh_tools`
    namespace
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - moves modules/imports
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - none beyond import/static check cost
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/_ops_mesh.py`
  - candidate `src/impression/modeling/mesh_tools/`
- Chosen defaults / parameters:
  - retained mesh tools are explicit and never imported as authored surface
    execution
- Test strategy:
  - static import tests plus unit tests for retained mesh tool behavior
- Data ownership:
  - mesh tools own mesh-only inputs/outputs
- Routes:
  - explicit mesh tool API route
- Reuse/extraction decision:
  - extract or wrap retained mesh utilities into explicit namespace
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Static boundary enforcement should avoid blocking legitimate tessellation
  imports.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 4 x 1 = 4
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 1 x 3 = 3
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: the import boundary and module quarantine must be specified
  together to avoid moving code without enforcing the boundary.

### Candidate Spec: No Hidden Mesh Fallback Enforcement

Discovery purpose:
- Ensure unsupported surface operations fail with diagnostics rather than
  silently falling back to mesh.

Responsibilities:
- Functions/methods:
  - fallback detection tests
  - diagnostic assertion helpers
- Data structures/models:
  - unsupported operation diagnostic
  - no-fallback test fixture matrix
- Dependencies/services:
  - loft
  - CSG
  - primitives
  - text/drafting
  - heightmap/hinges
  - `.impress` serializer
- Returns/outputs/signals:
  - explicit unsupported result
  - failing test on mesh fallback
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: existing warnings and diagnostics
  - Additions to existing reusable library/module: test helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - test/spec only; implementation changes happen in dependent specs
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - fallback tests should use bounded fixtures
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - test suite under `tests/`
- Chosen defaults / parameters:
  - hidden mesh fallback is always test failure for authored surface APIs
- Test strategy:
  - automated acceptance tests across each affected subsystem
- Data ownership:
  - surface subsystem owns fallback policy; tests own evidence
- Routes:
  - test helper to subsystem API
- Reuse/extraction decision:
  - add reusable test helpers
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This spec should be implemented after at least the initial inventory so it
  knows the full subsystem list.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 6 x 1 = 6
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: the fixture matrix spans subsystems, but the actual behavior
  is one policy: no hidden mesh fallback.

### Split Review Record: Loft Public Surface API And Reference QA Boundary

Discovery purpose:
- Remove the remaining public loft convenience route that returns mesh output
  as the default and correct loft reference QA so surfaced annotations are not
  asserted as watertight model solids.

Responsibilities:
- Functions/methods:
  - `loft_sections`
  - `loft_execute_plan`
  - `loft_execute_plan_debug_mesh`
  - loft reference mesh-quality helpers
- Data structures/models:
  - `LoftPlan`
  - `SurfaceBody` loft result
  - reference scene item role record
- Dependencies/services:
  - `src/impression/modeling/loft.py`
  - `tests/test_loft.py`
  - loft real-world examples
- Returns/outputs/signals:
  - canonical `SurfaceBody`
  - explicit debug/tessellated mesh only through named boundary APIs
  - role-aware QA result
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: loft planning, surface executor, debug mesh
    executor, tessellation helper
  - Additions to existing reusable library/module: reference scene role checks
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes public loft behavior and test expectations
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - no additional tessellation unless test/export explicitly requests it
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`, loft examples, and `tests/test_loft.py`
- Chosen defaults / parameters:
  - authored loft APIs return `SurfaceBody`; mesh appears only through
    `*_debug_mesh`, tessellation, or explicitly named compatibility helpers
- Test strategy:
  - tests prove `loft_sections` no longer returns a mesh by default and the
    splitter-manifold example distinguishes model bodies from annotation labels
- Data ownership:
  - loft owns surface plan execution; tessellation/debug owns mesh; examples
    own scene item role metadata
- Routes:
  - public loft API to surface executor to optional tessellation/debug boundary
- Reuse/extraction decision:
  - keep debug mesh executor only as explicit boundary code
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- The observed splitter-manifold failure is not proof that the loft kernel must
  be mesh-primary. The failing assertion applies mesh watertightness checks to
  a surfaced annotation label and uses a public convenience route that still
  returns a debug mesh.

Pre-split score:
- Functions/methods: 4 x 2 = 8
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- Existing reusable code reused as-is: 4 x 0.5 = 2
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
- Pre-split total: 25

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Split required by score. The implementation spec is split into:
  - Loft Public Surface API Default
  - Loft Reference QA Role Boundary

### Candidate Spec: Loft Public Surface API Default

Discovery purpose:
- Make authored loft public APIs return `SurfaceBody` by default and reserve
  mesh output for explicitly named tessellation, debug, or compatibility APIs.

Responsibilities:
- Functions/methods:
  - `loft_sections`
  - `loft_execute_plan`
  - `loft_execute_plan_debug_mesh`
- Data structures/models:
  - `LoftPlan`
  - `SurfaceBody` loft result
- Dependencies/services:
  - `src/impression/modeling/loft.py`
  - tessellation boundary
- Returns/outputs/signals:
  - canonical `SurfaceBody`
  - explicit debug mesh
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: loft planning and surface executor
  - Additions to existing reusable library/module: public route migration tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes public loft return defaults
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - avoids implicit tessellation on default path
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - `loft_sections` returns `SurfaceBody` unless the caller chooses an
    explicitly named mesh/debug route
- Test strategy:
  - public API tests for `SurfaceBody` default and explicit mesh route
- Data ownership:
  - loft owns modeled surface result; tessellation/debug owns mesh output
- Routes:
  - public loft API to surface executor
- Reuse/extraction decision:
  - reuse existing surface executor and debug mesh executor
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Backward compatibility must be explicit; compatibility mesh helpers cannot
  remain the default authored path.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
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
- Total: 18.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this is one public return-contract change.

### Candidate Spec: Loft Reference QA Role Boundary

Discovery purpose:
- Make loft reference/example tests distinguish modeled solid bodies from
  annotations, labels, debug meshes, and tessellated views.

Responsibilities:
- Functions/methods:
  - loft scene role helper
  - reference mesh-quality assertion
- Data structures/models:
  - scene item role record
  - expected QA mode record
- Dependencies/services:
  - `tests/test_loft.py`
  - loft real-world examples
- Returns/outputs/signals:
  - watertight assertion for model bodies
  - role-specific assertion for annotations
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: existing example outputs
  - Additions to existing reusable library/module: role-aware test helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes tests/examples only
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded tessellation only for model body QA
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `tests/test_loft.py` and loft real-world example metadata
- Chosen defaults / parameters:
  - labels/annotations are not required to be watertight solids unless declared
    as model bodies
- Test strategy:
  - splitter-manifold example passes with role-aware checks and no hidden loft
    mesh fallback
- Data ownership:
  - examples own scene item roles; tests own QA interpretation
- Routes:
  - example scene output to role-aware verification
- Reuse/extraction decision:
  - reuse tessellation helpers only at verification boundary
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- The current failing splitter-manifold test tessellates a surfaced label and
  demands watertightness, producing boundary-edge failures unrelated to the
  loft manifold body.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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
- Total: 16.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this is one reference QA contract.

## Change History

- 2026-05-27: Removed external fastener-profile modeling from the
  Impression-owned mesh-boundary scope because that library is now a sibling
  project consumer.
- 2026-05-26: Added loft public surface API and reference QA boundary manifest
  entries after the splitter-manifold mesh-quality test exposed a default mesh
  route plus annotation-as-solid verification issue.
- 2026-05-26: Further split high-scoring manifest entries where review exposed hidden API, compatibility, operation, and policy boundaries.
- 2026-05-26: Split all manifest candidates that scored 25+ into smaller assessed candidates for spec promotion.
- 2026-05-26: Added template-assessed Specification Manifest for Discovery so
  mesh-boundary specifications live with the architecture they define.
- 2026-05-26: Added audit-driven architecture defining the tessellation
  boundary for mesh execution and identifying modeling modules that must move
  mesh generation behind that boundary.
