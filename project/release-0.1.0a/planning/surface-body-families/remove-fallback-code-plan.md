# Remove Fallback Code Plan

## Goal

Remove hidden mesh fallback, placeholder surface bodies, and silent degraded
behavior from all surface-body paths.

This plan does not delete retained mesh tools. Mesh remains allowed at explicit
preview, export, analysis, repair, and legacy-consumer boundaries. The target is
to eliminate paths where a surface-facing API secretly uses mesh as modeling
truth or returns placeholder geometry.

## Current Problem Areas

### Surface API Builds Mesh Then Rewraps Surface

- `heightmap(..., backend="surface")` builds a mesh and wraps triangle faces as
  planar patches.
- `displace_heightmap(..., backend="surface")` tessellates surface input to a
  mesh, displaces mesh vertices, then wraps triangle faces as planar patches.

Required outcome:

- surface heightfields must use a surface-native heightfield/displacement
  representation or return explicit unsupported status until that exists.
- no surface API may rewrap triangle meshes as canonical `SurfaceBody` truth.

### Placeholder Surface Bodies

- empty surface text returns a tiny hidden box
- empty surface heightmap wrapping returns a tiny hidden box

Required outcome:

- introduce an explicit empty surface result contract, nullable body result, or
  structured no-output record.
- remove hidden placeholder geometry from surface modeling APIs.

### Stubbed Surface Assemblies

- traditional hinge surface leaves create a pin bore but do not apply it.
- several thread surface functions return structured assembly descriptors
  rather than executable surface bodies.

Required outcome:

- either execute the surface operation with surface-native booleans or return a
  structured unsupported/deferred surface assembly status.
- no finished-looking surface output may omit required subtraction/cut geometry.

### Public Mesh Defaults

Many public APIs still default to mesh:

- primitives
- drafting
- text
- threading
- hinges
- heightmap
- CSG

Required outcome:

- decide which APIs are ready for `backend="surface"` default.
- keep `backend="mesh"` only as an explicit compatibility request.
- emit deprecation warnings only on explicit legacy mesh paths, not on canonical
  surface paths.

## Work Plan

### Phase 1: Classify Every Mesh Use

- [ ] inventory every modeling-layer `Mesh(...)`, `combine_meshes(...)`,
  `boolean_*`, `tessellate_surface_body(...)`, and `to_mesh()` call.
- [ ] classify each use as one of:
  - explicit output boundary
  - retained mesh tool
  - legacy compatibility path
  - hidden fallback
  - bug
- [ ] add a durable classification matrix under project research or
  architecture.

Exit criteria:

- every mesh use has an owner and classification.
- hidden fallback candidates are listed by API and line-level owner.

### Phase 2: Add Explicit No-Output And Unsupported Contracts

- [ ] define how empty surface results are represented.
- [ ] define how surface APIs report unsupported execution without mesh fallback.
- [ ] update text and heightmap surface paths to stop returning hidden tiny
  boxes.
- [ ] add tests that assert no hidden placeholder geometry is emitted.

Exit criteria:

- empty text and empty heightmap cases no longer create hidden boxes.
- unsupported surface operations return structured unsupported status or raise
  documented errors.

### Phase 3: Replace Heightmap Mesh Rewrap

- [ ] define a `HeightfieldSurfacePatch` or explicit unsupported status for
  surface heightmaps.
- [ ] replace triangle-face wrapping with a surface-native heightfield patch
  when output is a sheet.
- [ ] define solidified heightfield behavior separately from sheet heightfields.
- [ ] make `displace_heightmap(..., backend="surface")` operate on surface
  patches directly or return explicit unsupported status.

Exit criteria:

- surface heightmap paths do not call `_heightmap_mesh_impl()` as modeling
  truth.
- surface displacement does not call `tessellate_surface_body()` except at an
  explicit consumer boundary.

### Phase 4: Resolve Stubbed Assemblies

- [ ] implement traditional hinge pin bores with surface booleans or mark the
  bore subtraction unsupported in the returned surface assembly contract.
- [ ] decide whether thread surface assemblies are intermediate descriptors or
  executable `SurfaceBody` results.
- [ ] add handoff tests for every descriptor-only surface result.

Exit criteria:

- no surface result silently omits expected cut/subtract geometry.
- descriptor-only results are typed and documented as not-yet-executable.

### Phase 5: Promote Surface Defaults

- [ ] convert ready public APIs to surface-first default.
- [ ] retain explicit `backend="mesh"` for compatibility where still approved.
- [ ] update docs and examples to use surface-first APIs.
- [ ] add regression tests that canonical examples do not use mesh fallback.

Exit criteria:

- canonical modeling docs no longer teach mesh as default truth.
- tests fail if a surface path enters mesh construction outside approved
  tessellation/export/analysis boundaries.

## Required Specifications To Create Or Refine

- fallback classification and mesh-use matrix
- explicit empty surface result contract
- surface unsupported-result contract for modeling APIs
- heightfield surface patch or explicit unsupported replacement
- surface assembly descriptor versus executable body contract
- public API surface-default migration contract

## Verification Plan

- static grep/AST test for forbidden mesh calls inside surface branches
- unit tests for empty/no-output surface cases
- unit tests for unsupported surface operation reporting
- reference artifact tests for replacement heightfield, hinge, text, and thread
  outputs
- documentation tests ensuring examples use surface-first paths

## Done Definition

This track is complete when:

- no surface-facing API secretly builds mesh as modeling truth
- no surface-facing API emits hidden placeholder geometry
- unsupported cases fail explicitly
- mesh remains only at documented compatibility, preview, export, analysis, or
  repair boundaries
