# Full Surface Body Implementation Plan

## Goal

Implement complete surface-body modeling, including all current and deferred
patch families:

- planar
- ruled
- revolution
- Bezier
- B-spline
- NURBS
- sweep
- offset
- heightfield/displacement
- subdivision
- implicit

The final system should keep `SurfaceBody` as modeling truth and use mesh only
for explicit downstream boundaries.

## Kernel Foundation

### Phase 1: Complete Core Topology

- [ ] add topology-owned point identity and point correspondence tracks for loft
      and patch boundaries.
- [ ] add explicit `PatchBoundaryUse`.
- [ ] add explicit `SurfaceBoundaryLoop`.
- [ ] add seam-owned 3D boundary curves.
- [ ] add patch-local p-curves.
- [ ] support open, shared, periodic, and collapsed boundaries.
- [ ] support closed-shell classification from seam/use truth.

Exit criteria:

- loft no longer depends on independent loop anchoring/resampling as implicit
  point correspondence truth.
- shell closure does not depend on mesh analysis.
- every patch boundary can be addressed by a stable boundary-use record.

### Phase 2: Complete Shared Patch Contract

- [ ] require point evaluation.
- [ ] require first derivatives or explicit singularity declaration.
- [ ] define optional higher derivatives.
- [ ] define closest-point projection.
- [ ] define boundary curve extraction.
- [ ] define patch splitting/subdomain extraction.
- [ ] define trim application and validation.
- [ ] define continuity classification at seams.

Exit criteria:

- tessellation, booleans, selection, and projection use the same patch protocol.

### Phase 3: Complete Tessellation

- [ ] implement seam-first sampling for all boundary types.
- [ ] implement trim-aware tessellation.
- [ ] implement adaptive tessellation by curvature and tolerance.
- [ ] implement deterministic vertex reuse across seams.
- [ ] support preview/export/analysis presets without changing modeled truth.

Exit criteria:

- closed-valid bodies tessellate watertight under export/analysis presets.

## Required V1 Patch Families

### Planar

- [ ] exact evaluation and derivatives.
- [ ] 3D-to-UV projection.
- [ ] arbitrary outer/inner trims.
- [ ] polygon containment and coplanar overlap.
- [ ] split by line, p-curve, and intersection curve.
- [ ] merge/coalesce compatible coplanar patches.
- [ ] deterministic trimmed tessellation.

Primary unblockers:

- caps
- boxes
- polyhedra
- CSG cut faces
- text faces

### Ruled

- [ ] curve-backed guide representation.
- [ ] guide compatibility validation.
- [ ] derivative evaluation from guide curves.
- [ ] collapsed ruling detection.
- [ ] split in `u` and `v`.
- [ ] continuity classification across spans.
- [ ] adaptive tessellation respecting guide curvature.

Primary unblockers:

- linear extrude sidewalls
- simple loft sidewalls
- prism/frustum sides
- sweep approximations

### Revolution

- [ ] curve-backed profile representation.
- [ ] full and partial sweep support.
- [ ] periodic `u` behavior.
- [ ] pole/apex singularity declarations.
- [ ] robust derivatives near axis collapse.
- [ ] exact analytic subtypes or subtype metadata for cylinder, cone, sphere,
  and torus.
- [ ] cap seam connection for capped revolutions.

Primary unblockers:

- cylinders
- cones
- spheres
- tori
- rotate-extrude
- thread foundations

## Freeform Patch Families

### Bezier

- [ ] control-net representation.
- [ ] Bernstein/de Casteljau evaluation.
- [ ] derivatives and boundary curve extraction.
- [ ] subdivision and subpatch extraction.
- [ ] rational-weight option.
- [ ] adaptive tessellation.

Role:

- simplest freeform evaluator
- B-spline/NURBS span target
- localized smooth patches

### B-Spline

- [ ] control grid.
- [ ] knot vectors and multiplicities.
- [ ] de Boor evaluation.
- [ ] derivatives.
- [ ] knot insertion.
- [ ] degree elevation.
- [ ] Bezier span extraction.
- [ ] periodic support.
- [ ] subdomain extraction.

Role:

- smooth loft surfaces
- fair surfaces
- reconstruction surfaces
- smooth path/profile infrastructure

### NURBS

- [ ] rational B-spline representation.
- [ ] positive weight validation.
- [ ] homogeneous evaluation.
- [ ] rational derivatives.
- [ ] exact conic support.
- [ ] import/export compatible canonical payloads.

Role:

- CAD-compatible freeform and analytic surfaces
- exact circles/conics/quadrics where represented rationally

## Procedural Patch Families

### Sweep

- [ ] rail/path curve.
- [ ] profile curve or section.
- [ ] parallel-transport frame.
- [ ] twist/scale/orientation laws.
- [ ] self-intersection checks.
- [ ] conversion to ruled/B-spline/NURBS patches where possible.
- [ ] explicit unsupported status where exact surface emission is not possible.

Role:

- pipes
- helical threads
- path-driven solids
- sweep-like loft workflows

### Offset

- [ ] basis surface reference.
- [ ] normal-based evaluation.
- [ ] C1 basis validation.
- [ ] self-intersection detection.
- [ ] trim propagation/reconstruction.
- [ ] shell/thicken integration.

Role:

- clearance
- shelling
- thickening
- displacement foundation

### Heightfield / Displacement

- [ ] native heightfield patch representation.
- [ ] sampled grid or analytic height function.
- [ ] derivative estimation.
- [ ] adaptive tessellation by height variation.
- [ ] tile/boundary compatibility.
- [ ] surface-native displacement over basis surfaces.
- [ ] solidified heightfield sidewall/cap policy.

Role:

- terrain-like surfaces
- embossing
- relief
- image-derived geometry without mesh fallback

### Subdivision

- [ ] control cage representation.
- [ ] half-edge or equivalent topology.
- [ ] Catmull-Clark and/or Loop support.
- [ ] creases and boundary rules.
- [ ] limit-surface evaluation.
- [ ] extraordinary-vertex handling.
- [ ] adaptive tessellation.
- [ ] conversion/extraction to Bezier or B-spline patches where possible.

Role:

- organic shapes
- sculptural surfaces
- imported subdivision assets

### Implicit

- [ ] scalar field representation.
- [ ] gradient evaluation.
- [ ] bounding volume and interval estimates.
- [ ] clipping/trimming support.
- [ ] explicit/exact conversion where possible.
- [ ] contouring only as an output boundary, not as modeling truth.
- [ ] boolean/composition semantics.

Role:

- signed-distance fields
- metaballs
- soft blends
- procedural volumes

## Surface Operations

### Construction

- [ ] planar face from region
- [ ] linear extrude
- [ ] rotate extrude
- [ ] loft
- [ ] ruled bridge
- [ ] sweep
- [ ] cap construction
- [ ] shell/thicken
- [ ] transform and instance

### Editing

- [ ] apply trim
- [ ] split by curve
- [ ] merge/coalesce patches
- [ ] offset
- [ ] displace
- [ ] fillet/blend
- [ ] chamfer

### Boolean

- [ ] patch-pair intersection matrix
- [ ] cut-curve mapping into both parameter spaces
- [ ] fragment classification
- [ ] shell reconstruction
- [ ] bounded healing
- [ ] metadata/provenance propagation

Initial family-pair order:

1. planar/planar
2. planar/ruled
3. planar/revolution
4. ruled/ruled
5. ruled/revolution
6. revolution/revolution
7. Bezier/B-spline/NURBS interactions
8. procedural family interactions, each explicitly bounded

## Public Capability Completion

### Primitives

- [ ] box closed-valid
- [ ] cylinder closed-valid, caps seam-connected
- [ ] cone/frustum closed-valid
- [ ] prism closed-valid
- [ ] sphere closed-valid with pole handling
- [ ] torus closed-valid with periodic seams
- [ ] polyhedron closed-valid with arbitrary face seams

### Modeling Helpers

- [ ] public linear extrude
- [ ] public rotate extrude
- [ ] public sweep
- [ ] public loft surface output
- [ ] text extrusion
- [ ] drafting geometry
- [ ] threading
- [ ] hinges
- [ ] heightfields/displacement
- [ ] CSG

## Verification Matrix

Each patch family needs:

- [ ] construction validation tests
- [ ] evaluation tests
- [ ] derivative/normal/frame tests
- [ ] transform tests
- [ ] trim tests
- [ ] boundary-use tests
- [ ] seam compatibility tests
- [ ] tessellation determinism tests
- [ ] watertightness tests for closed bodies
- [ ] projection tests
- [ ] split/subdomain tests
- [ ] boolean family-pair tests
- [ ] reference artifacts for visible outputs

## Suggested Milestone Sequence

### Milestone A: No Hidden Fallback Foundation

- complete shared libraries for vectors, transforms, metadata, loops
- remove hidden placeholder geometry
- make unsupported surface operations explicit

### Milestone B: Required Family Completion

- finish planar, ruled, and revolution patch contracts
- close-valid primitives
- seam-first tessellation for required families

### Milestone C: Surface Operations Completion

- public surface extrude/rotate-extrude
- loft surface executor with full seam orchestration
- initial surface booleans for required family pairs

### Milestone D: Freeform Surface Completion

- Bezier
- B-spline
- NURBS
- smooth loft and reconstruction support

### Milestone E: Procedural Surface Completion

- sweep
- offset
- heightfield/displacement
- subdivision
- implicit

### Milestone F: Public Promotion

- surface-first public defaults
- mesh compatibility explicit only
- docs and reference artifacts updated
- promotion gates prove no hidden fallback

## Required Specifications To Create Or Refine

- full patch family scope and exclusions replacement for current incomplete
  deferred-family specs
- complete patch-family feature coverage matrix
- patch projection and split contract
- patch-family tessellation contracts
- boolean family-pair support matrix
- heightfield/displacement surface contract
- sweep surface emission contract
- offset/shell/thicken contract
- subdivision and implicit bounded-support contracts
- public surface-default promotion contract

## Done Definition

Full surface-body implementation is complete when:

- all listed patch families have explicit contracts
- all implemented patch families can evaluate, trim, project, split, tessellate,
  and participate in supported booleans
- closed-valid bodies tessellate watertight without repair fallback
- current public modeling helpers can produce surface-native output or explicit
  unsupported results
- mesh is no longer used as canonical modeling truth anywhere in surface paths
