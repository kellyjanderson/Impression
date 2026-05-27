# Surface Support Libraries Plan

## Goal

Create shared libraries for the geometry, topology, patch, transform, metadata,
and verification services needed by current code and planned surface-body work.

This track removes application-wide duplication while giving future patch
families common infrastructure instead of one-off implementations.

## Current Duplication To Retire

The current code repeats several low-level helpers:

- vector coercion and normalization
- attached-frame transform construction
- axis-angle and mirror matrix construction
- surface-body shell combination
- color-to-consumer metadata conversion
- stable identity hashing patterns
- loop area and winding helpers
- polyline arclength resampling
- B-spline 2D/3D evaluation wrappers

## Library Targets

### `geometry.vectors`

Responsibilities:

- vector coercion: `as_vec2`, `as_vec3`, `as_points2`, `as_points3`
- finite-value validation
- unit-vector normalization
- tolerance-aware vector comparison
- point hashing / rounded point cache keys

Consumers:

- surface
- surface primitives
- surface ops
- drafting
- primitives
- transforms
- group
- text
- heightmap
- loft

### `geometry.transforms`

Responsibilities:

- 4x4 matrix validation
- axis-angle rotation matrix
- Euler composition
- mirror matrix
- attached frame from origin/direction
- frame from two basis vectors
- point/vector transform helpers
- transform composition and baking utilities

Consumers:

- primitives
- drafting
- text orientation
- surface primitives
- surface ops
- group and transform APIs
- hinge placement
- loft stations

### `geometry.curves`

Responsibilities:

- polyline curve
- line segment
- circular arc
- Bezier curve
- B-spline curve
- NURBS curve placeholder or final type
- arclength sampling
- closest-point projection
- derivative/tangent evaluation
- closed-loop handling

Consumers:

- `Path`
- `Path3D`
- topology
- loft
- surface trims
- seam p-curves
- ruled patches
- revolution profiles
- sweep patches

### `geometry.loops`

Responsibilities:

- signed area
- winding classification
- orientation normalization
- loop closure
- point-in-loop and containment
- loop resampling
- deterministic loop correspondence helpers
- hole/outer categorization helpers

Consumers:

- drawing2d
- topology
- surface trims
- tessellation
- loft
- text
- CSG trim reconstruction

### `surface.boundaries`

Responsibilities:

- shared seam records
- patch-boundary use records
- boundary loop records
- p-curve and 3D curve compatibility checks
- open/shared boundary classification
- seam continuity classification

Consumers:

- `SurfaceShell`
- tessellation
- CSG
- loft surface executor
- primitives

### `surface.patches`

Responsibilities:

- base patch contract
- common parameter-domain handling
- boundary evaluator protocol
- trim support
- periodicity/singularity declarations
- split/subdomain protocol
- projection protocol

Consumers:

- all patch families
- tessellation
- booleans
- picking/selection

### `surface.tessellation`

Responsibilities:

- request normalization
- seam-first sampling
- adaptive patch tessellation
- trim-aware tessellation
- watertightness validation
- deterministic cache keys

Consumers:

- preview/export/analysis
- reference artifacts
- surface-to-mesh adapter

### `surface.boolean`

Responsibilities:

- operand eligibility
- patch-pair intersection dispatch
- cut-curve records
- trim fragment mapping
- fragment graph
- result shell reconstruction
- validity/healing gate
- unsupported-case matrix

Consumers:

- public CSG
- hinge/heightfield operations
- future shell/thicken/fillet/chamfer work

### `surface.metadata`

Responsibilities:

- kernel versus consumer metadata split
- color metadata helpers
- provenance propagation
- deterministic metadata merge policy
- stable identity payload normalization

Consumers:

- surface bodies
- surface scenes
- primitives
- drafting
- text
- hinges
- CSG

### `verification.surface`

Responsibilities:

- no-hidden-mesh-fallback checks
- closed/open shell checks
- seam completeness checks
- tessellation determinism checks
- reference artifact helper utilities
- fixture matrix execution for patch-family combinations

Consumers:

- unit tests
- reference image/STL tests
- future promotion gates

## Work Plan

### Phase 1: Extract Pure Helpers

- [ ] extract vector coercion and normalization.
- [ ] extract transform matrix helpers.
- [ ] extract metadata/color helpers.
- [ ] replace call sites without behavior changes.
- [ ] add parity tests for old behavior.

Exit criteria:

- repeated `_as_vec3`, `_normalize`, `_attached_transform`, and
  `_surface_metadata` helpers are gone.

### Phase 2: Extract Loop And Curve Foundations

- [ ] consolidate signed area and winding.
- [ ] consolidate loop closure/resampling/correspondence helpers.
- [ ] define curve protocol used by Path, Path3D, trims, profiles, and seams.
- [ ] migrate B-spline 2D/3D common evaluation into shared implementation.

Exit criteria:

- loft and topology no longer carry duplicate loop resampling logic.
- B-spline curve types share evaluator core.

### Phase 3: Build Boundary And Patch Protocol Libraries

- [ ] add explicit patch-boundary use record.
- [ ] add explicit surface-boundary loop record.
- [ ] add p-curve and 3D seam curve records.
- [ ] define patch projection/split/boundary protocols.
- [ ] retrofit planar, ruled, and revolution patches onto the protocol.

Exit criteria:

- patch families expose common boundary and split surfaces.
- tessellation can consume boundaries without family-specific guessing.

### Phase 4: Build Surface Boolean Library

- [ ] isolate operand prep from public CSG module.
- [ ] isolate patch intersection dispatch.
- [ ] isolate fragment graph and reconstruction.
- [ ] add unsupported-case registry.
- [ ] keep mesh boolean implementation in standalone retained mesh tool module.

Exit criteria:

- public CSG becomes a thin API layer over surface boolean library and retained
  mesh tool library.

### Phase 5: Verification Library

- [ ] add static forbidden-call checks.
- [ ] add common reference artifact harness helpers.
- [ ] add patch-family matrix test utilities.
- [ ] add seam/watertightness assertion helpers.

Exit criteria:

- new patch families get a standard test harness by default.

## Required Specifications To Create Or Refine

- shared vector and transform helper contract
- shared curve protocol contract
- loop and planar-region utility contract
- patch-boundary use and p-curve contract
- patch projection/split protocol
- surface boolean library boundary contract
- surface verification helper contract

## Done Definition

This track is complete when:

- shared geometry and surface services are implemented once
- all current surface-related modules use the shared libraries
- adding a new patch family does not require reimplementing transforms, trims,
  seams, projection, tessellation request handling, metadata, or test harnesses
