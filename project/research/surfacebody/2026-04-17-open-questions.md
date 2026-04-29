# SurfaceBody Open Questions Research

## Topic

Open research questions for the surface-first kernel, with emphasis on
`SurfaceBody`, seams, tessellation guarantees, and migration to a
surface-native internal model.

This document records what is already architecturally stable, what has partial
implementation evidence, and what should remain open research until more of the
surface program is built.

## Findings

### 1. The surface-first direction is architecturally stable

The high-level direction appears settled:

- `SurfaceBody` is the intended canonical 3D modeling representation
- `Mesh` is a derived artifact for preview, export, and analysis
- tessellation is a boundary subsystem, not the modeling kernel
- loft is a downstream client of the surface kernel, not the definer of it

This system-level direction does not appear to need more research before
implementation continues.

### 2. The current implementation proves useful scaffolding, not completion

There is already working code for:

- `SurfaceBody`
- `SurfaceShell`
- `SurfacePatch`
- `ParameterDomain`
- `TrimLoop`
- scene handoff
- tessellation request normalization
- planar patch tessellation
- mesh-analysis hookup

This is meaningful progress, but it does not yet close the core SurfaceBody
research questions.

In particular, the current implementation does not yet prove:

- seam-consistent shared-boundary sampling
- watertight tessellation for valid closed multi-patch bodies
- a final adjacency/seam source-of-truth model
- whole-program migration viability for primitives and modeling operations

### 3. Seam and adjacency law remains the largest unresolved kernel question

The most important unresolved surface-body area is still:

- who owns shared boundaries
- how seams are represented
- what counts as the same shared boundary
- what continuity/compatibility must hold before tessellation

This is the kernel law that later loft, extrude, revolve, and tessellation work
will all depend on.

Current architectural direction is now clearer:

- use one shared seam truth
- represent per-patch participation through oriented boundary-use records
- let shells own adjacency truth
- derive watertight tessellation from seam-first sampling rather than repair
- keep seam sampling owned by the tessellator and keyed from seam truth rather
  than storing baked seam samples in kernel state

The open question is no longer the broad direction.
It is now mostly the implementation policy and validation burden for that
model.

Industry evidence now strongly suggests the next-most-likely representation:

- explicit seam records
- oriented patch-boundary uses
- one canonical 3D boundary reference per seam
- one per-patch 2D trim reference per boundary use

This is the same core pattern used by STEP-style trimmed faces and by
industrial edge/coedge models.

### 4. Watertightness guarantees are not yet backed by full seam machinery

The current tessellation path triangulates patches independently and classifies
the body as open or closed using metadata.

That is enough to:

- exercise request normalization
- test patch tessellation
- keep preview/export policy work moving

It is not enough to honestly claim:

- strong seam-edge agreement across adjacent patches
- watertight tessellation of all valid closed bodies

This is an important research distinction:

- the contract exists in specs
- the implementation scaffold exists in code
- the seam/watertight proof path is still open

### 5. Patch-family scope is defined, but feature coverage needs validation

Current v1 patch-family scope in code is:

- `planar`
- `ruled`
- `revolution`

Deferred families include:

- `nurbs`
- `bspline`
- `subdivision`
- `implicit`
- `sweep`

This scope is reasonable, but still research-backed rather than fully proven.

Open questions remain around:

- whether ruled + planar + revolution are enough for the first surface-native
  primitive and modeling-op migration wave
- whether loft endcaps or later trim-heavy workflows will need earlier
  expansion of patch families

### 6. Trim semantics are only partially exercised

The current kernel has:

- trim-loop structure
- outer/inner categories
- orientation normalization
- domain validation

What is still insufficiently proven:

- shared trim ownership across adjacent patches
- trim interaction with seam law
- whether trim loops alone are the right first-class boundary representation
  for all v1 patch families
- how trim validity should fail in partially invalid bodies

This should remain research-backed until more non-planar and multi-patch cases
are exercised.

Industry references do give a strong directional answer, though:

- trimmed-surface definitions commonly use one outer loop and zero or more
  inner loops
- trim loops are oriented relative to the surface normal
- 2D trim curves in parameter space are paired with the face/surface they bound
- when both 2D and 3D boundary geometry exist, their parameterization should
  match to within tolerance

That suggests Impression should not treat trims as arbitrary patch decorations.
They are a first-class part of boundary truth on each patch.

### 7. Transform and identity policy are scaffolded but not yet stress-tested

The current kernel already supports:

- attached transforms
- stable hash-based identity
- metadata splitting between kernel and consumer namespaces

Open questions still remain:

- when baking must occur for downstream consumers
- how identity should behave through composition and later patch mutation
- whether identity and cache keys remain stable enough once adjacency and seam
  records become richer

This is especially important because many later tessellation and caching
decisions will depend on identity truth.

Industry evidence points toward an attached-transform pattern rather than eager
geometry baking:

- exact kernels commonly keep topology/geometry stable and carry placement
  through location/transform records
- tessellated assembly formats also support repositioned items rather than
  geometry duplication

For Impression, that strengthens the current architectural preference for
attached transforms and delayed baking.

The current preferred policy is now:

- attached-transform first
- baking only when required by downstream consumers
- body and seam caches keyed from structural identity plus request/transform
  state

### 8. Migration remains more open than the surface core

The surface core is ahead of the migration program.

The least-settled areas are now:

- which primitive APIs should switch first
- how modeling operations return surfaces without splitting public APIs
- how long compatibility adapters should remain visible
- what evidence burden is required before `SurfaceBody` is promoted as
  canonical across the project

These are not kernel-theory problems; they are rollout and evidence problems.

## Implications

### Immediate implication

The next surface work should prioritize kernel law over broader migration.

In practical terms, the highest-value unresolved surface questions are:

1. seam ownership and adjacency truth
2. shared-boundary sampling contract
3. watertightness conditions for closed valid bodies

Until those are stronger, broad API migration will risk building on a kernel
whose boundary rules are still provisional.

### Near-term implication

Loft should continue to wait on surface-body boundary law rather than inventing
its own seam model.

This confirms the current architectural split:

- loft defines structural evolution
- SurfaceBody defines patch/seam/shell truth

### Implementation implication

The best near-term implementation order appears to be:

1. finish seam and adjacency kernel rules
2. implement seam-consistent tessellation for shared boundaries
3. establish watertight closed-body guarantees
4. then migrate the first primitive/modeling API wave

## Open Questions Requiring Further Research

### Seam and Adjacency Truth

- The broad direction now appears settled in favor of explicit seam objects.
- The current preferred minimal field set is now proposed in the architecture.
- How much continuity information must be recorded in the kernel versus treated
  as optional metadata?

### Shared-Boundary Sampling

- What algorithm should guarantee that adjacent patches derive identical seam
  samples without post-stitch repair?
- Seam sampling now has a preferred ownership point:
  tessellator-owned, keyed by seam identity and tessellation request.
- The remaining question is the exact canonicalization and reuse algorithm.
- How should trim-driven boundaries participate in seam sampling?

### Watertightness

- What exact preconditions make a body “closed valid” in the surface kernel?
- When should tessellation fail fast versus emit an open classified result?
- How should watertightness QA interact with intentionally open bodies that
  still contain local shared boundaries?

### Trim Model

- Are trim loops alone sufficient for the v1 patch-family set?
- How should trim and seam ownership interact when a boundary is both trimmed
  and shared?
- What validation order is most useful:
  - domain validity first
  - orientation next
  - adjacency compatibility next
  - watertightness eligibility last

### Transform / Identity

- Which downstream operations require geometry baking rather than attached
  transforms?
- Are the currently proposed invalidation rules sufficient once seam and
  adjacency records are fully implemented?
- Which metadata changes, if any, should remain intentionally non-identity-
  affecting?

## Source-Supported Interim Conclusions

### Seam / Coedge Structure

The strongest industry pattern is:

- shared topological edge or seam
- oriented per-face/per-patch use
- shell-level adjacency derived from those shared seams

This supports the current architectural choice of explicit `SurfaceSeam` plus
`PatchBoundaryUse` records.

### 3D Boundary Plus 2D Trim Pairing

The strongest trimmed-surface pattern is:

- one shared 3D boundary representation
- one patch-local 2D trim representation per use
- matched parameterization between them

This strongly suggests that `shared_geometry_ref` should be a seam-owned 3D
boundary reference and `trim_ref` should be a patch-owned 2D trim reference.

### Watertight Tessellation

The strongest watertightness guidance found so far is:

- avoid duplicated boundary truth
- create explicit connecting/shared edge structures
- keep tessellation tied back to exact shared boundary truth where possible

That reinforces seam-first tessellation rather than independent edge sampling
plus weld/repair.

### Transform / Instance Handling

The strongest transform-handling pattern found so far is:

- preserve geometry/topology identity
- carry placement through attached transforms/locations
- compose transforms rather than eagerly baking geometry

This supports keeping Impression’s transform policy attached-first.

### Migration / Promotion

- Which primitive family is the best first proof point for surface-native
  returns?
- What verification matrix is sufficient to promote `SurfaceBody` from
  architectural target to canonical project-wide representation?
- What rollback triggers should remain in place during mesh-first
  decommissioning?

## Suggested Research Tasks

1. Prototype one explicit shared-boundary registry for multi-patch shells and
   compare it against purely patch-local seam references.
2. Build a minimal two-patch closed shell fixture and use it to test:
   - seam identity
   - edge agreement
   - watertightness failure modes
3. Exercise trim-heavy examples across planar and future ruled patches to see
   whether trim loops are sufficient as the only v1 trim primitive.
4. Run a first primitive migration spike on one simple primitive family to
   learn what public/internal API friction actually appears.
5. Record how current stable identities change under:
   - attached transform composition
   - shell ordering changes
   - future adjacency augmentation

## References

- `project/architecture/surface-first-internal-model.md`
- `project/architecture/surfacebody-seam-adjacency-architecture.md`
- `project/specifications/surface-01-surface-first-internal-model-program-v1_0.md`
- `project/specifications/surface-02-surface-core-data-model-v1_0.md`
- `project/specifications/surface-03-tessellation-boundary-v1_0.md`
- `project/specifications/surface-04-scene-and-modeling-api-adoption-v1_0.md`
- `project/specifications/surface-05-migration-and-compatibility-path-v1_0.md`
- `project/specifications/surface-09-surface-adjacency-and-seam-invariants-v1_0.md`
- `project/specifications/surface-13-seam-consistent-tessellation-watertightness-v1_0.md`
- `project/specifications/surface-38-shared-boundary-sampling-edge-agreement-v1_0.md`
- `project/specifications/surface-39-closed-body-watertight-tessellation-v1_0.md`
- `project/specifications/surface-44-primitive-api-surface-return-migration-v1_0.md`
- `project/specifications/surface-53-surface-migration-phase-ordering-v1_0.md`
- `src/impression/modeling/surface.py`
- `src/impression/modeling/tessellation.py`
- `src/impression/modeling/surface_scene.py`
- STEP topology schema  
  <https://www.steptools.com/stds/smrl/data/resource_docs/geometric_and_topological_representation/sys/5_schema.htm>
- JT B-rep / CoEdge usage pattern  
  <https://www.plm.automation.siemens.com/en_us/Images/JT-v10-file-format-reference-rev-B_tcm1023-233786.pdf>
- NVIDIA SMLib topology and trim-curve documentation  
  <https://docs.nvidia.com/smlib/manual/smlib/topology/index.html>
- NVIDIA SMLib trimmed-surface validity guidance  
  <https://docs.nvidia.com/smlib/manual/smlib/general-usage/index.html>
- Open CASCADE `TopLoc_Location` reference  
  <https://dev.opencascade.org/doc/refman/html/class_top_loc___location.html>
- CAx-IF recommended practices for watertight tessellation  
  <https://www.mbx-if.org/home/wp-content/uploads/2024/05/rec_prac_3dtess_geo_v11.pdf>
