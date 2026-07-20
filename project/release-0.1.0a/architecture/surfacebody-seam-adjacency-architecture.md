# SurfaceBody Seam and Adjacency Architecture

## Overview

This document defines the architectural direction for seam truth, adjacency,
and shared-boundary ownership in the surface-first kernel.

The recommended model is a lightweight surface-native boundary representation
that follows the same structural pattern used by industrial B-rep systems:

- one shared seam truth
- one or more oriented patch-boundary uses of that seam
- shell-level adjacency built from those shared seams

This is the architectural direction most likely to produce deterministic,
watertight tessellation without relying on post-mesh repair.

## Industry Pattern

The dominant industry pattern for exact surface topology is a B-rep-style
structure in which:

- faces or patches are the surface-carrying primitives
- edges are shared topological boundary objects
- each face has its own oriented use of a shared edge
- shells are collections of faces connected through those shared edge uses

For Impression, this does not require importing a full industrial CAD kernel.

It does mean we should adopt the core structural lesson:

> shared boundary truth should exist once in the kernel, and patch-local
> boundary usage should refer to that shared truth rather than independently
> recreating it.

## Components

### SurfaceBody

`SurfaceBody` remains the canonical 3D modeling representation.

It contains:

- one or more shells
- stable body-level identity
- body-level kernel metadata

It does not own seam-local orientation directly.

### SurfaceShell

`SurfaceShell` is the primary topology container for patch adjacency.

It owns:

- patch membership
- seam membership
- shell-level adjacency truth
- open-versus-closed shell classification inputs

Seams should be interpreted in shell context, because seam validity depends on
which patches are participating and how.

### SurfacePatch

`SurfacePatch` is the surface-carrying primitive.

A patch owns:

- one surface family instance
- one parameter domain
- patch-local trim/boundary information
- patch-local references to seam usage

A patch should not independently define shared boundary truth when a boundary is
shared with another patch.

### Shared Seam

A seam is the kernel object representing one shared boundary truth.

At minimum a seam should define:

- stable seam identity
- the participating patch-boundary uses
- the canonical shared-boundary geometric intent
- whether the seam is interior/shared or open/external
- any continuity classification carried as kernel truth

The seam is the source of truth for:

- whether two patch boundaries are actually the same boundary
- which side owns canonical sampling
- whether a shell can be considered closed along that boundary

### Patch-Boundary Use

Each patch participating in a seam should do so through an oriented boundary
use.

This boundary-use record should define:

- the patch reference
- the boundary/trim reference on that patch
- the orientation of traversal relative to the seam

This is the surface-kernel analogue of a coedge or oriented edge use in
industrial B-rep systems.

## Proposed Record Shapes

The recommended Pythonic representation for these kernel records is small typed
record objects, most likely immutable dataclass-style structures.

This is not a claim that Python has one universal industry-standard B-rep
object model.

It is a recommendation that Impression adopt:

- industry-standard topological structure
- Pythonic typed record encoding

### Proposed `SurfaceSeam` Record

Architecturally, a seam record is expected to carry at least:

- `seam_id`
  - stable seam identity within the shell
- `shell_id`
  - owning shell reference
- `classification`
  - `shared` or `open`
- `uses`
  - ordered patch-boundary uses participating in the seam
- `shared_geometry_ref`
  - canonical 3D geometric boundary reference used as the seam truth
- `continuity_class`
  - optional continuity metadata such as positional/tangent class
- `metadata`
  - kernel-native seam metadata only

This record is the shell-level source of truth for a shared boundary.

### Proposed `PatchBoundaryUse` Record

Architecturally, a patch-boundary use is expected to carry at least:

- `use_id`
  - stable boundary-use identity
- `patch_id`
  - participating patch reference
- `seam_id`
  - referenced seam
- `loop_id`
  - stable loop reference on the participating patch
- `boundary_id`
  - patch-local boundary identifier
- `trim_ref`
  - patch-local 2D trim or boundary reference where applicable
- `orientation`
  - traversal direction of this patch boundary relative to the seam
- `loop_role`
  - whether this use belongs to an outer or inner boundary context on the patch

Optional derived compatibility data such as parameter-alignment checks may be
added later, but they are not required in the minimal first-pass record.

This record is the oriented patch-local use of shared seam truth.

## Geometric Reference Recommendation

The recommended geometric-reference split follows the standard trimmed-surface
pattern used by exact surface modelers:

- the seam owns the canonical 3D boundary geometry
- each patch-boundary use owns the patch-local 2D trim reference in parameter
  space

This means Impression should strongly prefer:

- one shared 3D boundary representation for the seam
- one per-patch 2D trimcurve or equivalent parameter-space boundary reference

When both exist, they should be parameterization-compatible so that evaluating
the shared 3D boundary and evaluating the patch-local trim through the patch
surface describe the same boundary to within tolerance.

This is the most direct surface-kernel analogue of the industry-standard
edge/coedge plus trimcurve pattern.

### Proposed `SurfaceBoundaryLoop` Record

Where a loop-level record is needed, the kernel should prefer an explicit loop
structure rather than flattening all boundary-use membership into the patch.

Architecturally, such a loop record is expected to carry at least:

- `patch_id`
  - owning patch reference
- `loop_id`
  - stable patch-local loop identity
- `uses`
  - ordered boundary-use references making up the loop
- `loop_role`
  - `outer` or `inner`

This keeps loop-level topology explicit for trims, hole boundaries, and
executor/tessellator traversal.

### Proposed `SurfaceShellAdjacency` View

Adjacency may be derivable from seams rather than stored as a separate durable
kernel object.

If a separate adjacency view is materialized, it should be derived from seam
truth and should contain:

- `shell_id`
- `patch_pair`
- `seam_id`
- `boundary_use_pair`
- `classification`

The seam remains the source of truth. Derived adjacency views exist for lookup
and tooling convenience, not as competing topology truth.

## Preferred Minimal First-Pass Contract

The current recommended first-pass kernel contract is:

- one explicit `SurfaceSeam`
- one explicit `PatchBoundaryUse` per participating patch boundary
- one seam-owned canonical 3D boundary reference
- one patch-owned 2D trim reference per boundary use
- one seam classification of `shared` or `open`

This is the smallest record set that still strongly supports seam-first,
watertight tessellation.

### Open Boundary

Not every patch boundary is shared.

The kernel must distinguish:

- shared seam
- open external boundary

Open boundaries should not be represented as “degenerate seams.”

They should remain explicit as non-shared boundary truth.

## Relationships

- shells own seam truth and patch adjacency
- seams connect one or more oriented patch-boundary uses
- patches refer to shared seams through boundary-use records
- trims remain patch-local, but shared boundaries must still point to one shared
  seam truth
- tessellation consumes seams as the canonical source of shared-edge agreement

## Data Flow

### Kernel Topology Flow

```text
surface patches
-> patch-local boundaries / trims
-> shell-level seam assembly
-> shared seam truth + oriented boundary uses
-> SurfaceShell
-> SurfaceBody
```

### Tessellation Flow

```text
SurfaceBody
-> SurfaceShell
-> shared seams
-> canonical seam sampling
-> per-patch tessellation using reused seam samples
-> mesh
```

## Cross-Domain Solutions

### Shared Seam Truth Instead of Independent Patch Edges

Adjacent patches must not derive their common edge independently and then hope
to match after tessellation.

That approach leads naturally to:

- duplicate edge truth
- tolerance welding
- non-deterministic seam repair
- non-watertight output risk

The recommended architecture is:

- one shared seam truth
- oriented boundary use per participating patch
- seam-first tessellation agreement

This naturally maps to:

- one `SurfaceSeam`
- one or more `PatchBoundaryUse` records attached to that seam

### Seam-First Tessellation for Watertight Output

The best path to watertight tessellation is to tessellate the shared seam once
and reuse that sample set on all participating patches.

This means:

- no independent edge sampling on each adjacent patch
- no best-effort post-stitch welding as the primary contract
- watertightness follows from shared topological boundary truth rather than
  repair

The preferred ownership point for seam sampling is:

- tessellator-owned
- driven by seam truth
- keyed by seam identity plus tessellation request

This means the kernel should not need to store permanent seam samples as part
of the canonical seam record.

### Trim and Seam Cooperation

Patch-local trims remain important, but when a trimmed boundary is also a shared
boundary, trim information must cooperate with seam truth rather than replacing
it.

This implies:

- trims describe patch-local parameter-space boundaries
- seams describe shared topological and 3D geometric boundary truth across
  patches

Future specification work should define exactly how the two attach, but the
boundary of responsibility is now explicit.

### Attached Transform and Instance Pattern

The recommended transform policy remains attached rather than eagerly baked.

This follows the same broad pattern used by industrial kernels and exchange
formats:

- geometry/topology stay structurally stable
- placement is carried through attached transform or location records
- transforms compose without forcing geometry duplication

For Impression this means:

- seam, patch, shell, and body truth should remain geometry/topology-first
- placement should compose through attached transform records where possible
- baking should be treated as a downstream necessity, not the default kernel
  state

## Identity and Cache Guidance

The current preferred invalidation policy is:

### `SurfaceBody`

Body identity changes when:

- shell membership changes
- shell topology changes
- body-owned kernel geometry/topology changes

### `SurfaceShell`

Shell identity changes when:

- patch membership changes
- seam membership changes
- shell-owned adjacency or topology changes

### `SurfaceSeam`

Seam identity changes when:

- seam classification changes
- participating boundary uses change
- shared 3D boundary geometry changes
- continuity class changes

### `PatchBoundaryUse`

Boundary-use identity changes when:

- patch/seam association changes
- referenced loop changes
- referenced trim changes
- orientation changes
- loop role changes

### Tessellation Caches

The preferred first-pass cache structure is:

- body tessellation cache keyed by:
  - body identity
  - tessellation request
  - any attached transform state not yet baked
- seam sampling cache keyed by:
  - seam identity
  - tessellation request

This keeps cache invalidation aligned to kernel truth rather than consumer-side
repair behavior.

### Shell Closure Comes From Boundary Truth

Closed-shell classification should derive from seam/open-boundary truth at the
shell level.

It should not depend on:

- post-tessellation repair
- metadata-only declarations
- mesh analysis as the first source of truth

Mesh analysis remains useful as downstream verification, but closure should
already be known from kernel topology.

## Architectural Recommendation

For Impression, the recommended surface-body seam model is:

1. `SurfaceShell` owns adjacency and shared seam truth
2. shared seams are explicit first-class kernel records
3. patches participate through oriented boundary-use records
4. open boundaries remain explicit and distinct from seams
5. tessellation samples seams canonically and reuses those samples across
   patches

This is the surface-kernel direction most likely to yield watertight
tessellation while staying compatible with the broader surface-first program.

## Relationship to Loft

Loft should consume this seam contract, not define it.

That means:

- loft planner emits resolved structural operators
- loft executor creates patches and boundary-use intent
- shell/seam assembly follows SurfaceBody law
- watertightness depends on surface-body boundary truth, not loft-local repair

## References

Industry pattern references that informed this direction:

- STEP topology schema: oriented edges, face bounds, advanced faces  
  <https://www.steptools.com/stds/smrl/data/resource_docs/geometric_and_topological_representation/sys/5_schema.htm>
- JT B-rep / CoEdge usage pattern  
  <https://www.plm.automation.siemens.com/en_us/Images/JT-v10-file-format-reference-rev-B_tcm1023-233786.pdf>
- Open CASCADE topology model and shared-shape concepts  
  <https://dev.opencascade.org/doc/overview/html/occt_user_guides__modeling_data.html>
