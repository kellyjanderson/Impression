# SurfaceBody CSG Architecture

## Overview

This document defines the architecture for true `SurfaceBody` boolean
operations.

The current surfaced CSG branch already defines:

- input eligibility
- surfaced result envelopes
- public migration posture

What it does **not** yet define is the kernel law for actually executing
boolean operations on surface-native operands.

This architecture fills that gap.

The governing rule is:

> SurfaceBody CSG must operate on shells, patches, seams, trims, and
> classification truth directly.
>
> Mesh may appear only at preview, export, analysis, or explicit repair
> boundaries.

## Backlink

- [Surface-First Internal Model Architecture](surface-first-internal-model.md)
- [SurfaceBody Seam and Adjacency Architecture](surfacebody-seam-adjacency-architecture.md)
- [Surface-Native Capability Replacement Architecture](surface-native-capability-replacement-architecture.md)

## Components

### Operand Preparation

Operand preparation is responsible for turning caller-provided `SurfaceBody`
operands into canonical boolean inputs.

It owns:

- operand eligibility checks
- transform baking required for geometric comparison
- deterministic patch/shell ordering
- trim and seam validation required before execution

It does **not** decide the boolean result.

### Surface Intersection and Classification

The boolean kernel requires a surface-native intersection and classification
stage.

It owns:

- computing surface/surface intersection curves between operand patches
- mapping those intersections into patch-local trim space
- splitting affected operand boundaries into classified fragments
- determining which fragments lie inside, outside, or on the other operand

This stage is where boolean truth is discovered.

It should operate on:

- patch families
- canonical seam geometry
- patch-local trim references
- shell containment/classification queries

It must not degrade the problem into mesh-primary triangle clipping.

### Operand Fragment Graph

After intersections are discovered, each operand becomes a set of classified
surface fragments rather than a single untouched shell.

The fragment graph owns:

- surviving patch fragments
- discarded patch fragments
- newly created cut boundaries
- per-fragment provenance back to the source operand

This stage is temporary and executor-internal.

It exists so the result shell can be reconstructed deterministically.

### Result Topology Reconstructor

The reconstructor owns the actual result `SurfaceBody`.

It is responsible for:

- assembling surviving fragments into result shells
- constructing new trim loops from cut boundaries
- constructing new shared seams and open boundaries
- preserving or rebuilding adjacency truth
- deciding whether the result is one shell, multiple shells, or empty

This is where temporary fragment truth becomes durable kernel truth.

### Validity and Healing Gate

The boolean result must pass through an explicit validity gate.

This gate owns:

- trim validity checks on reconstructed patches
- seam pairing and open/shared classification checks
- closed-shell eligibility checks
- bounded healing or canonical cleanup where permitted

Healing here must remain narrow.

Allowed healing should be limited to:

- topological cleanup
- seam/use normalization
- trim canonicalization
- deterministic removal of zero-measure artifacts

It must not silently invent materially different geometry in order to force a
successful boolean result.

### Metadata and Provenance Propagation

Boolean execution also needs a durable rule for non-geometric information.

This stage owns:

- source provenance
- consumer metadata carry-forward
- operation metadata such as `union`, `difference`, or `intersection`
- color/material inheritance rules where relevant

Kernel-native topology truth must remain primary.

Consumer metadata propagation should be explicit and deterministic.

### Public CSG Surface

The public `csg.py` surface remains the stable caller boundary.

It owns:

- surfaced boolean entrypoints
- surfaced operation selection
- surfaced success/failure reporting
- migration and documentation posture

It should not reach through private kernel helpers in other modules.

## Relationships

- operand preparation produces canonical boolean operands
- intersection/classification consumes those operands and emits classified cut
  fragments
- the fragment graph feeds the result topology reconstructor
- the reconstructor emits candidate result shells and seams
- the validity/healing gate either accepts, rejects, or classifies the result
- metadata propagation finalizes surfaced result records
- the public CSG surface returns surfaced results to callers

The key relationship is:

```text
SurfaceBody operands
-> classified fragments
-> reconstructed SurfaceBody result
```

not:

```text
SurfaceBody operands
-> temporary mesh boolean
-> rewrapped surface result
```

## Data Flow

### Nominal Boolean Flow

```text
SurfaceBody operands
-> operand preparation
-> patch/patch intersection discovery
-> fragment classification
-> fragment graph
-> shell / seam / trim reconstruction
-> validity and bounded healing
-> SurfaceBody boolean result
-> tessellation on demand
```

### Failure Flow

```text
SurfaceBody operands
-> operand preparation
-> unsupported or invalid execution condition
-> surfaced boolean result with explicit failure / unsupported status
```

Failure must happen before any compatibility mesh shortcut is chosen as a
hidden fallback.

## Cross-Domain Solutions

### Split First, Then Reconstruct

The architectural heart of surfaced CSG is:

1. intersect and classify operands
2. split operands into fragments
3. reconstruct the result shell from those fragments

This is the surface-native analogue of industrial B-rep boolean behavior.

It keeps:

- patch meaning
- trim meaning
- seam truth
- shell truth

intact long enough for the result to remain surface-native.

### Seams Are Rebuilt as Kernel Truth

Boolean execution cannot treat seams as a post-process convenience.

When a cut boundary becomes part of the result, the resulting body must own:

- new seam identity where the boundary is shared
- new open-boundary truth where the boundary remains exposed
- new boundary-use records for participating patches

That keeps the result compatible with seam-first tessellation and watertight
classification.

### Trims Are the Result Boundary Law

For surfaced booleans, trims become the practical result-boundary mechanism.

This means the boolean result architecture depends on:

- patch-local trim reconstruction
- loop role classification
- boundary orientation correctness

The reconstructor should prefer trim reconstruction over inventing new patch
families or flattening the result into tessellated approximations.

### Bounded Healing, Not Mesh-Style Repair

SurfaceBody CSG will inevitably face near-degenerate cases.

The architecture permits bounded healing only when it preserves the intended
surface result.

Examples of allowed healing:

- removing duplicate seam uses
- removing zero-area trim slivers
- canonical loop orientation repair

Examples of disallowed healing:

- substituting a mesh boolean as hidden truth
- warping boundaries to force closure
- collapsing materially distinct patches into a fake success

### Initial Executable Scope Must Be Explicit

The first executable surfaced boolean slice should be intentionally bounded.

The architecture expects the initial executable scope to be defined in specs in
terms of:

- supported operation families
- supported operand shell classes
- supported patch families
- supported trim complexity
- unsupported cases that remain explicit

This keeps boolean implementation incremental without pretending the entire
general problem is solved at once.

### Metadata Follows Result Ownership

Boolean results should not simply concatenate operand metadata.

Instead:

- kernel metadata follows the reconstructed body/shell ownership
- provenance tracks source operands and operation type
- consumer metadata uses explicit carry-forward rules

This avoids the mesh-era pattern of “whichever side happened to win the union”
becoming the accidental metadata source.

## Related Architecture

- [Surface-First Internal Model Architecture](surface-first-internal-model.md)
- [SurfaceBody Seam and Adjacency Architecture](surfacebody-seam-adjacency-architecture.md)
- [Surface-Native Capability Replacement Architecture](surface-native-capability-replacement-architecture.md)

## Specifications

This architecture extends the existing surfaced boolean specification branch:

- [Surface Spec 102: Surface-Body Boolean Replacement (v1.0)](../specifications/surface-102-surface-body-boolean-replacement-v1_0.md)
- [Surface Spec 108: Surface Boolean Input Eligibility and Canonicalization (v1.0)](../specifications/surface-108-surface-boolean-input-eligibility-and-canonicalization-v1_0.md)
- [Surface Spec 109: Surface Boolean Result Contract and Failure Modes (v1.0)](../specifications/surface-109-surface-boolean-result-contract-and-failure-modes-v1_0.md)
- [Surface Spec 110: Surface Boolean Public API Migration and Reference Verification (v1.0)](../specifications/surface-110-surface-boolean-public-api-migration-and-reference-verification-v1_0.md)

The execution gap introduced by those earlier leaves is closed by these new
boolean execution leaves:

- [Surface Spec 117: Surface Boolean Intersection, Classification, and Operand Splitting (v1.0)](../specifications/surface-117-surface-boolean-intersection-classification-and-operand-splitting-v1_0.md)
- [Surface Spec 118: Surface Boolean Result Topology Reconstruction (v1.0)](../specifications/surface-118-surface-boolean-result-topology-reconstruction-v1_0.md)
- [Surface Spec 119: Surface Boolean Validity, Healing Limits, and Metadata Propagation (v1.0)](../specifications/surface-119-surface-boolean-validity-healing-limits-and-metadata-propagation-v1_0.md)
- [Surface Spec 120: Surface Boolean Initial Executable Scope and Reference Fixture Matrix (v1.0)](../specifications/surface-120-surface-boolean-initial-executable-scope-and-reference-fixture-matrix-v1_0.md)
