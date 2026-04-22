# Surface-First Internal Model Architecture

## Overview

Impression is moving from a mesh-first modeling kernel to a surface-first
modeling kernel.

The target architecture is:

```text
authored geometry
-> planar topology
-> surface body
-> tessellation
-> preview / render / export / analysis
```

In this architecture, `Mesh` is no longer the primary internal representation
of modeled geometry. Instead, the primary internal representation becomes a
surface-native body model that all modeling tools are expected to consume and
produce.

This architecture is whole-system in scope. It is not a loft-only change.

Loft remains one of the most important migration targets, but the surface
representation must be established before the dedicated loft refactor track can
be completed cleanly.

## Components

### Authored Geometry

The developer-facing authored geometry layer remains responsible for expressive,
ergonomic construction inputs such as:

- `drawing2d`
- `path3d`
- primitives
- transforms
- higher-level modeling helpers

This layer should remain authoring-friendly and should not expose tessellation
as its primary contract.

### Topology

The topology layer remains responsible for:

- planar loop / region / section truth
- winding
- containment
- correspondence preparation
- topology-event reasoning for tools such as loft

Topology remains upstream of surface realization.

### Surface Kernel

The new surface kernel becomes the core 3D modeling representation.

It must represent:

- surface bodies
- shells
- patches
- trims
- adjacency / seam ownership
- transforms and metadata

All modeling tools should ultimately consume and produce this surface-native
representation.

Its seam and adjacency truth should follow a shared-boundary model closer to
industrial B-rep practice:

- one shared seam truth
- oriented patch-boundary uses of that seam
- shell-level adjacency as the source of boundary truth

This is the kernel direction most likely to support deterministic, watertight
tessellation without repair-oriented fallback.

### Tessellation

Tessellation becomes a boundary subsystem rather than the modeling kernel.

It is responsible for:

- rendering tessellation
- export tessellation
- deterministic vertex/face generation
- seam-consistent tessellation across adjacent patches
- watertightness guarantees where the upstream surface body is valid

### Consumers

Downstream systems consume tessellated meshes rather than owning geometric
truth:

- preview
- render
- STL export
- mesh analysis
- any legacy mesh-only integrations

## Relationships

- authored geometry produces topology or directly authored surface-ready input
- topology informs surface construction where planar or cross-sectional truth is
  required
- modeling tools produce `SurfaceBody` rather than `Mesh`
- tessellation converts `SurfaceBody` into `Mesh` on demand
- preview/export/analysis consume tessellated meshes

Loft specifically should move from:

```text
planner -> mesh executor
```

to:

```text
planner -> surface executor -> tessellation
```

But this should happen only after the surface kernel and tessellation boundary
are defined for Impression as a whole.

## Data Flow

### Nominal Data Flow

```text
author input
-> normalized 2D/3D authored geometry
-> topology normalization where needed
-> modeling operation
-> surface body
-> tessellation request with quality/tolerance policy
-> mesh
-> preview/export/analysis
```

### Legacy Compatibility Data Flow

During migration, some callers may still request or expect mesh output.

In those cases:

```text
surface body
-> compatibility tessellation adapter
-> mesh consumer
```

Compatibility layers must not become the new architectural center.

## Cross-Domain Solutions

### Kernel Truth

The architectural truth for modeled 3D geometry becomes surfaces, not meshes.

This resolves the current conflict where:

- modeling wants geometric structure and continuity
- rendering/export want triangles

The system should preserve geometric meaning until the latest reasonable stage.

### Deterministic Tessellation Boundary

Tessellation must be deterministic so the system retains the current
determinism expectations established by the loft planner/executor work.

Same surface input + same tessellation settings must produce identical mesh
output.

For shared patch boundaries, this implies seam-first tessellation:

- tessellate the shared seam once
- reuse the same seam samples across adjacent patches
- do not rely on independent patch-edge sampling plus post-weld repair

### Quality as a Boundary Policy

Mesh quality settings should control tessellation density and tolerance, not the
meaning of the modeled object.

This allows:

- coarse preview
- higher-fidelity export
- shared geometry meaning across multiple delivery targets

### Loft as a Dedicated Surface Client

Loft should not define the surface architecture, even though it is one of the
highest-value consumers of it.

The correct order is:

1. establish surface kernel
2. establish tessellation boundary
3. migrate whole-system contracts
4. execute the dedicated loft refactor against that foundation

### Non-Goals for the First Surface Program

The first architecture pass does not require:

- full CAD-kernel replacement
- general surface booleans
- mandatory NURBS everywhere
- immediate elimination of all mesh adapters

The first program goal is to make surfaces the internal representation of
Impression, with clean tessellation boundaries and a migration path for
existing tools.

## Related Architecture

- [SurfaceBody Seam and Adjacency Architecture](surfacebody-seam-adjacency-architecture.md)

## Specifications

This architecture is implemented by the following specification branch:

- [Surface Spec 01: Surface-First Internal Model Program (v1.0)](../specifications/surface-01-surface-first-internal-model-program-v1_0.md)

That parent specification is decomposed into these first-generation child
branches:

- [Surface Spec 02: Surface Core Data Model (v1.0)](../specifications/surface-02-surface-core-data-model-v1_0.md)
- [Surface Spec 03: Tessellation Boundary and Rendering Contract (v1.0)](../specifications/surface-03-tessellation-boundary-v1_0.md)
- [Surface Spec 04: Scene and Modeling API Surface Adoption (v1.0)](../specifications/surface-04-scene-and-modeling-api-adoption-v1_0.md)
- [Surface Spec 05: Migration and Compatibility Path (v1.0)](../specifications/surface-05-migration-and-compatibility-path-v1_0.md)
- [Surface Spec 06: Loft Surface Refactor Track (v1.0)](../specifications/surface-06-loft-surface-refactor-track-v1_0.md)
