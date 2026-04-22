# Mesh Analysis and Repair Architecture

## Overview

Impression has a strong mesh legacy.

That legacy is not only technical debt. It also contains genuinely useful
tools.

This architecture defines what mesh should still mean in Impression now that
`SurfaceBody` is the canonical modeling truth.

The governing rule is:

> Mesh is no longer the primary modeling document.
>
> Mesh remains a first-class downstream tool representation for analysis,
> repair, preview, export, and other explicit inspection workflows.

This architecture therefore has two jobs:

1. define the mesh capabilities Impression still needs
2. define which mesh-era modeling capabilities should be deleted instead of
   preserved

## Backlink

- [Surface-First Internal Model Architecture](surface-first-internal-model.md)
- [Surface-Native Capability Replacement Architecture](surface-native-capability-replacement-architecture.md)

## What Mesh Is For

Mesh remains valuable in Impression for:

- preview and rendering payloads
- STL export and other mesh-based interchange
- mesh quality analysis
- watertightness and manifold checks
- explicit repair operations
- explicit analysis tooling such as slicing or sectioning
- explicit standalone mesh utilities that are useful as tools rather than as
  canonical modeling truth

The key idea is:

- mesh is a tool and boundary representation
- mesh is not the internal authored or modeled truth

## What Mesh Is Not For

Mesh is not the architectural home for the primary modeling stack anymore.

We should not continue implementing canonical modeling capability in mesh for:

- primitives
- extrude
- loft
- text generation
- drafting geometry
- hinges
- threading
- heightfield/displacement as modeled truth
- public boolean modeling truth

Those belong to topology-native and surface-native systems.

## Components

### Canonical Surfaced Modeling

This remains the primary modeling path.

It owns:

- authored geometry
- topology
- `SurfaceBody`
- shells, seams, trims, patches
- surfaced modeling operations

Mesh tools may consume this layer, but they must not replace it.

### Mesh Analysis Toolchain

This is the retained mesh-side subsystem for understanding geometric outputs.

It owns capabilities such as:

- degenerate-face detection
- boundary-edge and nonmanifold-edge detection
- watertightness checks
- mesh statistics and quality summaries
- sectioning or slicing analysis where mesh is an acceptable tool representation

This layer is especially valuable for:

- regression verification
- print/export QA
- loft reconstruction checks

### Mesh Repair Toolchain

This is the retained mesh-side subsystem for explicit cleanup and salvage work.

It owns capabilities such as:

- cleanup
- canonical weld or merge behavior where explicitly requested
- bounded hole filling or defect repair
- repair-oriented analysis feedback

Repair must remain explicit.

It must not silently redefine canonical surfaced truth.

### Standalone Mesh Tool Utilities

Some mesh operations remain useful even if they are not the canonical modeling
path.

Examples include:

- mesh plane sectioning / slicing
- mesh CSG for standalone analysis or repair workflows
- mesh hull or other inspection utilities

These can remain valuable if they are presented honestly as tools, not as the
main modeling kernel.

### Mesh Consumer Boundary

Preview, export, and downstream viewers still consume mesh payloads.

This boundary owns:

- surfaced tessellation into mesh
- preview payload generation
- STL writing
- mesh-based consumer bridges

This boundary is allowed to stay mesh-centric as long as it is downstream of
surfaced truth.

### Deletion Program

Not all mesh-era code should survive.

The deletion program owns:

- identifying mesh-era modeling code that should disappear
- separating retained toolchain code from removable modeling code
- sequencing deletion once surfaced replacements are stable

## Architectural Classes

### Deprecated Mesh-Primary Capability

This class contains public or user-facing capability whose modeling result is
still mesh-primary.

Examples:

- mesh-returning primitive backends
- mesh-returning modeling operations
- mesh-returning public loft entrypoints
- public APIs whose real execution still depends on mesh-first truth

These should be:

- replaced by surfaced capability
- deprecated where appropriate
- deleted when the surfaced path is stable enough

### Retained Mesh Toolchain

This class contains mesh-based capability we still want in Impression because
it is useful as tooling.

Examples:

- preview/export payload generation
- mesh QA and watertightness analysis
- explicit repair workflows
- explicit slicing or sectioning analysis
- explicit mesh-only inspection utilities

This class is not deprecated simply because it uses mesh.

It is retained because it serves a legitimate downstream or tooling purpose.

### Missing Surfaced-Adjacent Tooling

There is also a class of tools that may legitimately be implemented with mesh
internals first, but whose role is to help validate surfaced modeling.

The clearest example is plane sectioning for loft verification.

This class should be tracked because it helps us prove surfaced correctness even
if it is not itself a surfaced modeling feature.

## Relationships

- surfaced modeling produces canonical geometry
- tessellation converts surfaced geometry into mesh for downstream use
- mesh analysis and repair consume explicit mesh payloads
- standalone mesh tools operate on explicit mesh inputs or explicit
  surface-to-mesh analysis conversions
- the deletion program removes mesh-era modeling capability that is not part of
  the retained toolchain

The intended relationship is:

```text
surface truth
-> mesh toolchain
```

not:

```text
mesh toolchain
-> hidden modeling truth
```

## Data Flow

### Canonical Modeling Flow

```text
authored geometry
-> topology / surfaced modeling
-> SurfaceBody
-> tessellation on demand
-> preview / export / analysis / repair tooling
```

### Explicit Tool Flow

```text
SurfaceBody or imported mesh
-> explicit mesh representation
-> analysis / repair / sectioning / inspection tool
-> report, repaired mesh, or verification artifact
```

### Forbidden Legacy Drift

```text
authored geometry
-> hidden mesh modeling path
-> surfaced wrapper
```

That is the flow this architecture is explicitly trying to eliminate.

## Core Retained Mesh Needs For Impression

The retained mesh toolchain should at minimum cover:

- mesh statistics and quality reporting
- watertightness and manifold checks
- explicit surface-to-mesh consumer conversion
- STL export
- preview/render payload generation
- repair-oriented cleanup
- plane sectioning / slicing for analysis

The likely high-value optional retained tools are:

- standalone mesh CSG for analysis or repair
- standalone mesh hull
- standalone mesh inspection helpers for debugging surfaced results

Those are acceptable if they are framed as explicit tools rather than canonical
modeling features.

## Loft-Specific Tooling Need

Loft especially benefits from a retained analysis toolchain.

The highest-value missing tool is:

- plane sectioning of generated loft outputs at known progression positions

This enables:

- reconstruction-style comparison against authored station topology
- twist detection
- region permutation detection
- hole drift and correspondence drift detection

That tool can begin with mesh-based sectioning if necessary, as long as it is
presented as explicit analysis tooling rather than canonical modeling truth.

## Cross-Domain Solutions

### Keep Useful Mesh Tools, Delete Mesh Modeling

The architectural answer is not “remove all mesh code.”

The better answer is:

- retain mesh where it is the right downstream or tooling representation
- delete mesh where it is still acting as the hidden modeling kernel

This preserves useful debugging and repair power without undoing the surfaced
migration.

### Mesh CSG May Survive As A Tool

Mesh CSG is not acceptable as the canonical public modeling truth.

But it may still be useful as:

- explicit analysis tooling
- explicit repair tooling
- comparison or debugging infrastructure while surfaced CSG matures

If kept, it should be housed and documented as toolchain capability, not as the
primary modeling path.

### Sectioning Is A Verification Tool, Not Just A Feature

Plane sectioning is not only a convenience tool.

For Impression it is part of how we can verify loft correctness.

That means sectioning belongs in the retained mesh analysis architecture even if
its first implementation is mesh-based.

### Canonical Inventory Home

The distinction between:

- removable mesh-era modeling code
- retained mesh toolchain
- missing analysis/repair tooling

belongs to architecture and its supporting inventory.

The detailed inventory currently lives in:

- [2026-04-18 Legacy Mesh Audit](../research/2026-04-18-legacy-mesh-audit.md)

## Related Architecture

- [Surface-First Internal Model Architecture](surface-first-internal-model.md)
- [Surface-Native Capability Replacement Architecture](surface-native-capability-replacement-architecture.md)
- [SurfaceBody CSG Architecture](surfacebody-csg-architecture.md)

## Specifications

This architecture is implemented by the following new specification branch:

- [Surface Spec 121: Mesh Analysis and Repair Toolchain Program (v1.0)](../specifications/surface-121-mesh-analysis-and-repair-toolchain-program-v1_0.md)

That parent branch is decomposed into:

- [Surface Spec 122: Mesh Capability Retention and Deletion Matrix (v1.0)](../specifications/surface-122-mesh-capability-retention-and-deletion-matrix-v1_0.md)
- [Surface Spec 123: Mesh Analysis Toolchain Contract (v1.0)](../specifications/surface-123-mesh-analysis-toolchain-contract-v1_0.md)
- [Surface Spec 124: Mesh Repair Toolchain Contract (v1.0)](../specifications/surface-124-mesh-repair-toolchain-contract-v1_0.md)
- [Surface Spec 125: Standalone Mesh Utility Tool Contract (v1.0)](../specifications/surface-125-standalone-mesh-utility-tool-contract-v1_0.md)
