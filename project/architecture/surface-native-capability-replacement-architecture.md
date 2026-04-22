# Surface-Native Capability Replacement Architecture

## Overview

This document defines how deprecated mesh-primary capabilities are replaced by
surface-first equivalents.

The governing rule is:

> every deprecated capability needs a surface-first replacement path

This does **not** require every legacy file to gain a one-to-one sibling.

Some areas should be replaced in place, while others need dedicated
surface-native modules before they can later be folded back into the canonical
public API.

## Backlink

- [Surface-First Internal Model Architecture](surface-first-internal-model.md)

## Replacement Classes

### Promote In Place

These areas already have meaningful surface-native implementation and should
continue migrating within the existing public module:

- primitives
- extrude
- loft
- tessellation boundary

The public long-term outcome is:

- surface-first defaults
- mesh only at explicit consumer boundaries
- eventual removal of mesh-primary public return behavior

### Replace With Surface-Native Capability Modules

These areas remain mesh-only in their primary capability today and need
surface-native replacements before mesh decommission can proceed:

- drafting
- text
- booleans
- threading
- hinges
- heightfields and displacement

These may initially ship behind dedicated surface-native modules or helpers,
but they must terminate in canonical surface objects rather than mesh truth.

## Surface-First Output Rule

Replacement capabilities should terminate in one of:

- `SurfaceBody`
- `SurfaceConsumerCollection`
- topology-native intermediate structures when the capability is not yet at the
  surface-emission stage

They must not terminate in mesh as the primary modeling truth.

## Mesh Boundary Rule

Mesh remains a boundary artifact only for:

- preview
- export
- analysis
- explicit compatibility bridges during migration

Any new capability replacement should assume seam-first tessellation and
surface-native composition rather than internal mesh assembly.

## Capability Notes

### Drafting

Drafting replacements may use planar patches, polylines, or surface-backed
annotation bodies depending on the feature class, but should not rely on mesh as
the authored or canonical truth.

### Text

Text should remain topology-native as long as possible and emit surface-native
raised, inset, or embossed bodies through surface-first modeling ops.

### Booleans

Boolean replacement is expected to operate on surface bodies, shells, seams,
and trims rather than triangle meshes.

### Threading

Threading replacements should use analytic or structured surface patch families
where possible and only tessellate at consumer boundaries.

### Hinges

Hinge replacements should be built from reusable surface-native primitives and
surface-native boolean/composition capabilities.

### Heightfields

Heightfield and displacement replacements should define a surface-native
representation for height-derived geometry and deform existing surface-native
objects without making mesh the primary document.

## Verification

Each replacement branch must have:

- a feature specification
- a paired test specification
- durable docs

When visually meaningful outputs exist, the verification story should include
reference artifacts such as images and STL files.
