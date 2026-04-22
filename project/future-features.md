# Future Features

This document collects promising future directions that are worth preserving
before they are ready for full architecture and specification work.

Items here are intentionally exploratory. They are not commitments, and they do
not replace architecture or specifications once a feature becomes an active work
item.

## Control Station Inference

### Idea

Add a user-facing loft tool that starts from dense linear station space,
infers a smaller set of control stations, and returns a reduced progression
made of:

- topology stations
- control stations

This would preserve non-linear shape evolution without requiring the author to
brute-force the result with many tiny linear station spans.

### Why It Matters

- creates a sparse, more expressive loft authoring path
- separates structural anchors from shape-driving control samples
- fits between explicit station authoring and low-level surface patch work

### Architecture Note

- [Control Station Inference Architecture](future-features/control-station-inference-architecture.md)

## Spanwise Loft Consolidation

### Idea

Add a loft-aware capability that looks beyond the immediate station interval and
recognizes when a longer run of stations could be represented as one larger
coherent span in surfacebody terms.

This idea can be pursued in three distinct ways:

- inline loft enhancement
- post-loft optimization
- repair-oriented tooling

### Why It Matters

- dense station lofts can be locally valid but globally over-segmented
- larger-span reasoning may reduce realized patch complexity
- it complements control-station inference by simplifying realized surface
  structure rather than authored progression structure

### Architecture Notes

- [Spanwise Loft Consolidation Architecture](future-features/spanwise-loft-consolidation-architecture.md)
- [Spanwise Loft Inline Enhancement Architecture](future-features/spanwise-loft-inline-enhancement-architecture.md)
- [Spanwise Loft Postprocessing Optimization Architecture](future-features/spanwise-loft-postprocessing-optimization-architecture.md)
- [Spanwise Loft Repair Tool Architecture](future-features/spanwise-loft-repair-tool-architecture.md)

## Model-Assisted Mesh Repair

### Idea

Use Impression's surface-modeling stack as a repair method for damaged foreign
meshes.

Instead of treating repair as only triangle cleanup, the workflow would:

1. import a foreign mesh
2. analyze and localize a damaged region
3. section the mesh with planes or other probes near the damaged span
4. recover topology-like contour stations from healthy neighboring slices
5. remove the damaged mesh band
6. reconstruct that span with a surface-first modeling operation such as loft
7. tessellate the reconstructed surface back into a mesh repair result when
   needed

### Why It Matters

- turns Impression's modeling system into a high-value repair tool
- allows repair by geometric reconstruction instead of only local triangle
  surgery
- creates a strong bridge between the mesh analysis/repair branch and the
  surface-first modeling branch
- gives loft and related tools a practical workflow beyond authoring new parts

### Likely Building Blocks

- mesh import and defect localization
- plane sectioning / contour extraction
- contour cleanup and canonicalization
- section-to-station conversion
- damaged-band deletion
- loft or patch reconstruction between recovered stations
- post-repair validation:
  - watertightness
  - self-intersection
  - continuity / drift checks

## Patchwise Loft

### Idea

Add a loft-like reconstruction mode that does not require a complete recovered
topology for the full body. Instead, it captures the surface patch structure
around a local damaged area and rebuilds only the surfaces needed to span that
patch neighborhood.

This is especially useful when bad foreign-mesh triangles are not concentrated
in a clean band, but instead appear in scattered local patches.

### Why It Matters

- many damaged meshes do not fail in a single clean loft band
- localized patch reconstruction is often a better fit than full-section
  topological recovery
- it could enable repair of more irregular defects while still using
  surface-first reconstruction logic

### Intended Workflow Shape

1. identify the damaged patch region
2. recover neighboring patch boundaries and local boundary intent
3. capture enough local surface structure to define the missing patch network
4. use a loft-like surface construction to rebuild the span between those local
   patch boundaries
5. stitch the repaired patch set back into the surrounding shell or mesh repair
   result

### Open Questions

- what minimum local structure is needed to define a patchwise loft repair
- how patch adjacency and seam intent should be represented for local repair
- how patchwise repair relates to the broader SurfaceBody seam and adjacency
  model
- whether patchwise loft should be a standalone repair primitive or a mode of a
  more general surface reconstruction system
