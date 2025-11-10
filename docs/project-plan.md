# Impression Project Plan

This document tracks the near-term roadmap for Impression’s modeling toolkit. The goal is to provide a concise checklist we can revisit each sprint.

## Modeling Core

- **Primitives**
  - Uniform API (`make_box`, `make_cylinder`, etc.) with backend selection (`mesh` vs `cad`).
  - Consistent parameter names (center, dimensions, orientation) and metadata tagging (assign IDs/names to faces/edges as soon as they are created).
- **CSG Operations**
  - Wrapper helpers for `union`, `difference`, `intersection` that expose tolerance controls and auto-clean options.
  - Optional provenance tracking (node graph) so regenerations only recompute affected nodes.

## Advanced Modeling

- **Skinning & Lofting**
  - `loft(profiles, path=None)` helper that uses build123d loft when available and falls back to PyVista mesh-based approximations.
  - Support closed paths, ruled vs smooth, and per-profile parameter overrides.
- **Splines & Paths**
  - Canonical `Path` abstraction (polyline, Bezier, spline) that sweep, pipe, and array modifiers consume.
  - Extrude/sweep along path, including twist/scale controls.
- **Blobs / NURBS**
  - Implicit “blob” builder (metaballs via OpenVDB or marching cubes) with auto-conversion to PolyData.
  - NURBS surface helper leveraging OCC BSplines for precise surfacing.

## Advanced High-Level Modeling

- **Chamfers / Fillets**
  - Selection system: tag faces/edges on creation, expose selectors (by name, normal, area, adjacency, custom lambda).
  - Build123d-backed `fillet/chamfer` operations with fallback mesh bevel approximations where exact CAD isn’t available.
  - `round(face_selector, radius)` helper that evaluates face orientation and applies the requested fillet.
- **Auto-Round Helper**
  - `round_sharp_edges(radius)` routine that scans for faces whose adjacent dihedral angles exceed 89° and applies fillets automatically. (Still needs investigation to ensure it captures intent without over-rounding.)

## Tooling & Glue

- **Backends & Scene Graph**
  - Unified scene representation that can hold PyVista meshes, build123d solids, or implicit blobs, so preview/export pipelines remain backend agnostic.
- **Parameter Management**
  - Config-driven parameter overrides (YAML/JSON) and CLI flags to sweep model variants.
- **Caching & Tessellation**
  - Cache CAD tessellations keyed by parameter/tolerance to speed up preview/export in iterative workflows.

## Additional Nuances

- Units & tolerances: settle on default units (likely millimeters) and surface tolerances for booleans/meshing.
- Constraint-aware sketches for CAD workflows (equality, tangency, concentricity) to keep parametric edits stable.
- Validation hooks (minimum thickness, watertight checks) after heavy operations.
- Future format support: STEP/IGES export for CAD pipelines alongside STL for mesh workflows.

## Appendix: Gravitational Modeling

- **Concept overview**
  - Anchors (points or closed paths) remain fixed while interior surface vertices respond to gravitational or anti-gravitational primitives.
  - Surface is represented as a triangulated mesh; edges exceeding a maximum length trigger refinement so the mesh can stretch smoothly.
  - Force model combines attraction/repulsion from sources with internal tension (mass-spring) to keep the surface cohesive.
- **Prototype goals**
  - Implement a small solver that: initializes a mesh between anchors, applies forces iteratively, remeshes as needed, and outputs a PyVista surface.
  - Allow both positive (pull) and negative (push) sources; long-term, support animated sources for morphing effects.
  - Expose parameters for max triangle size, damping, and convergence tolerance.

## Appendix: Documentation & Presentation Features

- **Text rendering**
  - Support vector-based text primitives that can be extruded or engraved, enabling technical annotations directly in scenes.
  - Plan for both mesh-only text (triangulated glyphs) and CAD text (OCC-based fonts) with consistent styling controls.
- **2.5D (2D objects in 3D space)**
  - Provide utilities for lines, planes, arrows, and dimension markers that exist as thin geometry for drafting-like overlays.
  - Allow writers to place these references on any plane/section to document assemblies without leaving Impression.
- **Image rendering**
  - Embed raster images (screenshots, scanned drawings) as textured planes or billboards for hybrid documentation.
  - Long-term: export “technical sheets” that mix text, images, 2D overlays, and 3D previews for documentation handoffs.
