# Impression Project Plan

This document tracks the near-term roadmap for Impression’s modeling toolkit. The goal is to provide a concise checklist we can revisit each sprint.

Current execution order is captured in `docs/feature-pipeline.md` and should be reviewed each planning cycle.

## Modeling Core

- [ ] **Primitives**
  - [x] Uniform mesh-backend API (`make_box`, `make_cylinder`, etc.) with consistent naming.
  - [ ] CAD backend parity for primitives (`cad` backend).
  - [ ] Metadata tagging (assign IDs/names to faces/edges as soon as they are created).
- [ ] **Shape library + color support**
  - [ ] Curate a lightweight library of reusable shapes (fasteners, brackets, gears) under permissive licenses; expose helper loaders (`load_shape("gear_m12")`).
  - [x] Per-object RGBA color pipeline.
  - [ ] Boolean color inheritance (new cut surfaces adopt the subtracting object’s color).
  - [ ] Per-face/per-vertex colors for documentation callouts (API + docs).
- [x] **Custom primitive tessellation**
  - [x] Generate core primitives (box, cylinder, torus, sphere, cone, prism) via internal tessellation so preview/export align.
  - [ ] Investigate caching of generated templates for performance.
- [ ] **CSG Operations**
  - [x] Wrapper helpers for `union`, `difference`, `intersection` with tolerance controls.
  - [ ] Auto-clean options for mesh hygiene/repair during booleans.
  - [ ] Optional provenance tracking (node graph) so regenerations only recompute affected nodes.

## Advanced Modeling

- [ ] **Skinning & Lofting**
  - `loft(profiles, path=None)` helper that uses build123d loft when available and falls back to PyVista mesh-based approximations.
  - Support closed paths, ruled vs smooth, and per-profile parameter overrides.
- [ ] **Splines & Paths**
  - [x] Canonical `Path` abstraction (polyline) with length/closed utilities.
  - [ ] Bezier/spline `Path` types that sweep, pipe, and array modifiers consume.
  - [ ] Extrude/sweep along path, including twist/scale controls.
- [ ] **Blobs / NURBS**
  - Implicit “blob” builder (metaballs via OpenVDB or marching cubes) with auto-conversion to PolyData.
  - NURBS surface helper leveraging OCC BSplines for precise surfacing.

## Advanced High-Level Modeling

- [ ] **Chamfers / Fillets**
  - Selection system: tag faces/edges on creation, expose selectors (by name, normal, area, adjacency, custom lambda).
  - Build123d-backed `fillet/chamfer` operations with fallback mesh bevel approximations where exact CAD isn’t available.
  - `round(face_selector, radius)` helper that evaluates face orientation and applies the requested fillet.
- [ ] **Auto-Round Helper**
  - `round_sharp_edges(radius)` routine that scans for faces whose adjacent dihedral angles exceed 89° and applies fillets automatically. (Still needs investigation to ensure it captures intent without over-rounding.)

## Tooling & Glue

- [ ] **Backends & Scene Graph**
  - Unified scene representation that can hold mesh, CAD, or implicit blobs; internal Mesh/Polyline pipeline is in place.
- [ ] **Parameter Management**
  - Config-driven parameter overrides (YAML/JSON) and CLI flags to sweep model variants.
- [ ] **Caching & Tessellation**
  - Cache CAD tessellations keyed by parameter/tolerance to speed up preview/export in iterative workflows.
- [ ] **IDE Integration**
  - [x] VS Code extension commands for “Preview current model”, “Export STL”, and test helpers.
  - [ ] Wrap `impression preview` in a panel/webview and stream logs from the editor.
  - [ ] Quick links to docs/examples inside the extension UI.

## Additional Nuances

- [x] Default units configurable via `~/.impression/impression.cfg` (mm/m/in).
- [ ] Tolerance policy for booleans/meshing (surface, edge, and voxel tolerances).
- [ ] Constraint-aware sketches for CAD workflows (equality, tangency, concentricity) to keep parametric edits stable.
- [x] Mesh analysis warnings (boundary, nonmanifold, degenerate) on load.
- [ ] Validation hooks (minimum thickness, watertight checks, repair) after heavy operations.
- [ ] Future format support: STEP/IGES export for CAD pipelines alongside STL for mesh workflows.

## Testing & QA

- [x] STL regression tests verifying watertightness/manifoldness for exported meshes (`scripts/run_stl_tests.py`).
- [ ] Adopt pytest as the standard test harness for CLI + unit tests.
  - pytest offers concise fixtures, better parameterization, and aligns with our preference; unittest is batteries-included but more verbose.
  - Decision factors: integration with existing scripts (call via subprocess vs plugins), fixture complexity (e.g., temporary workspaces, font downloads), and ecosystem tooling (coverage, plugin availability).

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

- [x] **Text rendering**
  - Support vector-based text primitives that can be extruded or engraved, enabling technical annotations directly in scenes.
  - Plan for both mesh-only text (triangulated glyphs) and CAD text (OCC-based fonts) with consistent styling controls.
- [x] **2.5D (2D objects in 3D space)**
  - Provide utilities for lines, planes, arrows, and dimension markers that exist as thin geometry for drafting-like overlays.
  - Allow writers to place these references on any plane/section to document assemblies without leaving Impression.
- [ ] **Image rendering**
  - **Image import (future):** allow referencing external imagery as textured planes for contextual documentation.
  - **Image export (near-term):** render high-resolution images from scenes (orthographic/perspective) with overlays for technical documentation. Batch rendering hooks for versioned snapshots.
