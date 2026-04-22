# Changelog

## Unreleased

## 0.0.3a0

### Surface-First Modeling

- Added the core `SurfaceBody` / `SurfaceShell` / `SurfacePatch` surfaced
  modeling stack plus surfaced tessellation, scene, and primitive helpers.
- Reworked loft around the newer surfaced planner/executor and added stronger
  slice-based diagnostics and regression coverage.
- Added surfaced implementations for drafting, text, threading, hinges, and
  heightmaps.
- Removed public `morph` and public `extrude` from the supported modeling
  surface.

### Surfaced CSG

- Added a bounded surfaced boolean lane with structured results, explicit
  `succeeded` / `invalid` / `unsupported` posture, deterministic cleanup, and
  provenance metadata.
- Added stronger surfaced CSG overlap fixtures and promotion-gate coverage for
  `csg_union_box_post` and `csg_difference_slot`.
- Kept the `csg_intersection_box_sphere` promotion item explicitly open as a
  real kernel boundary rather than masking it with weak reference evidence.

### Testing Architecture

- Added shared reference-image/reference-STL lifecycle helpers with bootstrap,
  invalidation, grouped completeness, and promotion-gate behavior.
- Added computer-vision-oriented testing helpers for slice silhouette
  comparison, text verification, camera/framing contracts, canonical
  object-view bundles, handedness checks, and honest diagnostic triptychs.
- Added stronger text verification and glyph-capable text reference fixtures so
  bad fallback-box output now fails instead of silently passing.

### Project Structure And Docs

- Added the `project/` planning workspace for architecture, specifications,
  test-specifications, progression, release definitions, research, and future
  features.
- Added repository and project-specific agent guidance under `agents/` and
  `project/agents/`.
- Retired a large set of stale legacy planning docs that no longer matched the
  active system.

### Loft Topology Transition v1

- Added deterministic station-based topology lofting via `loft_sections(...)` with explicit station frames (`origin`, `u`, `v`, `n`).
- Added deterministic split/merge controls:
  - `split_merge_mode` (`"fail"` or `"resolve"`)
  - `split_merge_steps`
  - `split_merge_bias`
- Supported transition classes in resolve mode:
  - region split (`1->N`)
  - region merge (`N->1`)
  - hole split (`1->N`)
  - hole merge (`N->1`)
- Supported stable/event transitions:
  - region/hole stable
  - region birth/death
  - hole birth/death
- Explicitly unsupported:
  - many-to-many ambiguity (`N->M`, where `N>1` and `M>1`) for regions or holes.
