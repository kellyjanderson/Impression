# Release 0.0.3a0

## Intent

`0.0.3a0` is the release that turns Impression into a much more explicit
surface-first system while making its verification story durable enough to
trust the migration work.

This release is centered on three visible outcomes:

- surfaced modeling is no longer only an architectural direction; it is a real
  executable lane across loft, drafting, hinges, text, heightmaps, and bounded
  surfaced CSG
- reusable testing architecture now exists as a first-class project concern,
  including reference-artifact lifecycle rules and computer-vision-oriented
  verification helpers
- the project documents, progression, and research trail have been rebuilt so
  the current system is understandable without depending on legacy planning
  artifacts

## Delivered Features

- surface-first core data structures and tessellation helpers now exist in
  `src/impression/modeling/surface.py`, `surface_scene.py`,
  `_surface_primitives.py`, `_surface_ops.py`, and `tessellation.py`
- loft has been reworked around the newer surfaced planner/executor stack, with
  substantially broader regression coverage and diagnostic slice comparison
  support
- drafting, threading, hinges, heightmaps, and text all have surfaced support
  and surfaced regression coverage
- public `morph` and public `extrude` have been removed from the supported
  modeling surface
- surfaced CSG has an honest bounded execution lane with structured results,
  explicit invalid/unsupported posture, overlap diagnostics, and reference
  fixtures for the currently-supported overlap evidence cases
- reference-image and reference-STL testing now has first-run bootstrap,
  invalidation, grouped completeness, and promotion-gate behavior
- computer-vision-oriented verification helpers now exist for:
  - slice silhouette comparison
  - orientation-sensitive witness fixtures
  - text/glyph verification
  - camera/framing contracts
  - canonical object-view bundles
  - handedness/mirror checks
  - honest diagnostic triptych rules

## Major Code Changes

### Surface Modeling

- added the surfaced modeling modules and helper layers:
  - `src/impression/modeling/surface.py`
  - `src/impression/modeling/surface_scene.py`
  - `src/impression/modeling/_surface_primitives.py`
  - `src/impression/modeling/_surface_ops.py`
  - `src/impression/modeling/tessellation.py`
- expanded surfaced implementations in:
  - `src/impression/modeling/loft.py`
  - `src/impression/modeling/drafting.py`
  - `src/impression/modeling/heightmap.py`
  - `src/impression/modeling/hinges.py`
  - `src/impression/modeling/text.py`
  - `src/impression/modeling/threading.py`
  - `src/impression/modeling/primitives.py`
  - `src/impression/modeling/ops.py`

### Bounded Surfaced CSG

- added surfaced operand preparation, intersection staging, overlap fragments,
  bounded orthogonal reconstruction, validity gating, and provenance metadata
  in `src/impression/modeling/csg.py`
- added shared CSG reference fixtures and stronger overlap evidence in:
  - `tests/csg_reference_fixtures.py`
  - `tests/test_surface_csg.py`
  - `tests/test_reference_images.py`
- kept the surfaced intersection box/sphere matrix item explicitly open as a
  kernel boundary rather than hiding it behind weak evidence

### Testing Architecture

- added shared reference-artifact tooling and CV-oriented test helpers in:
  - `tests/reference_images.py`
  - `tests/text_font_fixtures.py`
- added and expanded regression suites:
  - `tests/test_reference_images.py`
  - `tests/test_surface.py`
  - `tests/test_surface_csg.py`
  - `tests/test_surface_csg_docs.py`
  - `tests/test_surface_hinges.py`
  - `tests/test_surface_threading.py`
  - `tests/test_surface_replacements.py`
  - `tests/test_loft_api.py`
  - `tests/test_loft_correspondence.py`
  - `tests/test_loft_kernel.py`
  - `tests/test_loft_suite.py`
  - `tests/test_text.py`
- added targeted developer scripts for focused and full validation under
  `scripts/dev/`

### Public API Narrowing

- removed:
  - `src/impression/modeling/extrude.py`
  - `src/impression/modeling/morph.py`
- narrowed the exported modeling surface in
  `src/impression/modeling/__init__.py`
- retained explicit mesh tooling and mesh-analysis helpers without treating
  them as canonical surfaced truth

### Documentation And Project Structure

- introduced the `project/` planning structure as the durable home for:
  - architecture
  - specifications
  - test-specifications
  - planning
  - research
  - future features
  - release definitions
- added root and project-specific agent guidance under `agents/` and
  `project/agents/`
- retired or removed a large set of stale legacy planning/docs under `docs/`
  that no longer matched the active project structure

## Planned Architecture

- [Surface Mesh Decommission Architecture](../architecture/surface-mesh-decommission-architecture.md)
- [Testing Architecture](../architecture/testing-architecture.md)
- [Model Output Reference Verification](../architecture/model-output-reference-verification.md)
- [Computer Vision Verification Architecture](../architecture/computer-vision-verification-architecture.md)

## Planned UI Definitions

This release does not introduce a standalone app UI-definition branch.

The primary user-visible surface in `0.0.3a0` is instead:

- the modeling API and its examples
- the surfaced reference fixtures and diagnostic artifacts
- the updated documentation and tutorials that describe the supported lanes

In practice that means the visible behavior for this release is expressed
through:

- [docs/modeling/loft.md](../../docs/modeling/loft.md)
- [docs/modeling/csg.md](../../docs/modeling/csg.md)
- [docs/modeling/text.md](../../docs/modeling/text.md)
- [docs/tutorials/getting-started.md](../../docs/tutorials/getting-started.md)
- [docs/tutorials/serious-modeling.md](../../docs/tutorials/serious-modeling.md)

## Planned Specifications

This release is primarily represented by the completed surfaced, loft, and
testing/tooling branches now reflected in progression, especially:

- the surface core, surfaced modeling, and surfaced consumer branches
- the newer loft planner/executor and diagnostic branches
- the bounded surfaced CSG execution and reference-evidence branches
- the top-level testing architecture and CV tooling branches

The detailed execution order remains in
[progression.md](../planning/progression.md).

## Verification

At the close of this tranche:

- the progression document is fully checked
- the full test suite passes
- full coverage was refreshed under:
  - [coverage.xml](../coverage/coverage.xml)
  - [html](../coverage/html)

Latest full-suite result at release definition time:

- `430 passed`
- total coverage: `73.3%`

## Notes

- surfaced CSG is still intentionally bounded; the named
  `surfacebody/csg_intersection_box_sphere` promotion item remains an honest
  kernel boundary rather than a disguised harness pass
- reference images and reference STLs should still be read primarily as change
  detectors unless a fixture also carries stronger explicit truth checks
- future loft ideas such as control-station inference and spanwise
  consolidation were intentionally preserved in `project/future-features/`
  rather than mixed into the active delivered branch
- this is an alpha release: it meaningfully expands the surfaced contract, but
  it should still be read as a migration-forward build rather than a claim that
  every historical mesh-era workflow is already replaced
