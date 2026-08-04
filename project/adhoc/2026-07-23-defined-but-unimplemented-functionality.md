# Defined but Unimplemented Functionality

Date: 2026-07-23

Status: Initial evidence-backed baseline; exhaustive spec conformance audit
still required

This report distinguishes implementation gaps from stale project records. It
uses current source, tests, architecture, future-feature documents, and
planning records. It does not treat an unchecked box as sufficient proof of a
missing feature, and it does not treat an implemented helper as proof of an
integrated product feature.

## Status Language

- `Verified gap`: current source or tests explicitly identify the capability as
  unsupported or not yet implemented.
- `No implementation located`: a focused source/test search found no matching
  implementation; confirm during domain audit.
- `Partial or uncertain`: implementation artifacts exist, but route,
  conformance, or completion evidence is incomplete.
- `Stale-document conflict`: project text says future/proposed while current
  code or progression indicates implementation.

## Verified Gaps

### Higher-order seam continuity

Defined by:

- `higher-order-seam-continuity-architecture.md`
- surface seam continuity records and related surface specifications

Current evidence:

- `surface_continuity_support()` explicitly returns
  `not-yet-implemented` for `C1`, `G1`, `C2`, and `G2`.
- Current supported seam validation is effectively positional `C0`.

Status: `Verified gap`.

Required resolution:

- keep higher-order continuity architecture active or convert it to an active
  ACD if the canonical document currently reads as implemented truth;
- identify final leaf specs and solver ownership;
- do not archive continuity specs merely because the data records exist.

### Unsupported surface-intersection family pairs

Defined by:

- `exact-surface-intersection-kernel-architecture.md`
- `higher-order-surface-csg-solver-architecture.md`
- surface-family intersection and CSG specs

Current evidence:

- the default solver registry explicitly creates `unsupported` entries with
  “is not implemented in this registry” diagnostics for family pairs that lack
  exact, declared-tolerance, or adapter routes;
- support is a matrix, not a single “CSG implemented” boolean.

Status: `Verified gap`, with supported subsets already implemented.

Required resolution:

- generate the current solver matrix into the conformance ledger;
- map each unsupported row to an active spec, an explicit non-goal, or a future
  feature;
- ensure canonical architecture states supported and unsupported routes
  precisely.

### Sampled and implicit CSG completion

Defined by:

- `sampled-implicit-surface-csg-support-architecture.md`
- `sampled-implicit-csg-unsupported-row-implementation-architecture.md`
- related surface CSG specifications

Current evidence:

- `test_sampled_implicit_csg_unsupported_row_tracker_covers_153_in_progress_rows`
  asserts 153 rows;
- every row has `route_status == "in-progress"` and
  `support_state == "unsupported"`;
- no hidden mesh fallback is attempted.

Status: `Verified gap`.

Required resolution:

- keep the unsupported-row work active;
- split “safe refusal exists” from “operation is implemented” in architecture
  and spec status;
- use the 153-row matrix as a generated evidence artifact rather than repeating
  the same status across many narrative documents.

### Global surface fillet, chamfer, and auto-round operations

Defined by:

- the active planning roadmap’s chamfer/fillet selection system and
  `round_sharp_edges(radius)` concept

Current evidence:

- focused searches find 2D offset chamfer behavior and loft end-cap modes;
- no public general surface-body fillet/chamfer selection system or
  `round_sharp_edges` implementation was located.

Status: `No implementation located`.

Boundary:

- existing planar offset and loft cap features must not be mislabeled as the
  missing general 3D operation.

### Simulation project family

Defined by:

- `impression-time`
- `impression-physics`
- `impression-interactive`
- `impression-sim`
- shared `SceneState`, timeline, keyframe, replay, collider, and solver concepts

Current evidence:

- the project roadmap lists all four projects and milestones as open;
- no `SceneState`, keyframe/timeline runtime, physics/collider layer, or
  integrated simulation runtime was located in `src/` or `tests/`.

Status: `No implementation located`.

Required resolution:

- keep as future product planning, not current Impression architecture;
- avoid mixing these external project concepts into the canonical modeling
  kernel architecture until a release adopts them.

### STEP/IGES export

Defined by:

- the planning roadmap’s future CAD exchange support

Current evidence:

- current public documentation centers on STL and `.impress`;
- no STEP/IGES writer or export route was located.

Status: `No implementation located`.

### Textured-plane image import and high-resolution image export workflow

Defined by:

- the planning roadmap’s image rendering section

Current evidence:

- no textured-plane/image-plane modeling primitive was located;
- preview screenshots and reference images exist, but they are not the defined
  image-import product feature.

Status: `No implementation located`.

### Reusable shape library

Defined by:

- the planning roadmap’s proposed `load_shape("gear_m12")` library

Current evidence:

- no `load_shape` API or curated reusable-shape package was located.

Status: `No implementation located`.

### Config-driven model parameter sweeps

Defined by:

- the planning roadmap’s YAML/JSON parameter overrides and CLI variant-sweep
  behavior

Current evidence:

- unit configuration exists;
- no general YAML/JSON model-parameter override and variant-sweep route was
  located.

Status: `No implementation located`.

## Future Loft Functionality Without Located Implementation

The following documents explicitly call themselves future features:

- `spanwise-loft-consolidation-architecture.md`
- `spanwise-loft-inline-enhancement-architecture.md`
- `spanwise-loft-postprocessing-optimization-architecture.md`
- `spanwise-loft-repair-tool-architecture.md`

A focused source/test search found no spanwise consolidation, postprocessing
optimizer, or repair-tool implementation.

Status: `No implementation located`.

Recommended treatment:

- preserve these as future-feature definitions;
- consolidate the four documents into one future-feature document with three
  clearly labeled implementation strategies;
- do not place them in canonical current architecture;
- do not confuse existing reversed-winding repair fixtures with the proposed
  general loft repair tool.

## Partial or Uncertain Areas Requiring Conformance Review

### Loft CSG ACD family

All loft CSG ACDs remain `Proposed` or `Manifesting`, but current
`src/impression/modeling/csg.py` contains many corresponding records and
execution helpers, including cut-loop, cap, fragment topology, seam, shell,
adjacency, validity, persistence, provenance, color, and no-hidden-mesh
structures.

Status: `Partial or uncertain`, not automatically unimplemented.

Audit requirement:

- verify public execution route, not only record/helper existence;
- verify focused tests and reference evidence;
- verify ACD closure checklist;
- merge and archive only the ACDs that are conformant.

### Reference Review hybrid stabilization

The stabilization plan still has 15 unchecked items, but the repository
contains substantial async, lifecycle, preview-payload, Qt/QML, notes,
promotion, and UI implementation with dedicated tests.

Status: `Partial or uncertain`.

Audit requirement:

- distinguish helper implementation, QML/Qt wiring, launch route, real-render
  smoke, packaging, and user accessibility;
- reconcile the plan rather than assuming either “done” or “not done.”

### Full surface-body family plans

The surface-body family planning folder contains 188 unchecked tasks across
three plans, while current source contains B-spline, NURBS, sweep, subdivision,
implicit, sampled, displacement, persistence, tessellation, and CSG structures.

Status: `Stale plan or partial implementation; item-by-item verification
required`.

Audit requirement:

- generate capability matrices from current code;
- replace broad checklist claims with code/test/route evidence;
- archive obsolete implementation plans once the verified gap list is
  transferred to active leaves or code-improvement work.

### Mesh boundary and cleanup plans

The mesh extraction audit has 14 unchecked and 4 checked items; the mesh
reference/cruft checklist has 42 unchecked items. Current code intentionally
retains explicit mesh compatibility modules and standalone mesh tools.

Status: `Partial or uncertain`.

Audit requirement:

- apply the six-lane mesh/surface classification from the cleanup plan;
- do not equate retained compatibility or tessellation code with failed
  surface-body migration;
- remove only hidden modeled-mesh authority and obsolete duplicated artifacts.

## Stale-Document Conflicts

### Control-station inference is still filed as a future feature

`project/future-features/control-station-inference-architecture.md` says the
feature is not in the active architecture/spec tree. The repository now has:

- active inference architecture and specifications;
- `src/impression/modeling/control_station_inference.py`;
- related control-station, diagnostic, and reporting modules;
- a completed inference progression.

Status: `Stale-document conflict`.

Recommended treatment:

- archive the old future-feature document after verifying that the active
  inference architecture covers all unique responsibilities;
- preserve any uncovered concept as an active ACD or future-feature subsection;
- do not keep two architecture narratives for the same feature.

### Trajectory-guided loft overlaps active shared-trajectory work

The future-feature trajectory document says the idea is outside active
architecture, while the repository contains shared trajectory, guidance,
curve-intent, and progression implementation and specs.

Status: `Stale-document conflict`, with possible remaining future scope.

Recommended treatment:

- compare the future document responsibility-by-responsibility against active
  Feature 07 architecture/specs and current code;
- archive covered responsibilities;
- retain only genuinely unimplemented user-facing trajectory authoring or
  inference behavior.

## Spec Status Anomalies

The active spec directory contains:

- 51 files explicitly marked `Proposed`;
- 10 final-leaf files from a recent review family;
- 9 explicit split/superseded parents;
- 1 retired mesh-executor spec;
- hundreds of older specs without consistent lifecycle metadata.

Meanwhile, the main progression marks 594 entries complete.

This is evidence of lifecycle drift, not evidence that 51 features are all
missing. The exhaustive audit must classify every spec using code, tests,
route, docs, and surface-transition evidence.

## Initial Priority Order

1. Generate current surface-family and CSG support matrices.
2. Reconcile the 12 ACDs against current loft/reference-review code.
3. Resolve future-feature conflicts for control-station and trajectory work.
4. Turn verified gaps into active leaf specs or clearly scoped future features.
5. Verify proposed specs domain by domain.
6. Archive superseded parents only after 100% child coverage proof.

This report should become smaller as the cleanup proceeds. Confirmed
implementation moves items into canonical architecture; confirmed gaps move
into final active specs or future-feature definitions; stale conflicts move to
the versioned archive.
