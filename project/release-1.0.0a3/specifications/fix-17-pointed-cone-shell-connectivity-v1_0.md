# Fix 17: Pointed Cone Shell Connectivity Specification

Date: 2026-08-04
Status: Final
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `not applicable - primitive topology correction`
Source artifact: full a3 release qualification failure
Split provenance: `none`
Canonical status: `Canonical`
Review Score: 5.5
Prerequisites: none

## Purpose

Report a pointed cone as one connected surface shell when exactly one end radius
collapses to the apex. Preserve the existing disconnected status for a frustum
whose two cap patches do not yet carry explicit cap-to-side adjacency.

## Scope

Owns:

- The `SurfaceShell.connected` value produced by `make_surface_cone(...)`.
- Both upward and downward pointed-cone orientations.
- Regression coverage that distinguishes pointed cones from two-radius frustums.

Does not own:

- Cone tessellation, dimensions, transforms, or public primitive routing.
- New cap seams or a general surface-shell connectivity inference system.
- Cylinder connectivity behavior.

## Implementation Routing

- `src/impression/modeling/_surface_primitives.py` - cone shell construction.
- `tests/test_surface_kernel.py` - pointed-cone and frustum connectivity assertions.

## Chosen Behavior

- One zero-radius end and one positive-radius end: `connected=True`.
- Two positive-radius ends: `connected=False` until cap adjacency is explicitly modeled.
- Two zero-radius ends remain invalid input.

## Data Ownership And Route

- Source of truth: normalized bottom and top radii in `make_surface_cone(...)`.
- Write owner: the primitive constructor sets the immutable `SurfaceShell.connected` field.
- Invocation route: cone parameters -> surface patches/caps -> shell construction -> `SurfaceBody`.
- Observable result: downstream topology consumers receive the correct connectivity signal.
- Derived/cache data, external services, database state, UI state, privacy, and security behavior: not applicable.

## Error And Performance Behavior

- Existing validation for non-positive total radius, non-positive height, and low resolution remains unchanged.
- The correction is a constant-time boolean decision and adds no tessellation work.

## Test Strategy

- Assert the bottom-apex and top-apex orientations are connected.
- Assert a two-radius frustum remains disconnected.
- Run the full release suite so downstream surface, drafting, export, and reference routes consume the corrected signal.

## Acceptance Criteria

- `make_surface_cone(bottom_diameter=0, top_diameter=1)` reports one connected shell.
- `make_surface_cone(bottom_diameter=1, top_diameter=0)` reports one connected shell.
- `make_surface_cone(bottom_diameter=1, top_diameter=1)` remains explicitly disconnected.
- The complete release suite passes with no connectivity regression.

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../planning/spec-review-ledger-20260804-pointed-cone.md` | 1 | Fix 17 | none | reached |

## Readiness Checklist

- [x] Ancestor, source, canonical status, routing, ownership, behavior, and proof are explicit.
- [x] No unresolved deferral, missing prerequisite, or readiness blocker remains.
- [x] The leaf describes one primitive connectivity decision and does not hide unrelated work.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; independently scored from the current specification.
- Adversarial rescore basis: checked for hidden tessellation, adjacency, public API, performance, persistence, and UI work; only one constructor output signal changes.
- Functions/methods: 1 x 2 = 2
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 0 x 1 = 0
- Returns/outputs/signals: 1 x 1 = 1
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 0 x 2 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 5.5
- Split decision: cohesive leaf below the mandatory split-review band.
