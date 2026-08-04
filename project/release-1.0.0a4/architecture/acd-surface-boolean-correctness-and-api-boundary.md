# Surface Boolean Correctness And API Boundary Architectural Change Document

Date: 2026-08-04
Status: Drafting Specs
Canonical architecture targets:

- `project/release-0.1.0a/architecture/csg-coincident-contact-architecture.md`
- `project/release-0.1.0a/architecture/lofted-body-csg-reference-architecture.md`
- `project/release-0.1.0a/architecture/surface-mesh-decommission-architecture.md`

Related:

- Release / plan / issue: `project/release-1.0.0a4/README.md`; GitHub #243, #247, #248
- Parent ACD, if any: none

## Change Intent

Make public surface booleans truthful: face-touch union must remove coincident
interior faces and assemble one shell; accepted difference must reconstruct
actual cut geometry; unchanged or partial results must not report success; and
the public modeling API must no longer accept mesh operands.

## Current Architecture

The `v1.0.0a3` union validator correctly rejects retained overlapping shells but
has no face-touch merger. Loft-pair union combines bodies without deleting the
coincident interior patch pair. The loft cut executor can validate an evidence
payload yet currently clones the base body for difference. Branching lofts are
refused before decomposition. Public boolean annotations still mix `Mesh`,
`MeshGroup`, and `SurfaceBody` despite the family gate requiring surface bodies.

## Target Architecture

- Coincident-contact classification yields deterministic patch-pair records
  with orientation, overlap extent, and tolerance evidence.
- Face-touch union removes exactly opposite-oriented coincident interior patch
  pairs, merges remaining patches, rebuilds seams/adjacency, and passes one-shell
  validity and operand-witness gates.
- Difference execution turns intersection curves into patch-local trims,
  classifies retained fragments, incorporates reversed cutter fragments/caps,
  rebuilds seams, and validates the resulting shell.
- Branching loft difference uses validated branch-graph decomposition, applies
  bounded sub-body cuts, and recomposes one surface body only when provenance
  and seam validity are complete.
- A public postcondition compares modeled surface evidence before and after a
  difference. Verified overlap plus unchanged geometry returns `invalid` with a
  no-cut diagnostic, never `succeeded`.
- Public `boolean_union`, `boolean_difference`, and `boolean_intersection`
  accept `SurfaceBody` operands and return `SurfaceBooleanResult`. Mesh utilities
  remain available only through explicitly non-modeling boundaries.

## Non-Goals

- Hidden mesh fallback, automatic mesh-to-surface conversion, or universal CSG
  support for every patch-family pair.

## Canonical Document Impact

- Architecture docs to update on closure:
  - coincident-contact architecture - accepted face-touch merge path.
  - lofted-body CSG architecture - cut reconstruction and branch decomposition.
  - mesh decommission architecture - public boolean API boundary.
- Specs or plans affected:
  - Fixes 02, 07, 08, and 09.

## Readiness Blocker Resolution

- Blocker being resolved: the issues required success but did not distinguish
  coincident union, cut construction, branch decomposition, no-op validation,
  and public compatibility ownership.
- Source artifact: GitHub #243, #247, #248 and current CSG code.
- Resolution provided by this ACD: four independently verifiable boundaries
  with explicit sequencing.
- Follow-on artifact: Fixes 02, 07, 08, and 09.
- Resolution status: resolved.

## Compatibility And Migration Strategy

The surface-only API change is intentional for the alpha line. Calls passing
mesh operands receive `TypeError` at the public boundary. Standalone mesh
analysis/export functions remain unchanged. The API migration lands after
surface union/difference correction so callers have a complete surfaced route.

## Application Integration Contract

- App type: library-only with console preview/export consumers.
- User/caller surface: public boolean functions; downstream preview and export.
- Invocation route: operand preparation -> route selection -> surface-native
  executor -> postcondition/validity gate -> `SurfaceBooleanResult` -> consumer.
- Wiring owner/module: `src/impression/modeling/csg.py` and public exports in
  `src/impression/modeling/__init__.py`.
- Observable result: one closed surfaced body, an explicitly empty result, or a
  structured unsupported/invalid result; never unchanged false success.
- Integration validation: minimal fixtures, audio-cube compositions, public API
  signature tests, and preview/export consumer smoke.

## Specification Sources

- Fix 02: coincident classification and face-touch union shell merger.
- Fix 07: surface-only public boolean signatures, runtime checks, docs/examples.
- Fix 08: real loft cut reconstruction and validated branching decomposition.
- Fix 09: geometry-change and operand-witness postcondition for difference.
- Reuse existing tolerance policy, CSG route records, trim-fragment records,
  seam/adjacency rebuild, validity gate, result envelope, and no-mesh proof.

## Specification Conformance

- Parent specs created or affected: none.
- Canonical child specs: none; draft Fixes 02, 07-09 await independent review.
- Paired test specs: draft Fixes 02, 07-09 test specifications.

## Conformance Checklist

- [ ] Implementation conforms to the target architecture.
- [ ] Draft leaves are independently reviewed and canonicalized.
- [ ] Paired test specs point to canonical leaves.
- [ ] Final progression preserves no-op gate and API migration prerequisites.
- [ ] Canonical CSG/API architecture is reconciled after implementation.

## Closure Criteria

Close after all release fixtures pass surfaced modeling, preview, and export
routes; mesh operands are absent from the public modeling API; and canonical
architecture records the conformed solver and compatibility boundaries.

## Closure Notes

- Canonical architecture updated: none yet.
- Archived or removed scaffolding: none.
- Follow-up ACDs: none.

## Change History

- 2026-08-04 - Initial draft. Reason: plan issues #243, #247, and #248 for `v1.0.0a4`.
