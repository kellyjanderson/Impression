# Surface CSG Pairwise Composition And Result Re-entry Architectural Change Document

Date: 2026-08-09
Status: Closed
Canonical architecture targets:

- `project/release-0.1.0a/architecture/surfacebody-csg-architecture.md`
- `project/release-0.1.0a/architecture/surfacebody-seam-adjacency-architecture.md`
- `project/release-0.1.0a/architecture/surface-csg-executable-completion-architecture.md`

Related:

- Release / plan / issue: `project/release-0.1.0a/planning/progression.md`
- Release / plan / issue: `https://github.com/kellyjanderson/Impression/issues/267`
- Release / plan / issue: `https://github.com/kellyjanderson/Impression/issues/268`
- Release / plan / issue: `project/release-0.1.0a/adhoc/2026-07-09-csg-reference-03-multi-operand-boolean-composition.md`
- Parent ACD, if any: `project/release-0.1.0a/architecture/acd-single-shell-loft-csg-operation-route.md`

## Change Intent

Define how the public surface CSG API composes polygon-loft operands through a
canonical declarative field route, and define the representation-specific
topology and provenance invariants that make every successful `SurfaceBody`
result eligible for a subsequent surface CSG call.

## Current Architecture

The public API accepts multiple union operands and multiple difference cutters,
but the general loft-pair executor accepts exactly two operands. The existing
polygon-loft difference route already promotes ruled loft cells into one
declarative `ImplicitSurfacePatch`. A subsequent call incorrectly classifies
that one-patch result as an explicit loft shell and rejects it for missing seams
that do not exist in the implicit representation.

The missing policy affects two caller-visible cases:

- union of one enclosure shell with several attached polygon-loft features;
- repeated pairwise differences in which each successful result becomes the
  base for the next copied snap-groove cutter.

## Target Architecture

The polygon-loft field executor is N-ary for union requests with at least three
operands and for difference requests with one or more cutters. Established
two-body union dispatch remains authoritative. At the declarative geometry boundary:

1. adapt each original closed polygon-loft body to a bounded
   `polygon_loft_body` field node;
2. for union, require one connected operand-contact graph and sort operands by
   stable body identity before composing one hard field-union node;
3. for difference, preserve the authored base/cutter order and compose one hard
   field-difference node;
4. emit one bounded `ImplicitSurfacePatch` in one connected closed shell;
5. retain canonical execution-order operand ids and route metadata on the body,
   while `SurfaceBooleanResult.operands` retains the caller's prepared request;
6. return structured refusal with no partial body when any operand cannot be
   adapted, a union contact graph is disconnected, or final validity fails.

No mesh execution or tessellation participates in modeling composition.
Tessellation remains a preview/export boundary.

Successful-result re-entry is representation-specific:

- explicit patch-shell results require canonical seams and seam-derived
  adjacency before re-entry;
- a one-patch polygon-loft field result re-enters through its validated
  declarative field root and `polygon_loft_field_csg` provenance;
- an implicit result must not be rejected for lacking explicit patch seams and
  must not receive invented seam records;
- Boolean provenance, operand ids, bounds, and no-hidden-mesh evidence remain
  mandatory for both representations;
- each new difference composition supplies a declarative field-graph geometry
  change witness to the public difference success gate.

Batch difference is permitted when all cutters adapt to this same bounded
field route, but six explicit sequential public calls remain the required
re-entry acceptance proof. No new public DTO is introduced.

## Non-Goals

- General N-ary intersection execution.
- General batch-difference lowering outside the polygon-loft field route.
- Branching-loft decomposition or recomposition.
- Mesh fallback, export tessellation, or changes to authored cutter geometry.
- Fabricated seams or adjacency for a one-patch implicit field result.
- New public CSG result or diagnostic DTOs.

## Canonical Document Impact

- Architecture docs to update on closure:
  - `project/release-0.1.0a/architecture/surfacebody-csg-architecture.md` - document deterministic N-ary polygon-loft field composition.
  - `project/release-0.1.0a/architecture/surfacebody-seam-adjacency-architecture.md` - document representation-specific successful-result re-entry.
  - `project/release-0.1.0a/architecture/surface-csg-executable-completion-architecture.md` - identify declarative field composition as the polygon-loft N-ary route.
- Specs or plans affected:
  - `project/release-0.1.0a/specifications/surface-432-attached-polygon-loft-surface-union-completion-v1_0.md` - owns canonical polygon-loft field union composition for the attached-feature fixture.
  - `project/release-0.1.0a/specifications/surface-433-repeated-snap-groove-surface-difference-provenance-preservation-v1_0.md` - owns successful-result re-entry for repeated differences.
  - `project/release-0.1.0a/planning/progression.md` - sequences both conformance leaves.

## Readiness Blocker Resolution

- Blocker being resolved:
  - Surface Specs 432 and 433 lacked architecture for multi-operand composition and successful-result re-entry.
- Source artifact:
  - `project/release-0.1.0a/specifications/surface-432-attached-polygon-loft-surface-union-completion-v1_0.md`
  - `project/release-0.1.0a/specifications/surface-433-repeated-snap-groove-surface-difference-provenance-preservation-v1_0.md`
- Resolution provided by this ACD:
  - deterministic field-composition semantics, structured route refusal, and an explicit representation-specific re-entry invariant.
- Follow-on artifact:
  - Surface Specs 432 and 433 and their paired test specifications.
- Resolution status:
  - resolved.

## Compatibility And Migration Strategy

- Existing two-operand public CSG behavior remains unchanged for non-polygon-loft routes.
- Equivalent polygon-loft union operand sets normalize to the same identity order;
  canonical-order provenance is attached to the body, and the original prepared
  request remains attached to the public result.
- Unsupported field adaptation or disconnected union contact exposes no partial body.
- Existing batch-difference refusal remains compatible for non-adaptable families.
- Existing no-hidden-mesh guards remain mandatory at the route and public result boundary.

## Application Integration Contract

- App type: library-only
- User/caller surface: consumers of `boolean_union` and `boolean_difference`
- Invocation route: public Boolean call to prepared operands, polygon-loft field adaptation/composition, result finalization, and returned `SurfaceBooleanResult`
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: one fused union body, a reusable sequential-difference body, or a structured unsupported result with no partial body
- Integration validation: public API regression fixtures for attached snap tabs, microphone rails, and six sequential snap-groove cutters

## Specification Sources

### Attached polygon-loft union composition

- Functions/methods: dispatch connected polygon-loft union operands to one canonically ordered field composition.
- Data structures/models: reuse `SurfaceBooleanOperands` and `SurfaceBooleanResult`.
- Dependencies/services: polygon-loft field adaptation, hard implicit union, and final validity gate.
- Returns/outputs/signals: final fused body or structured refusal.
- Owner/module: `src/impression/modeling/csg.py`.
- Test fixture owner: `tests/csg_reference_fixtures.py`.
- Final spec: `project/release-0.1.0a/specifications/surface-432-attached-polygon-loft-surface-union-completion-v1_0.md`.

### Successful-result topology and provenance re-entry

- Functions/methods: recognize accepted polygon-loft field roots and produce declarative field-change evidence on re-entry.
- Data structures/models: reuse `ImplicitFieldNode`, `ImplicitSurfacePatch`, `SurfaceBody`, and `SurfaceBooleanResult`.
- Dependencies/services: field provenance recognition, implicit composition, result validity, and difference success evidence.
- Returns/outputs/signals: one closed reusable body or a structured validity refusal.
- Owner/module: `src/impression/modeling/csg.py`.
- Test fixture owner: `tests/csg_reference_fixtures.py`.
- Final spec: `project/release-0.1.0a/specifications/surface-433-repeated-snap-groove-surface-difference-provenance-preservation-v1_0.md`.

Known readiness blockers: none. This ACD defines the target route, ownership,
failure behavior, data boundaries, and final spec split.

## Specification Conformance

- Parent specs created or affected:
  - none.
- Canonical child specs:
  - `project/release-0.1.0a/specifications/surface-432-attached-polygon-loft-surface-union-completion-v1_0.md` - primary ancestor is this ACD; split provenance is none.
  - `project/release-0.1.0a/specifications/surface-433-repeated-snap-groove-surface-difference-provenance-preservation-v1_0.md` - primary ancestor is this ACD; split provenance is none.
- Paired test specs:
  - `project/release-0.1.0a/test-specifications/surface-432-attached-polygon-loft-surface-union-completion-v1_0.md` - verifies Surface Spec 432.
  - `project/release-0.1.0a/test-specifications/surface-433-repeated-snap-groove-surface-difference-provenance-preservation-v1_0.md` - verifies Surface Spec 433.

## Conformance Checklist

- [x] Implementation conforms to the target architecture.
- [x] Parent specs are 100% represented by canonical child specs, or no parent split applies.
- [x] Superseded parent specs are archived, or none exist.
- [x] Canonical child specs point to architecture or this active ACD as primary ancestor.
- [x] Paired test specs point to canonical child specs.
- [x] Progression and indexes point to canonical child specs.
- [x] Completed process scaffolding is absent from active canonical architecture docs.
- [x] Canonical architecture docs describe the conformed architecture.

## Closure Criteria

- Both paired feature/test specs are implemented and pass their public-route acceptance checks.
- Canonical architecture documents incorporate field composition and representation-specific result re-entry contracts.
- Active specs and progression no longer require this ACD as transition authority.

## Closure Notes

- Canonical architecture updated:
  - `surfacebody-csg-architecture.md` now defines connected canonical N-ary polygon-loft field union.
  - `surfacebody-seam-adjacency-architecture.md` now distinguishes explicit-shell seam re-entry from one-patch implicit field re-entry.
  - `surface-csg-executable-completion-architecture.md` now records polygon-loft field composition and sequential re-entry.
- Archived or removed scaffolding:
  - none; this closed ACD remains durable decision history for the implementation-time architecture correction.
- Follow-up ACDs:
  - none.

## Change History

- 2026-08-09 - Initial draft. Reason: independent review of Surface Specs 432 and 433 found missing pairwise composition and result re-entry architecture.
- 2026-08-09 - Corrected the target after implementation discovery: polygon-loft results are declarative implicit patches, so field provenance rather than fabricated explicit seams owns re-entry; canonical reconciliation remains pending.
