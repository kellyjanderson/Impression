# Surface Spec 421: Loft Primitive Trim-Fragment Adapter (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-trim-fragment-execution-adapter.md`
Architecture ancestor: `../architecture/acd-loft-primitive-trim-fragment-execution-adapter.md`
Manifest source: `Loft Primitive Trim-Fragment Adapter`
Split provenance: `../specifications/surface-406-single-shell-loft-primitive-csg-execution-v1_0.md`
Canonical status: Canonical
Prerequisites:
- `../specifications/surface-420-loft-primitive-exact-reuse-execution-v1_0.md` - exact reuse/refusal route must exist before intersecting adapter work changes cut-case behavior.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one loft patch trim-fragment adapter and classification route.

## Purpose

Adapt loft side and cap patches into the existing surface CSG trim/fragment pipeline for intersecting loft/primitive cases.

## Scope

Owns:

- `LoftPrimitiveTrimAdapterRecord`.
- `LoftPrimitiveFragmentClassificationRecord`.
- Patch-local trim curve mapping and survive/discard/cut-cap classification for loft side/cap fragments.

Does not own:

- Result shell assembly and validity; see Surface Spec 422.
- Exact reuse execution; see Surface Spec 420.
- Provenance/color ownership resolution.

## Split Coverage

- Parent spec: `../specifications/surface-406-single-shell-loft-primitive-csg-execution-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 420, 421, and 422.
- Parent responsibilities owned by this child:
  - loft patch trim adapter
  - primitive intersection request construction for loft patches
  - loft fragment classification records and refusal diagnostics
- Parent responsibilities still missing from children:
  - none.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - loft patch adapter, classification records, refusal diagnostics.
- Supporting modules/files:
  - `src/impression/modeling/loft.py` - source patch role metadata and closure/cap evidence.
  - `src/impression/modeling/surface.py` - patch boundary and trim-loop primitives.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - adapter helpers consumed by cut-shell assembly.
- Tests:
  - `tests/test_surface_csg.py` - adapter records and public API refusal/adapter evidence tests for intersecting loft/box cases.

## Chosen Defaults / Parameters

- Only executor-authored loft bodies with boundary graph and cap evidence enter the adapter.
- Unsupported patch/primitive intersections refuse with structured diagnostics.
- Cap trim-loop orientation and station seam ownership are preserved in records for later provenance/color work.
- No mesh fallback or tessellation-as-execution is allowed.

## Data Ownership

- Source of truth: CSG adapter records derived from loft patch metadata and surface intersection records.
- Read ownership: Surface Spec 422 shell assembly and later provenance/color specs may read adapter records.
- Write ownership: CSG writes adapter and fragment classification records.
- Derived/cache data: records can be recomputed from operands, route selection, and loft evidence.
- Privacy/logging constraints: diagnostics must not dump full geometry arrays.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 420 exact reuse/refusal route
  - surface intersection request normalization
  - CSG curve mapping
  - loft boundary and cap evidence
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Prerequisite Handling

- Already implemented prerequisites:
  - `../specifications/surface-420-loft-primitive-exact-reuse-execution-v1_0.md` - exact reuse/refusal route is implemented and progression-complete.
- Missing prerequisite architecture:
  - none.
- Missing prerequisite specifications:
  - none.
- Unimplemented prerequisite specifications:
  - none.
- Progression handling:
  - current item may proceed because the prerequisite implementation is complete.

## Application Integration

- App type: library-only
- User/caller surface: public boolean API consumers using intersecting loft/primitive operands
- Invocation route: selected loft route to trim-fragment adapter
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: adapter emits trim and fragment records or refuses before shell execution
- Integration validation: public API tests for intersecting loft/box adapter records and refusal diagnostics
- Incomplete status risk: helper-only implementation if records are not consumed by Surface Spec 422

App-type-specific proof:

- Library-only: public boolean API tests validate the adapter route and downstream consumer handoff.

## Reuse And Extraction Plan

- Existing code to reuse:
  - surface intersection request normalization - patch pair request shape.
  - CSG curve mapping helpers - patch-local curve representation.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - loft primitive trim adapter and fragment classification helpers.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftPrimitiveTrimAdapterRecord` - maps a loft patch and primitive operand to trim curves and patch-local ownership.
  - `LoftPrimitiveFragmentClassificationRecord` - records survive, discard, and cut-cap decisions for loft fragments.
- Functions/methods:
  - `adapt_loft_patch_for_primitive_csg(...) -> LoftPrimitiveTrimAdapterRecord` - builds patch-local trim records.
  - `classify_loft_primitive_fragments(...) -> tuple[LoftPrimitiveFragmentClassificationRecord, ...]` - emits fragment decisions.
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by loft patch count and primitive patch count.
- No global tessellation or mesh conversion is permitted.

## Error And State Behavior

- Unsupported patch/primitive pairs refuse before shell assembly.
- Missing loft boundary/cap evidence refuses before adapter work begins.
- Ambiguous or degenerate trim curves emit deterministic diagnostics.

## Test Strategy

- Unit tests:
  - adapter records for loft side and cap patches.
  - fragment classification for survive/discard/cut-cap decisions.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - public boolean API tests proving intersecting loft/box adapter evidence or structured refusal.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Intersecting loft/primitive operands emit adapter records through the public CSG route.
- Fragment classifications preserve loft side/cap role and station seam ownership.
- Unsupported adapter states refuse without mesh fallback.

## Rescore And Split Review

- Manifest score: 18.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Review for split; remains cohesive because trim mapping and fragment classification share one public adapter verification surface.
- Review update: post-promotion review confirmed Surface Spec 406 split coverage is complete through Surface Specs 420, 421, and 422; score 18 still requires split review but no child split.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Canonical status is explicit.
- [x] Prerequisites are linked, implemented, or marked not applicable.
- [x] Missing prerequisite architecture has an ACD link, or is marked not applicable.
- [x] Missing prerequisite behavior has a final spec link, or is marked not applicable.
- [x] Split coverage is complete, or marked not applicable.
- [x] Implementation owner/module is named.
- [x] Existing code reuse/extraction decision is explicit.
- [x] Existing library/module additions or new reusable module boundaries are named, or marked not applicable.
- [x] UI fields/elements are listed, or marked not applicable.
- [x] Chosen defaults are explicit.
- [x] Data source of truth and write owner are explicit.
- [x] GUI/concurrency route is explicit, or marked not applicable.
- [x] App type and application integration route are explicit.
- [x] Integrated route validation is named.
- [x] GUI/console/API-service/mixed/library-only proof matches the app type.
- [x] Performance bounds are explicit, or marked not applicable.
- [x] Privacy/logging constraints are explicit, or marked not applicable.
- [x] Test strategy does not depend on production data.
- [x] Acceptance criteria are testable.

## Review History

- Pass 1 - Template: promoted from ACD manifest with all readiness fields.
- Pass 2 - Prerequisites: linked Surface Spec 420 as prerequisite implementation.
- Pass 3 - Rescore: manifest score 18; split review required.
- Pass 4 - Split Review: trim mapping and fragment classification remain cohesive because neither has a public route alone.
- Pass 5 - Final: ready as a canonical implementation leaf after Surface Spec 420.
- Post-promotion review - 2026-07-16: rechecked IWU count, split coverage, prerequisites, and readiness blockers; remains score 18, 1 IWU, no split.
