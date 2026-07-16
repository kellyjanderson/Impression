# Surface Spec 411: Loft CSG Fragment Provenance Map (v1.0)

Date: 2026-07-15
Status: Proposed
Primary ancestor: `../architecture/acd-loft-csg-result-provenance-and-color-propagation.md`
Architecture ancestor: `../architecture/acd-loft-csg-result-provenance-and-color-propagation.md`
Manifest source: `Loft CSG Fragment Provenance Map`
Split provenance: none
Canonical status: Canonical

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one loft CSG fragment provenance lineage contract.

## Purpose

Record source/output fragment lineage for loft CSG results.

## Scope

Owns:

- `LoftCSGSourceFragmentRecord`.
- `LoftCSGResultFragmentRecord`.
- Provenance mapping and missing provenance diagnostics.

Does not own:

- CSG execution; see Surface Spec 406.
- Color ownership; see Surface Spec 412.

## Split Coverage

- Parent spec: none
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - provenance mapper and result attachment.
- Supporting modules/files:
  - `src/impression/modeling/loft.py` - loft patch role metadata.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - loft-specific provenance records.
- Tests:
  - `tests/test_surface_csg.py` - public result provenance tests.

## Chosen Defaults / Parameters

- Every output fragment has a source ownership chain or explicit generated
  fragment reason.
- Source fragment ids are stable within one runtime result unless persistence
  later defines stronger identity.
- Missing provenance is diagnostic evidence.

## Data Ownership

- Source of truth: CSG result assembly.
- Read ownership: color resolver and reference fixtures read provenance maps.
- Write ownership: CSG writes result provenance.
- Derived/cache data: provenance maps derive from CSG fragments and loft patch
  role metadata.
- Privacy/logging constraints: not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 406 result geometry.
  - CSG result fragments.
  - loft patch role metadata.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Application Integration

- App type: library-only
- User/caller surface: public boolean API consumers inspecting loft CSG result
  metadata and reference fixtures generated from those results
- Invocation route: CSG result assembly after successful loft boolean operation
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: every loft CSG output fragment has source lineage or
  explicit generated-fragment reason
- Integration validation: public boolean API result tests that inspect
  provenance on generated loft CSG outputs
- Incomplete status risk: implemented in isolation if records exist but result
  assembly does not attach them to returned bodies

App-type-specific proof:

- Library-only: public boolean API result tests inspect returned metadata.

## Reuse And Extraction Plan

- Existing code to reuse:
  - general CSG provenance records.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - loft-specific provenance records.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftCSGSourceFragmentRecord` - source patch/role/fragment ownership.
  - `LoftCSGResultFragmentRecord` - output fragment lineage or generated reason.
- Functions/methods:
  - `map_loft_csg_fragment_provenance(...) -> list[LoftCSGResultFragmentRecord]`
  - `resolve_output_fragment_ownership(...) -> Diagnostic | Record`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by output fragment count.
- Provenance mapping must not rerun CSG.

## Error And State Behavior

- Missing source ownership produces deterministic diagnostics.
- Generated fragments must state their generated reason.
- Non-loft CSG results may omit loft-specific provenance.

## Test Strategy

- Unit tests:
  - retained loft fragments, cutter fragments, and generated fragments.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - public boolean API result provenance tests.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Loft CSG results carry source/output provenance maps.
- Missing provenance is explicit diagnostic evidence.
- Surface Spec 412 can consume provenance maps for color ownership.

## Rescore And Split Review

- Manifest score: 17.5.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Review for split; source mapping and missing-provenance diagnostics validate the same lineage contract.
- Review update: checked after promotion from ACD manifest; no child spec is required before implementation.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Canonical status is explicit.
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

- Pass 1 - Template: Promoted from ACD manifest.
- Pass 2 - Routing: Confirmed provenance attaches during result assembly.
- Pass 3 - Rescore: Manifest score 17.5; review-for-split band retained.
- Pass 4 - Split Review: Provenance diagnostics remain one lineage contract.
- Pass 5 - Final: Ready as a final leaf specification.
