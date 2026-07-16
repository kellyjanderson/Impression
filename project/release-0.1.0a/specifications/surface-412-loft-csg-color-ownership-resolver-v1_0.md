# Surface Spec 412: Loft CSG Color Ownership Resolver (v1.0)

Date: 2026-07-15
Status: Proposed
Primary ancestor: `../architecture/acd-loft-csg-result-provenance-and-color-propagation.md`
Architecture ancestor: `../architecture/acd-loft-csg-result-provenance-and-color-propagation.md`
Manifest source: `Loft CSG Color Ownership Resolver`
Split provenance: none
Canonical status: Canonical

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one loft CSG color ownership resolver.

## Purpose

Preserve authored colors and deterministic generated-surface colors through
loft CSG results.

## Scope

Owns:

- `LoftCSGColorOwnershipRecord`.
- `LoftCSGGeneratedSurfaceStylePolicy`.
- Color/material ownership resolver and fallback diagnostics.

Does not own:

- Fragment provenance map; see Surface Spec 411.
- Review UI color controls.

## Split Coverage

- Parent spec: none
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - color ownership resolver.
- Supporting modules/files:
  - `src/impression/modeling/loft.py` - source patch color metadata.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - loft CSG color resolver.
- Tests:
  - `tests/test_surface_csg.py` - authored and generated color metadata tests.

## Chosen Defaults / Parameters

- Fallback colors are explicit metadata, never silent authored-color claims.
- Retained loft and cutter fragments preserve authored color metadata.
- Generated cut/cap surfaces receive deterministic fallback style metadata.

## Data Ownership

- Source of truth: CSG result provenance.
- Read ownership: reference fixtures and render/review paths read output color
  lineage.
- Write ownership: CSG result assembly writes output color ownership.
- Derived/cache data: color ownership derives from provenance and source patch
  metadata.
- Privacy/logging constraints: not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 411 provenance map.
  - source patch color metadata.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Application Integration

- App type: library-only
- User/caller surface: public boolean API consumers and reference fixtures that
  inspect authored or generated output colors
- Invocation route: provenance map to color ownership resolver during CSG result
  assembly
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: retained loft and cutter fragments preserve authored
  colors; generated surfaces receive fallback style metadata
- Integration validation: public boolean API result tests covering authored
  loft colors, cutter colors, and generated-surface fallback metadata
- Incomplete status risk: implemented in isolation if fallback style metadata
  is calculated but not attached to result fragments

App-type-specific proof:

- Library-only: public boolean API result tests inspect output metadata.

## Reuse And Extraction Plan

- Existing code to reuse:
  - color metadata helpers.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - loft CSG color resolver.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftCSGColorOwnershipRecord` - output color lineage and source ownership.
  - `LoftCSGGeneratedSurfaceStylePolicy` - fallback style rules.
- Functions/methods:
  - `resolve_loft_csg_color_ownership(...) -> list[LoftCSGColorOwnershipRecord]`
  - `resolve_generated_surface_style(...) -> LoftCSGGeneratedSurfaceStylePolicy`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by output fragment count.
- Resolver must not rerun CSG or geometry classification.

## Error And State Behavior

- Missing authored color falls back with explicit metadata.
- Missing provenance refuses color ownership resolution.
- Non-loft CSG metadata remains unchanged.

## Test Strategy

- Unit tests:
  - authored loft color, cutter color, generated fallback color.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - public boolean API color metadata assertions.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Authored colors are preserved for retained source fragments.
- Generated surfaces carry explicit fallback style metadata.
- Reference fixtures can inspect color ownership evidence.

## Rescore And Split Review

- Manifest score: 16.5.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Review for split; authored color and fallback style are branches of one resolver.
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
- Pass 2 - Routing: Confirmed color resolver depends on provenance map.
- Pass 3 - Rescore: Manifest score 16.5; review-for-split band retained.
- Pass 4 - Split Review: Authored and fallback style stay in one resolver.
- Pass 5 - Final: Ready as a final leaf specification.
