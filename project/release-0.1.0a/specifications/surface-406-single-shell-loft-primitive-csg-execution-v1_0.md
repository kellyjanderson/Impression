# Surface Spec 406: Single-Shell Loft Primitive CSG Execution (v1.0)

Date: 2026-07-15
Status: Proposed
Primary ancestor: `../architecture/acd-single-shell-loft-csg-operation-route.md`
Architecture ancestor: `../architecture/acd-single-shell-loft-csg-operation-route.md`
Manifest source: `Single-Shell Loft Primitive CSG Execution`
Split provenance: none
Canonical status: Canonical

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one single-shell loft primitive CSG execution route.

## Purpose

Execute the first surface-native CSG route for an eligible single-shell loft and
analytic primitive operand pair.

## Scope

Owns:

- `LoftPatchFragmentParticipationRecord`.
- `LoftCSGResultGeometryRecord`.
- Loft patch classification adapter and surface-native CSG execution.

Does not own:

- Route selection; see Surface Spec 405.
- Reference proof artifacts; see Surface Spec 407.
- Provenance/color resolution; see Surface Specs 411 and 412.

## Split Coverage

- Parent spec: none
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - executor and loft patch adapter.
- Supporting modules/files:
  - `src/impression/modeling/loft.py` - loft patch role metadata.
  - `src/impression/modeling/surface.py` - shell assembly and validation.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - loft patch classification adapter.
- Tests:
  - `tests/test_surface_csg.py` - public boolean API operation tests.

## Chosen Defaults / Parameters

- No hidden mesh fallback.
- Unsupported pairings refuse before execution.
- The route returns `SurfaceBody`, never tessellated mesh data.

## Data Ownership

- Source of truth: CSG execution result geometry.
- Read ownership: downstream provenance, reference proof, and section evidence
  read returned result records.
- Write ownership: CSG writes result geometry and execution diagnostics.
- Derived/cache data: fragment participation records derive from route records
  and trim/fragment reconstruction.
- Privacy/logging constraints: not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 405 route selection.
  - surface CSG trim fragment reconstruction.
  - shell assembly validator.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Application Integration

- App type: library-only
- User/caller surface: public boolean API consumers using supported
  loft/primitive pairings
- Invocation route: selected loft CSG route to surface-native CSG execution
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: public boolean API returns a valid `SurfaceBody` for the
  supported loft/primitive pairing
- Integration validation: public boolean API tests covering union, difference,
  and intersection for the first supported loft/primitive case
- Incomplete status risk: implemented in isolation if executor bypasses the
  public boolean API or returns tessellated mesh data

App-type-specific proof:

- Library-only: public boolean API operation tests prove integration.

## Reuse And Extraction Plan

- Existing code to reuse:
  - surface CSG trim/fragment reconstruction.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - loft patch classification adapter.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftPatchFragmentParticipationRecord` - participating loft side/cap fragments.
  - `LoftCSGResultGeometryRecord` - shell count, validity summary, fragment
    count, and no-hidden-mesh proof.
- Functions/methods:
  - `execute_single_shell_loft_primitive_csg(...) -> SurfaceBody`
  - `classify_loft_patch_fragments(...) -> LoftPatchFragmentParticipationRecord`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by loft patch, primitive patch, and fragment counts.
- No tessellation before the explicit export/reference boundary.

## Error And State Behavior

- Invalid execution plans refuse before solver execution.
- Invalid result geometry produces explicit diagnostics.
- Mesh fallback is always a failure for this route.

## Test Strategy

- Unit tests:
  - loft patch fragment classification and result geometry records.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - public boolean API union, difference, and intersection tests.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- The supported loft/primitive pairing returns valid `SurfaceBody` geometry.
- Public boolean API tests cover union, difference, and intersection.
- No-hidden-mesh-fallback assertions pass.

## Rescore And Split Review

- Manifest score: 14.5.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Small; execution stays separate from route selection, proof, and provenance.
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
- Pass 2 - Routing: Confirmed executor is separate from route selector and proof.
- Pass 3 - Rescore: Manifest score 14.5; no split required.
- Pass 4 - Blocker Review: Predecessor Surface Spec 405 is explicit.
- Pass 5 - Final: Ready as a final leaf specification.
