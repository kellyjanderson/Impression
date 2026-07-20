# Surface Spec 422: Loft Primitive Cut Shell Assembly And Validity (v1.0)

Date: 2026-07-16
Status: Proposed
Primary ancestor: `../architecture/acd-loft-primitive-trim-fragment-execution-adapter.md`
Architecture ancestor: `../architecture/acd-loft-primitive-trim-fragment-execution-adapter.md`
Manifest source: `Loft Primitive Cut Shell Assembly And Validity`
Split provenance: `../specifications/surface-406-single-shell-loft-primitive-csg-execution-v1_0.md`
Canonical status: Canonical
Prerequisites:
- `../specifications/surface-421-loft-primitive-trim-fragment-adapter-v1_0.md` - trim-fragment adapter records must exist before result shell assembly can consume them.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one loft primitive cut-shell assembly and validity handoff route.

## Purpose

Assemble cut-producing loft/primitive Boolean results from classified loft fragments and generated primitive cap fragments.

## Scope

Owns:

- `LoftPrimitiveCutShellAssemblyRecord`.
- Result shell assembly from loft fragment classifications and generated cap fragments.
- Validity-gate handoff for cut-producing loft/primitive union, difference, and intersection.

Does not own:

- Trim-fragment adapter record construction; see Surface Spec 421.
- Exact reuse execution; see Surface Spec 420.
- Provenance/color ownership resolution or reference proof artifacts.

## Split Coverage

- Parent spec: `../specifications/surface-406-single-shell-loft-primitive-csg-execution-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 420, 421, and 422.
- Parent responsibilities owned by this child:
  - result shell assembly
  - generated cap seam rebuild inputs
  - validity gate handoff and public API result proof for cut-producing cases
- Parent responsibilities still missing from children:
  - none.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - cut-shell assembler, assembly record, public result route.
- Supporting modules/files:
  - `src/impression/modeling/surface.py` - shell assembly primitives and seam validation.
  - `src/impression/modeling/loft.py` - loft source patch role metadata.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - assembler consumed by public boolean API and later proof/provenance specs.
- Tests:
  - `tests/test_surface_csg.py` - public API cut-producing union, difference, and intersection tests.

## Chosen Defaults / Parameters

- Invalid result geometry refuses with structured diagnostics; no fallback is attempted.
- Generated primitive cap fragments must participate in seam rebuild evidence.
- The public API is the only completion route; helper-only shell assembly is not complete.

## Data Ownership

- Source of truth: CSG assembled result body and assembly diagnostics.
- Read ownership: Surface Spec 407 reference proof and Surface Specs 411-412 provenance/color may read assembly and result records.
- Write ownership: CSG writes assembled result geometry and diagnostics.
- Derived/cache data: assembly records can be recomputed from adapter records and operands.
- Privacy/logging constraints: diagnostics must avoid full geometry dumps.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 421 trim-fragment adapter records
  - surface shell assembly helpers
  - CSG validity gate
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Prerequisite Handling

- Already implemented prerequisites:
  - `../specifications/surface-421-loft-primitive-trim-fragment-adapter-v1_0.md` - trim-fragment adapter records are implemented and progression-complete.
- Missing prerequisite architecture:
  - `../architecture/acd-loft-primitive-cut-shell-geometric-kernel.md` - defines the cut boundary loop, generated cap, seam orientation, and public executor kernel needed before truthful cut-shell assembly can be implemented.
- Missing prerequisite specifications:
  - none.
- Unimplemented prerequisite specifications:
  - final specifications derived from `../architecture/acd-loft-primitive-cut-shell-geometric-kernel.md`.
- Progression handling:
  - leave this item unchecked until the geometric kernel ACD is promoted into final specs and those prerequisite specs are implemented.

## Application Integration

- App type: library-only
- User/caller surface: public boolean API consumers using cut-producing loft/primitive operands
- Invocation route: selected loft route to trim adapter to cut-shell assembler to result validity gate
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: public boolean API returns a valid new `SurfaceBody` for intersecting loft/primitive union, difference, and intersection
- Integration validation: public boolean API cut-producing operation tests and validity-gate assertions
- Incomplete status risk: incomplete if shell assembly exists but is not reachable from the public API

App-type-specific proof:

- Library-only: public boolean API tests validate the consuming route.

## Reuse And Extraction Plan

- Existing code to reuse:
  - surface shell assembly helpers - result shell construction.
  - CSG validity gate - accepted/invalid result classification.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - loft cut-shell assembler and assembly record.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftPrimitiveCutShellAssemblyRecord` - records result-shell assembly, generated caps, seam rebuild inputs, and diagnostics.
  - `LoftCSGResultGeometryRecord` - records shell count, patch count, classification, fragment count, and no-mesh proof.
- Functions/methods:
  - `assemble_loft_primitive_cut_shell(...) -> SurfaceBody | Diagnostic` - assembles cut-producing result geometry.
  - `execute_loft_primitive_trim_fragment_csg(...) -> SurfaceBooleanResult` - public route executor for cut-producing cases.
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by classified fragment and seam counts.
- No tessellation or mesh conversion is permitted before explicit reference export boundaries.

## Error And State Behavior

- Missing adapter records refuse before shell assembly.
- Invalid seam rebuild or open result boundaries produce deterministic diagnostics.
- The result validity gate is authoritative for accepted versus invalid result geometry.

## Test Strategy

- Unit tests:
  - generated cap seam rebuild inputs.
  - cut-shell assembly record payloads.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - public boolean API cut-producing union, difference, and intersection tests.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Intersecting loft/primitive union, difference, and intersection return valid surface-native `SurfaceBody` results through the public boolean API.
- Assembly records include generated caps, seam rebuild inputs, and no-hidden-mesh proof.
- Invalid cut-shell results refuse with structured diagnostics.

## Rescore And Split Review

- Manifest score: 18.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Review for split; remains cohesive because shell assembly and validity handoff share one public API completion route.
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
- Pass 2 - Prerequisites: linked Surface Spec 421 as prerequisite implementation.
- Pass 3 - Rescore: manifest score 18; split review required.
- Pass 4 - Split Review: shell assembly and validity handoff remain cohesive because public API completion requires both.
- Pass 5 - Final: ready as the final canonical child covering Surface Spec 406.
- Post-promotion review - 2026-07-16: rechecked IWU count, split coverage, prerequisites, and readiness blockers; remains score 18, 1 IWU, no split.
