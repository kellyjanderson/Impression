# Surface Spec 428: Loft Primitive Runtime Validity And Persistence Gate (v1.0)

Date: 2026-07-16
Status: Superseded
Primary ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Manifest source: `Loft Primitive Runtime Validity And Persistence Gate`
Split provenance: `../specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
Canonical status: Superseded parent
Prerequisites:
- `../specifications/surface-427-loft-primitive-seam-and-shell-assembly-v1_0.md` - validity consumes assembled candidate shells.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 3 IWU.
Basis: parent branch split into runtime validity, persistence/tessellation-boundary readiness, and no-hidden-mesh proof.

## Manifest Field Carryover

- Discovery purpose: accept or reject assembled cut-shell candidates before public return, persistence, tessellation, or reference generation.
- Manifest responsibilities by category:
  - Functions/methods: cut-shell runtime validity checker, persistence/tessellation-boundary proof collector, no-hidden-mesh gate.
  - Data structures/models: `LoftPrimitiveCutShellValidityRecord`, persistence evidence record, no-hidden-mesh proof record.
  - Dependencies/services: candidate `SurfaceBody`, runtime validity gate, `.impress` persistence and tessellation boundary helpers.
  - Returns/outputs/signals: accepted result body signal, invalid/unsupported diagnostics, persistence and no-mesh proof payload.
  - UI surfaces/components and fields: not applicable.
  - Reusable code plan: reuse runtime validity, persistence, and tessellation evidence helpers; add validity payload builder.
  - Database/async/write/security/cross-screen behavior: none.
  - Performance-sensitive behavior: bounded by result shell, seam, trim, and patch counts.
- Manifest open questions / nuance discovered:
  - Final spec should choose a focused persistence proof that avoids broad filesystem churn; this spec requires in-memory proof plus focused round-trip smoke only where needed by tests.
- Manifest score at promotion: 19.5 on 2026-07-16.
- Manifest readiness blockers and resolution: none; resolved.
- Manifest split decision: review for split; cohesive result-acceptance gate.
- Manifest cleanup state: ready after spec promotion.

## Purpose

Ensure candidate loft/primitive cut-shell bodies are valid, persistent, and mesh-free before they can be returned or consumed by reference workflows.

## Scope

Owns:

- `LoftPrimitiveCutShellValidityRecord`.
- Persistence readiness and tessellation-boundary proof.
- No-hidden-mesh result gate.

Does not own:

- Candidate shell assembly.
- Public `SurfaceBooleanResult` wiring.
- Reference artifact writing.

## Split Coverage

- Parent spec: `../specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 428a-428c.
- Parent responsibilities owned by this child:
  - none; superseded parent only.
- Parent responsibilities still missing from children:
  - none.
  - Runtime validity is covered by `surface-428a-loft-primitive-runtime-validity-checker-v1_0.md`.
  - Persistence and tessellation-boundary readiness are covered by `surface-428b-loft-primitive-persistence-and-tessellation-readiness-v1_0.md`.
  - No-hidden-mesh acceptance proof is covered by `surface-428c-loft-primitive-no-hidden-mesh-acceptance-proof-v1_0.md`.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Manifest promotion and sizing | this spec and paired test spec | Promoted manifest candidate and set IWU to 1 | score 19.5 | 1 IWU | no split | none | not applicable | fixed-point ledger readback |
| 2 | 2026-07-16 | Critical sizing review | this parent and child specs 428a-428c | Marked parent superseded and moved all executable work to children | 1 IWU | 3 IWU branch rollup | split | 428a, 428b, 428c | complete | child specs |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - validity record and result acceptance gate.
- Supporting modules/files:
  - `src/impression/modeling/surface.py` - runtime validity helpers.
  - `src/impression/modeling/tessellation.py` - boundary proof only, not execution fallback.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - validity payload consumed by public executor and reference handoff.
- Tests:
  - `tests/test_surface_csg.py` - accepted result, invalid refusal, persistence readiness, no-hidden-mesh evidence.

## Chosen Defaults / Parameters

- Tessellation is proof/export only; mesh-backed fragments refuse.
- Persistence proof must be focused and deterministic.
- Invalid candidate bodies cannot reach public result success.

## Data Ownership

- Source of truth: CSG validity proof and accepted candidate body.
- Read ownership: public executor and reference handoff read validity payload.
- Write ownership: `src/impression/modeling/csg.py`.
- Derived/cache data: validity proof recomputable from candidate shell records.
- Privacy/logging constraints: diagnostics avoid full geometry dumps.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 427 candidate shells.
  - runtime validity helpers.
  - persistence and tessellation-boundary helpers.
- Database dependencies: not applicable.
- GUI route, if applicable: not applicable.
- Background/concurrency route, if applicable: not applicable.

## Prerequisite Handling

- Already implemented prerequisites: none.
- Missing prerequisite architecture: none.
- Missing prerequisite specifications: none.
- Unimplemented prerequisite specifications:
  - `../specifications/surface-427-loft-primitive-seam-and-shell-assembly-v1_0.md` - must be implemented first.
- Progression handling:
  - sequence after Surface Spec 427.

## Application Integration

- App type: library-only
- User/caller surface: public Boolean API result metadata and downstream reference workflow readiness
- Invocation route: assembled candidate body to validity/persistence gate to result finalizer
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: accepted body metadata or invalid/unsupported result
- Integration validation: public API tests proving accepted body metadata, invalid refusal, persistence readiness, and no-hidden-mesh evidence
- Incomplete status risk: reference workflows could receive bodies before durability and no-hidden-mesh proof are established

App-type-specific proof:

- Library-only: public Boolean API tests prove the validity gate is on the integrated path.

## Reuse And Extraction Plan

- Existing code to reuse:
  - runtime validity, persistence, and tessellation evidence helpers.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - validity record and proof collector.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftPrimitiveCutShellValidityRecord` - validity, persistence readiness, no-hidden-mesh evidence.
- Functions/methods:
  - `validate_loft_primitive_cut_shell_result(...) -> LoftPrimitiveCutShellValidityRecord | Diagnostic`.
  - `collect_loft_primitive_no_hidden_mesh_proof(...) -> NoHiddenMeshProof`.
- UI fields / visible data, UI elements, and UI components:
  - not applicable.

## Performance Contract

- Bounded by result shell, seam, trim, and patch counts.
- Must not rerun CSG or write broad filesystem artifacts as part of ordinary validation.

## Error And State Behavior

- Invalid candidate bodies refuse with diagnostics.
- Missing persistence or no-hidden-mesh proof prevents success.

## Test Strategy

- Unit tests:
  - validity payload, persistence readiness, no-hidden-mesh proof.
- Service/DB tests: not applicable.
- GUI/controller tests, if applicable: not applicable.
- Integrated route tests:
  - public API accepted and invalid result route tests.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Accepted candidates carry validity, persistence readiness, tessellation-boundary readiness, and no-hidden-mesh evidence.
- Invalid candidates cannot return success.
- Reference workflows can distinguish accepted result geometry from invalid/unsupported diagnostics.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Work Units count and basis are explicit.
- [x] Manifest fields are carried into spec sections or preserved as explicit provenance/history.
- [x] Canonical status is explicit.
- [x] Prerequisites are linked, implemented, or marked not applicable.
- [x] Split coverage is complete, or marked not applicable.
- [x] Refinement history records the latest completed review/update/rescore/split iteration and the files written before its write barrier.
- [x] Implementation owner/module is named.
- [x] Existing code reuse/extraction decision is explicit.
- [x] Existing library/module additions or new reusable module boundaries are named, or marked not applicable.
- [x] UI fields/elements are listed, or marked not applicable.
- [x] Chosen defaults are explicit.
- [x] Data source of truth and write owner are explicit.
- [x] App type and application integration route are explicit.
- [x] Integrated route validation is named.
- [x] Performance bounds are explicit, or marked not applicable.
- [x] Privacy/logging constraints are explicit, or marked not applicable.
- [x] Test strategy does not depend on production data.
- [x] Acceptance criteria are testable.
