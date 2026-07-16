# Surface Spec 424: Loft Patch-Local Cut Loop Construction (v1.0)

Date: 2026-07-16
Status: Superseded
Primary ancestor: `../architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`
Architecture ancestor: `../architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`
Manifest source: `Loft Patch-Local Cut Loop Construction`
Split provenance: `../specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
Canonical status: Superseded parent
Prerequisites:
- `../specifications/surface-423-loft-primitive-intersection-source-normalization-v1_0.md` - normalized source records are the input to patch-local curve inversion and loop closure.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 3 IWU.
Basis: parent branch split into source curve inversion, loop closure/boundary participation, and degeneracy diagnostics.

## Manifest Field Carryover

- Discovery purpose: construct patch-local cut loops for loft patches from normalized primitive source records.
- Manifest responsibilities by category:
  - Functions/methods: patch-local curve inverter, loop closure builder, tangent/grazing diagnostic builder.
  - Data structures/models: `LoftPatchLocalCutLoopRecord`, loop segment record, loop closure diagnostic.
  - Dependencies/services: source records, loft patch parameter domains, patch-local curve mapper.
  - Returns/outputs/signals: closed cut loops, residual/tolerance diagnostics, no-mesh proof.
  - UI surfaces/components: not applicable.
  - UI fields/elements: not applicable.
  - Reusable code plan: reuse surface CSG patch-local curve mapping; add loft cut-loop helpers in `src/impression/modeling/csg.py`; no new module.
  - Database/async/write/security/cross-screen behavior: none.
  - Performance-sensitive behavior: bounded by affected patch and curve segment count.
- Manifest open questions / nuance discovered:
  - Existing cap trims and station seams must participate in loop closure; this spec requires loop closure to include those boundaries.
- Manifest score at promotion: 19 on 2026-07-16.
- Manifest readiness blockers and resolution: none; resolved.
- Manifest split decision: review for split; cohesive cut-loop construction boundary.
- Manifest cleanup state: ready after spec promotion.

## Purpose

Build deterministic patch-local cut-loop records for loft/primitive CSG without falling back to mesh or tessellated execution.

## Scope

Owns:

- `LoftPatchLocalCutLoopRecord`.
- Patch-local inversion of source curves into loft parameter domains.
- Loop closure and tangent/grazing/zero-area refusal diagnostics.

Does not own:

- Source normalization.
- Primitive cap generation.
- Fragment topology, shell assembly, or public result finalization.

## Split Coverage

- Parent spec: `../specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 424a-424c.
- Parent responsibilities owned by this child:
  - none; superseded parent only.
- Parent responsibilities still missing from children:
  - none.
  - Source curve inversion is covered by `surface-424a-loft-patch-local-source-curve-inversion-v1_0.md`.
  - Loop closure and boundary participation is covered by `surface-424b-loft-cut-loop-closure-and-boundary-participation-v1_0.md`.
  - Tangent, grazing, zero-area, and open-loop diagnostics are covered by `surface-424c-loft-cut-loop-degeneracy-diagnostics-v1_0.md`.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Manifest promotion and sizing | this spec and paired test spec | Promoted manifest candidate and set IWU to 1 | score 19 | 1 IWU | no split | none | not applicable | fixed-point ledger readback |
| 2 | 2026-07-16 | Critical sizing review | this parent and child specs 424a-424c | Marked parent superseded and moved all executable work to children | 1 IWU | 3 IWU branch rollup | split | 424a, 424b, 424c | complete | child specs |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - cut-loop records, loop builder, loop diagnostics.
- Supporting modules/files:
  - `src/impression/modeling/loft.py` - loft patch domains and station seam metadata.
  - `src/impression/modeling/surface.py` - parameter-domain and trim-loop helpers.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - reusable loop builder consumed by cap and topology stages.
- Tests:
  - `tests/test_surface_csg.py` - crossing, partial crossing, tangent/grazing refusal, cap-loop interaction.

## Chosen Defaults / Parameters

- Tangent and zero-area loops refuse unless classified earlier as no-cut/touching.
- CSG tolerance policy is inherited from the public Boolean route.
- Existing cap trims and station seams must participate in closure.

## Data Ownership

- Source of truth: cut-loop records owned by CSG.
- Read ownership: generated cap and topology specs consume cut loops.
- Write ownership: `src/impression/modeling/csg.py`.
- Derived/cache data: recomputable from source records and loft patch domains.
- Privacy/logging constraints: diagnostics must avoid full geometry dumps.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 423 source records.
  - loft patch parameter domains.
  - surface CSG patch-local curve mapper.
- Database dependencies: not applicable.
- GUI route, if applicable: not applicable.
- Background/concurrency route, if applicable: not applicable.

## Prerequisite Handling

- Already implemented prerequisites:
  - none.
- Missing prerequisite architecture:
  - none.
- Missing prerequisite specifications:
  - none.
- Unimplemented prerequisite specifications:
  - `../specifications/surface-423-loft-primitive-intersection-source-normalization-v1_0.md` - must be implemented first.
- Progression handling:
  - sequence immediately after Surface Spec 423.

## Application Integration

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: source normalizer to patch-local cut-loop builder through the loft CSG route
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: closed cut loops or deterministic loop/inversion diagnostics
- Integration validation: cut-loop tests plus public-route refusal tests for tangent, grazing, and open-loop cases
- Incomplete status risk: cap and topology work could consume synthetic or open boundaries instead of route-produced cut loops

App-type-specific proof:

- Library-only: public Boolean API route diagnostics prove loop construction is reachable.

## Reuse And Extraction Plan

- Existing code to reuse:
  - surface CSG patch-local curve mapper.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - cut-loop record and builder helpers.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftPatchLocalCutLoopRecord` - closed loop in patch-local coordinates.
  - `LoftPatchLocalCutLoopDiagnostic` - inversion, tolerance, and closure refusal.
- Functions/methods:
  - `build_loft_patch_local_cut_loops(...) -> Sequence[LoftPatchLocalCutLoopRecord] | Diagnostic`.
  - `validate_loft_cut_loop_closure(...) -> Diagnostic | None`.
- UI fields / visible data, if applicable: not applicable.
- UI elements / controls, if applicable: not applicable.
- UI components, if applicable: not applicable.

## Performance Contract

- Bounded by affected patch count and curve segment count.
- No hidden mesh, tessellation, or raster path is allowed.

## Error And State Behavior

- Open loops, tangent ambiguity, grazing ambiguity, and zero-area loops refuse deterministically.
- Diagnostics identify the failed patch and loop class without dumping raw geometry.

## Test Strategy

- Unit tests:
  - crossing, partial crossing, closed-loop construction, tangent/grazing refusal.
- Service/DB tests: not applicable.
- GUI/controller tests, if applicable: not applicable.
- Integrated route tests:
  - public-route refusal tests for open and ambiguous loops.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Supported source records produce closed patch-local cut-loop records.
- Ambiguous or invalid loop conditions refuse with deterministic diagnostics.
- No cut-loop path uses mesh or tessellation as an execution fallback.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Work Units count and basis are explicit.
- [x] Manifest fields are carried into spec sections or preserved as explicit provenance/history.
- [x] Canonical status is explicit.
- [x] Prerequisites are linked, implemented, or marked not applicable.
- [x] Missing prerequisite architecture has an ACD link, or is marked not applicable.
- [x] Missing prerequisite behavior has a final spec link, or is marked not applicable.
- [x] Split coverage is complete, or marked not applicable.
- [x] Refinement history records the latest completed review/update/rescore/split iteration and the files written before its write barrier.
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
