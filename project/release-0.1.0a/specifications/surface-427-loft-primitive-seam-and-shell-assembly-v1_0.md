# Surface Spec 427: Loft Primitive Seam And Shell Assembly (v1.0)

Date: 2026-07-16
Status: Superseded
Primary ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`
Manifest source: `Loft Primitive Seam And Shell Assembly`
Split provenance: `../specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
Canonical status: Superseded parent
Prerequisites:
- `../specifications/surface-426-loft-primitive-fragment-topology-and-operation-selection-v1_0.md` - shell assembly consumes selected fragments and topology records.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 3 IWU.
Basis: parent branch split into seam/use pairing, candidate shell assembly, and adjacency rebuild diagnostics.

## Manifest Field Carryover

- Discovery purpose: build candidate result shells and seam/use pair records from selected fragments and generated caps.
- Manifest responsibilities by category:
  - Functions/methods: seam/use pairing builder, cut-shell assembler, adjacency rebuild diagnostic builder.
  - Data structures/models: `LoftPrimitiveCutShellAssemblyRecord`, `LoftPrimitiveSeamUsePairRecord`, adjacency rebuild diagnostic.
  - Dependencies/services: topology selection records, generated cap records, SurfaceBody constructors and seam helpers.
  - Returns/outputs/signals: candidate `SurfaceBody` shells, seam/adjacency records, assembly diagnostics.
  - UI surfaces/components and fields: not applicable.
  - Reusable code plan: reuse `make_surface_shell`, `make_surface_body`, seam/adjacency helpers; add assembler to `src/impression/modeling/csg.py`.
  - Database/async/write/security/cross-screen behavior: none.
  - Performance-sensitive behavior: bounded by fragment, loop, and seam count.
- Manifest open questions / nuance discovered:
  - Continuity beyond C0/G0 must be recorded as future capability, not silently claimed.
- Manifest score at promotion: 19 on 2026-07-16.
- Manifest readiness blockers and resolution: none; resolved.
- Manifest split decision: review for split; cohesive seam/shell assembly boundary.
- Manifest cleanup state: ready after spec promotion.

## Purpose

Assemble topology-selected loft fragments and generated primitive caps into candidate `SurfaceBody` shells with explicit seam/use pairing.

## Scope

Owns:

- `LoftPrimitiveSeamUsePairRecord`.
- `LoftPrimitiveCutShellAssemblyRecord`.
- Candidate shell assembly and adjacency rebuild diagnostics.

Does not own:

- Operation topology selection.
- Runtime validity/persistence gating.
- Public result envelope integration.

## Split Coverage

- Parent spec: `../specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 427a-427c.
- Parent responsibilities owned by this child:
  - none; superseded parent only.
- Parent responsibilities still missing from children:
  - none.
  - Seam/use pairing is covered by `surface-427a-loft-primitive-seam-use-pairing-v1_0.md`.
  - Candidate shell assembly is covered by `surface-427b-loft-primitive-candidate-shell-assembly-v1_0.md`.
  - Adjacency rebuild diagnostics are covered by `surface-427c-loft-primitive-adjacency-rebuild-diagnostics-v1_0.md`.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Manifest promotion and sizing | this spec and paired test spec | Promoted manifest candidate and set IWU to 1 | score 19 | 1 IWU | no split | none | not applicable | fixed-point ledger readback |
| 2 | 2026-07-16 | Critical sizing review | this parent and child specs 427a-427c | Marked parent superseded and moved all executable work to children | 1 IWU | 3 IWU branch rollup | split | 427a, 427b, 427c | complete | child specs |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - seam/use records and cut-shell assembler.
- Supporting modules/files:
  - `src/impression/modeling/surface.py` - shell constructors and seam helpers.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - shell assembler consumed by validity gate.
- Tests:
  - `tests/test_surface_csg.py` - cap-loop pairing, unpaired seam refusal, cavity assembly, no mesh fragments.

## Chosen Defaults / Parameters

- Required generated cap loops pair exactly once with retained loft loops.
- Candidate shells record continuity truth; they do not claim unsupported continuity classes.
- Mesh fragments are refused.

## Data Ownership

- Source of truth: CSG assembly records and candidate `SurfaceBody` shells.
- Read ownership: validity gate consumes candidate shells and assembly diagnostics.
- Write ownership: `src/impression/modeling/csg.py`.
- Derived/cache data: recomputable from topology and cap records.
- Privacy/logging constraints: diagnostics avoid full geometry dumps.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 426 topology records.
  - SurfaceBody constructors and seam helpers.
- Database dependencies: not applicable.
- GUI route, if applicable: not applicable.
- Background/concurrency route, if applicable: not applicable.

## Prerequisite Handling

- Already implemented prerequisites: none.
- Missing prerequisite architecture: none.
- Missing prerequisite specifications: none.
- Unimplemented prerequisite specifications:
  - `../specifications/surface-426-loft-primitive-fragment-topology-and-operation-selection-v1_0.md` - must be implemented first.
- Progression handling:
  - sequence after Surface Spec 426.

## Application Integration

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: topology selector to seam/use pairing to candidate `SurfaceBody`
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: candidate result shells or assembly diagnostics
- Integration validation: seam assembly tests plus public invalid-seam refusal tests
- Incomplete status risk: candidate bodies could be constructed without public-route proof or with unpaired seams

App-type-specific proof:

- Library-only: public Boolean route refusal tests prove seam assembly is reachable.

## Reuse And Extraction Plan

- Existing code to reuse:
  - SurfaceBody constructors and seam/adjacency helpers.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - seam/use pairing and shell assembler.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftPrimitiveSeamUsePairRecord` - paired loop-use identity, orientation, tolerance, provenance.
  - `LoftPrimitiveCutShellAssemblyRecord` - candidate shells, seam-use pairs, topology class, diagnostics.
- Functions/methods:
  - `assemble_loft_primitive_cut_shell(...) -> LoftPrimitiveCutShellAssemblyRecord | Diagnostic`.
  - `pair_loft_primitive_seam_uses(...) -> Sequence[LoftPrimitiveSeamUsePairRecord] | Diagnostic`.
- UI fields / visible data, UI elements, and UI components:
  - not applicable.

## Performance Contract

- Bounded by fragment, loop, and seam counts.
- Assembly must not rerun intersection or topology selection.

## Error And State Behavior

- Unpaired seams, duplicate pairings, open boundaries, and mesh fragments refuse deterministically.
- Assembly diagnostics identify the failed seam/use boundary.

## Test Strategy

- Unit tests:
  - seam pairing, shell assembly, cavity assembly, unpaired seam refusal.
- Service/DB tests: not applicable.
- GUI/controller tests, if applicable: not applicable.
- Integrated route tests:
  - public invalid-seam refusal tests.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Selected fragments and caps produce candidate `SurfaceBody` shells with seam/use pairing records.
- Invalid seam or adjacency conditions refuse before validity finalization.
- Assembly records prove no mesh fragments participate.

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
