# Surface Spec 426: Loft Primitive Fragment Topology And Operation Selection (v1.0)

Date: 2026-07-16
Status: Superseded
Primary ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
Architecture ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`
Manifest source: `Loft Primitive Fragment Topology And Operation Selection`
Split provenance: `../specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
Canonical status: Superseded parent
Prerequisites:
- `../specifications/surface-425-loft-primitive-generated-cap-construction-v1_0.md` - topology selection consumes generated cap records.

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 3 IWU.
Basis: parent branch split into retained-fragment selection, topology-class classification, and orientation/refusal diagnostics.

## Manifest Field Carryover

- Discovery purpose: select fragments and topology outcome for loft/primitive union, difference, and intersection.
- Manifest responsibilities by category:
  - Functions/methods: topology selector, operation fragment chooser, orientation diagnostic builder.
  - Data structures/models: `LoftPrimitiveFragmentTopologyRecord`, operation selection record, topology diagnostic.
  - Dependencies/services: fragment classifications, generated caps, operation policy.
  - Returns/outputs/signals: selected fragments, topology records, orientation diagnostics.
  - UI surfaces/components and fields: not applicable.
  - Reusable code plan: add to existing CSG operation selection helpers; no new module.
  - Database/async/write/security/cross-screen behavior: none.
  - Performance-sensitive behavior: bounded by fragment, cap, and loop counts.
- Manifest open questions / nuance discovered:
  - Difference must distinguish internal cavity boundaries from exterior shell edits; this spec makes topology class explicit.
- Manifest score at promotion: 19 on 2026-07-16.
- Manifest readiness blockers and resolution: none; resolved.
- Manifest split decision: review for split; cohesive topology-selection boundary.
- Manifest cleanup state: ready after spec promotion.

## Purpose

Define the topology decision stage that turns classified fragments and generated caps into a shell-assembly plan for a specific Boolean operation.

## Scope

Owns:

- `LoftPrimitiveFragmentTopologyRecord`.
- Operation-specific retained-fragment selection.
- Empty, exterior-shell, interior-cavity, multi-shell, and refused topology classification.

Does not own:

- Generated cap construction.
- Seam/use pairing, shell assembly, result validity, or reference handoff.

## Split Coverage

- Parent spec: `../specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
- Parent coverage status: 100% covered by Surface Specs 426a-426c.
- Parent responsibilities owned by this child:
  - none; superseded parent only.
- Parent responsibilities still missing from children:
  - none.
  - Operation-specific fragment retention is covered by `surface-426a-loft-primitive-operation-fragment-retention-v1_0.md`.
  - Topology class classification is covered by `surface-426b-loft-primitive-result-topology-classification-v1_0.md`.
  - Orientation and refusal diagnostics are covered by `surface-426c-loft-primitive-topology-orientation-and-refusal-diagnostics-v1_0.md`.

## Refinement History

| Iteration | Date | Scope reviewed | Files written before barrier | Updates made | IWU/score before | IWU/score after | Split decision | Split artifacts | Child re-review status | Next scope after readback |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | 2026-07-16 | Manifest promotion and sizing | this spec and paired test spec | Promoted manifest candidate and set IWU to 1 | score 19 | 1 IWU | no split | none | not applicable | fixed-point ledger readback |
| 2 | 2026-07-16 | Critical sizing review | this parent and child specs 426a-426c | Marked parent superseded and moved all executable work to children | 1 IWU | 3 IWU branch rollup | split | 426a, 426b, 426c | complete | child specs |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - topology records and operation selector.
- Supporting modules/files:
  - `src/impression/modeling/surface.py` - orientation and shell identity helpers.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - selector consumed by seam/shell assembly.
- Tests:
  - `tests/test_surface_csg.py` - difference cavity, partial-overlap difference, union, intersection, touching/no-cut, orientation refusal.

## Chosen Defaults / Parameters

- Touching cases route to exact/no-cut outcomes.
- True cut cases require explicit topology records.
- Multi-shell cases must be represented or refused deterministically; they may not collapse into a single-shell approximation.

## Data Ownership

- Source of truth: CSG topology records.
- Read ownership: seam/shell assembly consumes topology records.
- Write ownership: `src/impression/modeling/csg.py`.
- Derived/cache data: recomputable from fragment classification and cap records.
- Privacy/logging constraints: diagnostics avoid full geometry dumps.

## Dependencies And Routes

- Domain/service dependencies:
  - generated cap records.
  - fragment classifications.
  - Boolean operation policy.
- Database dependencies: not applicable.
- GUI route, if applicable: not applicable.
- Background/concurrency route, if applicable: not applicable.

## Prerequisite Handling

- Already implemented prerequisites: none.
- Missing prerequisite architecture: none.
- Missing prerequisite specifications: none.
- Unimplemented prerequisite specifications:
  - `../specifications/surface-425-loft-primitive-generated-cap-construction-v1_0.md` - must be implemented first.
- Progression handling:
  - sequence after Surface Spec 425.

## Application Integration

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: generated cap and fragment classification records to operation topology selector
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: selected fragments and topology records, or orientation diagnostics
- Integration validation: topology tests plus public API route tests for difference, union, intersection, touching, and orientation refusal
- Incomplete status risk: seam assembly could build the wrong exterior, cavity, empty, or multi-shell topology

App-type-specific proof:

- Library-only: public Boolean API route tests prove operation topology is reachable.

## Reuse And Extraction Plan

- Existing code to reuse:
  - CSG operation selection helpers and orientation helpers.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - topology records and selector.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftPrimitiveFragmentTopologyRecord` - selected fragments, caps, orientation, topology class.
  - `LoftPrimitiveOperationSelectionDiagnostic` - topology/orientation refusal payload.
- Functions/methods:
  - `select_loft_primitive_fragment_topology(...) -> LoftPrimitiveFragmentTopologyRecord | Diagnostic`.
  - `classify_loft_primitive_operation_topology(...) -> TopologyClass`.
- UI fields / visible data, UI elements, and UI components:
  - not applicable.

## Performance Contract

- Bounded by fragment, cap, and loop counts.
- Selector must not rerun intersection or tessellation work.

## Error And State Behavior

- Unsupported topology and orientation ambiguity refuse deterministically.
- Diagnostics must distinguish empty, exterior, cavity, multi-shell, and refused outcomes.

## Test Strategy

- Unit tests:
  - operation topology for difference cavity, partial overlap, union, intersection, no-cut/touching.
- Service/DB tests: not applicable.
- GUI/controller tests, if applicable: not applicable.
- Integrated route tests:
  - public API route tests for accepted and refused topology outcomes.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Boolean operation topology records are explicit for supported cut cases.
- Difference distinguishes cavity topology from exterior shell edits.
- Unsupported or ambiguous topology refuses with deterministic diagnostics.

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
