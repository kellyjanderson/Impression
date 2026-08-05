# Fix 05B: Synthetic Station Identity Lineage

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Architecture ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Source artifact: [Split parent](./fix-05-count-changing-region-identity-preservation-v1_0.md)
Split provenance: `fix-05-count-changing-region-identity-preservation-v1_0.md` from GitHub issue #246
Canonical status: Canonical
Review Score: 17.5
Prerequisites:
- Fix 05a exact region pairing

## Source Field Carryover

- Source purpose: Carry exact predecessor/successor region and loop identities through every synthetic station so expansion never rebuilds an anonymous section.
- Source responsibilities by category:
  - Functions/methods: synthetic lineage constructor, lineage completeness validator, identity-bearing expansion handoff
  - Data structures/models: synthetic station lineage record
  - Dependencies/services: Fix 05a transition result, existing station expansion, planned loop refs
  - Returns/outputs/signals: expanded plan with complete identity lineage
  - UI surfaces/components: not applicable
  - UI fields/elements: not applicable
  - Reusable code plan: add the named responsibilities to existing owner modules
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: not applicable
  - Destructive/write behavior: no destructive runtime writes
  - Security/privacy-sensitive behavior: local paths may appear in diagnostics where applicable; source contents are not logged
  - Performance-sensitive behavior: lineage propagation is linear in regions and loops
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none.
- Source split/provenance notes: this child owns only the parent responsibilities listed below.

## Purpose

Carry exact predecessor/successor region and loop identities through every synthetic station so expansion never rebuilds an anonymous section.

## Scope

- Owns:
  - stable synthetic ids and predecessor/successor lineage
  - direction reversal and multi-stage expansion invariants
  - executor handoff and microphone rail-pair regression

- Does not own:
  - initial exact region matching, owned by Fix 05a
  - planner option propagation, owned by Fix 06

## Split Coverage

- Parent spec: `fix-05-count-changing-region-identity-preservation-v1_0.md`
- Parent coverage status: 100% covered collectively by the parent's listed children
- Parent responsibilities owned by this child:
  - stable synthetic ids and predecessor/successor lineage
  - direction reversal and multi-stage expansion invariants
  - executor handoff and microphone rail-pair regression
- Parent responsibilities still missing from children:
  - none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../spec-refinement-history/a4-open-issues-20260804-165103.md` | 4 | nineteen-leaf active set | none | reached |

Pass 4 split decision: retained. Cohesion reason: one synthetic-lineage construction and validation boundary.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/loft.py` - implementation and wiring owner
- Supporting modules/files:
  - use only the dependencies named below
- GUI/QML files, if applicable:
  - not applicable
- Reusable library/module files:
  - `src/impression/modeling/loft.py` - extend the existing reusable boundary
- Tests:
  - `tests/test_loft_point_lifecycle_records.py`
  - `tests/test_loft.py` rail-pair regression

## Chosen Defaults / Parameters

- synthetic identity derives from explicit transition lineage, never geometry alone
- reverse direction inverts lineage without changing identity

## Data Ownership

- Source of truth: `src/impression/modeling/loft.py`
- Read ownership: the named caller route reads immutable request/plan/surface evidence through the existing public boundary.
- Write ownership: `src/impression/modeling/loft.py` creates only its derived records/results; input source and operands remain unchanged.
- Derived/cache data: all records are recomputable from the declared inputs.
- Privacy/logging constraints: do not log model source contents; otherwise not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - Fix 05a transition result
  - existing station expansion
  - planned loop refs
- Database dependencies:
  - none
- GUI route, if applicable:
  - not applicable
- Background/concurrency route, if applicable:
  - not applicable

## Prerequisite Handling

- Architecture feedback artifacts:
  - `../architecture/acd-loft-identity-and-junction-correctness.md` - target ownership and route are resolved
- Architecture feedback status:
  - tracked in active ACD
- Already implemented prerequisites:
  - existing records/routes named under Dependencies And Routes
- Missing prerequisite architecture:
  - none
- Missing prerequisite specifications:
  - none
- Unimplemented prerequisite specifications:
  - Fix 05a exact region pairing
- Progression handling:
  - prerequisites listed above run first; otherwise this child may proceed after canonical review

## Application Integration

- App type: library-only
- User/caller surface: expanded loft plan consumed by `Loft(...)`
- Invocation route: expanded loft plan consumed by `Loft(...)` -> `src/impression/modeling/loft.py` -> expanded plan with complete identity lineage
- Wiring owner/module: `src/impression/modeling/loft.py`
- Observable result: expanded plan with complete identity lineage
- Integration validation: `tests/test_loft_point_lifecycle_records.py`; `tests/test_loft.py` rail-pair regression
- Incomplete status risk: completion requires the caller route and paired test specification to pass

App-type-specific proof:

- GUI: not applicable
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: expanded loft plan consumed by `Loft(...)` is the consuming route

## Reuse And Extraction Plan

- Existing code to reuse:
  - Fix 05a transition result
  - existing station expansion
  - planned loop refs
- Current reuse readiness:
  - add responsibilities to the existing owner module
- Extraction/wrapping needed:
  - only the named DTO/helper boundary; no parallel subsystem
- Additions to existing library/modules:
  - `src/impression/modeling/loft.py` - synthetic lineage constructor, lineage completeness validator, identity-bearing expansion handoff
- New reusable modules to expose:
  - none
- One-off code justification, if any:
  - none

## Required DTOs / Functions / Components

- DTOs/models:
  - synthetic station lineage record
- Functions/methods:
  - synthetic lineage constructor
  - lineage completeness validator
  - identity-bearing expansion handoff
- UI fields / visible data, if applicable:
  - not applicable
- UI elements / controls, if applicable:
  - not applicable
- UI components, if applicable:
  - none

## Performance Contract

- lineage propagation is linear in regions and loops

## Error And State Behavior

- missing, duplicate, or conflicting derived lineage fails before execution

## Test Strategy

- Unit tests:
  - `tests/test_loft_point_lifecycle_records.py`
  - `tests/test_loft.py` rail-pair regression
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - not applicable
- Integrated route tests:
  - expanded loft plan consumed by `Loft(...)` must exercise expanded plan with complete identity lineage
- Production-data rule:
  - deterministic project fixtures and temporary directories only

## Acceptance Criteria

- stable synthetic ids and predecessor/successor lineage is implemented and asserted through the declared route.
- direction reversal and multi-stage expansion invariants is implemented and asserted through the declared route.
- executor handoff and microphone rail-pair regression is implemented and asserted through the declared route.
- The paired test specification [Synthetic Station Identity Lineage Test](../test-specifications/fix-05b-synthetic-station-identity-lineage-v1_0.md) passes without helper-only substitution.

## Readiness Checklist

- [x] Ancestors, source parent, split provenance, and 100% collective parent coverage are explicit.
- [x] Numeric Review Score was supplied by the terminal fresh child-review pass.
- [x] Current template path is recorded below.
- [x] Architecture, owner, routes, defaults, data, reuse, prerequisites, performance, privacy, tests, and acceptance are explicit.
- [x] No parent responsibility owned by this child remains only in the parent.
- [x] Child was independently rescored and canonicalized after ledger pass 1 readback.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: 17.5; adversarial input, not trusted.
- Adversarial rescore basis: fresh terminal recount checked split lineage, UI/control inventory, routes, reuse, prerequisites, writes, concurrency, performance, and deferral markers.
- Functions/methods: 3 x 2 = 6
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 1 x 1 = 1
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 1 x 2 = 2
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 17.5
- If total matches prior score, adversarial survival reason: the score survived a complete terminal recount; no omitted responsibility, blocker, or route was found.
