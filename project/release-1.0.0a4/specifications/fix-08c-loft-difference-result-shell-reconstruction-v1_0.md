# Fix 08C: Loft Difference Result Shell Reconstruction

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Source artifact: [Split parent](./fix-08-loft-surface-difference-cut-execution-v1_0.md)
Split provenance: `fix-08-loft-surface-difference-cut-execution-v1_0.md` from GitHub issue #248
Canonical status: Canonical
Review Score: 22.5
Prerequisites:
- Fix 08a trim fragments
- Fix 08b branch decomposition
- Fix 09b public success gate

## Source Field Carryover

- Source purpose: Classify and assemble trim fragments plus reversed cutter boundary patches into validated closed result shells for the USB-C, acoustic, and snap-pocket fixtures.
- Source responsibilities by category:
  - Functions/methods: retained-fragment classifier, cutter boundary patch builder, result shell assembler/validator
  - Data structures/models: result-shell reconstruction evidence
  - Dependencies/services: Fix 08a fragments, Fix 08b recomposition map, seam/body validators
  - Returns/outputs/signals: closed changed `SurfaceBody` or precise refusal
  - UI surfaces/components: not applicable
  - UI fields/elements: not applicable
  - Reusable code plan: add the named responsibilities to existing owner modules
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: not applicable
  - Destructive/write behavior: no destructive runtime writes
  - Security/privacy-sensitive behavior: local paths may appear in diagnostics where applicable; source contents are not logged
  - Performance-sensitive behavior: assembly scales with produced fragments and avoids tessellation
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none.
- Source split/provenance notes: this child owns only the parent responsibilities listed below.

## Purpose

Classify and assemble trim fragments plus reversed cutter boundary patches into validated closed result shells for the USB-C, acoustic, and snap-pocket fixtures.

## Scope

- Owns:
  - retained-fragment classification and cutter boundary-patch orientation
  - seam/adjacency and closed-shell reconstruction
  - public fixture, validity, witness, preview/export, and no-mesh proof

- Does not own:
  - fragment creation, owned by Fix 08a
  - branch-plan creation, owned by Fix 08b
  - shared unchanged-result policy, owned by Fix 09b

## Split Coverage

- Parent spec: `fix-08-loft-surface-difference-cut-execution-v1_0.md`
- Parent coverage status: 100% covered collectively by the parent's listed children
- Parent responsibilities owned by this child:
  - retained-fragment classification and cutter boundary-patch orientation
  - seam/adjacency and closed-shell reconstruction
  - public fixture, validity, witness, preview/export, and no-mesh proof
- Parent responsibilities still missing from children:
  - none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../spec-refinement-history/a4-open-issues-20260804-165103.md` | 4 | nineteen-leaf active set | none | reached |

Pass 4 split decision: retained. Cohesion reason: one fragment-to-closed-result-shell assembly outcome and consumer proof.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - implementation and wiring owner
- Supporting modules/files:
  - use only the dependencies named below
- GUI/QML files, if applicable:
  - not applicable
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - extend the existing reusable boundary
- Tests:
  - `tests/test_surface_csg.py` public fixtures
  - preview/export consumer smoke

## Chosen Defaults / Parameters

- retain minuend-outside fragments and correctly oriented cutter boundaries
- success requires complete seams, closure, operands, and Fix 09 evidence

## Data Ownership

- Source of truth: `src/impression/modeling/csg.py`
- Read ownership: the named caller route reads immutable request/plan/surface evidence through the existing public boundary.
- Write ownership: `src/impression/modeling/csg.py` creates only its derived records/results; input source and operands remain unchanged.
- Derived/cache data: all records are recomputable from the declared inputs.
- Privacy/logging constraints: do not log model source contents; otherwise not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - Fix 08a fragments
  - Fix 08b recomposition map
  - seam/body validators
- Database dependencies:
  - none
- GUI route, if applicable:
  - not applicable
- Background/concurrency route, if applicable:
  - not applicable

## Prerequisite Handling

- Architecture feedback artifacts:
  - `../architecture/acd-surface-boolean-correctness-and-api-boundary.md` - target ownership and route are resolved
- Architecture feedback status:
  - tracked in active ACD
- Already implemented prerequisites:
  - existing records/routes named under Dependencies And Routes
- Missing prerequisite architecture:
  - none
- Missing prerequisite specifications:
  - none
- Unimplemented prerequisite specifications:
  - Fix 08a trim fragments
  - Fix 08b branch decomposition
  - Fix 09b public success gate
- Progression handling:
  - prerequisites listed above run first; otherwise this child may proceed after canonical review

## Application Integration

- App type: library-only
- User/caller surface: public `boolean_difference` and preview/export consumers
- Invocation route: public `boolean_difference` and preview/export consumers -> `src/impression/modeling/csg.py` -> closed changed `SurfaceBody` or precise refusal
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: closed changed `SurfaceBody` or precise refusal
- Integration validation: `tests/test_surface_csg.py` public fixtures; preview/export consumer smoke
- Incomplete status risk: completion requires the caller route and paired test specification to pass

App-type-specific proof:

- GUI: not applicable
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: public `boolean_difference` and preview/export consumers is the consuming route

## Reuse And Extraction Plan

- Existing code to reuse:
  - Fix 08a fragments
  - Fix 08b recomposition map
  - seam/body validators
- Current reuse readiness:
  - add responsibilities to the existing owner module
- Extraction/wrapping needed:
  - only the named DTO/helper boundary; no parallel subsystem
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - retained-fragment classifier, cutter boundary patch builder, result shell assembler/validator
- New reusable modules to expose:
  - none
- One-off code justification, if any:
  - none

## Required DTOs / Functions / Components

- DTOs/models:
  - result-shell reconstruction evidence
- Functions/methods:
  - retained-fragment classifier
  - cutter boundary patch builder
  - result shell assembler/validator
- UI fields / visible data, if applicable:
  - not applicable
- UI elements / controls, if applicable:
  - not applicable
- UI components, if applicable:
  - none

## Performance Contract

- assembly scales with produced fragments and avoids tessellation

## Error And State Behavior

- ambiguous classification, open seams, invalid closure, or no change cannot succeed

## Test Strategy

- Unit tests:
  - `tests/test_surface_csg.py` public fixtures
  - preview/export consumer smoke
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - not applicable
- Integrated route tests:
  - public `boolean_difference` and preview/export consumers must exercise closed changed `SurfaceBody` or precise refusal
- Production-data rule:
  - deterministic project fixtures and temporary directories only

## Acceptance Criteria

- retained-fragment classification and cutter boundary-patch orientation is implemented and asserted through the declared route.
- seam/adjacency and closed-shell reconstruction is implemented and asserted through the declared route.
- public fixture, validity, witness, preview/export, and no-mesh proof is implemented and asserted through the declared route.
- The paired test specification [Loft Difference Result Shell Reconstruction Test](../test-specifications/fix-08c-loft-difference-result-shell-reconstruction-v1_0.md) passes without helper-only substitution.

## Readiness Checklist

- [x] Ancestors, source parent, split provenance, and 100% collective parent coverage are explicit.
- [x] Numeric Review Score was supplied by the terminal fresh child-review pass.
- [x] Current template path is recorded below.
- [x] Architecture, owner, routes, defaults, data, reuse, prerequisites, performance, privacy, tests, and acceptance are explicit.
- [x] No parent responsibility owned by this child remains only in the parent.
- [x] Child was independently rescored and canonicalized after ledger pass 1 readback.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: 22.5; adversarial input, not trusted.
- Adversarial rescore basis: fresh terminal recount checked split lineage, UI/control inventory, routes, reuse, prerequisites, writes, concurrency, performance, and deferral markers.
- Functions/methods: 3 x 2 = 6
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
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
- Missing prerequisites: 3 x 2 = 6
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 22.5
- If total matches prior score, adversarial survival reason: the score survived a complete terminal recount; no omitted responsibility, blocker, or route was found.
