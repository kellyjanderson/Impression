# Fix 09A: Difference Geometry Change Evidence

Date: 2026-08-04
Status: Final
Primary ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Source artifact: [Split parent](./fix-09-surface-difference-no-op-result-gate-v1_0.md)
Split provenance: `fix-09-surface-difference-no-op-result-gate-v1_0.md` from GitHub issue #248
Canonical status: Canonical
Review Score: 19.5
Prerequisites:
- none

## Source Field Carryover

- Source purpose: Normalize executor output into inspectable geometry-change witnesses and compare result topology/domains against the minuend with bounded fallback checks.
- Source responsibilities by category:
  - Functions/methods: executor evidence normalizer, geometry-change witness validator, unchanged-result comparator
  - Data structures/models: `GeometryChangeWitness`, normalized difference evidence
  - Dependencies/services: patch provenance/domains, surface topology, tolerance policy
  - Returns/outputs/signals: validated change witnesses or unchanged/ambiguous comparison
  - UI surfaces/components: not applicable
  - UI fields/elements: not applicable
  - Reusable code plan: add the named responsibilities to existing owner modules
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: not applicable
  - Destructive/write behavior: no destructive runtime writes
  - Security/privacy-sensitive behavior: local paths may appear in diagnostics where applicable; source contents are not logged
  - Performance-sensitive behavior: no whole-body dense sampling; comparison is bounded to candidate changed patches
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none.
- Source split/provenance notes: this child owns only the parent responsibilities listed below.

## Purpose

Normalize executor output into inspectable geometry-change witnesses and compare result topology/domains against the minuend with bounded fallback checks.

## Scope

- Owns:
  - geometry-change witness model and executor evidence normalization
  - provenance/domain/topology unchanged comparison
  - bounded localized geometric fallback and tolerance cases

- Does not own:
  - public success/no-cut classification, owned by Fix 09b

## Split Coverage

- Parent spec: `fix-09-surface-difference-no-op-result-gate-v1_0.md`
- Parent coverage status: 100% covered collectively by the parent's listed children
- Parent responsibilities owned by this child:
  - geometry-change witness model and executor evidence normalization
  - provenance/domain/topology unchanged comparison
  - bounded localized geometric fallback and tolerance cases
- Parent responsibilities still missing from children:
  - none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../spec-refinement-history/a4-open-issues-20260804-165103.md` | 4 | nineteen-leaf active set | none | reached |

Pass 4 split decision: retained. Cohesion reason: one normalized evidence/comparison result consumed by every gate.

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
  - `tests/test_surface_csg.py` witness/comparator matrix

## Chosen Defaults / Parameters

- object identity or cloning is never change evidence
- provenance/topology checks precede localized geometry

## Data Ownership

- Source of truth: `src/impression/modeling/csg.py`
- Read ownership: the named caller route reads immutable request/plan/surface evidence through the existing public boundary.
- Write ownership: `src/impression/modeling/csg.py` creates only its derived records/results; input source and operands remain unchanged.
- Derived/cache data: all records are recomputable from the declared inputs.
- Privacy/logging constraints: do not log model source contents; otherwise not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - patch provenance/domains
  - surface topology
  - tolerance policy
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
  - none
- Progression handling:
  - prerequisites listed above run first; otherwise this child may proceed after canonical review

## Application Integration

- App type: library-only
- User/caller surface: all surfaced difference executors
- Invocation route: all surfaced difference executors -> `src/impression/modeling/csg.py` -> validated change witnesses or unchanged/ambiguous comparison
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: validated change witnesses or unchanged/ambiguous comparison
- Integration validation: `tests/test_surface_csg.py` witness/comparator matrix
- Incomplete status risk: completion requires the caller route and paired test specification to pass

App-type-specific proof:

- GUI: not applicable
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: all surfaced difference executors is the consuming route

## Reuse And Extraction Plan

- Existing code to reuse:
  - patch provenance/domains
  - surface topology
  - tolerance policy
- Current reuse readiness:
  - add responsibilities to the existing owner module
- Extraction/wrapping needed:
  - only the named DTO/helper boundary; no parallel subsystem
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - executor evidence normalizer, geometry-change witness validator, unchanged-result comparator
- New reusable modules to expose:
  - none
- One-off code justification, if any:
  - none

## Required DTOs / Functions / Components

- DTOs/models:
  - `GeometryChangeWitness`
  - normalized difference evidence
- Functions/methods:
  - executor evidence normalizer
  - geometry-change witness validator
  - unchanged-result comparator
- UI fields / visible data, if applicable:
  - not applicable
- UI elements / controls, if applicable:
  - not applicable
- UI components, if applicable:
  - none

## Performance Contract

- no whole-body dense sampling
- comparison is bounded to candidate changed patches

## Error And State Behavior

- missing or contradictory evidence is ambiguous, never changed

## Test Strategy

- Unit tests:
  - `tests/test_surface_csg.py` witness/comparator matrix
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - not applicable
- Integrated route tests:
  - all surfaced difference executors must exercise validated change witnesses or unchanged/ambiguous comparison
- Production-data rule:
  - deterministic project fixtures and temporary directories only

## Acceptance Criteria

- geometry-change witness model and executor evidence normalization is implemented and asserted through the declared route.
- provenance/domain/topology unchanged comparison is implemented and asserted through the declared route.
- bounded localized geometric fallback and tolerance cases is implemented and asserted through the declared route.
- The paired test specification [Difference Geometry Change Evidence Test](../test-specifications/fix-09a-difference-geometry-change-evidence-v1_0.md) passes without helper-only substitution.

## Readiness Checklist

- [x] Ancestors, source parent, split provenance, and 100% collective parent coverage are explicit.
- [x] Numeric Review Score was supplied by the terminal fresh child-review pass.
- [x] Current template path is recorded below.
- [x] Architecture, owner, routes, defaults, data, reuse, prerequisites, performance, privacy, tests, and acceptance are explicit.
- [x] No parent responsibility owned by this child remains only in the parent.
- [x] Child was independently rescored and canonicalized after ledger pass 1 readback.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: 19.5; adversarial input, not trusted.
- Adversarial rescore basis: fresh terminal recount checked split lineage, UI/control inventory, routes, reuse, prerequisites, writes, concurrency, performance, and deferral markers.
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
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
- Performance-sensitive behavior: 2 x 2 = 4
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 19.5
- If total matches prior score, adversarial survival reason: the score survived a complete terminal recount; no omitted responsibility, blocker, or route was found.
