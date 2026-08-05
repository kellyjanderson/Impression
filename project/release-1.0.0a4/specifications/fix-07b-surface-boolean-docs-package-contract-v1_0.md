# Fix 07B: Surface Boolean Docs And Package Contract

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Source artifact: [Split parent](./fix-07-surface-only-public-boolean-api-v1_0.md)
Split provenance: `fix-07-surface-only-public-boolean-api-v1_0.md` from GitHub issue #247
Canonical status: Canonical
Review Score: 17.5
Prerequisites:
- Fix 07a runtime API

## Source Field Carryover

- Source purpose: Make documentation, examples, type guards, and clean installed artifacts expose exactly the surfaced runtime API defined by Fix 07a.
- Source responsibilities by category:
  - Functions/methods: documentation/example migration, API inventory guard, clean-wheel smoke
  - Data structures/models: not applicable
  - Dependencies/services: Fix 07a public API, documentation corpus, wheel build/install
  - Returns/outputs/signals: consistent source/docs/wheel contract and migration guidance
  - UI surfaces/components: not applicable
  - UI fields/elements: not applicable
  - Reusable code plan: add the named responsibilities to existing owner modules
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: not applicable
  - Destructive/write behavior: no destructive runtime writes
  - Security/privacy-sensitive behavior: local paths may appear in diagnostics where applicable; source contents are not logged
  - Performance-sensitive behavior: not applicable
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none.
- Source split/provenance notes: this child owns only the parent responsibilities listed below.

## Purpose

Make documentation, examples, type guards, and clean installed artifacts expose exactly the surfaced runtime API defined by Fix 07a.

## Scope

- Owns:
  - CSG reference/tutorial/example migration
  - source/docs/export inventory guard
  - clean-wheel signature/import/runtime smoke and migration guidance

- Does not own:
  - runtime boundary implementation, owned by Fix 07a

## Split Coverage

- Parent spec: `fix-07-surface-only-public-boolean-api-v1_0.md`
- Parent coverage status: 100% covered collectively by the parent's listed children
- Parent responsibilities owned by this child:
  - CSG reference/tutorial/example migration
  - source/docs/export inventory guard
  - clean-wheel signature/import/runtime smoke and migration guidance
- Parent responsibilities still missing from children:
  - none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../spec-refinement-history/a4-open-issues-20260804-165103.md` | 4 | nineteen-leaf active set | none | reached |

Pass 4 split decision: retained. Cohesion reason: one source/docs/wheel conformance outcome with one migration corpus.

## Implementation Routing

- Primary modules/files:
  - `docs/modeling/csg.md`, `docs/examples/csg/`, package smoke tests - implementation and wiring owner
- Supporting modules/files:
  - use only the dependencies named below
- GUI/QML files, if applicable:
  - not applicable
- Reusable library/module files:
  - `docs/modeling/csg.md`, `docs/examples/csg/`, package smoke tests - extend the existing reusable boundary
- Tests:
  - `tests/test_surface_csg_docs.py`
  - clean-wheel smoke

## Chosen Defaults / Parameters

- all modeling examples use surfaced operands
- retained mesh utilities are described as non-modeling

## Data Ownership

- Source of truth: `docs/modeling/csg.md`, `docs/examples/csg/`, package smoke tests
- Read ownership: the named caller route reads immutable request/plan/surface evidence through the existing public boundary.
- Write ownership: `docs/modeling/csg.md`, `docs/examples/csg/`, package smoke tests creates only its derived records/results; input source and operands remain unchanged.
- Derived/cache data: all records are recomputable from the declared inputs.
- Privacy/logging constraints: do not log model source contents; otherwise not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - Fix 07a public API
  - documentation corpus
  - wheel build/install
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
  - Fix 07a runtime API
- Progression handling:
  - prerequisites listed above run first; otherwise this child may proceed after canonical review

## Application Integration

- App type: library-only
- User/caller surface: installed package, API documentation, tutorials, and examples
- Invocation route: installed package, API documentation, tutorials, and examples -> `docs/modeling/csg.md`, `docs/examples/csg/`, package smoke tests -> consistent source/docs/wheel contract and migration guidance
- Wiring owner/module: `docs/modeling/csg.md`, `docs/examples/csg/`, package smoke tests
- Observable result: consistent source/docs/wheel contract and migration guidance
- Integration validation: `tests/test_surface_csg_docs.py`; clean-wheel smoke
- Incomplete status risk: completion requires the caller route and paired test specification to pass

App-type-specific proof:

- GUI: not applicable
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: installed package, API documentation, tutorials, and examples is the consuming route

## Reuse And Extraction Plan

- Existing code to reuse:
  - Fix 07a public API
  - documentation corpus
  - wheel build/install
- Current reuse readiness:
  - add responsibilities to the existing owner module
- Extraction/wrapping needed:
  - only the named DTO/helper boundary; no parallel subsystem
- Additions to existing library/modules:
  - `docs/modeling/csg.md`, `docs/examples/csg/`, package smoke tests - documentation/example migration, API inventory guard, clean-wheel smoke
- New reusable modules to expose:
  - none
- One-off code justification, if any:
  - none

## Required DTOs / Functions / Components

- DTOs/models:
  - not applicable
- Functions/methods:
  - documentation/example migration
  - API inventory guard
  - clean-wheel smoke
- UI fields / visible data, if applicable:
  - not applicable
- UI elements / controls, if applicable:
  - not applicable
- UI components, if applicable:
  - none

## Performance Contract

- not applicable

## Error And State Behavior

- stale mesh signature/example or package mismatch fails validation

## Test Strategy

- Unit tests:
  - `tests/test_surface_csg_docs.py`
  - clean-wheel smoke
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - not applicable
- Integrated route tests:
  - installed package, API documentation, tutorials, and examples must exercise consistent source/docs/wheel contract and migration guidance
- Production-data rule:
  - deterministic project fixtures and temporary directories only

## Acceptance Criteria

- CSG reference/tutorial/example migration is implemented and asserted through the declared route.
- source/docs/export inventory guard is implemented and asserted through the declared route.
- clean-wheel signature/import/runtime smoke and migration guidance is implemented and asserted through the declared route.
- The paired test specification [Surface Boolean Docs And Package Contract Test](../test-specifications/fix-07b-surface-boolean-docs-package-contract-v1_0.md) passes without helper-only substitution.

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
- Data structures/models: 0 x 1 = 0
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 1 x 1 = 1
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 0 x 2 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 1 x 2 = 2
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 17.5
- If total matches prior score, adversarial survival reason: the score survived a complete terminal recount; no omitted responsibility, blocker, or route was found.
