# Fix 08B: Loft Difference Branch Decomposition

Date: 2026-08-04
Status: Final
Primary ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Source artifact: [Split parent](./fix-08-loft-surface-difference-cut-execution-v1_0.md)
Split provenance: `fix-08-loft-surface-difference-cut-execution-v1_0.md` from GitHub issue #248
Canonical status: Canonical
Review Score: 17.5
Prerequisites:
- Fix 05b synthetic station lineage

## Source Field Carryover

- Source purpose: Decompose validated branching loft topology into bounded cut sub-bodies and recompose provenance only when lineage and branch seams are complete.
- Source responsibilities by category:
  - Functions/methods: branch graph validator, bounded sub-body decomposition, recomposition-map validator
  - Data structures/models: branch decomposition/recomposition record
  - Dependencies/services: loft topology lineage, CSG route registry, configured bounds
  - Returns/outputs/signals: validated sub-body cut plan and recomposition map
  - UI surfaces/components: not applicable
  - UI fields/elements: not applicable
  - Reusable code plan: add the named responsibilities to existing owner modules
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: not applicable
  - Destructive/write behavior: no destructive runtime writes
  - Security/privacy-sensitive behavior: local paths may appear in diagnostics where applicable; source contents are not logged
  - Performance-sensitive behavior: branch work is bounded by configured planning/CSG limits
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none.
- Source split/provenance notes: this child owns only the parent responsibilities listed below.

## Purpose

Decompose validated branching loft topology into bounded cut sub-bodies and recompose provenance only when lineage and branch seams are complete.

## Scope

- Owns:
  - branch-graph eligibility and bounded decomposition
  - sub-body provenance and cutter routing
  - validated recomposition map for result-shell assembly

- Does not own:
  - patch intersection/fragments, owned by Fix 08a
  - final surface shell assembly, owned by Fix 08c

## Split Coverage

- Parent spec: `fix-08-loft-surface-difference-cut-execution-v1_0.md`
- Parent coverage status: 100% covered collectively by the parent's listed children
- Parent responsibilities owned by this child:
  - branch-graph eligibility and bounded decomposition
  - sub-body provenance and cutter routing
  - validated recomposition map for result-shell assembly
- Parent responsibilities still missing from children:
  - none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../spec-refinement-history/a4-open-issues-20260804-165103.md` | 4 | nineteen-leaf active set | none | reached |

Pass 4 split decision: retained. Cohesion reason: one branch-plan decomposition and recomposition-map outcome.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` with lineage read from `src/impression/modeling/loft.py` - implementation and wiring owner
- Supporting modules/files:
  - use only the dependencies named below
- GUI/QML files, if applicable:
  - not applicable
- Reusable library/module files:
  - `src/impression/modeling/csg.py` with lineage read from `src/impression/modeling/loft.py` - extend the existing reusable boundary
- Tests:
  - `tests/test_surface_csg.py` branch fixtures
  - audio-cube branched cutter regression

## Chosen Defaults / Parameters

- underconstrained branch graphs refuse
- decomposition preserves operand and topology provenance

## Data Ownership

- Source of truth: `src/impression/modeling/csg.py` with lineage read from `src/impression/modeling/loft.py`
- Read ownership: the named caller route reads immutable request/plan/surface evidence through the existing public boundary.
- Write ownership: `src/impression/modeling/csg.py` with lineage read from `src/impression/modeling/loft.py` creates only its derived records/results; input source and operands remain unchanged.
- Derived/cache data: all records are recomputable from the declared inputs.
- Privacy/logging constraints: do not log model source contents; otherwise not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - loft topology lineage
  - CSG route registry
  - configured bounds
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
  - Fix 05b synthetic station lineage
  - existing records/routes named under Dependencies And Routes
- Missing prerequisite architecture:
  - none
- Missing prerequisite specifications:
  - none
- Unimplemented prerequisite specifications:
  - none; Fix 05b synthetic station lineage is implemented
- Progression handling:
  - prerequisites listed above run first; otherwise this child may proceed after canonical review

## Application Integration

- App type: library-only
- User/caller surface: branching loft eligibility consumed by difference execution
- Invocation route: branching loft eligibility consumed by difference execution -> `src/impression/modeling/csg.py` with lineage read from `src/impression/modeling/loft.py` -> validated sub-body cut plan and recomposition map
- Wiring owner/module: `src/impression/modeling/csg.py` with lineage read from `src/impression/modeling/loft.py`
- Observable result: validated sub-body cut plan and recomposition map
- Integration validation: `tests/test_surface_csg.py` branch fixtures; audio-cube branched cutter regression
- Incomplete status risk: completion requires the caller route and paired test specification to pass

App-type-specific proof:

- GUI: not applicable
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: branching loft eligibility consumed by difference execution is the consuming route

## Reuse And Extraction Plan

- Existing code to reuse:
  - loft topology lineage
  - CSG route registry
  - configured bounds
- Current reuse readiness:
  - add responsibilities to the existing owner module
- Extraction/wrapping needed:
  - only the named DTO/helper boundary; no parallel subsystem
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` with lineage read from `src/impression/modeling/loft.py` - branch graph validator, bounded sub-body decomposition, recomposition-map validator
- New reusable modules to expose:
  - none
- One-off code justification, if any:
  - none

## Required DTOs / Functions / Components

- DTOs/models:
  - branch decomposition/recomposition record
- Functions/methods:
  - branch graph validator
  - bounded sub-body decomposition
  - recomposition-map validator
- UI fields / visible data, if applicable:
  - not applicable
- UI elements / controls, if applicable:
  - not applicable
- UI components, if applicable:
  - none

## Performance Contract

- branch work is bounded by configured planning/CSG limits

## Error And State Behavior

- invalid lineage, duplicate ownership, or open recomposition seam refuses

## Test Strategy

- Unit tests:
  - `tests/test_surface_csg.py` branch fixtures
  - audio-cube branched cutter regression
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - not applicable
- Integrated route tests:
  - branching loft eligibility consumed by difference execution must exercise validated sub-body cut plan and recomposition map
- Production-data rule:
  - deterministic project fixtures and temporary directories only

## Acceptance Criteria

- branch-graph eligibility and bounded decomposition is implemented and asserted through the declared route.
- sub-body provenance and cutter routing is implemented and asserted through the declared route.
- validated recomposition map for result-shell assembly is implemented and asserted through the declared route.
- The paired test specification [Loft Difference Branch Decomposition Test](../test-specifications/fix-08b-loft-difference-branch-decomposition-v1_0.md) passes without helper-only substitution.

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
