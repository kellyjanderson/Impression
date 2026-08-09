# Fix 01A: Preview Watch Request Coordination

Date: 2026-08-04
Status: Final
Primary ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)
Architecture ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)
Source artifact: [Split parent](./fix-01-preview-watch-and-forced-refresh-v1_0.md)
Split provenance: `fix-01-preview-watch-and-forced-refresh-v1_0.md` from GitHub issue #242
Canonical status: Canonical
Review Score: 23.5
Prerequisites:
- none

## Source Field Carryover

- Source purpose: Normalize filesystem events into bounded latest-wins build requests with one active build, one retained replacement, and a 250 ms delivery bound.
- Source responsibilities by category:
  - Functions/methods: `submit_reload(request)`, `begin_next_build()`, `complete_build(generation)`
  - Data structures/models: `ReloadRequest(reason, force, changed_paths, generation)`
  - Dependencies/services: watchdog adapter, existing preview build executor, watched-path provider
  - Returns/outputs/signals: one current build request or one latest replacement
  - UI surfaces/components: not applicable
  - UI fields/elements: not applicable
  - Reusable code plan: add the named responsibilities to existing owner modules
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: watcher callbacks submit only; one background build is active; stale completion cannot consume newer replacement
  - Destructive/write behavior: no destructive runtime writes
  - Security/privacy-sensitive behavior: local paths may appear in diagnostics where applicable; source contents are not logged
  - Performance-sensitive behavior: 250 ms pre-build delivery bound; constant request storage
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none.
- Source split/provenance notes: this child owns only the parent responsibilities listed below.

## Purpose

Normalize filesystem events into bounded latest-wins build requests with one active build, one retained replacement, and a 250 ms delivery bound.

## Scope

- Owns:
  - watcher event normalization and dependency-path matching
  - one-active/one-latest request state and forced-intent-preserving merge
  - filesystem-event to build-submission timing and burst behavior

- Does not own:
  - Python module cache invalidation, owned by Fix 01b
  - keyboard/visible scene integration, owned by Fix 01c

## Split Coverage

- Parent spec: `fix-01-preview-watch-and-forced-refresh-v1_0.md`
- Parent coverage status: 100% covered collectively by the parent's listed children
- Parent responsibilities owned by this child:
  - watcher event normalization and dependency-path matching
  - one-active/one-latest request state and forced-intent-preserving merge
  - filesystem-event to build-submission timing and burst behavior
- Parent responsibilities still missing from children:
  - none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../spec-refinement-history/a4-open-issues-20260804-165103.md` | 4 | nineteen-leaf active set | none | reached |

Pass 4 split decision: retained. Cohesion reason: one bounded request-state machine owns watcher delivery and burst behavior.

## Implementation Routing

- Primary modules/files:
  - `src/impression/preview.py` - implementation and wiring owner
- Supporting modules/files:
  - use only the dependencies named below
- GUI/QML files, if applicable:
  - not applicable
- Reusable library/module files:
  - `src/impression/preview.py` - extend the existing reusable boundary
- Tests:
  - `tests/test_preview_controller.py`
  - real temporary-filesystem watcher fixture

## Chosen Defaults / Parameters

- request storage is O(1)
- duplicate events coalesce by latest generation while preserving force
- eligible local events reach build submission within 250 ms

## Data Ownership

- Source of truth: `src/impression/preview.py`
- Read ownership: the named caller route reads immutable request/plan/surface evidence through the existing public boundary.
- Write ownership: `src/impression/preview.py` creates only its derived records/results; input source and operands remain unchanged.
- Derived/cache data: all records are recomputable from the declared inputs.
- Privacy/logging constraints: do not log model source contents; otherwise not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - watchdog adapter
  - existing preview build executor
  - watched-path provider
- Database dependencies:
  - none
- GUI route, if applicable:
  - not applicable
- Background/concurrency route, if applicable:
  - watcher callbacks submit only; one background build is active; stale completion cannot consume newer replacement

## Prerequisite Handling

- Architecture feedback artifacts:
  - `../architecture/acd-preview-reload-coordination.md` - target ownership and route are resolved
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
- User/caller surface: preview watcher and build scheduler consumed by `PyVistaPreviewer`
- Invocation route: preview watcher and build scheduler consumed by `PyVistaPreviewer` -> `src/impression/preview.py` -> one current build request or one latest replacement
- Wiring owner/module: `src/impression/preview.py`
- Observable result: one current build request or one latest replacement
- Integration validation: `tests/test_preview_controller.py`; real temporary-filesystem watcher fixture
- Incomplete status risk: completion requires the caller route and paired test specification to pass

App-type-specific proof:

- GUI: not applicable
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: preview watcher and build scheduler consumed by `PyVistaPreviewer` is the consuming route

## Reuse And Extraction Plan

- Existing code to reuse:
  - watchdog adapter
  - existing preview build executor
  - watched-path provider
- Current reuse readiness:
  - add responsibilities to the existing owner module
- Extraction/wrapping needed:
  - only the named DTO/helper boundary; no parallel subsystem
- Additions to existing library/modules:
  - `src/impression/preview.py` - `submit_reload(request)`, `begin_next_build()`, `complete_build(generation)`
- New reusable modules to expose:
  - none
- One-off code justification, if any:
  - none

## Required DTOs / Functions / Components

- DTOs/models:
  - `ReloadRequest(reason, force, changed_paths, generation)`
- Functions/methods:
  - `submit_reload(request)`
  - `begin_next_build()`
  - `complete_build(generation)`
- UI fields / visible data, if applicable:
  - not applicable
- UI elements / controls, if applicable:
  - not applicable
- UI components, if applicable:
  - none

## Performance Contract

- 250 ms pre-build delivery bound
- constant request storage

## Error And State Behavior

- watcher errors are reported without destroying request state
- shutdown rejects new requests and prevents stale apply

## Test Strategy

- Unit tests:
  - `tests/test_preview_controller.py`
  - real temporary-filesystem watcher fixture
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - not applicable
- Integrated route tests:
  - preview watcher and build scheduler consumed by `PyVistaPreviewer` must exercise one current build request or one latest replacement
- Production-data rule:
  - deterministic project fixtures and temporary directories only

## Acceptance Criteria

- watcher event normalization and dependency-path matching is implemented and asserted through the declared route.
- one-active/one-latest request state and forced-intent-preserving merge is implemented and asserted through the declared route.
- filesystem-event to build-submission timing and burst behavior is implemented and asserted through the declared route.
- The paired test specification [Preview Watch Request Coordination Test](../test-specifications/fix-01a-preview-watch-request-coordination-v1_0.md) passes without helper-only substitution.

## Readiness Checklist

- [x] Ancestors, source parent, split provenance, and 100% collective parent coverage are explicit.
- [x] Numeric Review Score was supplied by the terminal fresh child-review pass.
- [x] Current template path is recorded below.
- [x] Architecture, owner, routes, defaults, data, reuse, prerequisites, performance, privacy, tests, and acceptance are explicit.
- [x] No parent responsibility owned by this child remains only in the parent.
- [x] Child was independently rescored and canonicalized after ledger pass 1 readback.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: 23.5; adversarial input, not trusted.
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
- Async/concurrency behavior: 2 x 3 = 6
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 2 x 2 = 4
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 23.5
- If total matches prior score, adversarial survival reason: the score survived a complete terminal recount; no omitted responsibility, blocker, or route was found.
