# Fix 01B: Preview Module Cache Invalidation

Date: 2026-08-04
Status: Final
Primary ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)
Architecture ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)
Source artifact: [Split parent](./fix-01-preview-watch-and-forced-refresh-v1_0.md)
Split provenance: `fix-01-preview-watch-and-forced-refresh-v1_0.md` from GitHub issue #242
Canonical status: Canonical
Review Score: 17
Prerequisites:
- Fix 01a supplies typed forced intent

## Source Field Carryover

- Source purpose: Make each forced reload generation invalidate the entry model and transitive local Python dependencies even when mtime evidence is unchanged.
- Source responsibilities by category:
  - Functions/methods: `advance_reload_generation()`, generation-aware scene factory load, local dependency rediscovery/eviction
  - Data structures/models: reload generation and cached local-module set
  - Dependencies/services: Python `sys.modules`, CLI scene factory
  - Returns/outputs/signals: fresh model module and updated watched paths
  - UI surfaces/components: not applicable
  - UI fields/elements: not applicable
  - Reusable code plan: add the named responsibilities to existing owner modules
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: loader state is read on the background build lane
  - Destructive/write behavior: no destructive runtime writes
  - Security/privacy-sensitive behavior: local paths may appear in diagnostics where applicable; source contents are not logged
  - Performance-sensitive behavior: cache work is bounded to the discovered local module graph
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none.
- Source split/provenance notes: this child owns only the parent responsibilities listed below.

## Purpose

Make each forced reload generation invalidate the entry model and transitive local Python dependencies even when mtime evidence is unchanged.

## Scope

- Owns:
  - monotonic forced-reload generation
  - entry/transitive module discovery and cache eviction
  - mtime-neutral forced refresh and watched-path rediscovery

- Does not own:
  - watcher request coalescing, owned by Fix 01a
  - visible `R` and scene-apply state, owned by Fix 01c

## Split Coverage

- Parent spec: `fix-01-preview-watch-and-forced-refresh-v1_0.md`
- Parent coverage status: 100% covered collectively by the parent's listed children
- Parent responsibilities owned by this child:
  - monotonic forced-reload generation
  - entry/transitive module discovery and cache eviction
  - mtime-neutral forced refresh and watched-path rediscovery
- Parent responsibilities still missing from children:
  - none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../spec-refinement-history/a4-open-issues-20260804-165103.md` | 4 | nineteen-leaf active set | none | reached |

Pass 4 split decision: retained. Cohesion reason: one loader cache-generation boundary owns transitive invalidation.

## Implementation Routing

- Primary modules/files:
  - `src/impression/cli.py` - implementation and wiring owner
- Supporting modules/files:
  - use only the dependencies named below
- GUI/QML files, if applicable:
  - not applicable
- Reusable library/module files:
  - `src/impression/cli.py` - extend the existing reusable boundary
- Tests:
  - `tests/test_cli_preview.py`
  - temporary entry/helper module fixture

## Chosen Defaults / Parameters

- forced generation change always reloads
- automatic reload may reuse cache only when generation and tracked file evidence are unchanged
- only project-local transitive modules are evicted

## Data Ownership

- Source of truth: `src/impression/cli.py`
- Read ownership: the named caller route reads immutable request/plan/surface evidence through the existing public boundary.
- Write ownership: `src/impression/cli.py` creates only its derived records/results; input source and operands remain unchanged.
- Derived/cache data: all records are recomputable from the declared inputs.
- Privacy/logging constraints: do not log model source contents; otherwise not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - Python `sys.modules`
  - CLI scene factory
- Database dependencies:
  - none
- GUI route, if applicable:
  - not applicable
- Background/concurrency route, if applicable:
  - loader state is read on the background build lane

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
  - Fix 01a supplies typed forced intent
- Progression handling:
  - prerequisites listed above run first; otherwise this child may proceed after canonical review

## Application Integration

- App type: library-only
- User/caller surface: CLI scene factory consumed by live preview
- Invocation route: CLI scene factory consumed by live preview -> `src/impression/cli.py` -> fresh model module and updated watched paths
- Wiring owner/module: `src/impression/cli.py`
- Observable result: fresh model module and updated watched paths
- Integration validation: `tests/test_cli_preview.py`; temporary entry/helper module fixture
- Incomplete status risk: completion requires the caller route and paired test specification to pass

App-type-specific proof:

- GUI: not applicable
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: CLI scene factory consumed by live preview is the consuming route

## Reuse And Extraction Plan

- Existing code to reuse:
  - Python `sys.modules`
  - CLI scene factory
- Current reuse readiness:
  - add responsibilities to the existing owner module
- Extraction/wrapping needed:
  - only the named DTO/helper boundary; no parallel subsystem
- Additions to existing library/modules:
  - `src/impression/cli.py` - `advance_reload_generation()`, generation-aware scene factory load, local dependency rediscovery/eviction
- New reusable modules to expose:
  - none
- One-off code justification, if any:
  - none

## Required DTOs / Functions / Components

- DTOs/models:
  - reload generation and cached local-module set
- Functions/methods:
  - `advance_reload_generation()`
  - generation-aware scene factory load
  - local dependency rediscovery/eviction
- UI fields / visible data, if applicable:
  - not applicable
- UI elements / controls, if applicable:
  - not applicable
- UI components, if applicable:
  - none

## Performance Contract

- cache work is bounded to the discovered local module graph

## Error And State Behavior

- failed import/build does not mark the new generation successfully loaded
- diagnostics may name local paths but never log source contents

## Test Strategy

- Unit tests:
  - `tests/test_cli_preview.py`
  - temporary entry/helper module fixture
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - not applicable
- Integrated route tests:
  - CLI scene factory consumed by live preview must exercise fresh model module and updated watched paths
- Production-data rule:
  - deterministic project fixtures and temporary directories only

## Acceptance Criteria

- monotonic forced-reload generation is implemented and asserted through the declared route.
- entry/transitive module discovery and cache eviction is implemented and asserted through the declared route.
- mtime-neutral forced refresh and watched-path rediscovery is implemented and asserted through the declared route.
- The paired test specification [Preview Module Cache Invalidation Test](../test-specifications/fix-01b-preview-module-cache-invalidation-v1_0.md) passes without helper-only substitution.

## Readiness Checklist

- [x] Ancestors, source parent, split provenance, and 100% collective parent coverage are explicit.
- [x] Numeric Review Score was supplied by the terminal fresh child-review pass.
- [x] Current template path is recorded below.
- [x] Architecture, owner, routes, defaults, data, reuse, prerequisites, performance, privacy, tests, and acceptance are explicit.
- [x] No parent responsibility owned by this child remains only in the parent.
- [x] Child was independently rescored and canonicalized after ledger pass 1 readback.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: 17; adversarial input, not trusted.
- Adversarial rescore basis: fresh terminal recount checked split lineage, UI/control inventory, routes, reuse, prerequisites, writes, concurrency, performance, and deferral markers.
- Functions/methods: 3 x 2 = 6
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 17
- If total matches prior score, adversarial survival reason: the score survived a complete terminal recount; no omitted responsibility, blocker, or route was found.
