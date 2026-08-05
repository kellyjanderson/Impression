# Fix 01C1: Preview Refresh Input Wiring

Date: 2026-08-04
Status: Final
Primary ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)
Architecture ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)
Source artifact: [Split parent](./fix-01c-preview-refresh-route-integration-v1_0.md)
Split provenance: `fix-01c-preview-refresh-route-integration-v1_0.md` from GitHub issue #242
Canonical status: Canonical
Review Score: 22
Prerequisites:
- Fix 01a preview watch request coordination
- Fix 01b preview module cache invalidation

## Source Field Carryover

- Source purpose: Wire saved-file and `R` events into the typed coordinator and forced module generation without owning build completion or scene state.
- Source responsibilities by category:
  - Functions/methods: save-event route adapter, `R` forced-refresh route adapter
  - Data structures/models: not applicable
  - Dependencies/services: Fix 01a coordinator, Fix 01b generation callback
  - Returns/outputs/signals: typed automatic or forced reload request
  - UI surfaces/components: saved-file events and the existing preview-window `R` binding
  - UI fields/elements: existing `R` binding
  - Reusable code plan: extend the existing preview/CLI boundaries named below
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: event callbacks submit only and perform no model build
  - Destructive/write behavior: no destructive writes
  - Security/privacy-sensitive behavior: do not log model source contents
  - Performance-sensitive behavior: input wiring adds no polling delay
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none.
- Source split/provenance notes: this child owns the listed half of Fix 01c.

## Purpose

Wire saved-file and `R` events into the typed coordinator and forced module generation without owning build completion or scene state.

## Scope

- Owns:
  - save-event submission into Fix 01a
  - `R` event submission with forced intent and Fix 01b generation advance
  - command/window route tests proving each input reaches the intended boundary

- Does not own:
  - request coalescing internals, owned by Fix 01a
  - module eviction internals, owned by Fix 01b
  - scene application and recovery state, owned by Fix 01c2

## Split Coverage

- Parent spec: `fix-01c-preview-refresh-route-integration-v1_0.md`
- Parent coverage status: 100% covered collectively by Fix 01c1 and Fix 01c2
- Parent responsibilities owned by this child:
  - save-event submission into Fix 01a
  - `R` event submission with forced intent and Fix 01b generation advance
  - command/window route tests proving each input reaches the intended boundary
- Parent responsibilities still missing from children:
  - none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../spec-refinement-history/a4-open-issues-20260804-165103.md` | 4 | nineteen-leaf active set | none | reached |

Pass 4 split decision: retained. Cohesion reason: one event-to-request wiring boundary across the coordinated command/window route.

## Implementation Routing

- Primary modules/files:
  - `src/impression/preview.py` with `src/impression/cli.py` generation callback - implementation/wiring owner
- Supporting modules/files:
  - only declared dependencies
- GUI/QML files, if applicable:
  - no QML; existing Python preview window
- Reusable library/module files:
  - `src/impression/preview.py` with `src/impression/cli.py` generation callback
- Tests:
  - `tests/test_preview_controller.py` input route
  - `tests/test_cli_preview.py` generation callback
  - offscreen/real command key-event smoke

## Chosen Defaults / Parameters

- `R` always sets forced intent
- save events use automatic intent
- both routes share one coordinator boundary

## Data Ownership

- Source of truth: `src/impression/preview.py` with `src/impression/cli.py` generation callback
- Read ownership: declared event/result route.
- Write ownership: only the preview owner mutates request or visible state assigned to this child.
- Derived/cache data: recomputable from coordinator generation and build result.
- Privacy/logging constraints: source contents are not logged.

## Dependencies And Routes

- Domain/service dependencies:
  - Fix 01a coordinator
  - Fix 01b generation callback
- Database dependencies:
  - none
- GUI route, if applicable:
  - saved-file events and the existing preview-window `R` binding -> `src/impression/preview.py` with `src/impression/cli.py` generation callback
- Background/concurrency route, if applicable:
  - event callbacks submit only and perform no model build

## Prerequisite Handling

- Architecture feedback artifacts:
  - `../architecture/acd-preview-reload-coordination.md` - route ownership resolved
- Architecture feedback status:
  - tracked in active ACD
- Already implemented prerequisites:
  - existing preview window, scene controller, and CLI command
- Missing prerequisite architecture:
  - none
- Missing prerequisite specifications:
  - none
- Unimplemented prerequisite specifications:
  - Fix 01a preview watch request coordination
  - Fix 01b preview module cache invalidation
- Progression handling:
  - listed prerequisites run first

## Application Integration

- App type: mixed
- User/caller surface: saved-file events and the existing preview-window `R` binding
- Invocation route: saved-file events and the existing preview-window `R` binding -> `src/impression/preview.py` with `src/impression/cli.py` generation callback
- Wiring owner/module: `src/impression/preview.py` with `src/impression/cli.py` generation callback
- Observable result: typed automatic or forced reload request
- Integration validation: `tests/test_preview_controller.py` input route; `tests/test_cli_preview.py` generation callback; offscreen/real command key-event smoke
- Incomplete status risk: completion requires the integrated route and paired test contract

App-type-specific proof:

- GUI: saved-file events and the existing preview-window `R` binding event/state proof
- Console: real `impression preview` command route
- API/service: not applicable
- Mixed: event submission and command callback asserted separately
- Library-only: not applicable

## Reuse And Extraction Plan

- Existing code to reuse:
  - Fix 01a coordinator
  - Fix 01b generation callback
- Current reuse readiness:
  - add to existing preview/CLI modules
- Extraction/wrapping needed:
  - only the two named route/state methods
- Additions to existing library/modules:
  - `src/impression/preview.py` with `src/impression/cli.py` generation callback - save-event route adapter, `R` forced-refresh route adapter
- New reusable modules to expose:
  - none
- One-off code justification, if any:
  - none

## Required DTOs / Functions / Components

- DTOs/models:
  - not applicable
- Functions/methods:
  - save-event route adapter
  - `R` forced-refresh route adapter
- UI fields / visible data, if applicable:
  - existing `R` binding
- UI elements / controls, if applicable:
  - existing `R` key
- UI components, if applicable:
  - existing preview window

## Performance Contract

- input wiring adds no polling delay

## Error And State Behavior

- unavailable/shutting-down coordinator rejects the event visibly without mutating scene state

## Test Strategy

- Unit tests:
  - `tests/test_preview_controller.py` input route
  - `tests/test_cli_preview.py` generation callback
  - offscreen/real command key-event smoke
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - offscreen/real route proof through the preview window
- Integrated route tests:
  - saved-file events and the existing preview-window `R` binding asserts typed automatic or forced reload request
- Production-data rule:
  - temporary model modules and deterministic fixtures only

## Acceptance Criteria

- save-event submission into Fix 01a is implemented and asserted.
- `R` event submission with forced intent and Fix 01b generation advance is implemented and asserted.
- command/window route tests proving each input reaches the intended boundary is implemented and asserted.
- The paired test spec passes through the real route.

## Readiness Checklist

- [x] Ancestors, split provenance, ownership, routes, defaults, data, reuse, prerequisites, performance, privacy, and tests are explicit.
- [x] Numeric Review Score was supplied by the terminal fresh child pass.
- [x] Parent coverage is 100% across Fix 01c1 and Fix 01c2.
- [x] Child was independently rescored and canonicalized after pass 2 readback.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: 22; adversarial input, not trusted.
- Adversarial rescore basis: fresh terminal recount checked split lineage, UI/control inventory, routes, reuse, prerequisites, writes, concurrency, performance, and deferral markers.
- Functions/methods: 2 x 2 = 4
- Data structures/models: 0 x 1 = 0
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 2 x 2 = 4
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 22
- If total matches prior score, adversarial survival reason: the score survived a complete terminal recount; no omitted responsibility, blocker, or route was found.
