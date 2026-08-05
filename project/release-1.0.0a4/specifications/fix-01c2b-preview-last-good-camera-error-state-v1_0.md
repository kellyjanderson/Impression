# Fix 01C2B: Preview Last-Good Camera And Error State

Date: 2026-08-04
Status: Final
Primary ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)
Architecture ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)
Source artifact: [Fix 01c2 split parent](./fix-01c2-preview-scene-state-application-v1_0.md)
Split provenance: `fix-01c2-preview-scene-state-application-v1_0.md` from GitHub issue #242
Canonical status: Canonical
Review Score: 20
Prerequisites:
- Fix 01c2a current-generation scene apply

## Source Field Carryover

- Source purpose: Preserve camera and last-good scene across build failure while exposing error/recovery state and preventing late results from corrupting recovered state.
- Source responsibilities by category:
  - Functions/methods: successful-state commit, failure/recovery state transition
  - Data structures/models: last-good scene, camera snapshot, and visible error state
  - Dependencies/services: Fix 01c2a admitted result, existing scene controller/status surface
  - Returns/outputs/signals: preserved camera/last-good scene, visible error and recovery status
  - UI surfaces/components: preview window
  - UI fields/elements: camera and status/error output
  - Reusable code plan: extend existing preview controller/scene state
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: state transitions occur on the UI/render thread after generation admission
  - Destructive/write behavior: no destructive writes
  - Security/privacy-sensitive behavior: source contents are not logged
  - Performance-sensitive behavior: not applicable
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none.
- Source split/provenance notes: this child owns only the listed Fix 01c2 state boundary.

## Purpose

Preserve camera and last-good scene across build failure while exposing error/recovery state and preventing late results from corrupting recovered state.

## Scope

- Owns:
  - camera preservation across ordinary rebuilds
  - last-good scene retention on failure
  - visible error, subsequent recovery, and late-result state protection

- Does not own:
  - generation admission and renderer mutation boundary, owned by Fix 01c2a

## Split Coverage

- Parent spec: `fix-01c2-preview-scene-state-application-v1_0.md`
- Parent coverage status: 100% covered collectively by Fix 01c2a and Fix 01c2b
- Parent responsibilities owned by this child:
  - camera preservation across ordinary rebuilds
  - last-good scene retention on failure
  - visible error, subsequent recovery, and late-result state protection
- Parent responsibilities still missing from children:
  - none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../spec-refinement-history/a4-open-issues-20260804-165103.md` | 4 | nineteen-leaf active set | none | reached |

Pass 4 split decision: retained. Cohesion reason: one last-good/camera/error state machine and recovery outcome.

## Implementation Routing

- Primary modules/files:
  - `src/impression/preview.py` - Preserve camera and last-good scene across build failure while exposing error/recovery state and preventing late results from corrupting recovered state.
- Supporting modules/files:
  - existing scene controller only
- GUI/QML files, if applicable:
  - no QML; existing Python preview window
- Reusable library/module files:
  - `src/impression/preview.py`
- Tests:
  - `tests/test_preview_controller.py` camera/error/recovery tests
  - offscreen preview failure/recovery smoke

## Chosen Defaults / Parameters

- ordinary rebuild preserves camera
- failure retains last-good scene
- successful subsequent build clears the error

## Data Ownership

- Source of truth: `src/impression/preview.py` preview state.
- Read ownership: UI/render-thread result/state handler.
- Write ownership: UI/render thread alone mutates the state owned by this child.
- Derived/cache data: recomputable from admitted build results and last-good state.
- Privacy/logging constraints: source contents are not logged.

## Dependencies And Routes

- Domain/service dependencies:
  - Fix 01c2a admitted result
  - existing scene controller/status surface
- Database dependencies:
  - none
- GUI route, if applicable:
  - background completion -> UI-thread preview state handler -> preview window
- Background/concurrency route, if applicable:
  - state transitions occur on the UI/render thread after generation admission

## Prerequisite Handling

- Architecture feedback artifacts:
  - `../architecture/acd-preview-reload-coordination.md` - state ownership resolved
- Architecture feedback status:
  - tracked in active ACD
- Already implemented prerequisites:
  - existing preview scene controller and UI-thread timer
- Missing prerequisite architecture:
  - none
- Missing prerequisite specifications:
  - none
- Unimplemented prerequisite specifications:
  - Fix 01c2a current-generation scene apply
- Progression handling:
  - prerequisite runs first

## Application Integration

- App type: GUI
- User/caller surface: preview window
- Invocation route: admitted build result -> UI-thread state handler -> preview window
- Wiring owner/module: `src/impression/preview.py`
- Observable result: preserved camera/last-good scene; visible error and recovery status
- Integration validation: `tests/test_preview_controller.py` camera/error/recovery tests; offscreen preview failure/recovery smoke
- Incomplete status risk: completion requires offscreen/real preview route proof

App-type-specific proof:

- GUI: visible scene/state and UI-thread handoff are asserted.
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: not applicable

## Reuse And Extraction Plan

- Existing code to reuse:
  - Fix 01c2a admitted result
  - existing scene controller/status surface
- Current reuse readiness:
  - add to existing preview module
- Extraction/wrapping needed:
  - only the named state methods
- Additions to existing library/modules:
  - `src/impression/preview.py` - successful-state commit, failure/recovery state transition
- New reusable modules to expose:
  - none
- One-off code justification, if any:
  - none

## Required DTOs / Functions / Components

- DTOs/models:
  - last-good scene, camera snapshot, and visible error state
- Functions/methods:
  - successful-state commit
  - failure/recovery state transition
- UI fields / visible data, if applicable:
  - camera and status/error
- UI elements / controls, if applicable:
  - existing preview surface
- UI components, if applicable:
  - existing preview window

## Performance Contract

- not applicable

## Error And State Behavior

- failure never clears the scene
- late completion cannot overwrite recovered state

## Test Strategy

- Unit tests:
  - `tests/test_preview_controller.py` camera/error/recovery tests
  - offscreen preview failure/recovery smoke
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - offscreen scene/state and UI-thread handoff
- Integrated route tests:
  - admitted result through the preview window
- Production-data rule:
  - deterministic build-result fixtures only

## Acceptance Criteria

- camera preservation across ordinary rebuilds is implemented and asserted.
- last-good scene retention on failure is implemented and asserted.
- visible error, subsequent recovery, and late-result state protection is implemented and asserted.
- Paired test specification passes through the preview route.

## Readiness Checklist

- [x] Ancestors, parent coverage, owner, route, state, defaults, concurrency, reuse, prerequisites, and tests are explicit.
- [x] Numeric Review Score was supplied by the terminal fresh child pass.
- [x] Child was independently rescored and canonicalized after pass 3 readback.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: pending child review; adversarial input, not trusted.
- Adversarial rescore basis: fresh terminal recount checked split lineage, UI/control inventory, routes, reuse, prerequisites, writes, concurrency, performance, and deferral markers.
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 0 x 2 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 1 x 2 = 2
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 20
- If total matches prior score, adversarial survival reason: not applicable; this pass supplied the first numeric score.
