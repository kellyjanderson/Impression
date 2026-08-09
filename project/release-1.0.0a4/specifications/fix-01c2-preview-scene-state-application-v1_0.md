# Fix 01C2: Preview Scene State Application

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)
Architecture ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)
Source artifact: [Split parent](./fix-01c-preview-refresh-route-integration-v1_0.md)
Split provenance: `fix-01c-preview-refresh-route-integration-v1_0.md` from GitHub issue #242
Canonical status: Archived
Review Score: 26
Prerequisites:
- Fix 01a request generations
- Fix 01c1 input wiring for integrated route proof

## Source Field Carryover

- Source purpose: Apply only current-generation build results on the UI/render thread while preserving camera and last-good scene across stale or failed builds.
- Source responsibilities by category:
  - Functions/methods: current-result acceptance check, scene/status state application
  - Data structures/models: preview applied-generation and last-good state
  - Dependencies/services: Fix 01a generation state, existing preview scene controller/render-thread timer
  - Returns/outputs/signals: current visible scene, preserved camera/last-good scene, visible error and recovery status
  - UI surfaces/components: preview window scene, camera, and status/error state
  - UI fields/elements: preview scene, camera, and status/error output
  - Reusable code plan: extend the existing preview/CLI boundaries named below
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: only UI/render thread mutates scene state
  - Destructive/write behavior: no destructive writes
  - Security/privacy-sensitive behavior: do not log model source contents
  - Performance-sensitive behavior: result acceptance is constant-time before existing scene application
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none.
- Source split/provenance notes: this child owns the listed half of Fix 01c.

## Purpose

Apply only current-generation build results on the UI/render thread while preserving camera and last-good scene across stale or failed builds.

## Scope

- Owns:
  - current-generation result acceptance and UI-thread scene apply
  - camera and last-good scene preservation
  - visible error/recovery state and shutdown rejection of late results

- Does not own:
  - input wiring, owned by Fix 01c1
  - module loading/build execution, owned by Fix 01a and Fix 01b

## Split Coverage

- Split parent: this specification
- Parent coverage status: 100% covered
- Coverage matrix:
  - `fix-01c2a-preview-current-generation-scene-apply-v1_0.md` - Covered: generation admission, UI-thread apply, stale and post-shutdown rejection.
  - `fix-01c2b-preview-last-good-camera-error-state-v1_0.md` - Covered: camera, last-good scene, error/recovery, and late-result state protection.
- Parent responsibilities still missing from children:
  - none
- Parent disposition: Archived after both children completed fresh review and canonicalization.

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../spec-refinement-history/a4-open-issues-20260804-165103.md` | 3 | eighteen-leaf active set | Fix 01c2a and Fix 01c2b | continue |

Pass 3 split decision: forced split into Fix 01c2a and Fix 01c2b.

## Implementation Routing

- Primary modules/files:
  - `src/impression/preview.py` - implementation/wiring owner
- Supporting modules/files:
  - only declared dependencies
- GUI/QML files, if applicable:
  - no QML; existing Python preview window
- Reusable library/module files:
  - `src/impression/preview.py`
- Tests:
  - `tests/test_preview_controller.py` state transitions
  - offscreen preview scene/camera/error smoke

## Chosen Defaults / Parameters

- stale results never apply
- ordinary rebuild preserves camera
- failure preserves last-good scene and subsequent success clears the error

## Data Ownership

- Source of truth: `src/impression/preview.py`
- Read ownership: declared event/result route.
- Write ownership: only the preview owner mutates request or visible state assigned to this child.
- Derived/cache data: recomputable from coordinator generation and build result.
- Privacy/logging constraints: source contents are not logged.

## Dependencies And Routes

- Domain/service dependencies:
  - Fix 01a generation state
  - existing preview scene controller/render-thread timer
- Database dependencies:
  - none
- GUI route, if applicable:
  - preview window scene, camera, and status/error state -> `src/impression/preview.py`
- Background/concurrency route, if applicable:
  - only UI/render thread mutates scene state

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
  - Fix 01a request generations
  - Fix 01c1 input wiring for integrated route proof
- Progression handling:
  - listed prerequisites run first

## Application Integration

- App type: GUI
- User/caller surface: preview window scene, camera, and status/error state
- Invocation route: preview window scene, camera, and status/error state -> `src/impression/preview.py`
- Wiring owner/module: `src/impression/preview.py`
- Observable result: current visible scene; preserved camera/last-good scene; visible error and recovery status
- Integration validation: `tests/test_preview_controller.py` state transitions; offscreen preview scene/camera/error smoke
- Incomplete status risk: completion requires the integrated route and paired test contract

App-type-specific proof:

- GUI: preview window scene, camera, and status/error state event/state proof
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: not applicable

## Reuse And Extraction Plan

- Existing code to reuse:
  - Fix 01a generation state
  - existing preview scene controller/render-thread timer
- Current reuse readiness:
  - add to existing preview/CLI modules
- Extraction/wrapping needed:
  - only the two named route/state methods
- Additions to existing library/modules:
  - `src/impression/preview.py` - current-result acceptance check, scene/status state application
- New reusable modules to expose:
  - none
- One-off code justification, if any:
  - none

## Required DTOs / Functions / Components

- DTOs/models:
  - preview applied-generation and last-good state
- Functions/methods:
  - current-result acceptance check
  - scene/status state application
- UI fields / visible data, if applicable:
  - scene, camera, status/error
- UI elements / controls, if applicable:
  - existing preview surface
- UI components, if applicable:
  - existing preview window

## Performance Contract

- result acceptance is constant-time before existing scene application

## Error And State Behavior

- stale/failed results cannot replace the scene
- shutdown ignores late completions

## Test Strategy

- Unit tests:
  - `tests/test_preview_controller.py` state transitions
  - offscreen preview scene/camera/error smoke
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - offscreen/real route proof through the preview window
- Integrated route tests:
  - preview window scene, camera, and status/error state asserts current visible scene; preserved camera/last-good scene; visible error and recovery status
- Production-data rule:
  - temporary model modules and deterministic fixtures only

## Acceptance Criteria

- current-generation result acceptance and UI-thread scene apply is implemented and asserted.
- camera and last-good scene preservation is implemented and asserted.
- visible error/recovery state and shutdown rejection of late results is implemented and asserted.
- The paired test spec passes through the real route.

## Readiness Checklist

- [x] Ancestors, split provenance, ownership, routes, defaults, data, reuse, prerequisites, performance, privacy, and tests are explicit.
- [ ] Numeric Review Score is supplied by the next fresh child pass.
- [x] Parent coverage is 100% across Fix 01c1 and Fix 01c2.
- [ ] Child is independently rescored and canonicalized after pass 2 readback.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: pending child review; adversarial input, not trusted.
- Adversarial rescore basis: fresh recount checked UI field inventory, renderer-thread state, prerequisites, reuse, performance, and deferral markers.
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 3 x 1 = 3
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
- Missing prerequisites: 2 x 2 = 4
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 26
- If total matches prior score, adversarial survival reason: not applicable; this pass replaced a nonnumeric or different prior score.
