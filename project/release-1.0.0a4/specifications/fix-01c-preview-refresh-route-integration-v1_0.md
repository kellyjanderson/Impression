# Fix 01C: Preview Refresh Route Integration

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)
Architecture ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)
Source artifact: [Split parent](./fix-01-preview-watch-and-forced-refresh-v1_0.md)
Split provenance: `fix-01-preview-watch-and-forced-refresh-v1_0.md` from GitHub issue #242
Canonical status: Archived
Review Score: 29.5
Prerequisites:
- Fix 01a request coordinator
- Fix 01b cache invalidation

## Source Field Carryover

- Source purpose: Wire watcher and `R` reload intent through the background loader/build lane to current-generation UI-thread scene application while preserving camera and last-good state.
- Source responsibilities by category:
  - Functions/methods: preview route wiring, current-generation scene apply/state update
  - Data structures/models: not applicable
  - Dependencies/services: Fix 01a coordinator, Fix 01b loader, existing scene controller/render-thread timer
  - Returns/outputs/signals: fresh visible scene, preserved camera/last-good scene, visible status and errors
  - UI surfaces/components: `impression preview` command, preview window, and existing `R` binding
  - UI fields/elements: existing `R` binding and preview status/error output
  - Reusable code plan: add the named responsibilities to existing owner modules
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: build/loader remain off UI thread; only current generation applies on UI thread
  - Destructive/write behavior: no destructive runtime writes
  - Security/privacy-sensitive behavior: local paths may appear in diagnostics where applicable; source contents are not logged
  - Performance-sensitive behavior: route adds no polling delay beyond Fix 01a event bound
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none.
- Source split/provenance notes: this child owns only the parent responsibilities listed below.

## Purpose

Wire watcher and `R` reload intent through the background loader/build lane to current-generation UI-thread scene application while preserving camera and last-good state.

## Scope

- Owns:
  - `R`-to-forced-generation wiring and save-to-request wiring
  - current-generation UI-thread scene apply and stale rejection
  - camera preservation, visible errors, last-good scene, recovery, and command shutdown

- Does not own:
  - request-state internals, owned by Fix 01a
  - module eviction internals, owned by Fix 01b

## Split Coverage

- Split parent: this specification
- Parent coverage status: 100% covered
- Coverage matrix:
  - `fix-01c1-preview-refresh-input-wiring-v1_0.md` - Covered: saved-file and `R` input wiring into Fix 01a/Fix 01b.
  - `fix-01c2a-preview-current-generation-scene-apply-v1_0.md` - Covered: current-generation UI-thread apply, stale-result rejection, and shutdown admission.
  - `fix-01c2b-preview-last-good-camera-error-state-v1_0.md` - Covered: camera, error, last-good, recovery, and late-result state protection.
- Parent responsibilities still missing from children:
  - none
- Parent disposition: Archived after all three final descendants completed fresh review and canonicalization.

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../spec-refinement-history/a4-open-issues-20260804-165103.md` | 2 | seventeen-leaf active set | Fix 01c1 and Fix 01c2 | continue |

Pass 2 split decision: forced split into Fix 01c1 and Fix 01c2.

## Implementation Routing

- Primary modules/files:
  - `src/impression/preview.py` with `src/impression/cli.py` wiring - implementation and wiring owner
- Supporting modules/files:
  - use only the dependencies named below
- GUI/QML files, if applicable:
  - none; the existing Python preview window owns the route
- Reusable library/module files:
  - `src/impression/preview.py` with `src/impression/cli.py` wiring - extend the existing reusable boundary
- Tests:
  - `tests/test_preview_controller.py`
  - `tests/test_cli_preview.py`
  - real command/offscreen preview smoke

## Chosen Defaults / Parameters

- ordinary rebuilds preserve camera
- failed/stale builds never replace the last good scene
- `R` always carries forced intent

## Data Ownership

- Source of truth: `src/impression/preview.py` with `src/impression/cli.py` wiring
- Read ownership: the named caller route reads immutable request/plan/surface evidence through the existing public boundary.
- Write ownership: `src/impression/preview.py` with `src/impression/cli.py` wiring creates only its derived records/results; input source and operands remain unchanged.
- Derived/cache data: all records are recomputable from the declared inputs.
- Privacy/logging constraints: do not log model source contents; otherwise not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - Fix 01a coordinator
  - Fix 01b loader
  - existing scene controller/render-thread timer
- Database dependencies:
  - none
- GUI route, if applicable:
  - `impression preview` command, preview window, and existing `R` binding -> `src/impression/preview.py` with `src/impression/cli.py` wiring
- Background/concurrency route, if applicable:
  - build/loader remain off UI thread; only current generation applies on UI thread

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
  - Fix 01a request coordinator
  - Fix 01b cache invalidation
- Progression handling:
  - prerequisites listed above run first; otherwise this child may proceed after canonical review

## Application Integration

- App type: mixed
- User/caller surface: `impression preview` command, preview window, and existing `R` binding
- Invocation route: `impression preview` command, preview window, and existing `R` binding -> `src/impression/preview.py` with `src/impression/cli.py` wiring -> fresh visible scene; preserved camera/last-good scene; visible status and errors
- Wiring owner/module: `src/impression/preview.py` with `src/impression/cli.py` wiring
- Observable result: fresh visible scene; preserved camera/last-good scene; visible status and errors
- Integration validation: `tests/test_preview_controller.py`; `tests/test_cli_preview.py`; real command/offscreen preview smoke
- Incomplete status risk: completion requires the caller route and paired test specification to pass

App-type-specific proof:

- GUI: existing preview event/state and UI-thread handoff
- Console: real `impression preview` command and status/error behavior
- API/service: not applicable
- Mixed: separate command/request and visible scene assertions
- Library-only: not applicable

## Reuse And Extraction Plan

- Existing code to reuse:
  - Fix 01a coordinator
  - Fix 01b loader
  - existing scene controller/render-thread timer
- Current reuse readiness:
  - add responsibilities to the existing owner module
- Extraction/wrapping needed:
  - only the named DTO/helper boundary; no parallel subsystem
- Additions to existing library/modules:
  - `src/impression/preview.py` with `src/impression/cli.py` wiring - preview route wiring, current-generation scene apply/state update
- New reusable modules to expose:
  - none
- One-off code justification, if any:
  - none

## Required DTOs / Functions / Components

- DTOs/models:
  - not applicable
- Functions/methods:
  - preview route wiring
  - current-generation scene apply/state update
- UI fields / visible data, if applicable:
  - existing status/error output
- UI elements / controls, if applicable:
  - existing `R` binding
- UI components, if applicable:
  - none

## Performance Contract

- route adds no polling delay beyond Fix 01a event bound

## Error And State Behavior

- error remains visible and a subsequent repair recovers
- shutdown prevents late scene application

## Test Strategy

- Unit tests:
  - `tests/test_preview_controller.py`
  - `tests/test_cli_preview.py`
  - real command/offscreen preview smoke
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - offscreen/real-route state and stale-result proof
- Integrated route tests:
  - `impression preview` command, preview window, and existing `R` binding must exercise fresh visible scene; preserved camera/last-good scene; visible status and errors
- Production-data rule:
  - deterministic project fixtures and temporary directories only

## Acceptance Criteria

- `R`-to-forced-generation wiring and save-to-request wiring is implemented and asserted through the declared route.
- current-generation UI-thread scene apply and stale rejection is implemented and asserted through the declared route.
- camera preservation, visible errors, last-good scene, recovery, and command shutdown is implemented and asserted through the declared route.
- The paired test specification [Preview Refresh Route Integration Test](../test-specifications/fix-01c-preview-refresh-route-integration-v1_0.md) passes without helper-only substitution.

## Readiness Checklist

- [x] Ancestors, source parent, split provenance, and 100% collective parent coverage are explicit.
- [ ] Numeric Review Score is supplied by the next fresh child-review pass.
- [x] Current template path is recorded below.
- [x] Architecture, owner, routes, defaults, data, reuse, prerequisites, performance, privacy, tests, and acceptance are explicit.
- [x] No parent responsibility owned by this child remains only in the parent.
- [ ] Child is independently rescored and canonicalized after ledger pass 1 readback.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: pending child review; adversarial input, not trusted.
- Adversarial rescore basis: recounted every category and checked hidden routing, reuse, UI/input inventory, concurrency, prerequisite scope, write behavior, performance, and deferral markers.
- Functions/methods: 2 x 2 = 4
- Data structures/models: 0 x 1 = 0
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 2 x 3 = 6
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 2 x 2 = 4
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 29.5
- If total matches prior score, adversarial survival reason: not applicable; this pass replaced a nonnumeric or different prior score.
