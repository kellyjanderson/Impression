# Fix 01: Preview Watch And Forced Refresh

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)
Architecture ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)
Source artifact: [GitHub issue #242](https://github.com/kellyjanderson/Impression/issues/242)
Split provenance: none
Canonical status: Archived
Review Score: 41
Prerequisites:
- none - the current `impression preview` command and existing renderer-thread scene-application contract are the baseline

## Source Field Carryover

- Source purpose: Restore the a3-missed live-preview contract by making filesystem delivery prompt and making `R` a definitive cache-invalidating rebuild and re-render.
- Source responsibilities by category:
  - Functions/methods: reload request submission/coalescing, module invalidation, dependency rediscovery, background build scheduling, UI-thread scene apply
  - Data structures/models: `ReloadRequest` plus a monotonic reload generation and latest-replacement state
  - Dependencies/services: watchdog/local filesystem events, Python module cache, executor and Qt/PyVista render handoff
  - Returns/outputs/signals: fresh visible scene, preserved camera, status/error output, retained last-good scene
  - UI surfaces/components: live preview window and `R` binding
  - UI fields/elements: status/error text; no new user-editable fields
  - Reusable code plan: extend the existing preview controller, watcher adapter, module loader, and scene application path
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: watcher and model builds remain off the UI/render thread; scene application remains on it
  - Destructive/write behavior: reads source files and module state; no destructive project writes
  - Security/privacy-sensitive behavior: local source paths may appear in diagnostics but source contents must not be logged
  - Performance-sensitive behavior: filesystem event to build submission is at most 250 ms, excluding build/render time
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none hidden; independent review may refine split cohesion and exact symbol names.
- Source split/provenance notes: none

## Purpose

Restore the a3-missed live-preview contract by making filesystem delivery prompt and making `R` a definitive cache-invalidating rebuild and re-render.

## Scope

- Owns:
  - typed reload intent for automatic and forced refresh
  - bounded one-active/one-latest request coalescing
  - transitive local-module discovery and generation-based invalidation
  - watcher-to-build latency measurement, camera preservation, failure retention, and real CLI/GUI route proof

- Does not own:
  - model construction or tessellation speed improvements
  - new preview controls, keybindings, or renderer replacement

## Split Coverage

- Split parent: this specification
- Parent coverage status: 100% covered
- Coverage matrix:
  - `fix-01a-preview-watch-request-coordination-v1_0.md` - Covered: watcher normalization, bounded request state, event latency, and bursts.
  - `fix-01b-preview-module-cache-invalidation-v1_0.md` - Covered: forced generation, entry/transitive cache eviction, mtime-neutral reload.
  - `fix-01c1-preview-refresh-input-wiring-v1_0.md` - Covered: saved-file and `R` input wiring into request coordination and cache invalidation.
  - `fix-01c2a-preview-current-generation-scene-apply-v1_0.md` - Covered: current-generation UI-thread scene application and stale-result rejection.
  - `fix-01c2b-preview-last-good-camera-error-state-v1_0.md` - Covered: camera preservation, last-good scene retention, and error/recovery state.
- Parent responsibilities still missing from children:
  - none
- Parent disposition: Archived after all five final descendants completed fresh review and canonicalization.

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../spec-refinement-history/a4-open-issues-20260804-165103.md` | 1 | Fixes 01-09 | Fix 01a, Fix 01b, and Fix 01c | continue |

Pass 1 split decision: forced split into Fix 01a, Fix 01b, and Fix 01c.

## Implementation Routing

- Primary modules/files:
  - `src/impression/preview.py` - reload coordinator, watcher event normalization, build replacement, key route, scene handoff
  - `src/impression/cli.py` - module discovery, cache generation, and CLI wiring
- Supporting modules/files:
  - none
- GUI/QML files, if applicable:
  - none; no QML is involved
- Reusable library/module files:
  - `src/impression/preview.py` - reload coordinator, watcher event normalization, build replacement, key route, scene handoff
- Tests:
  - `tests/test_preview_controller.py` - coalescing, forced intent, failure and camera state
  - `tests/test_cli_preview.py` - transitive module invalidation and CLI route
  - new real-filesystem integration fixture colocated with preview tests - 250 ms delivery proof

## Chosen Defaults / Parameters

- one active build and at most one latest replacement request
- forced intent survives coalescing and advances a monotonic cache generation
- 250 ms maximum watcher-delivery budget before build time
- last good scene and camera remain on failed or stale builds

## Data Ownership

- source of truth: preview controller reload state plus CLI loader generation
- read ownership: watcher, build lane, and renderer poll through typed coordinator boundaries
- write ownership: only the coordinator mutates request state; only the CLI loader mutates module cache generation
- derived/cache data: watched dependency set and cached module graph are rebuilt from the entry model
- privacy/logging: paths and failure summaries may be logged; model source contents are not

## Dependencies And Routes

- Domain/service dependencies:
  - `PyVistaPreviewer`, current executor/timer handoff, watchdog adapter, CLI scene factory, Python `sys.modules`
  - GUI route: `R` key event -> forced request -> background build -> UI-thread apply
  - background route: filesystem event -> normalized request -> one-active/one-latest coordinator -> build executor
- Database dependencies:
  - none
- GUI route, if applicable:
  - GUI route: `R` key event -> forced request -> background build -> UI-thread apply
- Background/concurrency route, if applicable:
  - background route: filesystem event -> normalized request -> one-active/one-latest coordinator -> build executor

## Prerequisite Handling

- Architecture feedback artifacts:
  - `../architecture/acd-preview-reload-coordination.md` - owns reload coordination and cache-invalidation transition
- Architecture feedback status:
  - tracked in active ACD
- Already implemented prerequisites:
  - existing preview command, renderer-thread scene application, watched-module discovery, and last-good scene retention
- Missing prerequisite architecture:
  - none
- Missing prerequisite specifications:
  - none
- Unimplemented prerequisite specifications:
  - none
- Progression handling:
  - this leaf may proceed after independent review canonicalizes it

## Application Integration

- App type: mixed
- User/caller surface: live `impression preview` command and preview window
- Invocation route: saved file or `R` key -> coordinator -> loader/build lane -> renderer-thread scene application
- Wiring owner/module: `src/impression/preview.py` with cache invalidation in `src/impression/cli.py`
- Observable result: changed model appears after its real build time; errors remain visible without destroying the last good scene
- Integration validation: real command smoke plus filesystem-event timing, transitive dependency, mtime-neutral `R`, burst, stale/failure, and camera assertions
- Incomplete status risk: completion requires the declared integrated route and prerequisite sequence to pass

App-type-specific proof:

- GUI: visible preview entrypoint, `R` event, UI-thread handoff, stale/failure behavior, and GUI route smoke
- Console: real `impression preview` command, filesystem side effects, status/error output, and process exit/shutdown behavior
- API/service: not applicable
- Mixed: separate command/watcher and visible renderer assertions
- Library-only: not applicable

## Reuse And Extraction Plan

- Existing code to reuse:
  - existing code: `PyVistaPreviewer`, scene factory cache, watched-module discovery, executor, timer handoff, and scene controller
- Current reuse readiness:
  - readiness: add typed coordination and generation state to existing modules
- Extraction/wrapping needed:
  - extraction: a small reload coordinator record/state machine inside the preview module; no parallel preview engine
- Additions to existing library/modules:
  - readiness: add typed coordination and generation state to existing modules
- New reusable modules to expose:
  - new reusable modules: none unless review proves the coordinator has an independent cross-app consumer
- One-off code justification, if any:
  - one-off justification: none

## Required DTOs / Functions / Components

- DTOs/models:
  - `ReloadRequest(reason, force, changed_paths, generation)` - immutable reload intent
- Functions/methods:
  - reload coordinator methods to submit, begin, complete, and take latest replacement
  - generation-aware scene factory invalidation and transitive dependency rediscovery
  - status events distinguishing queued, rebuilding, succeeded, failed, and stale
- UI fields / visible data, if applicable:
  - existing preview status/error output; no new field
- UI elements / controls, if applicable:
  - existing `R` key binding; no new control
- UI components, if applicable:
  - none

## Performance Contract

- eligible local filesystem events reach build submission within 250 ms
- request storage remains O(1): one active plus one latest replacement
- watcher callback performs no model construction or rendering

## Error And State Behavior

- watch/build failure leaves the last good scene and camera visible
- newer queued work is not consumed by an older failure
- forced intent cannot be downgraded by an adjacent automatic event
- shutdown stops watcher/build scheduling without applying stale results

## Test Strategy

- Unit tests:
  - request merging, force-bit retention, generation changes, stale completion, failure recovery, and camera-preservation state
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - offscreen/controller state, UI-thread apply, stale result, failure retention, and camera preservation
- Integrated route tests:
  - real top-level and transitive file writes through `impression preview`; mtime-neutral include edit followed by `R`; burst saves; visible status and scene value
- Production-data rule:
  - tests use project fixtures and temporary directories; they do not require user production data

## Acceptance Criteria

- A saved top-level or transitive local Python file submits the next preview build within 250 ms on a supported local filesystem, excluding build/render time.
- Rapid or duplicate events retain at most one latest replacement while exactly one build runs.
- Pressing `R` invalidates user modules, rediscovers dependencies, rebuilds, and re-renders even when mtime evidence is unchanged or ambiguous.
- The current camera survives ordinary rebuilds; failures preserve the last good scene and remain visible.
- Watcher/build work stays off the UI/render thread and current-generation scene application stays on it.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [ ] Review Score appears in front matter and matches a completed independent calculation.
- [x] Current implementation-spec template was loaded; its path is recorded below.
- [ ] Independent adversarial recount completed.
- [x] No unresolved placeholder is hidden as implementation-ready behavior.
- [x] Source responsibilities are carried into durable sections.
- [x] Canonical status is Draft.
- [x] Prerequisites are linked or marked not applicable.
- [x] Missing/stale architecture is tracked in the active ACD.
- [x] Missing prerequisite behavior is linked or marked not applicable.
- [x] Split coverage is recorded for issue-level splits.
- [x] Review ledger is marked not applicable before review.
- [x] Implementation owner/module and reuse/extraction decisions are named.
- [x] UI fields/elements and concurrency are explicit or not applicable.
- [x] Defaults, data ownership, app type, route, performance, privacy, and test strategy are explicit.
- [x] Acceptance criteria are observable and testable.
- [ ] Independent `review specs` confirms cohesion, scoring, canonical status, and final progression coverage.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: pending independent review; rejected as nonnumeric creation placeholder.
- Adversarial rescore basis: recounted every category from the current text; checked hidden route wiring, reuse, prerequisites, write behavior, concurrency, and performance.
- Functions/methods: 4 x 2 = 8
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 5 x 1 = 5
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Existing reusable code reused as-is: 6 x 0.5 = 3
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 3 x 3 = 9
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 3 x 2 = 6
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 41
- If total matches prior score, adversarial survival reason: not applicable; prior score was nonnumeric.
