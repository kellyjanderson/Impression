# Reference Review Hybrid Stabilization Architectural Change Document

Date: 2026-07-17
Status: Proposed
Canonical architecture targets:

- `project/release-0.1.0a/architecture/reference-review-workbench-architecture.md`
- `project/release-0.1.0a/architecture/reference-review-qt-workbench-ui.md`
- `project/release-0.1.0a/architecture/reference-review-async-concurrency.md`
- `project/release-0.1.0a/architecture/reference-review-preview-payload-boundary-architecture.md`
- `project/release-0.1.0a/architecture/reference-review-preview-engine-sharing-architecture.md`
- `project/release-0.1.0a/architecture/agentic-gui-shared-workbench-code-architecture.md`
- `project/release-0.1.0a/architecture/reference-review-impression-bench-gui-delta-review.md`

Related:

- Release / plan / issue: `project/release-0.1.0a/planning/reference-review-hybrid-stabilization-plan.md`
- Release / plan / issue: `project/release-0.1.0a/planning/reference-review-workbench-progression.md`
- Parent ACD, if any: none

## Change Intent

Stabilize the current Reference Review app from its in-between launch-crashing
state without attempting the full Workbench Kit migration.

The intended path is a hybrid stabilization: preserve the app-specific review
workflow, harden launch and preview ownership, and adopt only the lowest-risk
`impression-workbench-kit` helpers where they directly remove current
duplication or instability. Once the app launches, renders, and supports review
actions through the real entrypoint, the branch should be pushed, opened as a
pull request, and merged into the release working branch. The broader kit
migration is planned separately after that stable checkpoint.

## Current Architecture

The current code and documentation are out of phase:

- the Reference Review progression marks the workbench and preview
  remediation specs complete;
- the live app currently crashes, hangs, or beachballs on launch;
- the branch contains in-flight renderer and review-app changes;
- Reference Review still carries local copies of helpers that now exist in
  `impression-workbench-kit`;
- newer Impression Bench management code demonstrates stronger async and
  diagnostic patterns, but those patterns are not yet cleanly part of the
  Reference Review app.

This ACD exists because canonical architecture should not be edited as though
the stabilized hybrid structure is already true in code.

## Target Architecture

The stabilized Reference Review app keeps review-domain ownership local while
using shared workbench infrastructure only at safe boundaries.

### Stabilization Boundary

Reference Review continues to own:

- fixture discovery and fixture list state;
- fixture status: approved, declined, and unreviewed;
- notes loading and real-time persistence;
- dirty-to-gold artifact promotion;
- fixture context, purpose, methodology, and expected-render text;
- review-specific source and artifact payload decisions;
- the `impression-reference-review` console entrypoint.

Impression core continues to own:

- model construction APIs;
- `.impress` loading and serialization behavior;
- preview scene semantics shared with CLI preview;
- renderer interaction semantics that must remain CLI-preview-like.

`impression-workbench-kit` may own, during this stabilization pass:

- sanitized Qt handoff helpers;
- latest-request or staleness helpers;
- durable write lane helpers;
- preview display option records and icon-control state, if API-compatible;
- markdown context rendering and style records, if API-compatible.

The kit must not own, during this pass:

- fixture review status;
- promotion policy;
- fixture queue behavior;
- app-specific payload builder decisions;
- Bench file browser, code editor, chat, prompt injector, or proposal
  workflows.

### Launch Ownership

The app launch path must initialize only the shell, QML/widget hierarchy,
lightweight models, and deferred task lanes.

The launch path must not synchronously:

- create or mutate expensive renderer content before the preview widget is
  ready;
- import fixture source modules;
- build models;
- tessellate geometry;
- scan large fixture roots;
- write notes, promotion state, or fixture files.

Expensive work starts only after the event loop is alive and is routed through
owned task lanes with UI-thread handoff.

### Task Lane Model

Each background producer has an owner, queue policy, stale-result policy, and
UI handoff route.

| Lane | Owner | Work | Queue policy | UI mutation rule |
| --- | --- | --- | --- | --- |
| launch/bootstrap | shell | create app shell and register bridges | no background work during visible bootstrap | UI thread only |
| fixture list | fixture queue | load and filter fixture records | replace stale refreshes | model mutation through UI handoff |
| preview payload | preview pane | load artifact/source payloads and prepare render input | one selected-fixture request; newer selection stales older work | result handoff only; no renderer mutation |
| preview display command | preview pane | apply display toggles and camera/render options | coalesce to latest command set | renderer mutation on UI/render thread |
| notes | notes panel | save selected fixture notes | serialized durable writes | selected fixture state handoff only |
| promotion | review action panel | approve/decline and artifact move/write | serialized; never silently discard durable result | visible status handoff and diagnostic route |

Workers must not mutate QML state, Qt models, widgets, `QtInteractor`,
PyVista/VTK renderer objects, or live scene objects.

### Preview Lifecycle

The preview widget owns one live renderer for the widget lifetime.

Fixture changes replace scene content inside that renderer. Cancellation, stale
completion, payload failure, or non-renderable fixture selection must not
destroy and recreate the renderer as a side effect.

Preview behavior follows these rules:

- renderable artifact fixtures enter the render path;
- diagnostic or non-renderable fixtures enter a contextual message path;
- stale success cannot overwrite a newer selected fixture;
- stale failure cannot clear a newer good preview;
- same-fixture failure after a successful render preserves the last good
  render and marks the preview state stale or diagnostic;
- display-control commands affect only the current preview surface through the
  UI/render-thread command route.

### Kit Adoption Boundary

Shared-kit adoption is permitted only when all of the following are true:

- the kit module is importable from the `.venv` used by
  `impression-reference-review`;
- the imported API is already compatible or has a narrow adapter;
- focused tests cover the app route that consumes the helper;
- replacing the local copy does not require importing `impression_gui`;
- failure to import the kit cannot create a launch-time beachball.

If those conditions are not true, keep the review-local module during the
stabilization PR and record the migration as full-kit follow-up work.

## Data Flow

```text
entrypoint
-> shell bootstrap
-> event loop alive
-> fixture list refresh request
-> fixture list completion handoff
-> selected fixture changed
-> preview payload request
-> stale check
-> UI-thread preview command
-> renderer scene replacement or non-renderable message
```

Notes and status use a separate write flow:

```text
selected fixture
-> notes/status edit
-> serialized durable write lane
-> completion handoff
-> selected fixture stale check
-> visible status/diagnostic update
```

Approve uses an artifact promotion flow:

```text
approve action
-> promotion validation
-> serialized artifact move/write
-> fixture record status write
-> completion handoff
-> fixture list/status refresh
```

## Failure And Diagnostic Routing

Failures must stay near the route that produced them:

- launch/import failures surface as entrypoint or shell diagnostics;
- fixture load failures surface in the fixture list/context region;
- preview payload failures surface in the preview pane;
- non-renderable fixtures surface as intentional context states, not preview
  crashes;
- notes write failures surface in the notes/status area;
- promotion failures surface near approve/decline controls and preserve the
  previous durable state.

The UI must not silently turn failures into a blank preview or empty fixture
state unless the empty state is the correct domain result and a diagnostic
remains available.

## Application Integration Contract

- App type: mixed GUI and console entrypoint
- User/caller surface: `impression-reference-review` and the Reference Review
  desktop app window
- Invocation route: console command starts the Qt shell; fixture selection,
  preview controls, notes edits, approve, decline, and show-approved filtering
  drive the live GUI route
- Wiring owner/module: `src/impression/devtools/reference_review/ui/shell.py`,
  `src/impression/devtools/reference_review/ui/preview_widget.py`,
  `src/impression/devtools/reference_review/preview_payload_builder.py`, and
  the existing reference review lifecycle modules
- Observable result: the app launches responsively, selected renderable
  fixtures preview correctly, non-renderable fixtures show context instead of
  crashing, notes persist, and review status actions persist
- Integration validation: focused shell and payload tests, notes/status tests,
  display-control routing tests, `git diff --check`, and a manual smoke through
  the `.venv` console entrypoint with real fixture data

## Conformance Checklist

- [ ] The `.venv` entrypoint launches the app without immediate crash, hang, or
  beachball.
- [ ] Launch performs no expensive fixture import, model build, tessellation,
  renderer scene build, or durable write on the UI thread.
- [ ] Kit imports used by Reference Review are importable from the `.venv` or
  are deferred out of the stabilization PR.
- [ ] Reference Review imports no modules from `impression_gui`.
- [ ] Fixture selection routes preview payload work off the UI thread.
- [ ] Preview renderer lifetime is owned by the preview widget and remains
  stable across fixture selection, stale completion, and display-control
  commands.
- [ ] Non-renderable fixtures produce contextual visible state rather than
  blank preview or renderer failure.
- [ ] Notes and review status actions persist through durable write routes.
- [ ] Focused tests and real-entrypoint smoke support the merge.
- [ ] The stabilized branch is pushed, opened as a PR, and merged before the
  full kit migration begins.

## Non-Goals

- Full `impression-workbench-kit` migration.
- Copying the Impression Bench UI into Reference Review.
- Adding Bench workspace file browsing, code editing, chat, snapshots, prompt
  injection, or proposal workflows.
- Reworking CSG, fixture generation, or reference-test expansion.
- Replacing the Reference Review promotion and notes lifecycle.

## Canonical Document Impact

On closure, reconcile the canonical architecture documents so they describe
the conformed stabilization state:

- `reference-review-workbench-architecture.md` should name the stabilized app
  boundaries and the deferred full-kit migration.
- `reference-review-async-concurrency.md` should reflect the actual launch,
  task-lane, and stale-result routes.
- `reference-review-preview-payload-boundary-architecture.md` should reflect
  the conformed payload and non-renderable fixture paths.
- `reference-review-qt-workbench-ui.md` should reflect launch, status, notes,
  preview, and filter behavior that is user-accessible through the app.
- `agentic-gui-shared-workbench-code-architecture.md` should remain the broader
  shared-code direction and should not be treated as fully implemented by this
  stabilization pass.

## Specification Sources

### Candidate Spec: Launch Baseline And Import Boundary Stabilization

Discovery purpose:
- Establish the real launch failure mode and make the `.venv` entrypoint's
  import boundary deterministic.

Responsibilities:
- confirm branch and fixture command;
- classify the crash/hang root category;
- verify kit importability from the app `.venv`;
- prevent `impression_gui` imports from the review app;
- preserve or adapt local helper imports when kit import risk is too high.

Integration validation:
- offscreen or shell-level bootstrap test where possible;
- manual launch smoke through `impression-reference-review`.

### Candidate Spec: Non-Blocking Shell Bootstrap And Task Handoff

Discovery purpose:
- Ensure launch creates only the shell and deferred lanes, with no UI-thread
  blocking work.

Responsibilities:
- defer fixture scans, source imports, model builds, tessellation, preview
  payload work, and durable writes until after event-loop startup;
- ensure worker completions cross into UI through typed handoff;
- reject stale completions before UI mutation.

Integration validation:
- focused shell test proving bootstrap does not select/build a fixture;
- manual launch remains responsive before and after first fixture selection.

### Candidate Spec: Preview Lifecycle And Non-Renderable Fixture Handling

Discovery purpose:
- Stabilize the preview route without changing the review workflow or full
  renderer architecture.

Responsibilities:
- preserve one preview renderer per widget lifetime;
- ensure payload work is off-thread and renderer mutation is UI/render-thread
  only;
- route renderable artifacts to preview;
- route diagnostic/non-renderable fixtures to contextual message state;
- preserve last-good preview on stale or failed completions.

Integration validation:
- focused preview payload and shell routing tests;
- manual selection smoke across renderable and diagnostic fixtures.

### Candidate Spec: Review Workflow Persistence Smoke

Discovery purpose:
- Confirm the stabilization did not regress the review app's core purpose.

Responsibilities:
- notes load and save for the selected fixture;
- approve persists status and moves dirty artifacts to gold;
- decline persists status without moving artifacts;
- show-approved filtering reflects current fixture status;
- status badge reflects approved, declined, and unreviewed.

Integration validation:
- focused fixture-record tests;
- manual smoke against the active fixture file.

### Candidate Spec: Stabilization Merge Gate

Discovery purpose:
- Define the proof required before merging the stabilization branch.

Responsibilities:
- run focused tests for preview payload, UI shell routing, notes/status, and
  display-control command routing;
- run `git diff --check`;
- run a real-entrypoint manual smoke;
- commit, push, open PR, and merge only after the stabilization definition is
  satisfied.

Integration validation:
- PR checklist and merge evidence.

## Closure Criteria

This ACD can close only after:

- the stabilized app conforms to the target architecture above;
- the stabilization plan checklist is complete or superseded by a stricter
  validated route;
- the stabilization PR has merged;
- canonical architecture documents have been reconciled to the implemented
  state;
- remaining full-kit migration work is represented in a separate plan or ACD.

## Change History

- 2026-07-17: Initial ACD created to support the short hybrid stabilization
  path for the launch-crashing Reference Review app.
