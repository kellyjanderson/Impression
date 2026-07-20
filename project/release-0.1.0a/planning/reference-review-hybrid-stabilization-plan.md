# Reference Review Hybrid Stabilization Plan

Status: active stabilization plan
Created: 2026-07-17
Branch context: `codex/reference-review-diagnostic-render-state`

## Purpose

The Reference Review app is in an in-between state and currently crashes or
hangs on launch. This plan defines a short, bounded march to stabilize the live
review app while moving only the safest pieces toward `impression-workbench-kit`.

This is not the full Workbench Kit migration. The goal is to get the current
app responsive, rendering, and review-capable again, then push, open a pull
request, and merge into the release working branch. The full kit migration gets
planned after that stable checkpoint.

## Related Documents

- [Reference Review And Impression Bench GUI Delta Review](../architecture/reference-review-impression-bench-gui-delta-review.md)
- [Agentic GUI Shared Workbench Code Architecture](../architecture/agentic-gui-shared-workbench-code-architecture.md)
- [Reference Review Async Concurrency](../architecture/reference-review-async-concurrency.md)
- [Reference Review Preview Remediation Plan](../architecture/reference-review-preview-remediation-plan.md)
- [Reference Review Preview Payload Boundary Architecture](../architecture/reference-review-preview-payload-boundary-architecture.md)
- [Reference Review Workbench Progression](reference-review-workbench-progression.md)

## Stabilization Definition

The app is stable enough to merge when all of these are true:

- `impression-reference-review` launches through the installed `.venv`
  entrypoint without an immediate crash, hang, or beachball.
- The fixture list is populated and remains clickable after launch.
- Selecting a renderable STL or `.impress` fixture shows a live preview.
- Selecting a diagnostic or non-renderable fixture shows a contextual
  non-renderable state and does not crash.
- Preview interaction remains responsive and renderer lifetime stays owned by
  the UI/render thread.
- Preview display-control buttons still route to the preview surface.
- Notes load for the selected fixture and save back to the fixture record.
- Approve, decline, unreviewed status, status badge, and the show-approved
  filter still work.
- Focused tests for preview payloads, shell routing, notes/status behavior, and
  display controls pass.
- `git diff --check` is clean.
- The stabilized branch is pushed, a PR is opened, and the PR is merged into
  the release working branch.

## Explicit Non-Goals

- Do not port the Impression Bench UI wholesale.
- Do not add the Bench file browser, code editor, prompt injectors, snapshots,
  direct-edit proposal workflow, or chat workflow.
- Do not rewrite the fixture queue, promotion lifecycle, or review state model.
- Do not perform the full `impression-workbench-kit` migration in this pass.
- Do not broaden this work into CSG, fixture generation, or reference-test
  expansion.

## Hybrid Strategy

Use the kit only where it reduces active duplication or current instability.
Keep review-domain behavior in the review app.

| Area | Stabilization choice |
| --- | --- |
| Qt launch and shell ownership | Keep in Reference Review. Fix launch crash at the current app boundary. |
| Fixture discovery, notes, approve/decline, dirty/gold promotion | Keep in Reference Review. These are product-specific. |
| Preview payload builder and fixture-to-preview adapter | Keep local for now. Only harden imports, stale handling, and diagnostic fallback. |
| Preview renderer interaction semantics | Keep aligned with `impression.preview`; do not invent a second renderer paradigm. |
| Sanitized UI handoff, latest-request tracking, durable write lane | Prefer kit imports if packaging is already reliable; otherwise keep local temporarily and record the follow-up. |
| Preview display options and icon/control records | Prefer kit imports if they are a direct drop-in; otherwise preserve current local controls for the stabilization PR. |
| Markdown context rendering and style records | Adopt from kit only if it is low risk and does not affect launch stability. |
| Owner route registry, lane policy records, generic task records | Defer to full migration unless a narrow adapter is needed to fix the crash safely. |

## Work Plan

### Draft Specification Handoff

These draft specifications were created from the active hybrid stabilization
ACD and remain unchecked until an independent `review specs` pass verifies
their split, score, and readiness.

- [ ] [Reference Review Spec 76: Launch Baseline And Import Boundary Stabilization](../specifications/reference-review-76-launch-baseline-and-import-boundary-stabilization-v1_0.md)
- [ ] [Reference Review Spec 76 Test: Launch Baseline And Import Boundary Stabilization](../test-specifications/reference-review-76-launch-baseline-and-import-boundary-stabilization-v1_0.md)
- [x] [Reference Review Spec 76a: Launch Failure Baseline](../specifications/reference-review-76a-launch-failure-baseline-v1_0.md)
- [x] [Reference Review Spec 76a Test: Launch Failure Baseline](../test-specifications/reference-review-76a-launch-failure-baseline-v1_0.md)
- [x] [Reference Review Spec 76b: Import Boundary And Kit Availability](../specifications/reference-review-76b-import-boundary-and-kit-availability-v1_0.md)
- [x] [Reference Review Spec 76b Test: Import Boundary And Kit Availability](../test-specifications/reference-review-76b-import-boundary-and-kit-availability-v1_0.md)
- [ ] [Reference Review Spec 77: Non-Blocking Shell Bootstrap And Task Handoff](../specifications/reference-review-77-non-blocking-shell-bootstrap-and-task-handoff-v1_0.md)
- [ ] [Reference Review Spec 77 Test: Non-Blocking Shell Bootstrap And Task Handoff](../test-specifications/reference-review-77-non-blocking-shell-bootstrap-and-task-handoff-v1_0.md)
- [x] [Reference Review Spec 77a: Non-Blocking Shell Startup](../specifications/reference-review-77a-non-blocking-shell-startup-v1_0.md)
- [x] [Reference Review Spec 77a Test: Non-Blocking Shell Startup](../test-specifications/reference-review-77a-non-blocking-shell-startup-v1_0.md)
- [x] [Reference Review Spec 77b: UI Handoff And Stale Completion Guard](../specifications/reference-review-77b-ui-handoff-and-stale-completion-guard-v1_0.md)
- [x] [Reference Review Spec 77b Test: UI Handoff And Stale Completion Guard](../test-specifications/reference-review-77b-ui-handoff-and-stale-completion-guard-v1_0.md)
- [ ] [Reference Review Spec 78: Preview Lifecycle And Non-Renderable Fixture Handling](../specifications/reference-review-78-preview-lifecycle-and-non-renderable-fixture-handling-v1_0.md)
- [ ] [Reference Review Spec 78 Test: Preview Lifecycle And Non-Renderable Fixture Handling](../test-specifications/reference-review-78-preview-lifecycle-and-non-renderable-fixture-handling-v1_0.md)
- [x] [Reference Review Spec 78a: Renderable Preview Lifecycle](../specifications/reference-review-78a-renderable-preview-lifecycle-v1_0.md)
- [x] [Reference Review Spec 78a Test: Renderable Preview Lifecycle](../test-specifications/reference-review-78a-renderable-preview-lifecycle-v1_0.md)
- [x] [Reference Review Spec 78b: Non-Renderable Preview State And Last-Good Guard](../specifications/reference-review-78b-non-renderable-preview-state-and-last-good-guard-v1_0.md)
- [x] [Reference Review Spec 78b Test: Non-Renderable Preview State And Last-Good Guard](../test-specifications/reference-review-78b-non-renderable-preview-state-and-last-good-guard-v1_0.md)
- [ ] [Reference Review Spec 79: Review Workflow Persistence Smoke](../specifications/reference-review-79-review-workflow-persistence-smoke-v1_0.md)
- [ ] [Reference Review Spec 79 Test: Review Workflow Persistence Smoke](../test-specifications/reference-review-79-review-workflow-persistence-smoke-v1_0.md)
- [x] [Reference Review Spec 79a: Notes Persistence Route](../specifications/reference-review-79a-notes-persistence-route-v1_0.md)
- [x] [Reference Review Spec 79a Test: Notes Persistence Route](../test-specifications/reference-review-79a-notes-persistence-route-v1_0.md)
- [x] [Reference Review Spec 79b: Approve/Decline Persistence And Promotion Route](../specifications/reference-review-79b-approve-decline-persistence-and-promotion-route-v1_0.md)
- [x] [Reference Review Spec 79b Test: Approve/Decline Persistence And Promotion Route](../test-specifications/reference-review-79b-approve-decline-persistence-and-promotion-route-v1_0.md)
- [x] [Reference Review Spec 79c: Status Badge And Approved Filter Route](../specifications/reference-review-79c-status-badge-and-approved-filter-route-v1_0.md)
- [x] [Reference Review Spec 79c Test: Status Badge And Approved Filter Route](../test-specifications/reference-review-79c-status-badge-and-approved-filter-route-v1_0.md)
- [ ] [Reference Review Spec 80: Stabilization Merge Gate](../specifications/reference-review-80-stabilization-merge-gate-v1_0.md)
- [ ] [Reference Review Spec 80 Test: Stabilization Merge Gate](../test-specifications/reference-review-80-stabilization-merge-gate-v1_0.md)

### 0. Baseline And Branch Hygiene

- [x] Confirm the current branch and uncommitted change set.
- [x] Record the exact launch command, fixture file, and observed failure mode.
- [x] Classify the launch failure as one of:
  - import/dependency failure;
  - QML/bootstrap failure;
  - UI-thread blocking/handoff failure;
  - preview renderer construction failure;
  - fixture/payload selection failure.
- [x] Preserve current in-flight work on a named stabilization branch if the
  active branch name no longer describes the work.

### 1. Dependency And Import Stabilization

- [x] Verify whether `impression-workbench-kit` is importable from the
  repository `.venv` used by `impression-reference-review`.
- [x] If kit is not reliably importable, choose one stabilization route:
  - add the correct local development dependency; or
  - temporarily keep review-local modules and defer imports to the full
    migration.
- [x] Ensure Reference Review never imports from `impression_gui`; both apps
  must converge through `impression_workbench` or Impression core.
- [ ] Replace duplicated helper imports with kit imports only when the imported
  API is already compatible and covered by focused tests.

### 2. Launch And Event-Loop Stability

- [x] Make launch initialization cheap: no renderer build, fixture import,
  model build, tessellation, filesystem scan, or durable write may block the
  UI thread.
- [x] Keep all UI model mutation, QML state updates, renderer mutation, and
  widget disposal on the UI/render thread.
- [x] Route worker completions through typed handoff and reject stale successes,
  stale failures, and stale cancellations before mutating UI state.
- [x] Add or update a focused launch smoke that proves the app can bootstrap
  without selecting a fixture.

### 3. Preview Stabilization

- [x] Preserve one live renderer per preview widget and dispose it only when
  the widget/app closes.
- [x] Ensure fixture selection builds or loads payloads off the UI thread.
- [x] Ensure renderable artifact fixtures render from their artifact payload.
- [x] Ensure diagnostic/non-renderable fixtures show a clear contextual state
  without entering the render path.
- [x] Preserve last-good preview behavior: a stale or same-fixture failure must
  not clear a newer good render.
- [x] Confirm display-control commands mutate only the preview surface through
  the UI-thread route.

### 4. Review Workflow Smoke

- [x] Verify notes load for the selected fixture.
- [x] Verify notes save in real time to the fixture record or database.
- [x] Verify approve updates fixture status and moves dirty artifacts to gold
  using the same folder structure.
- [x] Verify decline updates fixture status without moving the artifact.
- [x] Verify unreviewed fixtures remain visible.
- [x] Verify the show-approved checkbox filters only approved fixtures when
  unchecked and includes them when checked.

### 5. Validation And Merge

- [x] Run focused tests for preview payload builder behavior.
- [x] Run focused tests for UI shell routing and launch behavior.
- [x] Run focused tests for notes/status/promotion behavior.
- [x] Run focused tests for display-control state and command routing.
- [x] Run `git diff --check`.
- [x] Manually smoke the real app through the `.venv` console entrypoint.
- [ ] Commit the stabilization unit.
- [ ] Push the branch.
- [ ] Open a PR.
- [ ] Merge the PR into the release working branch after validation.

## Stop Line

When the stabilization definition is satisfied, stop expanding the scope.
Any broader kit extraction, Bench management-code port, or shared-library
redesign discovered during this pass should be written as follow-up work for
the full Workbench Kit migration plan.

## Follow-Up: Full Workbench Kit Migration Plan

After the stabilized PR is merged, create a separate migration plan that covers:

- replacing remaining duplicated async helpers with kit modules;
- introducing owner-route and lane-policy records deliberately;
- moving shared preview state and display-control records into the kit where
  both apps can consume them;
- deciding whether shared preview command records belong in a new
  `impression_workbench.preview` package;
- bringing over useful Bench design-system and diagnostics placement rules;
- defining integration smokes for both Reference Review and Impression Bench.

## Change History

- 2026-07-17: Implemented the hybrid stabilization leaves through Spec 79c.
  Baseline launch command: `.venv/bin/impression-reference-review
  --fixture-file tests/reference_review_fixtures/dirty-stl-fixtures.json`.
  The active failure class was fixture/payload selection work starting during
  shell bootstrap; startup now opens with no selected fixture and no preview
  controller launch until the user selects a row. Validation commands run:
  `.venv/bin/python -m pytest tests/test_reference_review_ui_shell.py
  tests/test_reference_review_preview_payload_builder.py -q`,
  `.venv/bin/impression-reference-review --fixture-file
  tests/reference_review_fixtures/dirty-stl-fixtures.json --check`,
  `.venv/bin/impression-reference-review --fixture-file
  tests/reference_review_fixtures/dirty-stl-fixtures.json --check --offscreen`,
  timed real-platform Qt smoke with fixture selection, and `git diff --check`.
- 2026-07-17: Initial hybrid stabilization plan created after the review app
  entered an in-between launch-crashing state.
