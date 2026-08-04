# Fix 07: Reference Review Linux Lifecycle (v1.0)

Date: 2026-08-04
Status: Final
Issue: [#227](https://github.com/kellyjanderson/Impression/issues/227)
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `project/release-0.1.0a/architecture/reference-review-async-concurrency.md`
Source artifact: GitHub issue `#227`
Split provenance: `none`
Canonical status: `Canonical`
Prerequisites:
- `none` - the existing shell and headless CI fixture are sufficient starting points.

## Review Score

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; obsolete IWU metadata removed.
- Adversarial rescore basis: counted application/widget/renderer lifecycle methods,
  lifecycle state, Qt/VTK dependencies, process exit, one GUI shell, two reused owners,
  three existing-module changes, and one async teardown responsibility.
- Functions/methods: 3 x 2 = 6
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 3 x 1 = 3
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 0 x 2 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 19
- Split decision: remain whole after mandatory split review. Qt application, widget,
  renderer, and pending-work teardown share one process-lifetime owner; separate leaves
  could not independently prove the exit-139 correction.

## Source Field Carryover

- Source purpose: eliminate Linux headless hang/segfault while preserving macOS.
- Source responsibilities by category:
  - Functions/methods: application initialization, shell close, renderer/pending-work drain.
  - Data structures/models: lifecycle ownership/state.
  - Dependencies/services: Qt and VTK/PyVista.
  - Returns/outputs/signals: clean process exit.
  - UI surfaces/components: reference-review shell.
  - Reusable code plan: current shell and async ownership helpers.
  - Async/concurrency behavior: drain/cancel pending work before GUI/renderer destruction.
  - Database, write, security, performance, and cross-screen behavior: not applicable.
- Source open questions / nuance discovered: platform configuration must precede Qt/VTK import.
- Source split/provenance notes: 19-point leaf retained for process-lifecycle cohesion.

## Purpose

Make the supported reference-review GUI lifecycle terminate deterministically on Linux.

## Problem And Outcome

`tests/test_reference_review_ui_shell.py` can hang or exit 139 on Linux under
headless Qt. The supported test lane must initialize and tear down the review UI
in one process without timeout, orphan process, or segmentation fault.

## Scope

- Correct application/widget/renderer setup and teardown ownership for headless Linux.
- Make the supported Qt platform and graphics configuration explicit in tests.
- Restore the test module to normal CI execution.

Not in scope: redesigning the reference-review UI or weakening assertions by
skipping Linux behavior.

## Split Coverage

- Parent spec: `none`
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../planning/spec-review-ledger-20260804-040607.md` | 2 | a3 specs 01-12, 13A, 13B | none | reached |

## Implementation Routing

- `src/impression/devtools/reference_review/ui/` lifecycle owners.
- `tests/test_reference_review_ui_shell.py` and CI environment configuration.

## Chosen Defaults / Parameters

- One process owns one application instance; supported headless platform is configured pre-import.
- Close drains/cancels pending work before destroying widgets/renderers.
- Bounded test timeout detects hangs; crashes are failures, never skips.

## Data Ownership

- Source of truth: application/shell lifecycle state and registered pending work.
- Read ownership: shell controller and test fixture.
- Write ownership: lifecycle owner transitions state on the GUI thread.
- Derived/cache data: renderer/widget resources are disposable.
- Privacy/logging constraints: preserve diagnostics without model contents or user paths.

## Dependencies And Routes

- Domain/service dependencies: Qt application/event loop; VTK/PyVista renderer lifecycle.
- Database dependencies: none.
- GUI route: test creates shell, processes events, closes shell, drains resources.
- Background/concurrency route: pending work completes/cancels before GUI-owned destruction.

## Prerequisite Handling

- Architecture feedback artifacts: none; current async concurrency architecture covers ownership.
- Already implemented prerequisites: UI shell fixture and lifecycle helpers.
- Missing prerequisite architecture/specifications: none.
- Unimplemented prerequisite specifications: none.
- Progression handling: current item may proceed.

## Application Integration

- App type: GUI.
- User/caller surface: reference-review shell startup and close.
- Invocation route: test/user launch -> Qt event loop -> shell close -> renderer/work drain -> exit.
- Wiring owner/module: reference-review UI shell/controller.
- Observable result: responsive shell and normal process exit.
- Integration validation: full UI-shell module in one Linux/macOS process, repeated.
- Incomplete status risk: helper-only tests cannot prove native resource teardown.

## Reuse And Extraction Plan

- Existing code to reuse: current shell ownership and async drain/cancel helpers.
- Current reuse readiness: add to existing UI/controller/renderer modules.
- Extraction/wrapping/new reusable modules: none.
- Additions to existing library/modules: explicit close ordering and platform setup.
- One-off code justification: none.

## Required DTOs / Functions / Components

- DTOs/models: existing lifecycle/pending-work state; no persistent DTO.
- Functions/methods: application setup, shell close, renderer/work teardown.
- UI components: reference-review shell; no new fields or controls.

## Performance Contract

- Close completes within the CI timeout; no unbounded polling or sleeps.

## Error And State Behavior

- Construction failure and close-during-work release owned resources exactly once.
- Fatal native signals remain visible to CI through faulthandler/process status.

## Test Strategy

- Unit tests: ownership transitions where separable.
- GUI/controller tests: full offscreen shell lifecycle and failure paths.
- Integrated route tests: repeated Linux/macOS one-process module execution.
- Service/DB tests: not applicable; no production data.

## Contract

One test process owns one application lifecycle, closes all top-level UI and
graphics resources, and exits normally. Platform setup happens before Qt/VTK
initialization. The same ownership rules must preserve the existing macOS lane.

## Acceptance Criteria

- The full UI-shell test module completes on Linux with exit code 0.
- No timeout, orphan process, fatal Qt message, or exit 139 occurs.
- Repeated execution is stable and does not depend on test order.
- macOS UI-shell coverage remains green.

## Verification

[Paired test specification](../test-specifications/fix-07-reference-review-linux-lifecycle-v1_0.md)

## Readiness Checklist

- [x] Ancestors, full score, carryover, canonical status, and terminal ledger are explicit.
- [x] The 19-point split review documents indivisible process-lifetime ownership.
- [x] GUI/concurrency route, ownership, defaults, modules, errors, and proof are explicit.
- [x] No blocker, missing prerequisite, unresolved gap, or split coverage remains.
- [x] Cross-platform tests avoid production data and preserve crash evidence.
