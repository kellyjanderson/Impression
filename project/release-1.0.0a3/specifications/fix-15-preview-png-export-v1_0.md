# Fix 15: Preview PNG Export Specification

Date: 2026-08-04
Status: Final
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `not applicable - restores the existing preview command contract`
Source artifact: `docs/cli.md` and the existing `preview --screenshot PATH` option
Split provenance: `none`
Canonical status: `Canonical`
Review Score: 16
Prerequisites:
- The existing PyVista preview route and `--screenshot PATH` CLI option are implemented.

## Source Field Carryover

- Source purpose:
  - Restore PNG export from the preview command when a live watched preview already exists.
- Source responsibilities by category:
  - Functions/methods: the existing `preview(...)` command and `PyVistaPreviewer.show(...)` renderer entrypoint.
  - Data structures/models: not applicable.
  - Dependencies/services: the existing PyVista `Plotter` screenshot capability.
  - Returns/outputs/signals: one PNG at the requested path and a success message naming that path.
  - UI surfaces/components: not applicable; this is a console command route.
  - UI fields/elements: the `--screenshot PATH` help entry explains one-shot PNG output and live-preview isolation.
  - Reusable code plan: reuse the existing CLI option and preview renderer rather than adding another export subsystem.
  - Database queries/tables/migrations: not applicable.
  - Async/concurrency behavior: not applicable; screenshot mode is a one-shot render and does not start file watching.
  - Destructive/write behavior: the explicitly requested output path may be created or replaced.
  - Security/privacy-sensitive behavior: not applicable.
  - Performance-sensitive behavior: one bounded 1280 by 800 render is produced and the process exits.
  - Cross-screen reusable behavior: not applicable.
- Source open questions / nuance discovered:
  - Screenshot mode currently inherits watch mode, so an existing control file can redirect the live preview and exit before rendering.
- Source split/provenance notes:
  - No parent split; routing and rendering form one atomic PNG-export command transaction.

## Purpose

Make `impression preview MODEL --screenshot PATH` reliably produce a PNG and
exit without opening, redirecting, or disturbing an already running watched preview.

## Scope

Owns:

- Screenshot mode bypassing the live-preview control-file handoff.
- Screenshot mode disabling file watching for its one-shot render.
- Off-screen PyVista capture to the exact requested output path.
- Success output that names the written PNG.
- Discoverable `preview --help` text that states the one-shot PNG and live-preview behavior.

Does not own:

- Adding an interactive file picker or new in-window keyboard controls.
- Changing watched-preview model switching when `--screenshot` is absent.
- High-resolution, transparent-background, or camera-preset image export.

## Split Coverage

- Parent spec: `none`
- Parent coverage status: not applicable
- Parent responsibilities owned by this child:
  - not applicable
- Parent responsibilities still missing from children:
  - none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../planning/spec-review-ledger-20260804-preview-png-help.md` | 1 | Fix 15 | none | reached |

## Implementation Routing

- Primary modules/files:
  - `src/impression/cli.py` - select one-shot screenshot mode before control-file handoff.
  - `src/impression/preview.py` - construct an off-screen plotter and write the PNG.
- Supporting modules/files:
  - `docs/cli.md` - document independence from live watch sessions and success behavior.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/preview.py` - retain screenshot capture in the existing preview renderer boundary.
- Tests:
  - `tests/test_cli_preview.py` - prove help discoverability, route selection in the presence of a live control file, and real-command PNG output.

## Chosen Defaults / Parameters

- `--screenshot PATH` always means a one-shot, non-watching render even though `--watch` is the command default.
- The requested path is used exactly and its parent directory is created.
- The current preview window remains attached to its current model.
- The image uses the existing 1280 by 800 preview window size and current preview styling.

## Data Ownership

- Source of truth: the model passed to the screenshot command invocation.
- Read ownership: the CLI model loader and preview dataset collector.
- Write ownership: `PyVistaPreviewer.show(...)` writes only the requested screenshot path.
- Derived/cache data: the PNG is a derived user-requested artifact and may be regenerated from the model.
- Privacy/logging constraints: report only the requested local path; do not log model contents.

## Dependencies And Routes

- Domain/service dependencies:
  - existing model loading, dataset collection, and PyVista rendering.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable; screenshot mode bypasses watcher and control-file concurrency.

## Prerequisite Handling

- Architecture feedback artifacts:
  - none; this repairs behavior already documented by the CLI contract.
- Architecture feedback status:
  - not applicable.
- Already implemented prerequisites:
  - the `preview --screenshot PATH` option and PyVista scene renderer.
- Missing prerequisite architecture:
  - none.
- Missing prerequisite specifications:
  - none.
- Unimplemented prerequisite specifications:
  - none.
- Progression handling:
  - current item may proceed.

## Application Integration

- App type: console.
- User/caller surface: `impression preview MODEL --screenshot PATH`.
- Invocation route: CLI option parsing -> screenshot-mode route selection -> `PyVistaPreviewer.show(...)` -> off-screen PNG write.
- Wiring owner/module: `src/impression/cli.py` and `src/impression/preview.py`.
- Observable result: help describes the behavior; invocation returns exit code zero, a decodable PNG at `PATH`, a success message, and no mutation of the live preview control file.
- Integration validation: inspect `preview --help`, invoke the installed CLI while a live-looking control file exists, then decode and inspect the resulting image.
- Incomplete status risk: helper-only rendering would leave the control-file early exit in place and remain inaccessible through the real command.

App-type-specific proof:

- Console: assert command arguments, zero exit, output path side effect, success text, and unchanged live control-file content.

## Reuse And Extraction Plan

- Existing code to reuse:
  - `preview(...)` `--screenshot` option - existing public command contract.
  - `PyVistaPreviewer.show(...)` - existing scene collection, styling, and capture boundary.
- Current reuse readiness:
  - add to existing CLI and preview modules.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/cli.py` - effective watch-mode selection and success reporting.
  - `src/impression/preview.py` - explicit off-screen plotter construction for screenshot mode.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - not applicable.
- Functions/methods:
  - `preview(...) -> None` - route screenshot invocations independently from watched preview sessions.
  - `PyVistaPreviewer.show(...) -> None` - render and write a one-shot PNG off-screen.
- UI fields / visible data, if applicable:
  - `--screenshot PATH` help - states that the command renders once off-screen, saves a PNG, exits, and does not redirect a running preview.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Produce one 1280 by 800 frame and exit; do not start watcher threads or remain resident.

## Error And State Behavior

- Model-load and renderer errors remain non-zero CLI failures with the existing diagnostic route.
- Parent directories are created before capture.
- The screenshot invocation neither rewrites the live preview control file nor changes its model target.
- A successful command reports the resolved output artifact path.

## Test Strategy

- Unit tests:
  - verify `preview --help` exposes the screenshot behavior and screenshot mode passes `watch_files=False`, no control file, and the requested path to the preview renderer.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - invoke the real installed command against a box model with a live-looking control file, then assert zero exit and decode the PNG.
- Production-data rule:
  - tests use temporary models, control files, and output paths only.

## Acceptance Criteria

- `impression preview MODEL --screenshot PATH` writes a decodable PNG and exits without opening an interactive window.
- `impression preview --help` describes one-shot PNG output and live-preview isolation beside `--screenshot`.
- An existing live preview control file is unchanged and does not intercept screenshot mode.
- The CLI reports the written path and returns a non-zero result when rendering fails.
- Normal watched-preview control-file handoff remains unchanged when `--screenshot` is absent.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Review Score appears in the front matter and exactly matches the total in the final Review Score Calculation section.
- [x] The current implementation-spec template was loaded and its source path is recorded in the final Review Score Calculation section.
- [x] Review Score is adversarially recounted from the current spec text; prior scores are challenged instead of trusted.
- [x] Unresolved deferral/gap markers are absent or explicitly resolved.
- [x] Source fields are carried into spec sections or preserved as explicit provenance/history.
- [x] Canonical status is explicit.
- [x] Prerequisites are linked and implemented.
- [x] Missing or stale prerequisite architecture is marked not applicable.
- [x] Missing prerequisite behavior is marked not applicable.
- [x] Split coverage is marked not applicable.
- [x] Per-request review ledger records the terminal new-leaf list.
- [x] Implementation owner/module is named.
- [x] Existing code reuse/extraction decision is explicit.
- [x] Existing library/module additions are named and new modules are marked not applicable.
- [x] The screenshot help field is listed explicitly.
- [x] Chosen defaults are explicit.
- [x] Data source of truth and write owner are explicit.
- [x] GUI/concurrency routes are marked not applicable.
- [x] App type and application integration route are explicit.
- [x] Integrated route validation is named.
- [x] Console proof matches the app type.
- [x] Performance bounds are explicit.
- [x] Privacy/logging constraints are explicit.
- [x] Test strategy does not depend on production data.
- [x] Acceptance criteria are testable.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: 15; treated as an adversarial input rather than trusted.
- Adversarial rescore basis: recounted the user-visible command contract and found the previously omitted `--screenshot` help field; checked for a hidden interactive surface, control-file concurrency, missing overwrite ownership, separate image-export subsystems, omitted console side effects, and unnamed additional work. The score crosses into explicit split review, but help discoverability cannot be delivered independently from the same screenshot option contract, so the spec remains one cohesive command transaction.
- Functions/methods: 2 x 2 = 4
- Data structures/models: 0 x 1 = 0
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 1 x 1 = 1
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 16
- If total matches prior score, adversarial survival reason: not applicable because the fresh score increased after counting the omitted help field.
