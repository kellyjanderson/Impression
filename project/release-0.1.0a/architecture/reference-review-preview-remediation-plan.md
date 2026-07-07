# Reference Review Preview Remediation Plan

## Overview

This document records the current Reference Review Workbench preview failure
mode and the ordered remediation plan.

The immediate problem is not only a rendering bug. The workbench preview has
drifted into the wrong architecture for interactive graphics: Qt UI code,
process management, tessellation, PyVista scene construction, and legacy PNG
snapshot rendering are mixed together. The result is a launch path that can
beachball before user action and an interaction path that does not preserve the
same stable live-preview contract as `impression preview`.

The remediation starts by importing and reusing the existing
`impression.preview` scene behavior instead of continuing to rebuild a parallel
preview stack inside the review UI.

## Related Architecture

- [Reference Review Workbench Architecture](reference-review-workbench-architecture.md)
- [Reference Review Qt Workbench UI](reference-review-qt-workbench-ui.md)
- [Reference Review Async Concurrency](reference-review-async-concurrency.md)
- [Reference Review Fixture Source Contract](reference-review-fixture-source-contract.md)

## Current Failure Summary

- The application frame can appear while the UI thread beachballs.
- Launch can trigger preview work before the reviewer explicitly chooses a
  fixture.
- Fixture selection can block while the preview is built or applied.
- The preview path previously created and destroyed offscreen renderers for
  PNG snapshots; that pathway is not the desired product behavior.
- The embedded live preview still duplicates rendering and camera behavior
  already owned by `impression preview`.
- Interactive rendering must behave like the CLI preview: live, stable,
  camera-control compatible, and backed by a long-lived renderer rather than
  repeated snapshot generation.

## Non-Goals

- Do not use the PNG snapshot pathway as the review preview.
- Do not hand-roll separate mouse, camera, or edge-rendering behavior in the
  workbench when the CLI preview already owns those semantics.
- Do not run tessellation, source import, filesystem scans, or process startup
  on the Qt UI thread.
- Do not put multiprocessing worker targets in modules that import Qt widgets
  or PyVistaQt at module import time.

## Issues

| Severity | Issue | Current anchor | Effect | Target fix |
| --- | --- | --- | --- | --- |
| P0 | Preview build worker target lives in the UI shell module. | `src/impression/devtools/reference_review/ui/shell.py:181` | A process worker imports the UI shell, which imports Qt widget code and pulls UI concerns into worker startup. | Move preview build work to a pure non-UI module. |
| P0 | The shell auto-selects an initial fixture after launch. | `src/impression/devtools/reference_review/ui/shell.py:554` | Launch can immediately start heavy preview work and beachball before the user acts. | Launch into a responsive queue and placeholder state; load preview only after explicit selection or open action. |
| P1 | Qt async is implemented with `ProcessPoolExecutor` plus `QTimer` polling. | `src/impression/devtools/reference_review/ui/shell.py:7`, `src/impression/devtools/reference_review/ui/shell.py:275`, `src/impression/devtools/reference_review/ui/shell.py:278` | Polling a future from Qt hides process lifecycle, cancellation, stdout/stderr, and stale-result handling. | Use a Qt-owned `QProcess` controller with signals, request ids, cancellation, and surfaced diagnostics. |
| P1 | Heavy render preparation still happens on the UI thread. | `src/impression/devtools/reference_review/ui/shell.py:302`, `src/impression/devtools/reference_review/ui/shell.py:358` | Applying a fixture can convert meshes, extract edges, add actors, reset the camera, and render while the UI is blocked. | Define a render-ready payload boundary and keep UI-thread work bounded to scene application. |
| P1 | Workbench preview duplicates `impression.preview`. | `src/impression/devtools/reference_review/ui/shell.py:358`, `src/impression/preview.py:501`, `src/impression/preview.py:592` | Camera reset, dataset application, edge rendering, colors, and interaction behavior can diverge from the CLI preview. | Extract or expose reusable scene application from `impression.preview` and make both CLI and workbench use it. |
| P2 | Legacy PNG renderer remains exported from the UI package. | `src/impression/devtools/reference_review/ui/artifact_preview.py:155`, `src/impression/devtools/reference_review/ui/__init__.py:3` | The wrong preview paradigm remains easy to call and easy to reintroduce into the workbench. | Remove it from the live workbench path; keep only as a clearly named thumbnail/artifact utility if still needed. |

## Remediation Sequence

### 1. Import Preview Functionality From `impression.preview`

Create a reusable preview scene component from the CLI preview implementation.
The workbench must consume that component instead of duplicating mesh-to-PyVista,
edge extraction, camera reset, color, and actor setup logic.

The extracted component should own:

- plotter configuration needed for Impression previews
- dataset-to-scene application
- feature or object edge policy
- camera reset semantics
- compatible interaction defaults
- dark blue background and light orange object color defaults when used by the
  review workbench

Candidate shape:

```python
class PreviewSceneController:
    def configure_plotter(self, plotter) -> None: ...
    def apply_scene(self, plotter, datasets, *, clear: bool = True) -> None: ...
    def reset_camera(self, plotter, datasets) -> None: ...
```

Acceptance:

- The workbench no longer calls Impression mesh conversion or edge extraction
  directly from `src/impression/devtools/reference_review/ui/shell.py`.
- The CLI preview and workbench share the same scene application path.
- A focused test or import check prevents the workbench from regrowing a
  parallel renderer implementation.

### 2. Stop Startup Preview Auto-Load

The workbench should open to a responsive queue and preview placeholder. It may
remember or highlight a row, but it must not start preview construction until a
fixture is explicitly opened or selected through the intended review action.

Acceptance:

- Launching `impression-reference-review` does not create a `QtInteractor`.
- Launching does not start preview tessellation or a preview build process.
- Existing shell tests assert launch responsiveness before preview work begins.

### 3. Move Preview Build Work Out Of UI Modules

Create a pure preview build module, for example
`impression.devtools.reference_review.preview_build` or
`impression.devtools.reference_review.preview_worker`.

That module may load `.impress` records and prepare render payloads, but it must
not import PySide widgets, QML, PyVistaQt, or the workbench shell.

Acceptance:

- The worker module imports without importing `PySide6.QtWidgets` or
  `pyvistaqt`.
- Worker failures return structured diagnostics that include fixture id,
  operation, safe path context, and recovery text when known.

### 4. Replace `ProcessPoolExecutor` Polling With `QProcess`

Qt should own process supervision for external preview-building work. A
`PreviewBuildProcessController` should launch a Python worker through
`QProcess`, connect to `finished`, `errorOccurred`, `readyReadStandardOutput`,
and `readyReadStandardError`, and emit typed results back to the shell.

The controller must carry:

- owner
- kind
- request id
- fixture id
- artifact path or source record id
- cancellation state
- timeout state

Acceptance:

- Selecting fixture B before fixture A completes cannot apply fixture A's
  result.
- Cancelling or replacing a request terminates or stale-marks the old process.
- Worker stdout and stderr are available as diagnostics instead of disappearing
  behind a future.

### 5. Define The Render-Ready Payload Boundary

The worker/UI boundary must make it obvious what work happens off the UI thread
and what work remains on the UI thread.

The worker should own source loading, `.impress` parsing, model construction,
and tessellation. The UI should own only Qt object mutation and bounded scene
application through the shared preview controller.

The payload should not require sending live VTK or Qt objects across process
boundaries. Acceptable candidates include immutable mesh arrays in a temporary
file, an enriched preview payload file, or another existing Impression mesh
serialization path.

Acceptance:

- UI shell code does not build source models or tessellate fixtures.
- UI thread work for a completed preview is bounded and measurable.
- Stale payloads are discarded by request id before scene mutation.

### 6. Decide Embedded Preview Versus Supervised CLI Preview Adapter

After the shared preview component and process boundary are in place, test the
embedded `pyvistaqt.QtInteractor` route again. If it remains unstable on macOS,
switch the workbench preview pane to a supervised adapter around the existing
`impression preview` behavior rather than continuing to debug a separate
interactive renderer.

Acceptance:

- The review shell can use either an embedded adapter or a supervised preview
  adapter without changing fixture queue, notes, promotion, or artifact panels.
- Renderer lifetime is explicit: create once for the preview surface and
  destroy only when that surface closes.

### 7. Quarantine Or Remove The PNG Snapshot Path

The PNG renderer should not be part of the live interactive review preview.
If a thumbnail or artifact snapshot utility is still useful, rename and isolate
it so it cannot be mistaken for the preview backend.

Acceptance:

- `render_stl_preview` is not exported from the live UI package.
- Workbench runtime code does not call the PNG snapshot renderer.
- Tests that mention PNG previews are limited to artifact or thumbnail utility
  behavior.

### 8. Verification

Minimum automated checks:

- shell launch does not start preview work
- worker module imports without Qt widget or PyVistaQt imports
- process controller handles success, failure, cancellation, and stale results
- stale result from fixture A cannot overwrite fixture B
- CLI preview and workbench share the same scene application component

Manual smoke:

```bash
.venv/bin/impression-reference-review --fixture-file tests/reference_review_fixtures/dirty-impress-fixtures.json
```

Manual pass criteria:

- app opens without beachballing
- fixture queue is interactive before any preview is loaded
- selecting a real `.impress` fixture loads a live preview
- dragging renders continuously and follows CLI preview controls
- changing fixtures cancels or stales old work
- closing the app releases the preview renderer and worker process

## Change History

- 2026-07-07: Added remediation plan after live preview beachballing and render
  instability showed the workbench preview had diverged from the CLI preview
  architecture.
