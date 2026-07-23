# Reference Review Rendering Path Code Review

Date: 2026-07-23

## Context

The Reference Review rendering path went through many stabilization patches while the PySide6, PyVista, pyvistaqt, VTK, and manifold packages were also upgraded. This review separates current rendering-path behavior from stale compatibility patches that may have been valid for older package versions but are now risk or clutter.

Current package versions observed in the repo virtualenv:

| Package | Version |
| --- | --- |
| PySide6 | 6.11.1 |
| pyvista | 0.48.4 |
| pyvistaqt | 0.12.0 |
| vtk | 9.6.2 |
| manifold3d | 3.5.2 |

## Main Rendering Path

The current live Reference Review preview path is coherent:

1. The selected fixture launches preview payload work through `PreviewPayloadProcessController`.
2. The worker builds preview datasets outside the UI thread.
3. The worker serializes those datasets to a JSON preview payload.
4. The Qt thread drains `PreviewRenderCommandQueue`.
5. `PreviewRendererLifecycleWidget` applies payload/display commands.
6. `PyVistaQtPreviewSurface` wraps the shared `QtPreviewSurface`.
7. `QtPreviewSurface` owns one long-lived pyvistaqt `QtInteractor`.
8. Shared scene semantics come from `PreviewSceneController`, which is also used by CLI preview.

This is the correct direction: the UI owns UI/render mutations, payload building is off the UI thread, stale results are identity-gated, and rendering uses the shared CLI-style preview controller instead of a custom raster path.

## Findings

### P2: VTK Runtime Patch Is Now the Riskiest Stale Compatibility Shim

Location:

```code-location
file: src/impression/_vtk_runtime.py
lines: 23-112
symbol: ensure_vtk_runtime
```

`ensure_vtk_runtime()` does more than set environment flags. It can rename and symlink files inside the installed `vtkmodules/.dylibs` directory.

Current observed VTK 9.6.2 wheel state:

```text
libvtkRenderingUI.dylib
libvtkRenderingOpenGL2.dylib
```

There were no numbered duplicate dylibs in the current venv. That means the file-mutation path is presently fixing a non-issue while retaining the ability to reshape the installed package.

Recommendation: remove the file-mutation behavior by default, or gate it behind an explicit emergency environment variable. Keep only non-invasive runtime setup unless a failing wheel shape is actually detected.

### P2: Old Cached PNG/Offscreen Renderer Is Still Exported

Location:

```code-location
file: src/impression/devtools/reference_review/ui/artifact_preview.py
lines: 57-168
symbol: ArtifactPreviewRenderer
```

```code-location
file: src/impression/devtools/reference_review/ui/__init__.py
lines: 7-202
symbol: render_stl_preview export
```

`artifact_preview.py` still owns a persistent offscreen PyVista renderer, writes cached PNGs, and sets `PYVISTA_OFF_SCREEN`. The current widget shell passes an empty `artifact_previews` map and uses the live payload-worker plus `QtPreviewSurface` route instead.

This looks like an older renderer route kept alive by exports and tests. It is not the current interactive preview dependency. Keeping it exported makes it easy to accidentally reintroduce the PNG pathway.

Recommendation: delete or demote this module if no supported product surface still needs cached artifact thumbnails. If screenshots are needed, keep them as explicit capture/test evidence, not as a fallback preview renderer.

### P2: Object-Edge Extraction Still Tries an Old PyVista API First

Location:

```code-location
file: src/impression/preview.py
lines: 430-448
symbol: PreviewSceneController._extract_feature_edges
```

With pyvista 0.48.4, `extract_feature_edges(angle=...)` raises `TypeError`. The current code catches that and retries the modern `feature_angle=...` signature.

This is not a correctness bug, but it adds an exception to every object-edge render and is a clear sign of stale compatibility layering.

Recommendation: call the current `feature_angle` signature first and keep the older `angle` call only as fallback.

### P3: Qt Surface Format Hook Is Correct but Misleadingly Named

Location:

```code-location
file: src/impression/preview_qt.py
lines: 114-119
symbol: configure_qt_preview_surface_format
```

The function is now intentionally a no-op because forcing OpenGL profiles caused either Qt shader crashes or a live black render target. That behavior is correct for the current macOS/PySide6/VTK stack, but the function name invites future patches to mutate the surface format again.

Recommendation: rename it to something like `preserve_qt_preview_surface_format()` or remove the call sites once tests protect the no-forced-format contract.

### P3: Legacy Live-Preview Branches Remain Beside the Payload Route

Locations:

```code-location
file: src/impression/devtools/reference_review/ui/shell.py
lines: 478-530
symbol: LiveArtifactPreviewWidget
```

```code-location
file: src/impression/devtools/reference_review/ui/shell.py
lines: 1045-1048
symbol: ReferenceReviewWindow._apply_preview_load
```

`set_artifact()`, `_apply_build_result()`, and `_apply_preview_load()` look like remnants from a direct artifact-build route. The current selected-fixture path calls `prepare_artifact()`, launches a payload worker, and enqueues `PreviewRenderCommand.payload_ready(...)`.

These branches are not causing the current rendering behavior, but they make the lifecycle harder to reason about and can confuse future debugging.

Recommendation: remove unreachable branches or write an explicit test/product route showing why they are still needed.

## Current Good Shape

The following parts are worth keeping:

- `QtPreviewSurfaceConfig.workbench_default()` uses `qvtk_base="QWidget"`.
- The preview no longer forces `QT_WIDGETS_RHI=0`.
- The preview no longer forces an OpenGL profile/version.
- `QtPreviewSurface` owns one long-lived `QtInteractor`.
- `PreviewSceneController` is shared by CLI and embedded preview.
- Preview payload building is off the UI thread.
- Render mutations are drained on the Qt thread through `PreviewRenderCommandQueue`.
- The display controls route to `PreviewSceneApplyOptions` instead of directly manipulating VTK from the shell.

## Validation Run During Review

Focused tests:

```text
.venv/bin/python -m pytest tests/test_preview_controller.py tests/test_reference_review_ui_shell.py tests/test_reference_review_preview_payload_controller.py -q
103 passed
```

Real first-STL Qt preview smoke:

```text
fixture loft/anchor_shift_rectangle
unique sampled colors: 1129
```

The smoke confirmed the live Qt widget rendered nonblank pixels through the current route.
