# Camera reset on hot reload

## Summary
When using Impression Preview with hot reload enabled, interacting with the PyVista window (panning, zooming, rotating) is undone every time a reload occurs. The preview resets the camera and bounds on each reload, making it difficult to inspect particular areas while iterating on a model.

## Steps to reproduce
1. Run `impression preview` (or the VS Code task) on any model.
2. Adjust the camera (zoom in on a detail, rotate to a specific angle).
3. Save the model file to trigger a hot reload.

## Observed behavior
After the reload finishes, the preview jumps back to the default camera alignment, losing all custom adjustments.

## Expected behavior
Hot reload should only update the geometry/materials while keeping the current camera position. Users should be able to zoom into a detail and keep that view across reloads.

## Notes
The reload path in `PyVistaPreviewer.show()` always passes `align_camera=True` to `_apply_scene`, causing `_reset_camera()` to run on every update. We should skip the camera reset for hot reloads and only align it for the initial render (or when explicitly requested).
