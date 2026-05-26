# PR: Preserve camera on hot reload

- **Issue source**: retired local issue record; use GitHub Issues for new issue tracking.
- **Branch**: `bugfix/camera-reset-on-hot-reload`
- **Assignee/Reviewer**: @kellyjanderson

## Summary
1. Stop calling `_reset_camera()` during hot reload updates so user camera adjustments persist.
2. Keep the initial render aligned to the model to maintain a sane starting viewpoint.

## Original Issue Summary

When using Impression Preview with hot reload enabled, interacting with the
PyVista window by panning, zooming, or rotating was undone every time a reload
occurred. The preview reset the camera and bounds on each reload, making it
difficult to inspect particular areas while iterating on a model.

Expected behavior: hot reload should update geometry and materials while
preserving the current camera position.

Implementation note: the reload path in `PyVistaPreviewer.show()` should align
the camera only for the initial render or an explicit camera-reset request, not
for ordinary hot reloads.

## Testing
- Manual: Not run (PyVista preview requires GUI runtime).
- Automated: Not run (no preview tests available).
