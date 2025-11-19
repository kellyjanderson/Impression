# PR: Preserve camera on hot reload

- **Issue**: [docs/issues/camera-reset-on-hot-reload.md](../issues/camera-reset-on-hot-reload.md)
- **Branch**: `bugfix/camera-reset-on-hot-reload`
- **Assignee/Reviewer**: @kellyjanderson

## Summary
1. Stop calling `_reset_camera()` during hot reload updates so user camera adjustments persist.
2. Keep the initial render aligned to the model to maintain a sane starting viewpoint.

## Testing
- Manual: Not run (PyVista preview requires GUI runtime).
- Automated: Not run (no preview tests available).
