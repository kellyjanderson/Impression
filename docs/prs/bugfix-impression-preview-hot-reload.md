# PR: Keep Impression Preview hot reload alive after errors

- **Issue**: [docs/issues/impression-preview-hot-reload-fails.md](../issues/impression-preview-hot-reload-fails.md)
- **Branch**: `bugfix/impression-preview-hot-reload`
- **Assignee/Reviewer**: @k

## Summary
1. Fix the attribute error by calling `collect_datasets` during reloads.
2. Wrap the timer callback with a guard that logs failures to the console but prevents PyVista from dropping the callback, so future file changes still trigger reloads.

## Testing
- Manual: Not run (PyVista preview requires the GUI runtime).
- Automated: Not run (PyVista preview currently lacks automated coverage).
