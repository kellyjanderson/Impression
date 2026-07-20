# PR: Keep Impression Preview hot reload alive after errors

- **Issue source**: retired local issue record; use GitHub Issues for new issue tracking.
- **Branch**: `bugfix/impression-preview-hot-reload`
- **Assignee/Reviewer**: @k

## Summary
1. Fix the attribute error by calling `collect_datasets` during reloads.
2. Wrap the timer callback with a guard that logs failures to the console but prevents PyVista from dropping the callback, so future file changes still trigger reloads.

## Original Issue Summary

When running Impression Preview with hot reload enabled, touching the model file
stopped the preview loop. The UI showed a reload failure and subsequent file
changes were ignored until the preview process was restarted.

Observed failure:

```text
'PyVistaPreviewer' object has no attribute '_collect_datasets'
```

Expected behavior: hot reload should recover from errors, continue watching the
model file, and log issues without killing the preview loop.

Implementation note: the reload path should use `collect_datasets`, and the
PyVista timer callback should catch unhandled exceptions so the event loop keeps
running.

## Testing
- Manual: Not run (PyVista preview requires the GUI runtime).
- Automated: Not run (PyVista preview currently lacks automated coverage).
