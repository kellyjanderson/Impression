# Impression Preview hot reload fails

## Summary
When running the Impression Preview with hot reload enabled, touching the model file stops the preview loop. The UI shows a reload failure and subsequent file changes are ignored until the preview process is restarted.

## Steps to reproduce
1. Launch the preview CLI with `impression preview` (or through VS Code) using a model such as `examples/cylinder_torus_arch.py`.
2. Save the model file to trigger hot reload.
3. Observe the console output and whether further reloads happen.

## Observed behavior
```
Using model cylinder_torus_arch.py
Watching for changes — save to hot reload, close the window to stop.
...
Reloading cylinder_torus_arch.py…
╭──────────────────────── Reload failed ─────────────────────────╮
│ 'PyVistaPreviewer' object has no attribute '_collect_datasets' │
╰────────────────────────────────────────────────────────────────╯
```
After the first failure the watcher no longer responds to file saves.

## Expected behavior
Hot reload should recover from errors, continue watching the model file, and only log issues without killing the preview loop.

## Notes
The callback that processes reloads calls a non-existent method (`_collect_datasets`), which raises an `AttributeError`. The resulting exception bubbles out of the PyVista timer callback, so PyVista stops invoking it and no more reloads are processed.

## Proposed fix
- Use the existing `collect_datasets` helper instead of the missing `_collect_datasets` method.
- Wrap the timer callback in a guard that catches any unhandled exception (showing a console error panel) so the PyVista event loop keeps running even when rendering or scene execution fails.
