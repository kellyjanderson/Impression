# Preview Import Isolation For Switched Model Roots

Status: proposed
Discovered during: CLI preview `.impress` and include-watch spike
Severity: medium
Scope: cross-module

## Summary

The CLI preview now tracks and reloads the active model's transitive local include tree, but switching between separate model roots that reuse the same bare module names can still inherit stale `sys.modules` entries before the new tree is discovered.

## Locations

```code-location
file: src/impression/cli.py
lines: 55-82
symbol: _load_module
```

```code-location
file: src/impression/cli.py
lines: 324-366
symbol: _scene_factory_from_module
```

```code-location
file: src/impression/cli.py
lines: 572-585
symbol: preview._get_scene_factory
```

## Problem

The active preview factory drops modules it already tracked before rebuilding, which keeps same-model include edits fresh. When the preview switches to a different model root, the new factory starts without the previous factory's tracked module names. If both roots import a bare module name such as `parts` or `lib`, Python can resolve that name from an existing `sys.modules` entry before the new root's include tree is tracked.

## Why Not Fixed Now

The current spike is scoped to direct `.impress` preview loading and hot reload for the active model's imported include tree. Fully isolating switched model roots needs a small import-session boundary design so it does not accidentally evict unrelated test runner, plugin, or application modules.

## Proposed Improvement

Introduce a preview import session object that owns loaded user-module names across model switches. Route `_scene_factory_from_path` creation through that session so changing the preview target can dispose the previous root's user modules before importing the new root. Keep Impression package modules and environment/site-package modules outside the disposal set.

## Validation Needed

Add a regression test with two separate preview roots that both import the same bare local module name and verify switching roots loads the second root's module, not the first root's stale module. Keep the existing transitive include edit/reload coverage.
