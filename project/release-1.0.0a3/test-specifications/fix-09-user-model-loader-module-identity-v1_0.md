# Fix 09 Test: User-Model Loader Module Identity

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Verify canonical package identity across model reloads

- Input: temporary user models/helpers, canonical package modules, and unrelated modules.
- Work: perform sequential edited loads and syntax/import/runtime failure cases while
  comparing module/class identity and ownership cleanup.
- Output: a loader isolation and refresh regression suite.
- Complete when: user code refreshes without package identity change or module leakage.

## Backlink

[Fix 09 specification](../specifications/fix-09-user-model-loader-module-identity-v1_0.md)

## Manual Smoke

Load a temporary model, edit its local helper, reload, and confirm the changed
result while comparing its class objects to canonical `impression.modeling` imports.

## Automated Smoke

Import a modeling class before model load, return that class from a model, and
assert identity and `isinstance` still hold after load.

## Automated Acceptance

- Snapshot canonical `impression` module objects before and after repeated loads.
- Modify model and owned helper code between loads and assert refresh.
- Preload an unrelated same-named module and assert it is not removed.
- Exercise syntax/import/runtime failure cleanup without module leakage.
- Preserve dataclass execution and existing preview-isolation tests.

Use temporary source trees with explicit module ownership expectations.
