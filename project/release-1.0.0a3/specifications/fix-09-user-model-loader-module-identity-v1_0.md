# Fix 09: User-Model Loader Module Identity (v1.0)

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Isolate user-model reloads from Impression package modules

- Input: the canonical Impression runtime plus a reloadable user model and helpers.
- Work: track/clean only model-owned modules, retain Impression modules, and execute
  changed model-owned code on the next load.
- Output: refreshed user behavior that shares the runtime's canonical class objects.
- Complete when: sequential-load identity, isolation, refresh, and failure tests pass.

## Problem And Outcome

The user-model loader deletes `impression.modeling` and its submodules from
`sys.modules`. Existing objects can then belong to old class definitions while a
newly loaded model imports replacements, breaking `isinstance`, dispatch, and
serialization. Reloading a user model must not reload installed Impression code.

## Scope

- Give each loaded user model a controlled module namespace and cleanup set.
- Retain canonical `impression` package/module objects across loads.
- Refresh changed user-model code and its owned local helper modules.
- Preserve preview isolation and repeat-load behavior.

Not in scope: a general Python plugin sandbox or process isolation redesign.

## Implementation Routing

- `src/impression/cli.py`: model load, module tracking, cleanup, and finish path.
- `tests/test_preview_isolation.py`, CLI preview tests, and focused identity tests.

## Contract

Inputs are a model path and the already imported Impression runtime. Output is a
loaded user-model module whose Impression classes are object-identical to the
runtime's classes. Cleanup may remove only names owned by the prior user-model
load. A changed model/helper is re-executed on the next load.

## Acceptance Criteria

- Class identity from model output matches the caller's canonical imports.
- Two sequential model loads reflect edited user code without reloading Impression.
- Cleanup does not remove unrelated application or third-party modules.
- Existing dataclass, preview isolation, and error cleanup tests remain green.

## Verification

[Paired test specification](../test-specifications/fix-09-user-model-loader-module-identity-v1_0.md)
