# Fix 09 Test: User-Model Loader Module Identity

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-09-user-model-loader-module-identity-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: not applicable

## Overview

Verify user-model refresh and canonical Impression module identity across repeated loads.

## Application Integration Under Test

- App type: console.
- User/caller surface: preview/export commands loading model files.
- Invocation route: command -> `_load_module` -> model factory -> scene result.
- Wiring owner/module: `src/impression/cli.py`.
- Observable result: refreshed model behavior with canonical class identity.
- Integration validation: sequential real loader calls with edited model/helper files.

## Backlink

[Fix 09 specification](../specifications/fix-09-user-model-loader-module-identity-v1_0.md)

## Manual Smoke

Load a temporary model, edit its local helper, reload, and confirm the changed
result while comparing its class objects to canonical `impression.modeling` imports.

## Automated Smoke Tests

Import a modeling class before model load, return that class from a model, and
assert identity and `isinstance` still hold after load.

## Automated Acceptance Tests

- Snapshot canonical `impression` module objects before and after repeated loads.
- Modify model and owned helper code between loads and assert refresh.
- Preload an unrelated same-named module and assert it is not removed.
- Exercise syntax/import/runtime failure cleanup without module leakage.
- Preserve dataclass execution and existing preview-isolation tests.

Use temporary source trees with explicit module ownership expectations.

## App-Type Proof

- Console proof: real CLI loader path, returned result/error, and module-registry side effects.
- GUI, API/service, mixed, and library-only proof: not applicable.

## Fixtures And Data

- Temporary model/helper trees, canonical imports, and unrelated module sentinel.
- Production-data rule: temporary local code only.

## Acceptance

- [x] Feature spec is canonical and real loader route is exercised.
- [x] Refresh, identity, cleanup, and failure results are asserted.
- [x] Helper-only ownership tests cannot satisfy the contract.
