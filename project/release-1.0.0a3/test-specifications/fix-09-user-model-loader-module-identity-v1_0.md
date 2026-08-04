# Fix 09 Test: User-Model Loader Module Identity

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One sequential-load isolation suite proves canonical package identity and user-code refresh together.

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
