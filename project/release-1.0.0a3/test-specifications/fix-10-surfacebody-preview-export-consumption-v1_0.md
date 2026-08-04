# Fix 10 Test: SurfaceBody Preview and Export Consumption

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One primary-consumer integration matrix proves direct SurfaceBody preview/export without hidden model-side adapters.

## Backlink

[Fix 10 specification](../specifications/fix-10-surfacebody-preview-export-consumption-v1_0.md)

## Manual Smoke

Create a model returning one surface-first box body; preview it, export it, and
confirm no explicit tessellation call is present in the model.

## Automated Smoke

Pass a `SurfaceBody` through scene collection and assert it yields a non-empty
dataset using a spy that records the preview tessellation policy.

## Automated Acceptance

- Run CLI preview collection and STL export for a direct `SurfaceBody` result.
- Assert preview/export policies are distinct and invoked once per surface payload.
- Cover nested/mixed groups with surfaces, meshes, polylines, and transforms.
- Assert stable traversal order and exactly-once transform application.
- Assert unsupported payloads receive a named diagnostic.

Tests use deterministic primitive surfaces and do not require an interactive window.
