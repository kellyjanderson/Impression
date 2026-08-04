# Fix 10 Test: SurfaceBody Preview and Export Consumption

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Verify direct SurfaceBody preview/export consumption

- Input: direct-body and mixed nested-group model fixtures with transforms.
- Work: spy on tessellation policy/count and assert ordering, transforms, legacy
  payload behavior, and unsupported-payload refusal.
- Output: primary preview/export consumer integration coverage.
- Complete when: CLI preview collection and STL export pass without model-side adapters.

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
