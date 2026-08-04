# Fix 10 Test: SurfaceBody Preview and Export Consumption

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-10-surfacebody-preview-export-consumption-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: `project/release-0.1.0a/architecture/surface-first-internal-model.md`

## Overview

Verify direct `SurfaceBody` consumption through separate preview and export routes.

## Application Integration Under Test

- App type: mixed.
- User/caller surface: GUI preview viewport and console export command.
- Invocation route: model result -> shared collector -> route policy -> viewport/export.
- Wiring owner/module: `src/impression/preview.py`, called from `src/impression/cli.py`.
- Observable result: render dataset and export-ready mesh.
- Integration validation: separate preview and export route tests.

## Backlink

[Fix 10 specification](../specifications/fix-10-surfacebody-preview-export-consumption-v1_0.md)

## Manual Smoke

Create a model returning one surface-first box body; preview it, export it, and
confirm no explicit tessellation call is present in the model.

## Automated Smoke Tests

Pass a `SurfaceBody` through scene collection and assert it yields a non-empty
dataset using a spy that records the preview tessellation policy.

## Automated Acceptance Tests

- Run CLI preview collection and STL export for a direct `SurfaceBody` result.
- Assert preview/export policies are distinct and invoked once per surface payload.
- Cover nested/mixed groups with surfaces, meshes, polylines, and transforms.
- Assert stable traversal order and exactly-once transform application.
- Assert unsupported payloads receive a named diagnostic.

Tests use deterministic primitive surfaces and do not require an interactive window.

## App-Type Proof

- GUI proof: offscreen preview collection reaches the viewport payload with preview policy.
- Console proof: export command reaches shared collector with export policy and produces mesh.
- Mixed-surface proof: each route has a separate failure assertion.
- API/service and library-only proof: not applicable.

## Fixtures And Data

- Direct surface body and nested mixed groups with stable transforms/order.
- Production-data rule: temporary deterministic models only.

## Acceptance

- [x] Feature spec is canonical and both independently failing routes are covered.
- [x] Observable route outputs, policy, ordering, transforms, and refusal are asserted.
- [x] Shared-helper-only tests cannot satisfy the contract.
