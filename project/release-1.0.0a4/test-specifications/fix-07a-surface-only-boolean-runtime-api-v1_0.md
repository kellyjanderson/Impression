# Fix 07A Test: Surface-Only Boolean Runtime API

Date: 2026-08-04
Status: Final
Feature spec: [Fix 07A: Surface-Only Boolean Runtime API](../specifications/fix-07a-surface-only-boolean-runtime-api-v1_0.md)
Feature spec canonical status: Canonical
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)

## Overview

This canonical paired contract verifies the complete retained split-child boundary.

## Application Integration Under Test

- App type: library-only
- User/caller surface: public `impression.modeling` boolean functions
- Invocation route: public `impression.modeling` boolean functions -> `src/impression/modeling/csg.py` and `src/impression/modeling/__init__.py`
- Wiring owner/module: `src/impression/modeling/csg.py` and `src/impression/modeling/__init__.py`
- Observable result: surfaced result or actionable representation error
- Integration validation: `tests/test_surface_csg.py` runtime/signature matrix

## Manual Smoke

- Exercise public `impression.modeling` boolean functions with the parent issue fixture and inspect surfaced result or actionable representation error.

## Automated Smoke Tests

- A fast route-level test reaches `src/impression/modeling/csg.py` and `src/impression/modeling/__init__.py` through public `impression.modeling` boolean functions.

## Automated Acceptance Tests

- Unit/helper behavior:
  - surface operand validator
  - public boolean boundary update
  - mesh utility export separation
- Integrated route behavior:
  - public `impression.modeling` boolean functions asserts every child-owned acceptance outcome.
- Failure and stale-result behavior, if applicable:
  - mesh/mixed inputs identify the separate non-modeling utility

## App-Type Proof

- GUI proof:
  - not applicable
- Console proof:
  - not applicable
- API/service proof:
  - not applicable
- Mixed-surface proof:
  - not applicable
- Library-only proof:
  - public `impression.modeling` boolean functions is exercised as the real consuming route

## Fixtures And Data

- Parent issue #247 deterministic fixture and focused negative controls.
- Production-data rule: no user production data is required.

## Acceptance

- [x] Feature child is canonical.
- [x] Route-level proof exists for library-only.
- [x] Helper-only tests cannot satisfy the contract.
- [x] Observable results and failure behavior are asserted.

## Validation Evidence

- Runtime introspection asserts all three public parameter names, surfaced
  operand annotations, tolerance annotations, and the uniform
  `SurfaceBooleanResult` return annotation.
- Calls imported through `impression.modeling` prove valid surfaced operands
  still reach the real boolean routes and return structured results.
- Mesh, `MeshGroup`, base, cutter, and mixed-collection controls prove
  actionable `TypeError` occurs before family-gate dispatch.
- Export inspection proves `union_meshes` is absent from top-level modeling and
  available through `impression.modeling.mesh_tools`.
- Focused runtime/signature/export matrix: 8 passed.
- Surface CSG and retained mesh-tool group: 262 passed.
- Full repository suite: 1,778 passed.
