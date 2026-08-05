# Fix 09B Test: Difference Public Success Gate

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 09B: Difference Public Success Gate](../specifications/fix-09b-difference-public-success-gate-v1_0.md)
Feature spec canonical status: Canonical
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)

## Overview

This canonical paired contract verifies the complete retained split-child boundary.

## Application Integration Under Test

- App type: library-only
- User/caller surface: public `boolean_difference` and every registered surfaced executor
- Invocation route: public `boolean_difference` and every registered surfaced executor -> `src/impression/modeling/csg.py`
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: truthful `SurfaceBooleanResult` classification
- Integration validation: `tests/test_surface_csg.py` public/registry matrix; rotated snap-groove false-success regression

## Manual Smoke

- Exercise public `boolean_difference` and every registered surfaced executor with the parent issue fixture and inspect truthful `SurfaceBooleanResult` classification.

## Automated Smoke Tests

- A fast route-level test reaches `src/impression/modeling/csg.py` through public `boolean_difference` and every registered surfaced executor.

## Automated Acceptance Tests

- Unit/helper behavior:
  - public difference postcondition
  - no-cut classifier
  - executor registry gate assertion
- Integrated route behavior:
  - public `boolean_difference` and every registered surfaced executor asserts every child-owned acceptance outcome.
- Failure and stale-result behavior, if applicable:
  - executor bypass or missing evidence fails validation

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
  - public `boolean_difference` and every registered surfaced executor is exercised as the real consuming route

## Fixtures And Data

- Parent issue #248 deterministic fixture and focused negative controls.
- Production-data rule: no user production data is required.

## Acceptance

- [x] Feature child is canonical.
- [ ] Route-level proof exists for library-only.
- [ ] Helper-only tests cannot satisfy the contract.
- [ ] Observable results and failure behavior are asserted.
