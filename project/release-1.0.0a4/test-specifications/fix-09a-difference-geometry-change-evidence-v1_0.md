# Fix 09A Test: Difference Geometry Change Evidence

Date: 2026-08-04
Status: Final
Feature spec: [Fix 09A: Difference Geometry Change Evidence](../specifications/fix-09a-difference-geometry-change-evidence-v1_0.md)
Feature spec canonical status: Canonical
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)

## Overview

This canonical paired contract verifies the complete retained split-child boundary.

## Application Integration Under Test

- App type: library-only
- User/caller surface: all surfaced difference executors
- Invocation route: all surfaced difference executors -> `src/impression/modeling/csg.py`
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: validated change witnesses or unchanged/ambiguous comparison
- Integration validation: `tests/test_surface_csg.py` witness/comparator matrix

## Manual Smoke

- Exercise all surfaced difference executors with the parent issue fixture and inspect validated change witnesses or unchanged/ambiguous comparison.

## Automated Smoke Tests

- A fast route-level test reaches `src/impression/modeling/csg.py` through all surfaced difference executors.

## Automated Acceptance Tests

- Unit/helper behavior:
  - executor evidence normalizer
  - geometry-change witness validator
  - unchanged-result comparator
- Integrated route behavior:
  - all surfaced difference executors asserts every child-owned acceptance outcome.
- Failure and stale-result behavior, if applicable:
  - missing or contradictory evidence is ambiguous, never changed

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
  - all surfaced difference executors is exercised as the real consuming route

## Fixtures And Data

- Parent issue #248 deterministic fixture and focused negative controls.
- Production-data rule: no user production data is required.

## Acceptance

- [x] Feature child is canonical.
- [x] Route-level proof exists for library-only.
- [x] Helper-only tests cannot satisfy the contract.
- [x] Observable results and failure behavior are asserted.
