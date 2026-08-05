# Fix 08C Test: Loft Difference Result Shell Reconstruction

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 08C: Loft Difference Result Shell Reconstruction](../specifications/fix-08c-loft-difference-result-shell-reconstruction-v1_0.md)
Feature spec canonical status: Canonical
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)

## Overview

This canonical paired contract verifies the complete retained split-child boundary.

## Application Integration Under Test

- App type: library-only
- User/caller surface: public `boolean_difference` and preview/export consumers
- Invocation route: public `boolean_difference` and preview/export consumers -> `src/impression/modeling/csg.py`
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: closed changed `SurfaceBody` or precise refusal
- Integration validation: `tests/test_surface_csg.py` public fixtures; preview/export consumer smoke

## Manual Smoke

- Exercise public `boolean_difference` and preview/export consumers with the parent issue fixture and inspect closed changed `SurfaceBody` or precise refusal.

## Automated Smoke Tests

- A fast route-level test reaches `src/impression/modeling/csg.py` through public `boolean_difference` and preview/export consumers.

## Automated Acceptance Tests

- Unit/helper behavior:
  - retained-fragment classifier
  - cutter boundary patch builder
  - result shell assembler/validator
- Integrated route behavior:
  - public `boolean_difference` and preview/export consumers asserts every child-owned acceptance outcome.
- Failure and stale-result behavior, if applicable:
  - ambiguous classification, open seams, invalid closure, or no change cannot succeed

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
  - public `boolean_difference` and preview/export consumers is exercised as the real consuming route

## Fixtures And Data

- Parent issue #248 deterministic fixture and focused negative controls.
- Production-data rule: no user production data is required.

## Acceptance

- [x] Feature child is canonical.
- [ ] Route-level proof exists for library-only.
- [ ] Helper-only tests cannot satisfy the contract.
- [ ] Observable results and failure behavior are asserted.
