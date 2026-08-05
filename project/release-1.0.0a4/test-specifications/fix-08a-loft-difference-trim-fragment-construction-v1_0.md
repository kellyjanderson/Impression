# Fix 08A Test: Loft Difference Trim Fragment Construction

Date: 2026-08-04
Status: Final
Feature spec: [Fix 08A: Loft Difference Trim Fragment Construction](../specifications/fix-08a-loft-difference-trim-fragment-construction-v1_0.md)
Feature spec canonical status: Canonical
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)

## Overview

This canonical paired contract verifies the complete retained split-child boundary.

## Application Integration Under Test

- App type: library-only
- User/caller surface: loft surface-difference executor
- Invocation route: loft surface-difference executor -> `src/impression/modeling/csg.py`
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: validated trim fragments or precise refusal
- Integration validation: `tests/test_surface_csg.py` trim/fragment fixtures; `tests/csg_reference_fixtures.py` cutters

## Manual Smoke

- Exercise loft surface-difference executor with the parent issue fixture and inspect validated trim fragments or precise refusal.

## Automated Smoke Tests

- A fast route-level test reaches `src/impression/modeling/csg.py` through loft surface-difference executor.

## Automated Acceptance Tests

- Unit/helper behavior:
  - intersection candidate builder
  - closed trim constructor
  - patch fragmenter
- Integrated route behavior:
  - loft surface-difference executor asserts every child-owned acceptance outcome.
- Failure and stale-result behavior, if applicable:
  - missing/ambiguous trim closure refuses fragment construction

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
  - loft surface-difference executor is exercised as the real consuming route

## Fixtures And Data

- Parent issue #248 deterministic fixture and focused negative controls.
- Production-data rule: no user production data is required.

## Acceptance

- [x] Feature child is canonical.
- [x] Route-level proof exists for library-only.
- [x] Helper-only tests cannot satisfy the contract.
- [x] Observable results and failure behavior are asserted.
