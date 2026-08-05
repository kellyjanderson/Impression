# Fix 04A Test: Hole Junction Plan Records

Date: 2026-08-04
Status: Final
Feature spec: [Fix 04A: Hole Junction Plan Records](../specifications/fix-04a-hole-junction-plan-records-v1_0.md)
Feature spec canonical status: Canonical
Architecture ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)

## Overview

This canonical paired contract verifies the complete retained split-child boundary.

## Application Integration Under Test

- App type: library-only
- User/caller surface: loft planning consumed by the surface executor
- Invocation route: loft planning consumed by the surface executor -> `src/impression/modeling/loft.py`
- Wiring owner/module: `src/impression/modeling/loft.py`
- Observable result: validated junction event consumed by executor
- Integration validation: `tests/test_loft_point_birth_death_resolution.py`; `tests/test_loft_point_lifecycle_records.py`

## Manual Smoke

- Exercise loft planning consumed by the surface executor with the parent issue fixture and inspect validated junction event consumed by executor.

## Automated Smoke Tests

- A fast route-level test reaches `src/impression/modeling/loft.py` through loft planning consumed by the surface executor.

## Automated Acceptance Tests

- Unit/helper behavior:
  - junction event builder
  - junction lineage validator
  - junction boundary-input derivation
- Integrated route behavior:
  - loft planning consumed by the surface executor asserts every child-owned acceptance outcome.
- Failure and stale-result behavior, if applicable:
  - ambiguous or incomplete lineage fails before surface execution

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
  - loft planning consumed by the surface executor is exercised as the real consuming route

## Fixtures And Data

- Parent issue #245 deterministic fixture and focused negative controls.
- Production-data rule: no user production data is required.

## Acceptance

- [x] Feature child is canonical.
- [x] Route-level proof exists for library-only.
- [x] Helper-only tests cannot satisfy the contract.
- [x] Observable results and failure behavior are asserted.
