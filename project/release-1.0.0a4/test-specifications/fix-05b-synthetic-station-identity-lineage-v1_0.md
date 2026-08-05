# Fix 05B Test: Synthetic Station Identity Lineage

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 05B: Synthetic Station Identity Lineage](../specifications/fix-05b-synthetic-station-identity-lineage-v1_0.md)
Feature spec canonical status: Canonical
Architecture ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)

## Overview

This canonical paired contract verifies the complete retained split-child boundary.

## Application Integration Under Test

- App type: library-only
- User/caller surface: expanded loft plan consumed by `Loft(...)`
- Invocation route: expanded loft plan consumed by `Loft(...)` -> `src/impression/modeling/loft.py`
- Wiring owner/module: `src/impression/modeling/loft.py`
- Observable result: expanded plan with complete identity lineage
- Integration validation: `tests/test_loft_point_lifecycle_records.py`; `tests/test_loft.py` rail-pair regression

## Manual Smoke

- Exercise expanded loft plan consumed by `Loft(...)` with the parent issue fixture and inspect expanded plan with complete identity lineage.

## Automated Smoke Tests

- A fast route-level test reaches `src/impression/modeling/loft.py` through expanded loft plan consumed by `Loft(...)`.

## Automated Acceptance Tests

- Unit/helper behavior:
  - synthetic lineage constructor
  - lineage completeness validator
  - identity-bearing expansion handoff
- Integrated route behavior:
  - expanded loft plan consumed by `Loft(...)` asserts every child-owned acceptance outcome.
- Failure and stale-result behavior, if applicable:
  - missing, duplicate, or conflicting derived lineage fails before execution

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
  - expanded loft plan consumed by `Loft(...)` is exercised as the real consuming route

## Fixtures And Data

- Parent issue #246 deterministic fixture and focused negative controls.
- Production-data rule: no user production data is required.

## Acceptance

- [x] Feature child is canonical.
- [ ] Route-level proof exists for library-only.
- [ ] Helper-only tests cannot satisfy the contract.
- [ ] Observable results and failure behavior are asserted.
