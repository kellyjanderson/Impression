# Fix 05A Test: Count-Changing Exact Region Pairing

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 05A: Count-Changing Exact Region Pairing](../specifications/fix-05a-count-changing-exact-region-pairing-v1_0.md)
Feature spec canonical status: Canonical
Architecture ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)

## Overview

This canonical paired contract verifies the complete retained split-child boundary.

## Application Integration Under Test

- App type: library-only
- User/caller surface: public loft planning
- Invocation route: public loft planning -> `src/impression/modeling/loft.py`
- Wiring owner/module: `src/impression/modeling/loft.py`
- Observable result: stable exact pairs plus explicit residual births/deaths
- Integration validation: `tests/test_loft_identity_first_pairing.py`

## Manual Smoke

- Exercise public loft planning with the parent issue fixture and inspect stable exact pairs plus explicit residual births/deaths.

## Automated Smoke Tests

- A fast route-level test reaches `src/impression/modeling/loft.py` through public loft planning.

## Automated Acceptance Tests

- Unit/helper behavior:
  - exact region identity resolver
  - residual candidate constructor
- Integrated route behavior:
  - public loft planning asserts every child-owned acceptance outcome.
- Failure and stale-result behavior, if applicable:
  - duplicate or contradictory ids fail before scoring

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
  - public loft planning is exercised as the real consuming route

## Fixtures And Data

- Parent issue #246 deterministic fixture and focused negative controls.
- Production-data rule: no user production data is required.

## Acceptance

- [x] Feature child is canonical.
- [ ] Route-level proof exists for library-only.
- [ ] Helper-only tests cannot satisfy the contract.
- [ ] Observable results and failure behavior are asserted.
