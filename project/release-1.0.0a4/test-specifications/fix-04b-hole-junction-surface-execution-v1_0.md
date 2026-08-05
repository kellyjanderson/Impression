# Fix 04B Test: Hole Junction Surface Execution

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 04B: Hole Junction Surface Execution](../specifications/fix-04b-hole-junction-surface-execution-v1_0.md)
Feature spec canonical status: Canonical
Architecture ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)

## Overview

This canonical paired contract verifies the complete retained split-child boundary.

## Application Integration Under Test

- App type: library-only
- User/caller surface: `Loft(...)` and published split/merge examples
- Invocation route: `Loft(...)` and published split/merge examples -> `src/impression/modeling/loft.py`
- Wiring owner/module: `src/impression/modeling/loft.py`
- Observable result: closed `SurfaceBody` with valid evidence
- Integration validation: `tests/test_loft_surface_executor_correspondence.py`; `tests/test_loft_showcase.py`

## Manual Smoke

- Exercise `Loft(...)` and published split/merge examples with the parent issue fixture and inspect closed `SurfaceBody` with valid evidence.

## Automated Smoke Tests

- A fast route-level test reaches `src/impression/modeling/loft.py` through `Loft(...)` and published split/merge examples.

## Automated Acceptance Tests

- Unit/helper behavior:
  - junction patch builder
  - junction seam assembler
  - cap/closure result validator
- Integrated route behavior:
  - `Loft(...)` and published split/merge examples asserts every child-owned acceptance outcome.
- Failure and stale-result behavior, if applicable:
  - orientation, self-intersection, seam, or closure failure returns no body

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
  - `Loft(...)` and published split/merge examples is exercised as the real consuming route

## Fixtures And Data

- Parent issue #245 deterministic fixture and focused negative controls.
- Production-data rule: no user production data is required.

## Acceptance

- [x] Feature child is canonical.
- [ ] Route-level proof exists for library-only.
- [ ] Helper-only tests cannot satisfy the contract.
- [ ] Observable results and failure behavior are asserted.
