# Fix 08B Test: Loft Difference Branch Decomposition

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 08B: Loft Difference Branch Decomposition](../specifications/fix-08b-loft-difference-branch-decomposition-v1_0.md)
Feature spec canonical status: Canonical
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)

## Overview

This canonical paired contract verifies the complete retained split-child boundary.

## Application Integration Under Test

- App type: library-only
- User/caller surface: branching loft eligibility consumed by difference execution
- Invocation route: branching loft eligibility consumed by difference execution -> `src/impression/modeling/csg.py` with lineage read from `src/impression/modeling/loft.py`
- Wiring owner/module: `src/impression/modeling/csg.py` with lineage read from `src/impression/modeling/loft.py`
- Observable result: validated sub-body cut plan and recomposition map
- Integration validation: `tests/test_surface_csg.py` branch fixtures; audio-cube branched cutter regression

## Manual Smoke

- Exercise branching loft eligibility consumed by difference execution with the parent issue fixture and inspect validated sub-body cut plan and recomposition map.

## Automated Smoke Tests

- A fast route-level test reaches `src/impression/modeling/csg.py` with lineage read from `src/impression/modeling/loft.py` through branching loft eligibility consumed by difference execution.

## Automated Acceptance Tests

- Unit/helper behavior:
  - branch graph validator
  - bounded sub-body decomposition
  - recomposition-map validator
- Integrated route behavior:
  - branching loft eligibility consumed by difference execution asserts every child-owned acceptance outcome.
- Failure and stale-result behavior, if applicable:
  - invalid lineage, duplicate ownership, or open recomposition seam refuses

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
  - branching loft eligibility consumed by difference execution is exercised as the real consuming route

## Fixtures And Data

- Parent issue #248 deterministic fixture and focused negative controls.
- Production-data rule: no user production data is required.

## Acceptance

- [x] Feature child is canonical.
- [ ] Route-level proof exists for library-only.
- [ ] Helper-only tests cannot satisfy the contract.
- [ ] Observable results and failure behavior are asserted.
