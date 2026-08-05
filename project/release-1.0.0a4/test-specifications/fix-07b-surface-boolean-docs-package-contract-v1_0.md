# Fix 07B Test: Surface Boolean Docs And Package Contract

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 07B: Surface Boolean Docs And Package Contract](../specifications/fix-07b-surface-boolean-docs-package-contract-v1_0.md)
Feature spec canonical status: Canonical
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)

## Overview

This canonical paired contract verifies the complete retained split-child boundary.

## Application Integration Under Test

- App type: library-only
- User/caller surface: installed package, API documentation, tutorials, and examples
- Invocation route: installed package, API documentation, tutorials, and examples -> `docs/modeling/csg.md`, `docs/examples/csg/`, package smoke tests
- Wiring owner/module: `docs/modeling/csg.md`, `docs/examples/csg/`, package smoke tests
- Observable result: consistent source/docs/wheel contract and migration guidance
- Integration validation: `tests/test_surface_csg_docs.py`; clean-wheel smoke

## Manual Smoke

- Exercise installed package, API documentation, tutorials, and examples with the parent issue fixture and inspect consistent source/docs/wheel contract and migration guidance.

## Automated Smoke Tests

- A fast route-level test reaches `docs/modeling/csg.md`, `docs/examples/csg/`, package smoke tests through installed package, API documentation, tutorials, and examples.

## Automated Acceptance Tests

- Unit/helper behavior:
  - documentation/example migration
  - API inventory guard
  - clean-wheel smoke
- Integrated route behavior:
  - installed package, API documentation, tutorials, and examples asserts every child-owned acceptance outcome.
- Failure and stale-result behavior, if applicable:
  - stale mesh signature/example or package mismatch fails validation

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
  - installed package, API documentation, tutorials, and examples is exercised as the real consuming route

## Fixtures And Data

- Parent issue #247 deterministic fixture and focused negative controls.
- Production-data rule: no user production data is required.

## Acceptance

- [x] Feature child is canonical.
- [ ] Route-level proof exists for library-only.
- [ ] Helper-only tests cannot satisfy the contract.
- [ ] Observable results and failure behavior are asserted.
