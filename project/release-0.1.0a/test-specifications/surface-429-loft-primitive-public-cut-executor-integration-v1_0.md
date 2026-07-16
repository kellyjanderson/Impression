# Surface Spec 429 Test: Loft Primitive Public Cut Executor Integration

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-429-loft-primitive-public-cut-executor-integration-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`

## Overview

This test specification verifies the public `SurfaceBooleanResult` route for cut-producing loft/primitive CSG.

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API `SurfaceBooleanResult`
- Invocation route: `surface_boolean_result` to exact reuse or trim-fragment cut executor
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: public API accepted/unsupported/invalid results with cut-shell metadata
- Integration validation: public API integration tests for accepted cut cases, structured refusals, and route precedence

## Manual Smoke

- Run public loft/primitive union, difference, and intersection calls and confirm accepted cases return non-null surface-native bodies.

## Automated Smoke Tests

- `tests/test_surface_csg.py` includes one public accepted cut case and one structured refusal.

## Automated Acceptance Tests

- Unit/helper behavior:
  - result metadata builder and route precedence guard.
- Integrated route behavior:
  - public API accepted union, difference, intersection, invalid seam refusal, and exact-reuse precedence.
- Failure and stale-result behavior, if applicable:
  - unsupported and invalid cases return structured diagnostics without hidden mesh fallback.

## App-Type Proof

- GUI proof: not applicable.
- Console proof: not applicable.
- API/service proof: not applicable.
- Mixed-surface proof: not applicable.
- Library-only proof: public Boolean API integration tests are required.

## Fixtures And Data

- In-memory loft/primitive operands for accepted and refused cut cases.
- Production-data rule: tests must not require production data.

## Acceptance

- [ ] Feature spec is canonical, or this test spec is explicitly temporary while split coverage is incomplete.
- [ ] Route-level proof exists for the app type.
- [ ] Helper-only tests cannot satisfy this feature contract.
- [ ] Observable result is asserted or manually checked.
- [ ] Failure behavior is covered where applicable.
