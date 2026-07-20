# Surface Spec 428 Test: Loft Primitive Runtime Validity And Persistence Gate

Date: 2026-07-16
Status: Superseded
Feature spec: `../specifications/surface-428-loft-primitive-runtime-validity-and-persistence-gate-v1_0.md`
Feature spec canonical status: Superseded parent
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`

## Overview

This test specification verifies runtime validity, persistence readiness, tessellation-boundary readiness, and no-hidden-mesh proof for loft/primitive cut results.

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API result metadata and downstream reference workflow readiness
- Invocation route: assembled candidate body to validity/persistence gate to result finalizer
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: accepted body metadata or invalid/unsupported result
- Integration validation: public API tests proving accepted body metadata, invalid refusal, persistence readiness, and no-hidden-mesh evidence

## Manual Smoke

- Run an accepted cut case and confirm the public result carries validity and no-hidden-mesh evidence.

## Automated Smoke Tests

- `tests/test_surface_csg.py` validates one accepted candidate body through the gate.

## Automated Acceptance Tests

- Unit/helper behavior:
  - validity record, persistence readiness, tessellation-boundary readiness, no-hidden-mesh proof.
- Integrated route behavior:
  - public API result metadata reflects accepted or invalid gate outcome.
- Failure and stale-result behavior, if applicable:
  - invalid candidate bodies and mesh-backed fragments refuse.

## App-Type Proof

- GUI proof: not applicable.
- Console proof: not applicable.
- API/service proof: not applicable.
- Mixed-surface proof: not applicable.
- Library-only proof: public Boolean API route tests assert validity payloads.

## Fixtures And Data

- In-memory assembled candidate shells and invalid-shell fixtures.
- Production-data rule: tests must not require production data.

## Acceptance

- [ ] Feature spec is canonical, or this test spec is explicitly temporary while split coverage is incomplete.
- [ ] Route-level proof exists for the app type.
- [ ] Helper-only tests cannot satisfy this feature contract.
- [ ] Observable result is asserted or manually checked.
- [ ] Failure behavior is covered where applicable.
