# Surface Spec 423 Test: Loft Primitive Intersection Source Normalization

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-423-loft-primitive-intersection-source-normalization-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: `../architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`

## Overview

This test specification verifies normalized primitive source-region records for loft/primitive CSG.

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: public Boolean route to loft adapter to source normalizer
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: source records or unsupported diagnostics
- Integration validation: source-record tests plus public-route diagnostics

## Manual Smoke

- Run a public loft/primitive Boolean case and confirm unsupported source regions report structured diagnostics instead of generic adapter-only refusal.

## Automated Smoke Tests

- `tests/test_surface_csg.py` creates representative box, sphere, and cylinder source-region records.
- Public-route diagnostic smoke proves the normalizer is reachable.

## Automated Acceptance Tests

- Unit/helper behavior:
  - supported primitive source regions produce `LoftPrimitiveIntersectionSourceRecord` data.
  - unsupported regions produce deterministic diagnostics with no-hidden-mesh proof.
- Integrated route behavior:
  - public loft/primitive CSG reaches source normalization before later-stage refusal.
- Failure and stale-result behavior, if applicable:
  - missing adapter evidence refuses before normalization.

## App-Type Proof

- GUI proof: not applicable.
- Console proof: not applicable.
- API/service proof: not applicable.
- Mixed-surface proof: not applicable.
- Library-only proof: public Boolean API route tests assert source-normalization diagnostics.

## Fixtures And Data

- In-memory loft/primitive operands for box, sphere, cylinder, and unsupported region cases.
- Production-data rule: tests must not require production data.

## Acceptance

- [ ] Feature spec is canonical, or this test spec is explicitly temporary while split coverage is incomplete.
- [ ] Route-level proof exists for the app type.
- [ ] Helper-only tests cannot satisfy this feature contract.
- [ ] Observable result is asserted or manually checked.
- [ ] Failure behavior is covered where applicable.
