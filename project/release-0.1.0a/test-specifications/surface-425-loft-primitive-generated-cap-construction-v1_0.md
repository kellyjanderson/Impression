# Surface Spec 425 Test: Loft Primitive Generated Cap Construction

Date: 2026-07-16
Status: Superseded
Feature spec: `../specifications/surface-425-loft-primitive-generated-cap-construction-v1_0.md`
Feature spec canonical status: Superseded parent
Architecture ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`

## Overview

This test specification verifies generated primitive cap records and unsupported-cap refusals.

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: cut-loop records to generated cap builder through the loft CSG route
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: generated cap records or unsupported cap diagnostics
- Integration validation: cap-builder tests plus public-route unsupported-cap diagnostics

## Manual Smoke

- Run a supported cut case and confirm generated caps appear in route diagnostics or result evidence.

## Automated Smoke Tests

- `tests/test_surface_csg.py` creates generated cap records from closed cut loops.
- Public route smoke proves unsupported caps refuse before shell assembly.

## Automated Acceptance Tests

- Unit/helper behavior:
  - supported cap loops create generated cap records with source identity.
  - unsupported analytic cap regions refuse.
- Integrated route behavior:
  - public loft/primitive CSG reaches cap construction after closed cut loops.
- Failure and stale-result behavior, if applicable:
  - missing or unpaired cap loops refuse.

## App-Type Proof

- GUI proof: not applicable.
- Console proof: not applicable.
- API/service proof: not applicable.
- Mixed-surface proof: not applicable.
- Library-only proof: public Boolean API diagnostics prove cap construction is on route.

## Fixtures And Data

- In-memory cut-loop records for supported and unsupported primitive caps.
- Production-data rule: tests must not require production data.

## Acceptance

- [ ] Feature spec is canonical, or this test spec is explicitly temporary while split coverage is incomplete.
- [ ] Route-level proof exists for the app type.
- [ ] Helper-only tests cannot satisfy this feature contract.
- [ ] Observable result is asserted or manually checked.
- [ ] Failure behavior is covered where applicable.
