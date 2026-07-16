# Surface Spec 426 Test: Loft Primitive Fragment Topology And Operation Selection

Date: 2026-07-16
Status: Superseded
Feature spec: `../specifications/surface-426-loft-primitive-fragment-topology-and-operation-selection-v1_0.md`
Feature spec canonical status: Superseded parent
Architecture ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`

## Overview

This test specification verifies operation-specific topology selection for loft/primitive cut fragments.

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: generated cap and fragment classification records to operation topology selector
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: selected fragments and topology records, or orientation diagnostics
- Integration validation: topology tests plus public API route tests

## Manual Smoke

- Run representative union, difference, and intersection loft/primitive cases and confirm topology class diagnostics are explicit.

## Automated Smoke Tests

- `tests/test_surface_csg.py` proves one difference cavity and one intersection topology route.

## Automated Acceptance Tests

- Unit/helper behavior:
  - difference cavity, exterior shell edit, union, intersection, touching/no-cut, and multi-shell/refused classifications are deterministic.
- Integrated route behavior:
  - public API route exposes topology selection before shell assembly.
- Failure and stale-result behavior, if applicable:
  - orientation ambiguity refuses.

## App-Type Proof

- GUI proof: not applicable.
- Console proof: not applicable.
- API/service proof: not applicable.
- Mixed-surface proof: not applicable.
- Library-only proof: public Boolean API route tests assert topology class.

## Fixtures And Data

- In-memory fragment classifications and generated cap records.
- Production-data rule: tests must not require production data.

## Acceptance

- [ ] Feature spec is canonical, or this test spec is explicitly temporary while split coverage is incomplete.
- [ ] Route-level proof exists for the app type.
- [ ] Helper-only tests cannot satisfy this feature contract.
- [ ] Observable result is asserted or manually checked.
- [ ] Failure behavior is covered where applicable.
