# Surface Spec 426a Test: Loft Primitive Operation Fragment Retention

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-426a-loft-primitive-operation-fragment-retention-v1_0.md`
Feature spec canonical status: Canonical leaf
Architecture ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: paired cap loops and fragment classifications to operation retention
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: retained/excluded fragment records

## Automated Smoke Tests

- `tests/test_surface_csg.py` proves union, difference, and intersection retention through public route evidence.

## Automated Acceptance Tests

- Unit/helper behavior: each operation produces deterministic retained/excluded fragment sets.
- Integrated route behavior: public route exposes retention evidence before topology classification.
- Failure behavior: empty or conflicting retention refuses.

## Acceptance

- [ ] Operation-specific retention is asserted for all three Boolean operations.
- [ ] Retention records preserve source identity.
- [ ] Ambiguous retention refuses before topology classification.
