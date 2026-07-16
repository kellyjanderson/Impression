# Surface Spec 425b Test: Loft Primitive Generated Cap Record Construction

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-425b-loft-primitive-generated-cap-record-construction-v1_0.md`
Feature spec canonical status: Canonical leaf
Architecture ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: supported cap classifications to generated cap records
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: generated cap records with source identity and provenance

## Automated Smoke Tests

- `tests/test_surface_csg.py` proves a supported cap produces a generated cap record.

## Automated Acceptance Tests

- Unit/helper behavior: generated cap records preserve support class, source region, loop id, and cap id.
- Integrated route behavior: public route exposes generated cap evidence.
- Failure behavior: missing support classification refuses before construction.

## Acceptance

- [ ] Generated cap records are source-native.
- [ ] Provenance fields are asserted.
- [ ] Missing support classification refuses.
