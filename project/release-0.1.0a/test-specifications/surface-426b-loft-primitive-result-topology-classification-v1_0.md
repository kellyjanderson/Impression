# Surface Spec 426b Test: Loft Primitive Result Topology Classification

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-426b-loft-primitive-result-topology-classification-v1_0.md`
Feature spec canonical status: Canonical leaf
Architecture ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: retained fragments to topology classifier
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: topology class record

## Automated Smoke Tests

- `tests/test_surface_csg.py` proves one exterior edit and one interior cavity topology route.

## Automated Acceptance Tests

- Unit/helper behavior: empty, exterior edit, interior cavity, multi-shell, and refused classes.
- Integrated route behavior: public route exposes topology class before assembly.
- Failure behavior: unsupported topology refuses.

## Acceptance

- [ ] Supported topology classes are deterministic.
- [ ] Unsupported topology refuses before shell assembly.
- [ ] Topology records are observable through route evidence.
