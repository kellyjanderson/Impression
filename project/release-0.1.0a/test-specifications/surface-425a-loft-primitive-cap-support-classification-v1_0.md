# Surface Spec 425a Test: Loft Primitive Cap Support Classification

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-425a-loft-primitive-cap-support-classification-v1_0.md`
Feature spec canonical status: Canonical leaf
Architecture ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: cut loops and primitive source regions to cap support classifier
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: supported/unsupported cap classification diagnostics

## Automated Smoke Tests

- `tests/test_surface_csg.py` proves one supported cap and one unsupported cap route.

## Automated Acceptance Tests

- Unit/helper behavior: supported planar/box caps and available surface-native primitive caps classify as supported.
- Integrated route behavior: public route exposes unsupported cap refusal before record construction.
- Failure behavior: unsupported cap shapes refuse deterministically.

## Acceptance

- [ ] Cap support is explicit before generated cap record construction.
- [ ] Unsupported caps are refused through route-level evidence.
- [ ] Tests do not depend on production data.
