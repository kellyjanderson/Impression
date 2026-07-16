# Surface Spec 426c Test: Loft Primitive Topology Orientation And Refusal Diagnostics

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-426c-loft-primitive-topology-orientation-and-refusal-diagnostics-v1_0.md`
Feature spec canonical status: Canonical leaf
Architecture ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: topology records to orientation-readiness gate
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: orientation-ready evidence or refusal diagnostics

## Automated Smoke Tests

- `tests/test_surface_csg.py` proves one ready topology and one orientation refusal route.

## Automated Acceptance Tests

- Unit/helper behavior: inverted source normal, cap conflict, ambiguous inside/outside, and ready topology cases.
- Integrated route behavior: public route exposes orientation refusal.
- Failure behavior: refused topology does not reach seam/use pairing.

## Acceptance

- [ ] Orientation-ready and refused states are both covered.
- [ ] Diagnostics identify the blocking fragment or cap.
- [ ] Seam/use pairing is not attempted after refusal.
