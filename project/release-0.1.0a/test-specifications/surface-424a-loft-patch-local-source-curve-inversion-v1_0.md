# Surface Spec 424a Test: Loft Patch-Local Source Curve Inversion

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-424a-loft-patch-local-source-curve-inversion-v1_0.md`
Feature spec canonical status: Canonical leaf
Architecture ancestor: `../architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: loft/primitive intersection source normalization to patch-local inversion
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: patch-local intersection records or deterministic inversion diagnostics

## Automated Smoke Tests

- `tests/test_surface_csg.py` proves one supported inversion reaches the public route and one failed inversion refuses.

## Automated Acceptance Tests

- Unit/helper behavior: source curve to owning patch inversion preserves source ids and parameter-domain locations.
- Integrated route behavior: public route exposes inversion success and refusal diagnostics.
- Failure behavior: missing source record and out-of-domain inversion refuse before loop closure.

## Acceptance

- [ ] Route-level proof exists for the app type.
- [ ] Helper-only tests cannot satisfy this feature contract.
- [ ] Failure behavior is covered.
