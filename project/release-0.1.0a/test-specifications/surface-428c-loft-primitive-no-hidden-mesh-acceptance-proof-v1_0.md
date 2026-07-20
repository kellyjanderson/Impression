# Surface Spec 428c Test: Loft Primitive No Hidden Mesh Acceptance Proof

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-428c-loft-primitive-no-hidden-mesh-acceptance-proof-v1_0.md`
Feature spec canonical status: Canonical leaf
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: accepted persistence record to no-hidden-mesh proof
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: accepted-result evidence with mesh-fallback flag fixed false

## Automated Smoke Tests

- `tests/test_surface_csg.py` proves an accepted loft/primitive route reports no hidden mesh fallback.

## Automated Acceptance Tests

- Unit/helper behavior: proof construction and missing-proof refusal.
- Integrated route behavior: public route asserts no mesh fallback hook was invoked.
- Failure behavior: missing construction proof refuses before reference handoff.

## Acceptance

- [ ] No-hidden-mesh proof is present on accepted results.
- [ ] Accidental fallback invocation can be detected by tests.
- [ ] Missing proof refuses acceptance.
