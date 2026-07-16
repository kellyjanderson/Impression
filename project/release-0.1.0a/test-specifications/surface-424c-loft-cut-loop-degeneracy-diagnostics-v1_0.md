# Surface Spec 424c Test: Loft Cut Loop Degeneracy Diagnostics

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-424c-loft-cut-loop-degeneracy-diagnostics-v1_0.md`
Feature spec canonical status: Canonical leaf
Architecture ancestor: `../architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: cut-loop closure to degeneracy gate
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: deterministic degeneracy diagnostics

## Automated Smoke Tests

- `tests/test_surface_csg.py` proves at least one degeneracy refuses through the public route.

## Automated Acceptance Tests

- Unit/helper behavior: tangent, grazing, zero-area, duplicate-segment, and open-loop cases classify deterministically.
- Integrated route behavior: public route exposes the degeneracy refusal code.
- Failure behavior: degenerate loops do not reach generated cap construction.

## Acceptance

- [ ] Each degeneracy class has focused coverage.
- [ ] Route-level refusal is asserted.
- [ ] No generated cap records are produced after degeneracy refusal.
