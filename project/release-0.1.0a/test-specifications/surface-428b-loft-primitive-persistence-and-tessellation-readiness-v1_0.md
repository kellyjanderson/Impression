# Surface Spec 428b Test: Loft Primitive Persistence And Tessellation Readiness

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-428b-loft-primitive-persistence-and-tessellation-readiness-v1_0.md`
Feature spec canonical status: Canonical leaf
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: runtime-valid shell to accepted result persistence
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: accepted surface body and tessellation-readiness metadata

## Automated Smoke Tests

- `tests/test_surface_csg.py` proves a runtime-valid shell persists with readiness metadata.

## Automated Acceptance Tests

- Unit/helper behavior: valid persistence, stale evidence refusal, non-ready refusal.
- Integrated route behavior: public route returns accepted surface body and readiness metadata.
- Failure behavior: invalid or stale shells refuse before persistence.

## Acceptance

- [ ] Accepted results remain surface-body native.
- [ ] Readiness metadata does not eagerly tessellate.
- [ ] Stale or invalid shells refuse.
