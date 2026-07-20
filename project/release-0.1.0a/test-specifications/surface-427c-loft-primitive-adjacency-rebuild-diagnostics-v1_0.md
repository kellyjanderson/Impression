# Surface Spec 427c Test: Loft Primitive Adjacency Rebuild Diagnostics

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-427c-loft-primitive-adjacency-rebuild-diagnostics-v1_0.md`
Feature spec canonical status: Canonical leaf
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: candidate shell records to adjacency rebuild diagnostics
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: adjacency-complete evidence or diagnostics

## Automated Smoke Tests

- `tests/test_surface_csg.py` proves complete adjacency and one missing-link refusal route.

## Automated Acceptance Tests

- Unit/helper behavior: complete, missing, duplicate, and inconsistent adjacency links.
- Integrated route behavior: public route exposes adjacency evidence before validity checks.
- Failure behavior: invalid adjacency refuses before runtime validity.

## Acceptance

- [ ] Adjacency-complete evidence is asserted.
- [ ] Invalid adjacency refuses before validity.
- [ ] Diagnostics identify offending shell/use ids.
