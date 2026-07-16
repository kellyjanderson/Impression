# Surface Spec 428a Test: Loft Primitive Runtime Validity Checker

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-428a-loft-primitive-runtime-validity-checker-v1_0.md`
Feature spec canonical status: Canonical leaf
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: adjacency-complete candidate shell to runtime validity checker
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: runtime-valid evidence or validity diagnostics

## Automated Smoke Tests

- `tests/test_surface_csg.py` proves one valid shell and one invalid shell refusal route.

## Automated Acceptance Tests

- Unit/helper behavior: valid shell, open shell, non-manifold adjacency, inconsistent orientation.
- Integrated route behavior: public route exposes validity evidence before persistence.
- Failure behavior: invalid shells do not persist.

## Acceptance

- [ ] Runtime-valid and invalid states are both covered.
- [ ] Invalid diagnostics identify the blocking condition.
- [ ] Invalid shells are not persisted.
