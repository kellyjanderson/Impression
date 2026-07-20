# Surface Spec 424b Test: Loft Cut Loop Closure And Boundary Participation

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-424b-loft-cut-loop-closure-and-boundary-participation-v1_0.md`
Feature spec canonical status: Canonical leaf
Architecture ancestor: `../architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: patch-local intersection records to cut-loop closure
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: closed loop records with cap-trim and station-seam participation

## Automated Smoke Tests

- `tests/test_surface_csg.py` proves a public loft/primitive route reaches closed cut-loop evidence.

## Automated Acceptance Tests

- Unit/helper behavior: ordered segments close, open segments refuse, boundary participants are retained.
- Integrated route behavior: public route exposes loop closure and boundary participation evidence.
- Failure behavior: missing boundary participant refuses before cap construction.

## Acceptance

- [ ] Closed loops are asserted through route-level evidence.
- [ ] Open and incomplete loops are refused.
- [ ] Boundary participation survives into the observable record.
