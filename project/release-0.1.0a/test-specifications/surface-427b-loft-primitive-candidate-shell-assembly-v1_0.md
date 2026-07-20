# Surface Spec 427b Test: Loft Primitive Candidate Shell Assembly

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-427b-loft-primitive-candidate-shell-assembly-v1_0.md`
Feature spec canonical status: Canonical leaf
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: complete seam/use pairing to candidate shell assembly
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: candidate shell assembly record or refusal

## Automated Smoke Tests

- `tests/test_surface_csg.py` proves one supported candidate assembly route.

## Automated Acceptance Tests

- Unit/helper behavior: valid exterior edit, valid cavity, and missing participant refusal.
- Integrated route behavior: public route exposes candidate shell evidence.
- Failure behavior: incomplete assembly refuses before body creation.

## Acceptance

- [ ] Supported topology plus pairing produces candidate shell evidence.
- [ ] Missing participants refuse deterministically.
- [ ] Runtime validity is not run by this leaf.
