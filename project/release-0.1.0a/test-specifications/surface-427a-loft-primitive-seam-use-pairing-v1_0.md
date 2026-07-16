# Surface Spec 427a Test: Loft Primitive Seam Use Pairing

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-427a-loft-primitive-seam-use-pairing-v1_0.md`
Feature spec canonical status: Canonical leaf
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: orientation-ready topology to seam/use pairing
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: complete pairing records or pairing diagnostics

## Automated Smoke Tests

- `tests/test_surface_csg.py` proves valid pairing and one dangling-use refusal route.

## Automated Acceptance Tests

- Unit/helper behavior: valid, dangling, duplicate, and ambiguous use pairings.
- Integrated route behavior: public route exposes seam/use pairing evidence.
- Failure behavior: invalid pairing refuses before shell assembly.

## Acceptance

- [ ] Complete pairings are asserted.
- [ ] Invalid pairings refuse before assembly.
- [ ] Pairing records preserve source patch and cap ids.
