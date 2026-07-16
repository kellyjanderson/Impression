# Surface Spec 425c Test: Loft Primitive Cap Loop Pairing And Diagnostics

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-425c-loft-primitive-cap-loop-pairing-and-diagnostics-v1_0.md`
Feature spec canonical status: Canonical leaf
Architecture ancestor: `../architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: generated cap records and cut loops to cap-loop pairing
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: exactly paired cap loop records or diagnostics

## Automated Smoke Tests

- `tests/test_surface_csg.py` proves exact pairing and one pairing refusal route.

## Automated Acceptance Tests

- Unit/helper behavior: exact, missing, duplicate, and unpaired loops classify deterministically.
- Integrated route behavior: public route exposes pairing evidence before topology selection.
- Failure behavior: incomplete pairing refuses before topology selection.

## Acceptance

- [ ] Every generated cap loop pairs exactly once.
- [ ] Missing and duplicate pairings are refused.
- [ ] Topology selection cannot run after pairing refusal.
