# Surface Spec 424 Test: Loft Patch-Local Cut Loop Construction

Date: 2026-07-16
Status: Superseded
Feature spec: `../specifications/surface-424-loft-patch-local-cut-loop-construction-v1_0.md`
Feature spec canonical status: Superseded parent
Architecture ancestor: `../architecture/acd-loft-primitive-intersection-and-cut-loop-kernel.md`

## Overview

This test specification verifies patch-local cut-loop construction from normalized source records.

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: source normalizer to patch-local cut-loop builder through the loft CSG route
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: closed cut loops or deterministic loop/inversion diagnostics
- Integration validation: cut-loop tests plus public-route refusal tests

## Manual Smoke

- Run a representative intersecting loft/primitive case and confirm the diagnostic names a closed loop or specific loop refusal.

## Automated Smoke Tests

- `tests/test_surface_csg.py` builds a crossing case that produces closed patch-local loops.
- Public route smoke proves tangent/grazing refusal is reachable.

## Automated Acceptance Tests

- Unit/helper behavior:
  - crossing and partial-crossing source records produce closed cut loops.
  - tangent, grazing, zero-area, and open loops refuse deterministically.
- Integrated route behavior:
  - public loft/primitive CSG reports loop diagnostics before cap construction.
- Failure and stale-result behavior, if applicable:
  - malformed source records refuse before loop construction.

## App-Type Proof

- GUI proof: not applicable.
- Console proof: not applicable.
- API/service proof: not applicable.
- Mixed-surface proof: not applicable.
- Library-only proof: public Boolean API route tests assert loop diagnostics.

## Fixtures And Data

- In-memory normalized source records and loft patches with cap/station seams.
- Production-data rule: tests must not require production data.

## Acceptance

- [ ] Feature spec is canonical, or this test spec is explicitly temporary while split coverage is incomplete.
- [ ] Route-level proof exists for the app type.
- [ ] Helper-only tests cannot satisfy this feature contract.
- [ ] Observable result is asserted or manually checked.
- [ ] Failure behavior is covered where applicable.
