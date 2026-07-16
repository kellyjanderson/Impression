# Surface Spec 427 Test: Loft Primitive Seam And Shell Assembly

Date: 2026-07-16
Status: Superseded
Feature spec: `../specifications/surface-427-loft-primitive-seam-and-shell-assembly-v1_0.md`
Feature spec canonical status: Superseded parent
Architecture ancestor: `../architecture/acd-loft-primitive-seam-shell-validity-execution.md`

## Overview

This test specification verifies seam/use pairing and candidate shell assembly for loft/primitive cut results.

## Application Integration Under Test

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: topology selector to seam/use pairing to candidate `SurfaceBody`
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: candidate result shells or assembly diagnostics
- Integration validation: seam assembly tests plus public invalid-seam refusal tests

## Manual Smoke

- Run a cut-producing loft/primitive case and confirm candidate shell evidence includes seam/use pairing.

## Automated Smoke Tests

- `tests/test_surface_csg.py` assembles a simple candidate shell from selected fragments and generated caps.

## Automated Acceptance Tests

- Unit/helper behavior:
  - cap-loop pairing, cavity shell assembly, adjacency rebuild diagnostics, and no-mesh-fragment evidence.
- Integrated route behavior:
  - public route reaches seam/shell assembly and reports invalid-seam refusal.
- Failure and stale-result behavior, if applicable:
  - unpaired seams, duplicate pairings, and open boundaries refuse.

## App-Type Proof

- GUI proof: not applicable.
- Console proof: not applicable.
- API/service proof: not applicable.
- Mixed-surface proof: not applicable.
- Library-only proof: public Boolean API route tests assert seam/shell diagnostics.

## Fixtures And Data

- In-memory topology records, generated caps, and selected loft fragments.
- Production-data rule: tests must not require production data.

## Acceptance

- [ ] Feature spec is canonical, or this test spec is explicitly temporary while split coverage is incomplete.
- [ ] Route-level proof exists for the app type.
- [ ] Helper-only tests cannot satisfy this feature contract.
- [ ] Observable result is asserted or manually checked.
- [ ] Failure behavior is covered where applicable.
