# Fix 02 Test: Coplanar Loft Face-Touch Union

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 02: Coplanar Loft Face-Touch Union](../specifications/fix-02-coplanar-loft-face-touch-union-v1_0.md)
Feature spec canonical status: Draft
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)

## Overview

This paired contract verifies the complete draft feature boundary for Fix 02. It remains a draft until independent `review specs` canonicalizes the feature leaf.

## Application Integration Under Test

- App type: library-only
- User/caller surface: public `boolean_union` consumed by modeling, preview, and export
- Invocation route: two surfaced loft operands -> face-touch classifier -> interior-pair removal -> shell assembly -> public result gate
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: one closed surfaced shell or precise refusal
- Integration validation: public union fixture plus real enclosure preview/export

## Manual Smoke

- Build the issue's two closed loft bodies sharing a designed face.
- Call public `boolean_union`, inspect one-shell closure/seams and absence of the shared interior patches.
- Preview/export the composed enclosure and confirm no mesh modeling fallback.

## Automated Smoke Tests

- Exact face-touch fixture returns succeeded, non-null, one-shell `SurfaceBody`.
- Near-coplanar control does not take the exact face-touch route.

## Automated Acceptance Tests

- Unit/helper behavior:
  - bounds candidates, trimmed-domain equivalence, orientation, interior-pair removal, seam/adjacency reconstruction
- Integrated route behavior:
  - public union exact fixture, partial-overlap and near-coplanar controls, enclosure preview/export
- Failure and stale-result behavior, if applicable:
  - ambiguous/partial contact, open seams, duplicate shells, or missing witnesses cannot report success

## App-Type Proof

- GUI proof: not applicable
- Console proof: not applicable
- API/service proof:
  - not applicable
- Mixed-surface proof: not applicable
- Library-only proof: public union plus downstream preview/export consumer

## Fixtures And Data

- exact face-touch loft pair
- reversed orientation
- near-coplanar gap and partial-domain overlap
- Production-data rule: tests use project-local deterministic fixtures and temporary directories; no user production data is required.

## Acceptance

- [ ] Feature spec is canonical, or this test spec remains explicitly temporary while review/split coverage is incomplete.
- [ ] Route-level proof exists for the declared app type.
- [ ] Helper-only tests cannot satisfy this contract.
- [ ] Every observable result and feature acceptance criterion is asserted through the intended route.
- [ ] Failure, stale-result, refusal, or no-cut behavior is covered where applicable.
- [ ] Focused and full configured suites pass without mesh modeling fallback or test-model workaround geometry.

