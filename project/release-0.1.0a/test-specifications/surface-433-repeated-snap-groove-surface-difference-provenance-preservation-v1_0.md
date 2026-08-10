# Surface Spec 433 Test: Repeated Snap-Groove Surface Difference Provenance Preservation

Date: 2026-08-09
Status: Complete
Feature spec: `../specifications/surface-433-repeated-snap-groove-surface-difference-provenance-preservation-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: `../architecture/acd-surface-csg-pairwise-composition-and-result-reentry.md`

## Overview

Verify that every successful pairwise snap-groove difference returns a reusable
declarative field root and provenance evidence and can immediately serve as the base
for the next of six public difference calls.

## Application Integration Under Test

- App type: library-only
- User/caller surface: model modules consuming `boolean_difference`
- Invocation route: exported pairwise difference to field composition, validity/success gates, returned body, and the next exported pairwise difference
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: six consecutive succeeded/closed results ending in one surface body with six nested groove-cut field compositions
- Integration validation: one public-route test validates every intermediate body before passing it into the next call

## Manual Smoke

- From the sibling audio-cube project, run `references/reproduce_impression_open_issues.py` against the implementation checkout and confirm the first and second sequential groove calls report `succeeded`; then run the six-cutter model workflow and visually confirm all six receiving grooves are present.

## Automated Smoke Tests

- A focused `tests/test_surface_csg.py` test applies six self-contained groove cutters sequentially and asserts every public call succeeds with a closed, non-null body and changed-geometry evidence.

## Automated Acceptance Tests

- Unit/helper behavior:
  - original polygon lofts and accepted field results adapt deterministically;
  - validity finalization rejects invalid field result bodies;
  - difference evidence recognizes each newly composed declarative field graph as changed geometry.
- Integrated route behavior:
  - loop through all six cutters with `boolean_difference(current_body, (cutter,))`;
  - assert each result is succeeded/closed/non-null before assigning its body to `current_body`;
  - assert each intermediate body has one connected shell, one bounded implicit patch, Boolean/field provenance, and a reusable field root;
  - assert nested field-difference depth increases by one at every step and reaches six;
  - assert normalized declarative field-change evidence prevents unchanged-geometry false success;
  - assert every accepted result is surface-native, never `Mesh` or `MeshGroup`.
- Failure and stale-result behavior, if applicable:
  - batch input may succeed fully or return structured unsupported; an unsupported batch has no body and no hidden mesh fallback;
  - any failed sequential step stops the loop and exposes no next-step body;
  - stale-result and cancellation behavior are not applicable to this synchronous route.

## App-Type Proof

- GUI proof:
  - not applicable.
- Console proof:
  - not applicable.
- API/service proof:
  - not applicable.
- Mixed-surface proof:
  - not applicable.
- Library-only proof:
  - `tests/test_surface_csg.py` consumes each exported `boolean_difference` result in the next exported call and validates the public contract at all six steps.

## Fixtures And Data

- `tests/csg_reference_fixtures.py` provides a deterministic simplified northwest shell and six deep-copied rounded polygon-loft cutters with the issue's authored transforms and penetration depth.
- The fixture exposes stable cutter ids/order and expected cut evidence so repeated or skipped cutters cannot satisfy the test.
- Production-data rule: tests do not import the sibling audio-cube project and do not require production data.

## Acceptance

- [x] Feature spec remains canonical.
- [x] The full six-cutter self-contained and sibling public routes pass.
- [x] Every intermediate result is succeeded, closed, non-null, single-shell, and surface-native.
- [x] Every intermediate result has a reusable bounded field root and route provenance.
- [x] Geometry/provenance evidence changes at every step and identifies six nested cuts.
- [x] Adaptable batch difference succeeds through the same mesh-free field route.
