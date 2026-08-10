# Surface Spec 432 Test: Attached Polygon-Loft Surface Union Completion

Date: 2026-08-09
Status: Complete
Feature spec: `../specifications/surface-432-attached-polygon-loft-surface-union-completion-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: `../architecture/acd-surface-csg-pairwise-composition-and-result-reentry.md`

## Overview

Verify that the public surface-union route deterministically composes the complete
attached snap-tab and microphone-rail operand tuples into one closed,
surface-native body without leaking partial results.

## Application Integration Under Test

- App type: library-only
- User/caller surface: model modules consuming `boolean_union`
- Invocation route: public `boolean_union` to polygon-loft field adaptation, canonical hard union, and returned `SurfaceBooleanResult`
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: one closed fused body for each complete issue fixture, or structured refusal for non-adaptable/disconnected input
- Integration validation: public API tests call both complete fixture tuples and inspect final geometry/provenance, result shape, and failure behavior

## Manual Smoke

- From the sibling audio-cube project, run `references/reproduce_impression_open_issues.py` against the implementation checkout and confirm both attached-feature union cases report `succeeded`, one closed body, and no failure reason.

## Automated Smoke Tests

- A focused `tests/test_surface_csg.py` test unions the self-contained shell and microphone-rail fixture through `boolean_union` and asserts succeeded/closed/non-null surface output.

## Automated Acceptance Tests

- Unit/helper behavior:
  - original polygon lofts adapt to bounded declarative nodes;
  - equivalent N-operand sets normalize to the same stable-identity order while the final result retains the original request;
  - disconnected or non-adaptable input returns no falsely connected partial body.
- Integrated route behavior:
  - public union succeeds for the full northwest snap-tab operand tuple;
  - public union succeeds for the full microphone-rail operand tuple;
  - representative permutations of each equivalent operand set produce the same stable body identity, canonical body provenance, and classification while each result retains its original request order;
  - each final body has one shell and deterministic patch/provenance evidence from every attached feature;
  - the existing orthogonal coplanar union regression remains passing;
  - every accepted result is a `SurfaceBooleanResult` carrying `SurfaceBody`, never `Mesh` or `MeshGroup`.
- Failure and stale-result behavior, if applicable:
  - intentionally disconnected input does not return a false one-shell success;
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
  - `tests/test_surface_csg.py` calls the exported `boolean_union` function with both complete fixture tuples and asserts the public result contract.

## Fixtures And Data

- `tests/csg_reference_fixtures.py` provides a deterministic simplified northwest shell, attached polygon-loft snap tabs, and a two-body microphone rail fixture derived from GitHub issue #267.
- Fixtures assert feature contribution through stable patch/provenance evidence so a false success with unchanged or incomplete geometry fails.
- Production-data rule: tests do not import the sibling audio-cube project and do not require production data.

## Acceptance

- [x] Feature spec remains canonical.
- [x] Both complete sibling-project fixture tuples pass through exported `boolean_union`.
- [x] Final results are succeeded, closed, non-null, single-shell, and surface-native.
- [x] Field-graph/provenance assertions prove every attached feature contributed to the result.
- [x] Equivalent operand permutations produce the same stable result identity and classification.
- [x] Disconnected/non-adaptable input cannot expose a false partial success.
- [x] Existing orthogonal coplanar regression and full no-hidden-mesh suite pass in release validation.
