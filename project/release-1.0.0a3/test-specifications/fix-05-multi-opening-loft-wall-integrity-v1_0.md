# Fix 05 Test: Multi-Opening Loft Wall Integrity

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Verify direct lofts preserve multiple wall openings

- Input: one-, two-, and several-opening fixtures plus the original test-model wall.
- Work: measure loop ownership, opening witnesses, degenerates, normals, validity,
  invalid-hole refusals, and solid-section controls.
- Output: a multi-loop loft regression suite and manual multi-view smoke record.
- Complete when: direct wall automation passes and the louver defect is absent.

## Backlink

[Fix 05 specification](../specifications/fix-05-multi-opening-loft-wall-integrity-v1_0.md)

## Manual Smoke

Render the original wall as a direct multi-opening loft and inspect front, back,
and oblique views for preserved openings and absence of louver-like bridges.

## Automated Smoke

Loft a two-station rectangle with two disjoint inner loops and assert two openings,
zero degenerate cells, and expected watertight status.

## Automated Acceptance

- Run one-, two-, and several-opening fixtures with stable loop IDs.
- Assert cap loop count/orientation and no triangle crosses an opening witness area.
- Assert zero degenerates, finite normals, and mesh/body validity.
- Assert nested/overlapping invalid holes fail with explicit diagnostics.
- Execute the test-modeling wall without boolean-cut workaround.

Store exact coordinates and QA thresholds beside the fixture.
