# Fix 05 Test: Multi-Opening Loft Wall Integrity

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One multi-loop loft fixture and its controls prove openings remain holes with clean tessellation.

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
