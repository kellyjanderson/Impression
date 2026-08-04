# Fix 02 Test: Protected Loft Corner Tessellation

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One geometry regression matrix proves a protected loft corner survives tessellation across sampling densities.

## Backlink

[Fix 02 specification](../specifications/fix-02-protected-loft-corner-tessellation-v1_0.md)

## Manual Smoke

Render both diagonal audio-cube halves with fairness disabled and inspect the
diagonal corner and shared seam at low and high sample counts.

## Automated Smoke

Tessellate the smallest two-station fixture with one protected point and assert a
mesh vertex lies within the declared tolerance of that authored point.

## Automated Acceptance

- Parameterize preview/export requests and at least three sample densities.
- Assert protected vertex coordinates and body bounds remain within tolerance.
- Assert shared-boundary coincidence, watertightness, and zero degenerates.
- Add an unprotected control showing extra sampling remains free to vary.
- Execute the full audio-cube diagonal-halves fixture as a release regression.

Record numeric tolerances in the test rather than relying on rendered appearance.
