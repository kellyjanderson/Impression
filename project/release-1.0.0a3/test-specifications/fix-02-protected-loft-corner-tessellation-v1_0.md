# Fix 02 Test: Protected Loft Corner Tessellation

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Verify protected vertices across tessellation policies

- Input: a minimal protected-point loft and both audio-cube diagonal halves.
- Work: run preview/export policies across sampling densities and measure protected
  coordinates, bounds, seams, degenerates, and validity.
- Output: a parameterized numeric regression matrix plus rendered smoke evidence.
- Complete when: every numeric invariant passes and both corners are visibly present.

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
