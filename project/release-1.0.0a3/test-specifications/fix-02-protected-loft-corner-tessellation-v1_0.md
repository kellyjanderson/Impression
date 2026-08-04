# Fix 02 Test: Protected Loft Corner Tessellation

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-02-protected-loft-corner-tessellation-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md`

## Overview

Verify that protected loft points become stable mesh vertices across policies and sampling densities.

## Application Integration Under Test

- App type: library-only.
- User/caller surface: loft consumers requesting preview/export tessellation.
- Invocation route: `Loft` -> surface executor -> `tessellate_surface_body`.
- Wiring owner/module: `src/impression/modeling/tessellation.py`.
- Observable result: protected mesh vertex and stable seam/body bounds.
- Integration validation: both audio-cube halves under preview/export requests.

## Backlink

[Fix 02 specification](../specifications/fix-02-protected-loft-corner-tessellation-v1_0.md)

## Manual Smoke

Render both diagonal audio-cube halves with fairness disabled and inspect the
diagonal corner and shared seam at low and high sample counts.

## Automated Smoke Tests

Tessellate the smallest two-station fixture with one protected point and assert a
mesh vertex lies within the declared tolerance of that authored point.

## Automated Acceptance Tests

- Parameterize preview/export requests and at least three sample densities.
- Assert protected vertex coordinates and body bounds remain within tolerance.
- Assert shared-boundary coincidence, watertightness, and zero degenerates.
- Add an unprotected control showing extra sampling remains free to vary.
- Execute the full audio-cube diagonal-halves fixture as a release regression.

Record numeric tolerances in the test rather than relying on rendered appearance.

## App-Type Proof

- GUI, console, API/service, and mixed-surface proof: not applicable.
- Library-only proof: public loft and tessellation consumers are exercised end to end.

## Fixtures And Data

- Minimal protected-point loft and both committed audio-cube diagonal halves.
- Production-data rule: deterministic local geometry only.

## Acceptance

- [x] Feature spec is canonical and the public library route is covered.
- [x] Numeric observable results and rendered manual fallback are defined.
- [x] Sampling-policy failure behavior is covered.
