# Fix 17 Test: Pointed Cone Shell Connectivity

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-17-pointed-cone-shell-connectivity-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: not applicable

## Overview

Verify that pointed cones expose one connected shell while two-radius frustums
retain their explicit disconnected-cap status.

## Application Integration Under Test

- App type: library.
- User/caller surface: internal surface-native cone constructor and downstream primitive consumers.
- Invocation route: radii -> cone patches/caps -> `SurfaceShell.connected` -> `SurfaceBody`.
- Wiring owner/module: `src/impression/modeling/_surface_primitives.py`.
- Observable result: correct immutable connectivity metadata.
- Integration validation: focused constructor assertions plus the complete release suite.

## Automated Acceptance Tests

- Bottom apex and top apex each report `connected=True`.
- A two-radius frustum reports `connected=False`.
- Existing invalid-radius validation remains covered.
- The full suite reaches a zero-failure terminal result.

## Fixtures And Data

- Unit-scale synthetic cone dimensions only; no production or private data.

## Acceptance

- [x] Feature specification is canonical and the real constructor route is exercised.
- [x] Both pointed orientations and the frustum contrast are asserted.
- [x] Full release suite passes after the correction.
