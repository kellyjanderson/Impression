# Fix 16 Test: Preview Sharp-Edge Normals

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-16-preview-sharp-edge-normals-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: not applicable

## Overview

Verify that the shared preview actor route splits sharp render normals while
preserving smooth shading and that real audio-cube models render faithfully.

## Application Integration Under Test

- App type: mixed GUI and console.
- User/caller surface: interactive preview and `preview --screenshot PATH`.
- Invocation route: model -> shared scene controller -> PyVista actor -> visible preview/PNG.
- Wiring owner/module: `src/impression/preview.py`.
- Observable result: sharp CAD seams without false folds, spikes, or fragmented walls.
- Integration validation: actor-call assertions and three real PNG renders.

## Manual Smoke

- Inspect original, loft, and diagonal audio-cube assembly PNGs at full size.
- Confirm top plates and walls are planar, openings remain visible, and the diagonal halves remain distinguishable.

## Automated Smoke Tests

- Assert smooth uniform-color and per-face-color actors request sharp-edge splitting at 60 degrees.
- Assert flat-shaded actors disable the split.

## Automated Acceptance Tests

- Unit/helper behavior:
  - capture the exact PyVista actor arguments from the shared controller.
- Integrated route behavior:
  - invoke the installed screenshot command for all three assemblies and require valid PNG outputs.
- Failure and stale-result behavior, if applicable:
  - actor configuration errors remain explicit preview failures; model geometry is never rewritten.

## App-Type Proof

- GUI proof:
  - shared controller route and manual visible-state inspection.
- Console proof:
  - real command arguments, zero exits, PNG side effects, and visual outputs.
- API/service proof:
  - not applicable.
- Mixed-surface proof:
  - shared-controller unit proof plus console-render integration proof.
- Library-only proof:
  - not applicable.

## Fixtures And Data

- Fake plotter/actor capture and the three test-modeling assembly entrypoints.
- Production-data rule: repository models and disposable ignored PNGs only.

## Acceptance

- [x] Feature spec is canonical.
- [x] Route-level proof exists for shared controller and console rendering.
- [x] Helper-only tests cannot satisfy this feature contract.
- [x] Three visible model results are inspected.
- [x] Flat-shading behavior is covered.
