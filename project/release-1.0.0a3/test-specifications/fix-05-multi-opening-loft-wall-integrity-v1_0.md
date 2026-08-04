# Fix 05 Test: Multi-Opening Loft Wall Integrity

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-05-multi-opening-loft-wall-integrity-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: `project/release-0.1.0a/architecture/loft-tolerance-and-degeneracy-architecture.md`

## Overview

Verify that direct multi-opening wall lofts preserve holes and emit clean geometry.

## Application Integration Under Test

- App type: library-only.
- User/caller surface: model authors lofting multi-opening sections.
- Invocation route: `Loft` -> cap/side executor -> trim tessellation.
- Wiring owner/module: `src/impression/modeling/loft.py`.
- Observable result: direct wall body with preserved openings and clean QA.
- Integration validation: original test-model wall without boolean-cut workaround.

## Backlink

[Fix 05 specification](../specifications/fix-05-multi-opening-loft-wall-integrity-v1_0.md)

## Manual Smoke

Render the original wall as a direct multi-opening loft and inspect front, back,
and oblique views for preserved openings and absence of louver-like bridges.

## Automated Smoke Tests

Loft a two-station rectangle with two disjoint inner loops and assert two openings,
zero degenerate cells, and expected watertight status.

## Automated Acceptance Tests

- Run one-, two-, and several-opening fixtures with stable loop IDs.
- Assert cap loop count/orientation and no triangle crosses an opening witness area.
- Assert zero degenerates, finite normals, and mesh/body validity.
- Assert nested/overlapping invalid holes fail with explicit diagnostics.
- Execute the test-modeling wall without boolean-cut workaround.

Store exact coordinates and QA thresholds beside the fixture.

## App-Type Proof

- GUI, console, API/service, and mixed-surface proof: not applicable.
- Library-only proof: public loft output is measured for holes, topology, and mesh QA.

## Fixtures And Data

- One-, two-, several-opening, invalid-hole, solid, and original wall fixtures.
- Production-data rule: committed deterministic geometry only.

## Acceptance

- [x] Feature spec is canonical and route-level output is measured.
- [x] Valid, invalid, and regression behavior is covered.
- [x] Manual smoke supplements but does not replace automated geometry proof.
