# Fix 01 Test: TopologyPath Loft Input Preservation

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-01-topology-path-loft-input-preservation-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md`

## Overview

Verify that the public loft route accepts a closed `TopologyPath`, preserves every
authored identity field, and refuses invalid path topology.

## Application Integration Under Test

- App type: library-only.
- User/caller surface: model authors calling `Loft(...)`.
- Invocation route: `Loft` -> `as_section` -> topology adapter -> planner.
- Wiring owner/module: `src/impression/modeling/loft.py`.
- Observable result: surface plan/body with retained IDs or specific refusal.
- Integration validation: direct public call plus audio-cube model smoke.

## Backlink

[Fix 01 specification](../specifications/fix-01-topology-path-loft-input-preservation-v1_0.md)

## Manual Smoke

Run the audio-cube diagonal-halves model with its named `TopologyPath` passed
directly to `Loft`; confirm planning succeeds and its diagnostic record retains
the authored diagonal IDs and protection flag.

## Automated Smoke Tests

Add a minimal named rectangle `TopologyPath` fixture and assert `Loft` accepts it,
returns a `SurfaceBody`, and retains point/correspondence identity in the plan.

## Automated Acceptance Tests

- Reproduce the test-modeling diagonal path with exact point IDs and assert a
  lossless mapping to the section loop.
- Cover closed clockwise/counterclockwise paths, landmarks, roles, and anchor.
- Assert an open path and duplicate identity fail with stable diagnostics.
- Run the existing `Section`, `Path2D`, and planar-shape loft API suite.

Fixtures must be deterministic local geometry and require no production data.

## App-Type Proof

- GUI, console, API/service, and mixed-surface proof: not applicable.
- Library-only proof: `Loft(...)` is invoked through the public modeling import and
  the returned plan/body is inspected for canonical topology identity.

## Fixtures And Data

- Minimal closed/open/duplicate paths and committed audio-cube diagonal path.
- Production-data rule: no production data; deterministic local geometry only.

## Acceptance

- [x] Feature spec is canonical.
- [x] Public library route and observable result are asserted.
- [x] Helper-only tests cannot satisfy the contract.
- [x] Success and invalid-input behavior are covered.
