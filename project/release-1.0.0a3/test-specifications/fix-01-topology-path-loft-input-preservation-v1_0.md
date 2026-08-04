# Fix 01 Test: TopologyPath Loft Input Preservation

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Verify lossless TopologyPath-to-loft normalization

- Input: minimal and audio-cube paths with named/protected topology records.
- Work: exercise direct `Loft` input, every retained identity field, invalid-path
  refusals, and existing section-like inputs.
- Output: a focused identity-preservation regression module and manual smoke record.
- Complete when: the automated module and audio-cube smoke both pass.

## Backlink

[Fix 01 specification](../specifications/fix-01-topology-path-loft-input-preservation-v1_0.md)

## Manual Smoke

Run the audio-cube diagonal-halves model with its named `TopologyPath` passed
directly to `Loft`; confirm planning succeeds and its diagnostic record retains
the authored diagonal IDs and protection flag.

## Automated Smoke

Add a minimal named rectangle `TopologyPath` fixture and assert `Loft` accepts it,
returns a `SurfaceBody`, and retains point/correspondence identity in the plan.

## Automated Acceptance

- Reproduce the test-modeling diagonal path with exact point IDs and assert a
  lossless mapping to the section loop.
- Cover closed clockwise/counterclockwise paths, landmarks, roles, and anchor.
- Assert an open path and duplicate identity fail with stable diagnostics.
- Run the existing `Section`, `Path2D`, and planar-shape loft API suite.

Fixtures must be deterministic local geometry and require no production data.
