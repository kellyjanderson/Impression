# Fix 01 Test: TopologyPath Loft Input Preservation

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One verification surface proves identity-preserving TopologyPath normalization at the loft boundary.

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
