# Fix 01: TopologyPath Loft Input Preservation (v1.0)

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One input-normalization contract preserves authored topology identity through one loft-planning entry boundary.

## Problem And Outcome

`Loft(...)` accepts section-like inputs but its normalization through
`as_section` rejects `TopologyPath`; manually converting the path discards the
named point and protection metadata required by correspondence planning. A
closed `TopologyPath` must become a loft section without losing its stable IDs,
correspondence IDs, landmarks, segment roles, anchor, direction, or protected
point intent.

## Scope

- Extend the `Loft`/`as_section` input boundary for closed `TopologyPath` values.
- Define a deterministic mapping from path records to section loop records.
- Refuse open paths or invalid topology with a specific input diagnostic.
- Preserve all existing `Section`, `Region`, `Path2D`, and planar-shape behavior.

Not in scope: tessellation enforcement of protected vertices (Fix 02) or
correspondence inference policy (Fix 03).

## Implementation Routing

- `src/impression/modeling/topology.py`: `as_section`, path-to-section adapter.
- `src/impression/modeling/loft.py`: `Loft` normalization and diagnostics.
- `tests/test_loft_api.py` and a focused topology-path loft regression module.
- Reproduction: `testingImp/models/audio_cube_diagonal_halves.py`.

## Contract

Input is a closed `TopologyPath`; output is the canonical loft planning input
with an identity-preserving loop. Conversion must not resample or rename authored
topology. No unresolved design choice remains: direct acceptance is the public
behavior, with the existing section representation carrying mapped identity.

## Acceptance Criteria

- The test-modeling path can be passed directly to `Loft(...)`.
- Authored point/correspondence IDs and protection flags are observable unchanged
  by loft correspondence planning.
- Invalid open-path input fails before planning with a stable, actionable error.
- Existing section-like input tests remain green.

## Verification

[Paired test specification](../test-specifications/fix-01-topology-path-loft-input-preservation-v1_0.md)
