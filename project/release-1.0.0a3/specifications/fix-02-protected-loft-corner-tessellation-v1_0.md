# Fix 02: Protected Loft Corner Tessellation (v1.0)

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Carry protected loft points into the tessellated mesh

- Input: a valid loft plan containing an authored protected point.
- Work: propagate its record to surface boundaries, require it as a tessellation
  sample, and keep it fixed as unrelated sampling density changes.
- Output: tessellated geometry with a vertex at the protected authored coordinate.
- Complete when: both diagonal halves retain the corner and pass seam/mesh QA.

## Problem And Outcome

The protected diagonal corner in the audio-cube half can disappear or drift in
the tessellated body, and the result changes when sample count changes. A
protected loft point must survive planning and surface tessellation as a vertex
within the active geometric tolerance, independent of unrelated sampling density.

## Scope

- Propagate protected-point identity from loft plan through surface patches to
  the tessellation boundary.
- Constrain shared-boundary sampling so protected vertices are mandatory samples.
- Keep fairness disabled behavior deterministic; do not redesign fairness.

Not in scope: accepting `TopologyPath` input (Fix 01) or general adaptive
tessellation quality policy.

## Implementation Routing

- `src/impression/modeling/loft.py`: protected point lifecycle in the plan/executor.
- `src/impression/modeling/tessellation.py`: mandatory boundary sample handling.
- Focused regression tests plus `testingImp/models/audio_cube_diagonal_halves.py`.

## Contract

Input is a valid loft plan containing a protected authored point. Output is a
tessellated body with a corresponding vertex within tolerance and with bounds
that do not drift when only non-protected sampling density changes. The chosen
rule is vertex preservation, not proximity represented only by an edge crossing.

## Acceptance Criteria

- The diagonal corner is present in both lofted halves within declared tolerance.
- Sample-count changes add samples without moving or deleting protected vertices.
- Shared boundaries remain coincident and the closed result meets mesh QA.
- Non-protected loft behavior and performance remain within existing contracts.

## Verification

[Paired test specification](../test-specifications/fix-02-protected-loft-corner-tessellation-v1_0.md)
