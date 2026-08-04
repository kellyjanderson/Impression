# Fix 05: Multi-Opening Loft Wall Integrity (v1.0)

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Preserve multiple openings through loft assembly

- Input: valid loft stations with one outer loop and multiple disjoint inner loops.
- Work: retain loop ownership through caps/sides, prevent cross-connections, and
  refuse invalid nesting before tessellation.
- Output: a direct wall loft whose authored inner loops remain openings.
- Complete when: the original model needs no cut workaround and reports the expected
  openings with zero degenerate cells.

## Problem And Outcome

A wall section containing multiple openings can loft into louver-like faces and
approximately 502 degenerate cells. The same intended wall currently requires a
solid loft followed by boolean cuts. Multiple inner loops must remain holes
through cap construction, side-wall orchestration, and tessellation.

## Scope

- Preserve outer/inner loop classification across the reproduced wall loft.
- Build caps and side surfaces without cross-connecting separate openings.
- Reject invalid loop nesting explicitly rather than emitting degenerate geometry.

Not in scope: arbitrary self-intersecting profiles or a general boolean-cut
replacement program.

## Implementation Routing

- `src/impression/modeling/loft.py`: multi-region cap and side orchestration.
- `src/impression/modeling/tessellation.py`: trim-loop tessellation validation.
- Focused loft regression plus the test-modeling multi-opening wall fixture.

## Contract

Input is a valid wall section with one outer boundary and multiple disjoint inner
loops at each station. Output preserves the same opening count, contains no faces
bridging an opening, and reports zero degenerate cells under the release QA
tolerance. Invalid nesting is a diagnostic, not best-effort geometry.

## Acceptance Criteria

- The original multi-opening wall model works without solid-wall-plus-cuts.
- Opening count and loop ownership are stable at every station and cap.
- Tessellation has zero degenerate cells and passes the expected watertight check.
- Single-opening and solid-section loft regressions remain green.

## Verification

[Paired test specification](../test-specifications/fix-05-multi-opening-loft-wall-integrity-v1_0.md)
