# Fix 11: Export Manufacturing Integrity Gate (v1.0)

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Gate STL writes on manufacturing mesh integrity

- Input: collected model geometry and a requested STL output path.
- Work: run export tessellation/QA, reject invalid manufacturing meshes with measured
  diagnostics, and place valid results through an atomic write boundary.
- Output: a valid STL or a nonzero refusal that leaves the target untouched.
- Complete when: valid ASCII/binary output succeeds and every invalid fixture fails pre-write.

## Problem And Outcome

The CLI export path can merge data and call `write_stl` without requiring a
watertight, non-degenerate manufacturing result. Export must fail before writing
when the candidate mesh violates the supported STL integrity contract.

## Scope

- Use export tessellation policy for surface inputs.
- Validate non-empty geometry, finite coordinates, zero degenerate faces, and
  watertight/manifold status before STL write.
- Emit an actionable refusal with measured failure categories.
- Preserve an explicit opt-in path only for intentionally open/non-manufacturing
  output if one already exists; do not silently weaken the default.

Not in scope: automatic repair, slicer simulation, or non-STL interchange.

## Implementation Routing

- `src/impression/cli.py::export`.
- `src/impression/modeling/tessellation.py` existing export and QA records.
- `src/impression/io.py` write boundary and focused CLI export tests.

## Contract

Input is collected model geometry. Output is either an STL written atomically
after passing the integrity gate or a nonzero command failure with no new/partial
target file. Units and format behavior remain unchanged from the current CLI.

## Acceptance Criteria

- Valid watertight model export succeeds in binary and ASCII modes.
- Open, empty, non-finite, non-manifold, and degenerate fixtures fail pre-write.
- Failure reports the violated properties and leaves no partial output.
- The test-modeling release fixtures export with zero degenerates.

## Verification

[Paired test specification](../test-specifications/fix-11-export-manufacturing-integrity-gate-v1_0.md)
