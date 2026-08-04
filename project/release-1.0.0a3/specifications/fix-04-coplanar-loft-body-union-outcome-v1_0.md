# Fix 04: Coplanar Loft-Body Union Outcome (v1.0)

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Prevent operand loss in the coplanar enclosure union

- Input: the reproduced enclosure and loft bodies sharing coplanar contact.
- Work: validate successful `boolean_union` results against both operand witness
  regions and return a typed refusal when the kernel result is incomplete.
- Output: either one valid union containing both operands or an explicit refusal.
- Complete when: both operand orders produce that outcome and never a partial body.

## Problem And Outcome

Unioning a loft body into an enclosure across coplanar contact can collapse the
earlier enclosure, forcing the test model to return a group instead of one body.
The operation must either return a valid union containing both operands or refuse
with a specific unsupported/invalid-result diagnostic; operand loss is forbidden.

## Scope

- Add result validation for the confirmed coplanar loft-body union case.
- Return the valid combined body when the supported kernel path succeeds.
- Return a typed, actionable refusal before exposing an incomplete result.

Not in scope: universal coincident-face boolean support or automatic mesh repair.

## Implementation Routing

- `src/impression/modeling/csg.py`: `boolean_union` result classification/gate.
- Surface boolean helpers already used by the public union boundary.
- Focused CSG regression plus the test-modeling enclosure composition.

## Contract

Inputs are two valid bodies with the reproduced coplanar contact. A successful
output preserves the occupied volume and distinguishing bounds of both operands
and passes body validity checks. Otherwise the operation raises the documented
refusal; it never returns a body that silently omits either operand.

## Acceptance Criteria

- The test-model enclosure either forms one valid body or receives the declared
  coplanar-union refusal without data loss.
- Successful output includes both operand witness regions and is watertight.
- Operand order does not change success/refusal classification.
- Existing supported union fixtures remain green.

## Verification

[Paired test specification](../test-specifications/fix-04-coplanar-loft-body-union-outcome-v1_0.md)
