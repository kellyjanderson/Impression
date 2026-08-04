# Fix 04: Coplanar Loft-Body Union Outcome (v1.0)

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One boolean-union outcome contract prevents silent operand loss for a confirmed coplanar enclosure composition.

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
