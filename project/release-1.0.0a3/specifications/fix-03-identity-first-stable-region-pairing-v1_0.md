# Fix 03: Identity-First Stable Region Pairing (v1.0)

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One planner ordering change resolves explicit region identities before bounded ambiguity enumeration.

## Problem And Outcome

Adjacent stations with many identical regions can enumerate more than the
default 64 ambiguity branches even when every region has stable explicit
identity. Explicit one-to-one identity must remove candidates before subset
enumeration, leaving the branch limit for genuinely ambiguous residue.

## Scope

- Pair unique compatible region identities before geometric candidate creation.
- Remove resolved regions from ambiguity enumeration.
- Diagnose duplicate, missing, or contradictory identities rather than guessing.
- Preserve the existing branch limit for unresolved candidates.

Not in scope: raising or removing `ambiguity_max_branches`, or inventing identity
for anonymous regions.

## Implementation Routing

- `src/impression/modeling/loft.py`: region correspondence planning near bounded
  split/merge candidate enumeration.
- Focused tests in loft correspondence/inference modules.
- Reproduction from the stable multi-region station in the test-modeling issue list.

## Contract

Input is two station region sets. Unique matching identities are deterministic
assignments; only the unmatched residue is passed to bounded inference. Output
ordering remains stable. Identity contradictions are invalid input with named
source and target regions.

## Acceptance Criteria

- More than 64 identity-matched regions plan successfully at the default limit.
- The planner does not visit ambiguity branches for resolved pairs.
- Duplicate or contradictory IDs fail with deterministic diagnostics.
- Truly ambiguous anonymous input still obeys the configured branch limit.

## Verification

[Paired test specification](../test-specifications/fix-03-identity-first-stable-region-pairing-v1_0.md)
