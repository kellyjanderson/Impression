# Fix 03: Identity-First Stable Region Pairing (v1.0)

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Resolve stable region identities before ambiguity search

- Input: adjacent station regions containing explicit IDs and any unmatched residue.
- Work: pair compatible IDs first, remove those pairs from search, preserve order,
  and diagnose duplicate or contradictory identities.
- Output: a reduced ambiguity-search input containing only genuinely unresolved regions.
- Complete when: 65+ identified regions plan at the existing limit while anonymous
  ambiguity still stops at that limit.

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
