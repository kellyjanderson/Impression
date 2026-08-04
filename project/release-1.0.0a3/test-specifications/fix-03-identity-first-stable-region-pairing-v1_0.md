# Fix 03 Test: Identity-First Stable Region Pairing

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Verify identity pairing bypasses ambiguity enumeration

- Input: generated 1, 64, 65, and larger region sets with identified/anonymous variants.
- Work: measure pairing, search visits, ordering, mixed residue, and invalid-ID behavior.
- Output: a bounded planner regression matrix with positive and refusal controls.
- Complete when: identified sets bypass the limit and anonymous sets still obey it.

## Backlink

[Fix 03 specification](../specifications/fix-03-identity-first-stable-region-pairing-v1_0.md)

## Manual Smoke

Run the multi-region test model at the default branch limit and inspect the plan
diagnostic: explicit pairs should be resolved before candidate enumeration.

## Automated Smoke

Create 65 geometrically identical regions with unique matching IDs and assert
planning succeeds with zero ambiguous assignments visited for those pairs.

## Automated Acceptance

- Cover 1, 64, 65, and a larger bounded set of identity-matched regions.
- Shuffle input order and assert identity pairing/output ordering is deterministic.
- Assert duplicate and contradictory IDs produce named invalid-input diagnostics.
- Assert anonymous ambiguous regions still stop at `ambiguity_max_branches`.
- Assert a mixed fixture enumerates only the unmatched residue.

Fixtures use generated local regions and deterministic IDs.
