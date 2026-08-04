# Fix 03 Test: Identity-First Stable Region Pairing

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One planner test matrix distinguishes identity-resolved pairs from genuinely ambiguous candidate enumeration.

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
