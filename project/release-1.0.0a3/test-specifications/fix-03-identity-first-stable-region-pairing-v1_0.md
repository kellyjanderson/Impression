# Fix 03 Test: Identity-First Stable Region Pairing

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-03-identity-first-stable-region-pairing-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: `project/release-0.1.0a/architecture/loft-nm-mn-decomposition-architecture.md`

## Overview

Verify that explicit region identity is resolved before bounded ambiguity search.

## Application Integration Under Test

- App type: library-only.
- User/caller surface: `Loft(...)` with multi-region stations.
- Invocation route: station normalization -> identity pairing -> residual enumeration.
- Wiring owner/module: `src/impression/modeling/loft.py`.
- Observable result: deterministic plan or named conflict/limit refusal.
- Integration validation: public loft planning with 65+ identified regions.

## Backlink

[Fix 03 specification](../specifications/fix-03-identity-first-stable-region-pairing-v1_0.md)

## Manual Smoke

Run the multi-region test model at the default branch limit and inspect the plan
diagnostic: explicit pairs should be resolved before candidate enumeration.

## Automated Smoke Tests

Create 65 geometrically identical regions with unique matching IDs and assert
planning succeeds with zero ambiguous assignments visited for those pairs.

## Automated Acceptance Tests

- Cover 1, 64, 65, and a larger bounded set of identity-matched regions.
- Shuffle input order and assert identity pairing/output ordering is deterministic.
- Assert duplicate and contradictory IDs produce named invalid-input diagnostics.
- Assert anonymous ambiguous regions still stop at `ambiguity_max_branches`.
- Assert a mixed fixture enumerates only the unmatched residue.

Fixtures use generated local regions and deterministic IDs.

## App-Type Proof

- GUI, console, API/service, and mixed-surface proof: not applicable.
- Library-only proof: public loft planning is exercised, not only the enumerator helper.

## Fixtures And Data

- Generated identified, shuffled, mixed, contradictory, and anonymous region sets.
- Production-data rule: no production data.

## Acceptance

- [x] Feature spec is canonical and route-level behavior is asserted.
- [x] Success, conflict, and branch-limit outcomes are covered.
- [x] Observable search-visit and pairing results are measured.
