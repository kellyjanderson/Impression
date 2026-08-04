# Fix 05 Test: Count-Changing Region Identity Preservation

Date: 2026-08-04
Status: Proposed
Feature specification: [Fix 05: Count-Changing Region Identity Preservation](../specifications/fix-05-count-changing-region-identity-preservation-v1_0.md)
Canonical status: Draft

## Overview

This contract proves the user-visible behavior, internal invariants, failure behavior, and release regression boundary for Fix 05. It becomes binding only when the paired feature spec is independently reviewed and canonicalized.

## Application Integration Under Test

Planner integration proof: use the public loft plan route and serialize/inspect its real derived records.

## Manual Smoke

Inspect expanded split/merge plans and verify every synthetic station reports predecessor and successor identity. Confirm the last valid result remains usable after any deliberate failure.

## Automated Smoke Tests

Planner tests cover stable derived ids, path propagation, direction reversal, duplicates, and missing lineage.

## Automated Acceptance Tests

End-to-end planning of named count-changing sections must preserve lineage consumed by junction execution. Include deterministic positive, negative, and regression assertions and require actionable diagnostic content for refusals.

## App-Type Proof

Planner integration proof: use the public loft plan route and serialize/inspect its real derived records.

## Fixtures And Data

Named region/hole splits and merges; reverse direction; multiple synthetic stations; conflict fixtures. Fixtures must be deterministic, project-local, and small enough for normal CI. Preserve the exact issue reproduction where it is the acceptance fixture.

## Acceptance

- [ ] Manual smoke succeeds on a supported macOS development environment.
- [ ] Automated smoke covers the primary state transition and failure recovery.
- [ ] Automated acceptance proves every criterion in the paired implementation specification.
- [ ] The real application/public route is exercised; helper-only proof is rejected.
- [ ] The focused suite and full configured suite pass without workaround geometry or mesh fallback.
- [ ] Test names and failure output identify the violated contract and relevant fixture.

