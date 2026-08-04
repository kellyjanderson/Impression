# Fix 03 Test: Named Hole Identity Pairing

Date: 2026-08-04
Status: Proposed
Feature specification: [Fix 03: Named Hole Identity Pairing](../specifications/fix-03-named-hole-identity-pairing-v1_0.md)
Canonical status: Draft

## Overview

This contract proves the user-visible behavior, internal invariants, failure behavior, and release regression boundary for Fix 03. It becomes binding only when the paired feature spec is independently reviewed and canonicalized.

## Application Integration Under Test

Planner/public-API proof: construct sections through supported authored identity inputs and inspect the canonical plan.

## Manual Smoke

Loft stations whose named holes cross positions and inspect plan correspondence diagnostics. Confirm the last valid result remains usable after any deliberate failure.

## Automated Smoke Tests

Planner tests cover named-first resolution, mixed residue fallback, duplicate names, and missing required identities.

## Automated Acceptance Tests

The crossed fixture must pair by names while an equivalent anonymous fixture retains geometric assignment. Include deterministic positive, negative, and regression assertions and require actionable diagnostic content for refusals.

## App-Type Proof

Planner/public-API proof: construct sections through supported authored identity inputs and inspect the canonical plan.

## Fixtures And Data

Crossed named holes; mixed named/unnamed loops; duplicate and missing identities; anonymous compatibility fixture. Fixtures must be deterministic, project-local, and small enough for normal CI. Preserve the exact issue reproduction where it is the acceptance fixture.

## Acceptance

- [ ] Manual smoke succeeds on a supported macOS development environment.
- [ ] Automated smoke covers the primary state transition and failure recovery.
- [ ] Automated acceptance proves every criterion in the paired implementation specification.
- [ ] The real application/public route is exercised; helper-only proof is rejected.
- [ ] The focused suite and full configured suite pass without workaround geometry or mesh fallback.
- [ ] Test names and failure output identify the violated contract and relevant fixture.

