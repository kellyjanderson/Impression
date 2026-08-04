# Fix 04 Test: Hole Split/Merge Junction Surfaces

Date: 2026-08-04
Status: Proposed
Feature specification: [Fix 04: Hole Split/Merge Junction Surfaces](../specifications/fix-04-hole-split-merge-junction-surfaces-v1_0.md)
Canonical status: Draft

## Overview

This contract proves the user-visible behavior, internal invariants, failure behavior, and release regression boundary for Fix 04. It becomes binding only when the paired feature spec is independently reviewed and canonicalized.

## Application Integration Under Test

Surface-body proof: validate real patches and seams through the standard loft builder, not plan-only mocks.

## Manual Smoke

Build one-to-two and two-to-one hole lofts and inspect body closure, patch roles, cap count, and seam incidence. Confirm the last valid result remains usable after any deliberate failure.

## Automated Smoke Tests

Unit tests cover junction event construction, orientation, terminal-cap classification, and invalid lineage refusal.

## Automated Acceptance Tests

Real loft execution must return a closed body with exactly two terminal caps and no internal closure cap. Include deterministic positive, negative, and regression assertions and require actionable diagnostic content for refusals.

## App-Type Proof

Surface-body proof: validate real patches and seams through the standard loft builder, not plan-only mocks.

## Fixtures And Data

Split and merge fixtures, reversed station order, terminal birth/death, crossing/ambiguous lineage negatives. Fixtures must be deterministic, project-local, and small enough for normal CI. Preserve the exact issue reproduction where it is the acceptance fixture.

## Acceptance

- [ ] Manual smoke succeeds on a supported macOS development environment.
- [ ] Automated smoke covers the primary state transition and failure recovery.
- [ ] Automated acceptance proves every criterion in the paired implementation specification.
- [ ] The real application/public route is exercised; helper-only proof is rejected.
- [ ] The focused suite and full configured suite pass without workaround geometry or mesh fallback.
- [ ] Test names and failure output identify the violated contract and relevant fixture.

