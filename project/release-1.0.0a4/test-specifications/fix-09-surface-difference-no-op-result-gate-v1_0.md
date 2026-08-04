# Fix 09 Test: Surface Difference No-Op Result Gate

Date: 2026-08-04
Status: Proposed
Feature specification: [Fix 09: Surface Difference No-Op Result Gate](../specifications/fix-09-surface-difference-no-op-result-gate-v1_0.md)
Canonical status: Draft

## Overview

This contract proves the user-visible behavior, internal invariants, failure behavior, and release regression boundary for Fix 09. It becomes binding only when the paired feature spec is independently reviewed and canonicalized.

## Application Integration Under Test

Public-route proof: validate through `boolean_difference`; validator-only unit tests are necessary but insufficient.

## Manual Smoke

Exercise a true cut, a disjoint cutter, and an executor double that returns a cloned minuend with interaction evidence. Confirm the last valid result remains usable after any deliberate failure.

## Automated Smoke Tests

Validator tests cover each witness type, unchanged comparisons, disjoint classification, tangency, and tolerance-near changes.

## Automated Acceptance Tests

Every registered surface-difference executor passes the shared gate; cloned-minuend success is rejected. Include deterministic positive, negative, and regression assertions and require actionable diagnostic content for refusals.

## App-Type Proof

Public-route proof: validate through `boolean_difference`; validator-only unit tests are necessary but insufficient.

## Fixtures And Data

Changed result, cloned minuend, proven disjoint cutter, tangent contact, tolerance-near patch/domain changes. Fixtures must be deterministic, project-local, and small enough for normal CI. Preserve the exact issue reproduction where it is the acceptance fixture.

## Acceptance

- [ ] Manual smoke succeeds on a supported macOS development environment.
- [ ] Automated smoke covers the primary state transition and failure recovery.
- [ ] Automated acceptance proves every criterion in the paired implementation specification.
- [ ] The real application/public route is exercised; helper-only proof is rejected.
- [ ] The focused suite and full configured suite pass without workaround geometry or mesh fallback.
- [ ] Test names and failure output identify the violated contract and relevant fixture.

