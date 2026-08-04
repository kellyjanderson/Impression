# Fix 06 Test: Expanded Planner Configuration Propagation

Date: 2026-08-04
Status: Proposed
Feature specification: [Fix 06: Expanded Planner Configuration Propagation](../specifications/fix-06-expanded-planner-configuration-propagation-v1_0.md)
Canonical status: Draft

## Overview

This contract proves the user-visible behavior, internal invariants, failure behavior, and release regression boundary for Fix 06. It becomes binding only when the paired feature spec is independently reviewed and canonicalized.

## Application Integration Under Test

Planner integration proof: configure through the public entry point; direct helper invocation alone is insufficient.

## Manual Smoke

Run direct and expanded transitions with intentionally small branch caps and inspect refusal diagnostics. Confirm the last valid result remains usable after any deliberate failure.

## Automated Smoke Tests

Tests instrument attempted branches and cover limits below, at, and above the required search size.

## Automated Acceptance Tests

Every nested expansion must remain at or below the caller cap and report the effective value on refusal. Include deterministic positive, negative, and regression assertions and require actionable diagnostic content for refusals.

## App-Type Proof

Planner integration proof: configure through the public entry point; direct helper invocation alone is insufficient.

## Fixtures And Data

Direct/expanded ambiguity fixtures with multiple branch caps and invalid configuration values. Fixtures must be deterministic, project-local, and small enough for normal CI. Preserve the exact issue reproduction where it is the acceptance fixture.

## Acceptance

- [ ] Manual smoke succeeds on a supported macOS development environment.
- [ ] Automated smoke covers the primary state transition and failure recovery.
- [ ] Automated acceptance proves every criterion in the paired implementation specification.
- [ ] The real application/public route is exercised; helper-only proof is rejected.
- [ ] The focused suite and full configured suite pass without workaround geometry or mesh fallback.
- [ ] Test names and failure output identify the violated contract and relevant fixture.

