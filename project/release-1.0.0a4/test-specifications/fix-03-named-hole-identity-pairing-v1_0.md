# Fix 03 Test: Named Hole Identity Pairing

Date: 2026-08-04
Status: Final
Feature spec: [Fix 03: Named Hole Identity Pairing](../specifications/fix-03-named-hole-identity-pairing-v1_0.md)
Feature spec canonical status: Canonical
Architecture ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)

## Overview

This canonical paired contract verifies the complete retained feature boundary for Fix 03.

## Application Integration Under Test

- App type: library-only
- User/caller surface: `loft_plan_sections(...)` and `Loft(...)`
- Invocation route: named sections -> normalization -> identity-first loop pairing -> plan -> executor
- Wiring owner/module: `src/impression/modeling/loft.py`
- Observable result: plan metadata and executed surface preserve authored named-hole paths
- Integration validation: public planner and executor crossed-hole fixtures

## Manual Smoke

- Create named `hole-a` and `hole-b` that exchange positions between stations.
- Inspect the public plan and confirm names, not proximity, determine pairs.
- Execute the loft and confirm output correspondence matches the plan.

## Automated Smoke Tests

- Crossed named holes pair by identity.
- Equivalent unnamed holes retain geometric fallback.

## Automated Acceptance Tests

- Unit/helper behavior:
  - identity index, named-first resolution, mixed residue assignment, duplicate/missing/contradictory diagnostics
- Integrated route behavior:
  - public plan and `Loft` execution assert identical selected pairs
- Failure and stale-result behavior, if applicable:
  - identity conflicts fail before geometry scoring; executor cannot silently re-pair

## App-Type Proof

- GUI proof: not applicable
- Console proof: not applicable
- API/service proof:
  - not applicable
- Mixed-surface proof: not applicable
- Library-only proof: public planner and consuming loft executor

## Fixtures And Data

- crossed named rectangles
- mixed named/unnamed loops
- duplicate, missing, contradictory, and anonymous controls
- Production-data rule: tests use project-local deterministic fixtures and temporary directories; no user production data is required.

## Acceptance

- [x] Feature spec is canonical.
- [x] Route-level proof exists for the declared app type.
- [x] Helper-only tests cannot satisfy this contract.
- [x] Every observable result and feature acceptance criterion is asserted through the intended route.
- [x] Failure, stale-result, refusal, or no-cut behavior is covered where applicable.
- [x] Focused and full configured suites pass without mesh modeling fallback or test-model workaround geometry.
