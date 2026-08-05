# Fix 09 Test: Surface Difference No-Op Result Gate

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 09: Surface Difference No-Op Result Gate](../specifications/fix-09-surface-difference-no-op-result-gate-v1_0.md)
Feature spec canonical status: Archived
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)

## Overview

This paired contract verifies the complete draft feature boundary for Fix 09. It remains a draft until independent `review specs` canonicalizes the feature leaf.

## Application Integration Under Test

- App type: library-only
- User/caller surface: public `boolean_difference` and every surfaced executor
- Invocation route: executor body/evidence -> normalized change/no-cut gate -> public result
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: truthful changed success, documented disjoint no-cut, or invalid/unsupported outcome
- Integration validation: public result matrix across every registered surfaced difference executor

## Manual Smoke

- Run a true cut, a proven disjoint cutter, and the rotated false-success reproduction.
- Confirm only changed geometry succeeds and each outcome includes inspectable evidence/diagnostics.

## Automated Smoke Tests

- A cloned-minuend executor result is rejected.
- A true cut with a valid witness succeeds; proven disjoint remains a distinct no-cut.

## Automated Acceptance Tests

- Unit/helper behavior:
  - each witness kind, normalized evidence, unchanged comparison, disjoint/tangent/tolerance-near classification
- Integrated route behavior:
  - public route for every registered surfaced difference executor
- Failure and stale-result behavior, if applicable:
  - overlap plus unchanged or ambiguous evidence refuses success; no executor bypasses the gate

## App-Type Proof

- GUI proof: not applicable
- Console proof: not applicable
- API/service proof:
  - not applicable
- Mixed-surface proof: not applicable
- Library-only proof: public result gate exercised through every executor route

## Fixtures And Data

- changed result
- cloned minuend with interaction evidence
- disjoint, tangent, tolerance-near, and ambiguous cases
- Production-data rule: tests use project-local deterministic fixtures and temporary directories; no user production data is required.

## Acceptance

- [ ] Feature spec is canonical, or this test spec remains explicitly temporary while review/split coverage is incomplete.
- [ ] Route-level proof exists for the declared app type.
- [ ] Helper-only tests cannot satisfy this contract.
- [ ] Every observable result and feature acceptance criterion is asserted through the intended route.
- [ ] Failure, stale-result, refusal, or no-cut behavior is covered where applicable.
- [ ] Focused and full configured suites pass without mesh modeling fallback or test-model workaround geometry.
