# Fix 05 Test: Count-Changing Region Identity Preservation

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 05: Count-Changing Region Identity Preservation](../specifications/fix-05-count-changing-region-identity-preservation-v1_0.md)
Feature spec canonical status: Draft
Architecture ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)

## Overview

This paired contract verifies the complete draft feature boundary for Fix 05. It remains a draft until independent `review specs` canonicalizes the feature leaf.

## Application Integration Under Test

- App type: library-only
- User/caller surface: public loft planner and `Loft(...)`
- Invocation route: named count-changing stations -> exact region resolution -> synthetic lineage -> execution
- Wiring owner/module: `src/impression/modeling/loft.py`
- Observable result: stable exact pairs, explicit births, lineage-bearing synthetic stations, executable rail transition
- Integration validation: 2-to-3 planner metadata and microphone rail-pair execution

## Manual Smoke

- Plan named `shell`, `guide-a`, and born `guide-b` across 2-to-3 stations.
- Inspect exact pairs and every synthetic predecessor/successor identity.
- Execute the selected rail-pair transition.

## Automated Smoke Tests

- Exact stable identities pair before permutations.
- Synthetic stations retain complete lineage.

## Automated Acceptance Tests

- Unit/helper behavior:
  - exact region assignment, stable derived ids, reverse direction, multiple synthetic stations, conflict validation
- Integrated route behavior:
  - public plan and `Loft` execution of the named rail transition
- Failure and stale-result behavior, if applicable:
  - contradictory or incomplete lineage fails before execution with stable diagnostics

## App-Type Proof

- GUI proof: not applicable
- Console proof: not applicable
- API/service proof:
  - not applicable
- Mixed-surface proof: not applicable
- Library-only proof: public planner metadata plus consuming executor

## Fixtures And Data

- named 2-to-3 transition
- reversed transition
- multi-stage synthetic expansion
- conflict/incomplete-lineage controls
- Production-data rule: tests use project-local deterministic fixtures and temporary directories; no user production data is required.

## Acceptance

- [ ] Feature spec is canonical, or this test spec remains explicitly temporary while review/split coverage is incomplete.
- [ ] Route-level proof exists for the declared app type.
- [ ] Helper-only tests cannot satisfy this contract.
- [ ] Every observable result and feature acceptance criterion is asserted through the intended route.
- [ ] Failure, stale-result, refusal, or no-cut behavior is covered where applicable.
- [ ] Focused and full configured suites pass without mesh modeling fallback or test-model workaround geometry.

