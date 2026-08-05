# Fix 04 Test: Hole Split/Merge Junction Surfaces

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 04: Hole Split/Merge Junction Surfaces](../specifications/fix-04-hole-split-merge-junction-surfaces-v1_0.md)
Feature spec canonical status: Archived
Architecture ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)

## Overview

This paired contract verifies the complete draft feature boundary for Fix 04. It remains a draft until independent `review specs` canonicalizes the feature leaf.

## Application Integration Under Test

- App type: library-only
- User/caller surface: published split/merge example and `Loft(...)`
- Invocation route: identity-aware plan -> junction event -> surface patches/seams -> cap/closure validation
- Wiring owner/module: `src/impression/modeling/loft.py`
- Observable result: closed body, valid orientations/seams, exactly two terminal caps
- Integration validation: published many-to-one plus reversed one-to-many public execution

## Manual Smoke

- Run the published many-to-one example and reversed one-to-many variant.
- Inspect `cap_valid`, `closed_valid`, seam coverage, patch roles, and cap count.
- Confirm no interior planar closure cap or mesh fallback.

## Automated Smoke Tests

- Both transition directions return closed-valid surfaced bodies.
- Each has exactly two terminal caps.

## Automated Acceptance Tests

- Unit/helper behavior:
  - junction event direction, lineage, patch orientation, seam incidence, terminal/junction roles
- Integrated route behavior:
  - public `Loft` execution for published and reversed examples
- Failure and stale-result behavior, if applicable:
  - ambiguous lineage, invalid orientation, self-intersection, or non-manifold seams return no partial body

## App-Type Proof

- GUI proof: not applicable
- Console proof: not applicable
- API/service proof:
  - not applicable
- Mixed-surface proof: not applicable
- Library-only proof: published caller route through real surface executor

## Fixtures And Data

- published many-to-one
- reversed one-to-many
- crossing/ambiguous lineage and invalid orientation controls
- Production-data rule: tests use project-local deterministic fixtures and temporary directories; no user production data is required.

## Acceptance

- [ ] Feature spec is canonical, or this test spec remains explicitly temporary while review/split coverage is incomplete.
- [ ] Route-level proof exists for the declared app type.
- [ ] Helper-only tests cannot satisfy this contract.
- [ ] Every observable result and feature acceptance criterion is asserted through the intended route.
- [ ] Failure, stale-result, refusal, or no-cut behavior is covered where applicable.
- [ ] Focused and full configured suites pass without mesh modeling fallback or test-model workaround geometry.
