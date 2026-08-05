# Fix 08 Test: Loft Surface Difference Cut Execution

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 08: Loft Surface Difference Cut Execution](../specifications/fix-08-loft-surface-difference-cut-execution-v1_0.md)
Feature spec canonical status: Archived
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)

## Overview

This paired contract verifies the complete draft feature boundary for Fix 08. It remains a draft until independent `review specs` canonicalizes the feature leaf.

## Application Integration Under Test

- App type: library-only
- User/caller surface: public `boolean_difference` consumed by audio-cube preview/export
- Invocation route: surface base/cutters -> eligibility/decomposition -> intersection/trims -> shell rebuild -> shared result gates
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: changed closed surface with intended USB/acoustic/snap opening or precise refusal
- Integration validation: public issue fixtures followed by real preview/export with no workaround geometry

## Manual Smoke

- Run USB-C, acoustic, and rotated snap-pocket cuts through public surface inputs.
- Inspect new boundaries, geometry-change evidence, closure, cap, and seam validity.
- Preview/export the result without mesh construction or grouped-body workarounds.

## Automated Smoke Tests

- Each qualifying cutter changes geometry and yields a closed result.
- A branched fixture reaches validated decomposition/recomposition.

## Automated Acceptance Tests

- Unit/helper behavior:
  - intersection evidence, trim fragments, classification, cutter patch orientation, branch decomposition/recomposition, shell validation
- Integrated route behavior:
  - all three public issue fixtures and downstream preview/export
- Failure and stale-result behavior, if applicable:
  - missing trims, ambiguous classification, invalid branch graph, open seams, or unchanged result cannot succeed

## App-Type Proof

- GUI proof: not applicable
- Console proof: not applicable
- API/service proof:
  - not applicable
- Mixed-surface proof: not applicable
- Library-only proof: public difference plus downstream preview/export consumer

## Fixtures And Data

- USB-C and acoustic loft cutters
- rotated rounded-tab groove
- branched topology
- tangent, disjoint, and invalid controls
- Production-data rule: tests use project-local deterministic fixtures and temporary directories; no user production data is required.

## Acceptance

- [ ] Feature spec is canonical, or this test spec remains explicitly temporary while review/split coverage is incomplete.
- [ ] Route-level proof exists for the declared app type.
- [ ] Helper-only tests cannot satisfy this contract.
- [ ] Every observable result and feature acceptance criterion is asserted through the intended route.
- [ ] Failure, stale-result, refusal, or no-cut behavior is covered where applicable.
- [ ] Focused and full configured suites pass without mesh modeling fallback or test-model workaround geometry.
