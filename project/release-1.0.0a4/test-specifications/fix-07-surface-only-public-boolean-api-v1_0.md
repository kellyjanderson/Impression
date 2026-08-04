# Fix 07 Test: Surface-Only Public Boolean API

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 07: Surface-Only Public Boolean API](../specifications/fix-07-surface-only-public-boolean-api-v1_0.md)
Feature spec canonical status: Draft
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)

## Overview

This paired contract verifies the complete draft feature boundary for Fix 07. It remains a draft until independent `review specs` canonicalizes the feature leaf.

## Application Integration Under Test

- App type: library-only
- User/caller surface: installed `impression.modeling` public API
- Invocation route: import/call -> representation guard -> surfaced solver -> surfaced result
- Wiring owner/module: `src/impression/modeling/csg.py` and `src/impression/modeling/__init__.py`
- Observable result: consistent source/wheel signatures, surfaced results, and actionable mesh rejection
- Integration validation: source and clean-wheel signature/runtime matrix plus docs/example scan

## Manual Smoke

- Install the candidate wheel in a clean environment.
- Call each public boolean with surfaced operands, then mesh and mixed operands.
- Confirm surfaced results and early mesh migration errors; inspect docs/examples.

## Automated Smoke Tests

- Installed public signatures contain no `Mesh`/`MeshGroup`.
- Mesh inputs fail before kernel dispatch and separately named mesh utilities remain explicit if retained.

## Automated Acceptance Tests

- Unit/helper behavior:
  - annotations, parameter names, exports, runtime guards, error text, docs inventory
- Integrated route behavior:
  - clean wheel imports/calls, documentation examples, preview/export surface consumer
- Failure and stale-result behavior, if applicable:
  - source/docs/wheel mismatch or implicit conversion fails the contract

## App-Type Proof

- GUI proof: not applicable
- Console proof: not applicable
- API/service proof:
  - not applicable
- Mixed-surface proof: not applicable
- Library-only proof: installed public API and consuming docs/preview/export

## Fixtures And Data

- surface, mesh, and mixed operand matrix
- public export inventory
- docs/examples and clean wheel
- Production-data rule: tests use project-local deterministic fixtures and temporary directories; no user production data is required.

## Acceptance

- [ ] Feature spec is canonical, or this test spec remains explicitly temporary while review/split coverage is incomplete.
- [ ] Route-level proof exists for the declared app type.
- [ ] Helper-only tests cannot satisfy this contract.
- [ ] Every observable result and feature acceptance criterion is asserted through the intended route.
- [ ] Failure, stale-result, refusal, or no-cut behavior is covered where applicable.
- [ ] Focused and full configured suites pass without mesh modeling fallback or test-model workaround geometry.

