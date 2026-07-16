# Surface Spec 430 Test: Loft CSG Reference Geometry Handoff Proof

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-430-loft-csg-reference-geometry-handoff-proof-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: `../architecture/acd-loft-csg-reference-geometry-handoff.md`

## Overview

This test specification verifies dirty STL reference generation handoff from accepted public loft CSG result geometry.

## Application Integration Under Test

- App type: workflow/tooling
- User/caller surface: dirty STL reference generation workflow and review fixtures
- Invocation route: fixture source registry to public CSG result to STL export
- Wiring owner/module: `tests/reference_review_fixtures/stl_review_sources.py`
- Observable result: dirty STL generated only from accepted public result body
- Integration validation: accepted handoff test, adapter-only refusal test, no-synthetic-geometry test, dirty artifact smoke

## Manual Smoke

- Run the reference STL generation workflow for a loft CSG fixture and confirm only accepted public result geometry produces a dirty STL.

## Automated Smoke Tests

- Fixture registry smoke builds an accepted handoff and produces source readiness for STL export.

## Automated Acceptance Tests

- Unit/helper behavior:
  - accepted-result validator and refusal payloads.
- Integrated route behavior:
  - fixture source registry to STL export workflow accepts real result geometry and refuses adapter/synthetic/tessellation payloads.
- Failure and stale-result behavior, if applicable:
  - non-success result and missing body refuse before artifact writing.

## App-Type Proof

- GUI proof: not applicable.
- Console proof: not applicable.
- API/service proof: not applicable.
- Mixed-surface proof: workflow route through fixture registry and dirty artifact smoke.
- Library-only proof: not applicable.

## Fixtures And Data

- Temporary reference fixture records and accepted/refused public result doubles.
- Production-data rule: tests must not require production data.

## Acceptance

- [ ] Feature spec is canonical, or this test spec is explicitly temporary while split coverage is incomplete.
- [ ] Route-level proof exists for the app type.
- [ ] Helper-only tests cannot satisfy this feature contract.
- [ ] Observable result is asserted or manually checked.
- [ ] Failure behavior is covered where applicable.
