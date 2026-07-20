# Surface Spec 431 Test: Loft CSG Section Evidence Readiness Handoff

Date: 2026-07-16
Status: Proposed
Feature spec: `../specifications/surface-431-loft-csg-section-evidence-readiness-handoff-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: `../architecture/acd-loft-csg-reference-geometry-handoff.md`

## Overview

This test specification verifies section evidence readiness from accepted loft CSG result geometry and declared section inputs.

## Application Integration Under Test

- App type: workflow/tooling
- User/caller surface: reference section evidence workflow and review fixture artifacts tab
- Invocation route: fixture source registry to accepted CSG result to section evidence bundle
- Wiring owner/module: `tests/reference_images.py`; `tests/reference_review_fixtures/stl_review_sources.py`
- Observable result: section readiness bundle only from accepted result geometry and declared section inputs
- Integration validation: section readiness test, missing plane refusal test, adapter-only refusal test, fixture registry integration test

## Manual Smoke

- Run a section-evidence fixture generation workflow and confirm section bundles require accepted result geometry plus an explicit section plane.

## Automated Smoke Tests

- Fixture registry integration smoke builds a section readiness bundle from an accepted handoff.

## Automated Acceptance Tests

- Unit/helper behavior:
  - readiness validator, missing plane refusal, detached evidence refusal.
- Integrated route behavior:
  - fixture registry and section artifact workflow accept only accepted result geometry with declared section inputs.
- Failure and stale-result behavior, if applicable:
  - adapter-only, synthetic, detached, or missing-plane payloads refuse.

## App-Type Proof

- GUI proof: not applicable.
- Console proof: not applicable.
- API/service proof: not applicable.
- Mixed-surface proof: workflow route through fixture registry and section artifact bundle integration.
- Library-only proof: not applicable.

## Fixtures And Data

- Temporary section evidence fixture records, accepted handoff records, and missing-plane refusal fixtures.
- Production-data rule: tests must not require production data.

## Acceptance

- [ ] Feature spec is canonical, or this test spec is explicitly temporary while split coverage is incomplete.
- [ ] Route-level proof exists for the app type.
- [ ] Helper-only tests cannot satisfy this feature contract.
- [ ] Observable result is asserted or manually checked.
- [ ] Failure behavior is covered where applicable.
