# Fix 06 Test: Expanded Planner Configuration Propagation

Date: 2026-08-04
Status: Final
Feature spec: [Fix 06: Expanded Planner Configuration Propagation](../specifications/fix-06-expanded-planner-configuration-propagation-v1_0.md)
Feature spec canonical status: Canonical
Architecture ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)

## Overview

This canonical paired contract verifies the complete retained feature boundary for Fix 06.

## Application Integration Under Test

- App type: library-only
- User/caller surface: all public loft planner entry points and `Loft(...)`
- Invocation route: caller arguments -> immutable options -> direct/nested pairing and expansion
- Wiring owner/module: `src/impression/modeling/loft.py`
- Observable result: candidate search respects caller cap and reports effective configuration
- Integration validation: public direct and staged 1-to-4-to-7 tests below/at/above required cap

## Manual Smoke

- Run the staged transition with a deliberately small cap and inspect refusal.
- Repeat with `ambiguity_max_branches=4096` and confirm no internal diagnostic reports 64.

## Automated Smoke Tests

- Nested expansion observes the supplied non-default cap.
- Invalid option values fail at the public boundary.

## Automated Acceptance Tests

- Unit/helper behavior:
  - options construction, every helper handoff, hard cap enforcement, diagnostic payload
- Integrated route behavior:
  - public 1-to-4-to-7 planning with small and large limits
- Failure and stale-result behavior, if applicable:
  - limit exhaustion names the supplied cap/location; no nested reset or multiplication

## App-Type Proof

- GUI proof: not applicable
- Console proof: not applicable
- API/service proof:
  - not applicable
- Mixed-surface proof: not applicable
- Library-only proof: options supplied only through public planning route

## Fixtures And Data

- direct ambiguity fixture
- staged 1-to-4-to-7 expansion
- invalid and boundary option values
- Production-data rule: tests use project-local deterministic fixtures and temporary directories; no user production data is required.

## Acceptance

- [x] Feature spec is canonical.
- [x] Route-level proof exists for the declared app type.
- [x] Helper-only tests cannot satisfy this contract.
- [x] Every observable result and feature acceptance criterion is asserted through the intended route.
- [x] Failure, stale-result, refusal, or no-cut behavior is covered where applicable.
- [x] Focused and full configured suites pass without mesh modeling fallback or test-model workaround geometry.
