# Fix 12 Test: Documentation Policy Test Migration

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-12-documentation-policy-test-migration-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: not applicable

## Overview

Verify documentation governance tests target current authority and detect obligation removal.

## Application Integration Under Test

- App type: workflow.
- User/caller surface: repository pytest/CI governance lane.
- Invocation route: pytest -> documentation-rule module -> current authority files.
- Wiring owner/module: `tests/test_documentation_rules.py`.
- Observable result: focused pass or clear failure naming removed authority/obligation.
- Integration validation: focused module and normal CI invocation.

## Backlink

[Fix 12 specification](../specifications/fix-12-documentation-policy-test-migration-v1_0.md)

## Manual Smoke

Run `tests/test_documentation_rules.py` on a clean checkout and inspect failure
messages to ensure they name current managed skill/release files.

## Automated Smoke Tests

Assert every path opened by the test exists in the current tree and no source
literal references retired `agents/` or `project/agents/` authorities.

## Automated Acceptance Tests

- Run the migrated module as part of the normal suite.
- Against temporary copies, remove each required semantic obligation and assert
  the corresponding check fails.
- Change unrelated prose and assert semantic checks remain stable.
- Cover active release placement, archive immutability, durable documentation,
  and reference image/STL lifecycle obligations.

Tests inspect repository fixtures only and do not modify managed skill sources.

## App-Type Proof

- Workflow proof: test discovery and authority reads run through normal pytest/CI.
- GUI, console, API/service, mixed, and library-only proof: not applicable.

## Fixtures And Data

- Current repository authorities plus temporary mutated copies.
- Production-data rule: repository fixtures only; managed sources remain unchanged.

## Acceptance

- [x] Feature spec is canonical and normal workflow route is covered.
- [x] Current success and obligation-removal failure are asserted.
- [x] Path-existence-only tests cannot satisfy the contract.
