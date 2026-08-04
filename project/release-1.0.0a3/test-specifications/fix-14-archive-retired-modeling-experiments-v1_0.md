# Fix 14 Test: Archive Retired Modeling Experiments

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-14-archive-retired-modeling-experiments-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: not applicable

## Overview

Verify that the hinge and SDF experiments remain recoverable outside core while
the a3 source, documentation, test collection, and remote branch inventory no
longer treat them as Impression product scope.

## Application Integration Under Test

- App type: workflow and library package.
- User/caller surface: release maintainer workflow and `impression.modeling` consumers.
- Invocation route: archive repositories -> core package tree -> pytest
  collection and focused release regressions -> remote ref inventory.
- Wiring owner/module: Git repositories, Impression package tree, documentation indexes, and `tests/test_release_metadata.py`.
- Observable result: clean archives, successful collection, explicit package
  absence checks, and no SDF experiment branch on the Impression remote.
- Integration validation: Git fsck/ref queries, complete collection, and focused regression suites.

## Backlink

[Fix 14 specification](../specifications/fix-14-archive-retired-modeling-experiments-v1_0.md)

## Manual Smoke

- Run `git fsck --full --no-dangling` in both standalone archives.
- Inspect the SDF archive at original commit `3b35e4490be4c4de4592c0a5bb445655c4c1efe6`.
- Confirm `git ls-remote --heads origin feature/sdf-endcaps-shelved` is empty.

## Automated Smoke Tests

- Collect the full core suite and require zero experimental import errors.
- Run `tests/test_release_metadata.py` and assert the hinge/SDF module and export
  surfaces are absent.

## Automated Acceptance Tests

- Unit/helper behavior:
  - Assert source, exports, docs, examples, and dependencies contain no retired experiment payload.
- Integrated route behavior:
  - Run documentation-governance, modern-geometry, no-hidden-fallback, CSG
    caller-inventory, full collection, Git fsck, and remote-ref checks.
- Failure and stale-result behavior, if applicable:
  - A returned experiment path/export/ref or failed archive object check fails qualification explicitly.

## App-Type Proof

- GUI proof:
  - not applicable.
- Console proof:
  - not applicable; Git commands are maintainer workflow evidence, not a product console surface.
- API/service proof:
  - not applicable.
- Mixed-surface proof:
  - Archive/ref checks prove the workflow surface; package tests and collection independently prove the library surface.
- Library-only proof:
  - `tests/test_release_metadata.py` consumes the packaged modeling boundary and asserts retired exports are absent.

## Fixtures And Data

- Standalone archive Git object databases.
- Current core source/docs/tests and temporary pytest outputs.
- No production or private data.

## Acceptance

- [x] Feature spec is canonical and archive-before-removal order is proved.
- [x] Core collection and focused regression routes are asserted.
- [x] Helper-only tests cannot satisfy this feature contract.
- [x] Observable archive, package, and ref results are asserted.
- [x] Failure behavior is explicit.
