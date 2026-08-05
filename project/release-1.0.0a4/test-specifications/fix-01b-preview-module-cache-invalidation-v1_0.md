# Fix 01B Test: Preview Module Cache Invalidation

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 01B: Preview Module Cache Invalidation](../specifications/fix-01b-preview-module-cache-invalidation-v1_0.md)
Feature spec canonical status: Canonical
Architecture ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)

## Overview

This canonical paired contract verifies the complete retained split-child boundary.

## Application Integration Under Test

- App type: library-only
- User/caller surface: CLI scene factory consumed by live preview
- Invocation route: CLI scene factory consumed by live preview -> `src/impression/cli.py`
- Wiring owner/module: `src/impression/cli.py`
- Observable result: fresh model module and updated watched paths
- Integration validation: `tests/test_cli_preview.py`; temporary entry/helper module fixture

## Manual Smoke

- Exercise CLI scene factory consumed by live preview with the parent issue fixture and inspect fresh model module and updated watched paths.

## Automated Smoke Tests

- A fast route-level test reaches `src/impression/cli.py` through CLI scene factory consumed by live preview.

## Automated Acceptance Tests

- Unit/helper behavior:
  - `advance_reload_generation()`
  - generation-aware scene factory load
  - local dependency rediscovery/eviction
- Integrated route behavior:
  - CLI scene factory consumed by live preview asserts every child-owned acceptance outcome.
- Failure and stale-result behavior, if applicable:
  - failed import/build does not mark the new generation successfully loaded
  - diagnostics may name local paths but never log source contents

## App-Type Proof

- GUI proof:
  - not applicable
- Console proof:
  - not applicable
- API/service proof:
  - not applicable
- Mixed-surface proof:
  - not applicable
- Library-only proof:
  - CLI scene factory consumed by live preview is exercised as the real consuming route

## Fixtures And Data

- Parent issue #242 deterministic fixture and focused negative controls.
- Production-data rule: no user production data is required.

## Acceptance

- [x] Feature child is canonical.
- [ ] Route-level proof exists for library-only.
- [ ] Helper-only tests cannot satisfy the contract.
- [ ] Observable results and failure behavior are asserted.
