# Fix 01A Test: Preview Watch Request Coordination

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 01A: Preview Watch Request Coordination](../specifications/fix-01a-preview-watch-request-coordination-v1_0.md)
Feature spec canonical status: Canonical
Architecture ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)

## Overview

This canonical paired contract verifies the complete retained split-child boundary.

## Application Integration Under Test

- App type: library-only
- User/caller surface: preview watcher and build scheduler consumed by `PyVistaPreviewer`
- Invocation route: preview watcher and build scheduler consumed by `PyVistaPreviewer` -> `src/impression/preview.py`
- Wiring owner/module: `src/impression/preview.py`
- Observable result: one current build request or one latest replacement
- Integration validation: `tests/test_preview_controller.py`; real temporary-filesystem watcher fixture

## Manual Smoke

- Exercise preview watcher and build scheduler consumed by `PyVistaPreviewer` with the parent issue fixture and inspect one current build request or one latest replacement.

## Automated Smoke Tests

- A fast route-level test reaches `src/impression/preview.py` through preview watcher and build scheduler consumed by `PyVistaPreviewer`.

## Automated Acceptance Tests

- Unit/helper behavior:
  - `submit_reload(request)`
  - `begin_next_build()`
  - `complete_build(generation)`
- Integrated route behavior:
  - preview watcher and build scheduler consumed by `PyVistaPreviewer` asserts every child-owned acceptance outcome.
- Failure and stale-result behavior, if applicable:
  - watcher errors are reported without destroying request state
  - shutdown rejects new requests and prevents stale apply

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
  - preview watcher and build scheduler consumed by `PyVistaPreviewer` is exercised as the real consuming route

## Fixtures And Data

- Parent issue #242 deterministic fixture and focused negative controls.
- Production-data rule: no user production data is required.

## Acceptance

- [x] Feature child is canonical.
- [ ] Route-level proof exists for library-only.
- [ ] Helper-only tests cannot satisfy the contract.
- [ ] Observable results and failure behavior are asserted.
