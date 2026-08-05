# Fix 01C1 Test: Preview Refresh Input Wiring

Date: 2026-08-04
Status: Final
Feature spec: [Fix 01C1: Preview Refresh Input Wiring](../specifications/fix-01c1-preview-refresh-input-wiring-v1_0.md)
Feature spec canonical status: Canonical
Architecture ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)

## Overview

Canonical paired contract for the retained Fix 01c split child.

## Application Integration Under Test

- App type: mixed
- User/caller surface: saved-file events and the existing preview-window `R` binding
- Invocation route: saved-file events and the existing preview-window `R` binding -> `src/impression/preview.py` with `src/impression/cli.py` generation callback
- Wiring owner/module: `src/impression/preview.py` with `src/impression/cli.py` generation callback
- Observable result: typed automatic or forced reload request
- Integration validation: `tests/test_preview_controller.py` input route; `tests/test_cli_preview.py` generation callback; offscreen/real command key-event smoke

## Manual Smoke

- Exercise saved-file events and the existing preview-window `R` binding and inspect typed automatic or forced reload request.

## Automated Smoke Tests

- A fast offscreen/real route test reaches the declared owner.

## Automated Acceptance Tests

- Unit/helper behavior:
  - save-event route adapter
  - `R` forced-refresh route adapter
- Integrated route behavior:
  - the real preview route asserts every owned outcome
- Failure and stale-result behavior, if applicable:
  - unavailable/shutting-down coordinator rejects the event visibly without mutating scene state

## App-Type Proof

- GUI proof:
  - preview event/state and UI-thread behavior
- Console proof:
  - real command callback and status behavior
- API/service proof:
  - not applicable
- Mixed-surface proof:
  - command and visible route asserted separately
- Library-only proof:
  - not applicable

## Fixtures And Data

- Temporary entry/helper model and deterministic build results.
- Production-data rule: no user production data.

## Acceptance

- [x] Feature child is canonical.
- [x] Route-level proof exists.
- [x] Helper-only tests cannot satisfy the contract.
- [x] Observable and failure behavior is asserted.
