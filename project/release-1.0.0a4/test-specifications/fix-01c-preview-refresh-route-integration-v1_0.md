# Fix 01C Test: Preview Refresh Route Integration

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 01C: Preview Refresh Route Integration](../specifications/fix-01c-preview-refresh-route-integration-v1_0.md)
Feature spec canonical status: Archived
Architecture ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)

## Overview

This temporary paired contract verifies the split-child boundary until the child is independently rescored and canonicalized.

## Application Integration Under Test

- App type: mixed
- User/caller surface: `impression preview` command, preview window, and existing `R` binding
- Invocation route: `impression preview` command, preview window, and existing `R` binding -> `src/impression/preview.py` with `src/impression/cli.py` wiring
- Wiring owner/module: `src/impression/preview.py` with `src/impression/cli.py` wiring
- Observable result: fresh visible scene; preserved camera/last-good scene; visible status and errors
- Integration validation: `tests/test_preview_controller.py`; `tests/test_cli_preview.py`; real command/offscreen preview smoke

## Manual Smoke

- Exercise `impression preview` command, preview window, and existing `R` binding with the parent issue fixture and inspect fresh visible scene; preserved camera/last-good scene; visible status and errors.

## Automated Smoke Tests

- A fast route-level test reaches `src/impression/preview.py` with `src/impression/cli.py` wiring through `impression preview` command, preview window, and existing `R` binding.

## Automated Acceptance Tests

- Unit/helper behavior:
  - preview route wiring
  - current-generation scene apply/state update
- Integrated route behavior:
  - `impression preview` command, preview window, and existing `R` binding asserts every child-owned acceptance outcome.
- Failure and stale-result behavior, if applicable:
  - error remains visible and a subsequent repair recovers
  - shutdown prevents late scene application

## App-Type Proof

- GUI proof:
  - preview event/state and UI-thread scene apply
- Console proof:
  - real `impression preview` command, status/error, and shutdown
- API/service proof:
  - not applicable
- Mixed-surface proof:
  - command/request and visible scene paths are asserted separately
- Library-only proof:
  - not applicable

## Fixtures And Data

- Parent issue #242 deterministic fixture and focused negative controls.
- Production-data rule: no user production data is required.

## Acceptance

- [ ] Feature child is canonical, or this test remains explicitly temporary.
- [ ] Route-level proof exists for mixed.
- [ ] Helper-only tests cannot satisfy the contract.
- [ ] Observable results and failure behavior are asserted.
