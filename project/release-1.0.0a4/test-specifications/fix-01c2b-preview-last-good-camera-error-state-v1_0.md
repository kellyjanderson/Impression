# Fix 01C2B Test: Preview Last-Good Camera And Error State

Date: 2026-08-04
Status: Final
Feature spec: [Fix 01C2B: Preview Last-Good Camera And Error State](../specifications/fix-01c2b-preview-last-good-camera-error-state-v1_0.md)
Feature spec canonical status: Canonical
Architecture ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)

## Overview

Canonical paired GUI-state contract for the retained Fix 01c2 split child.

## Application Integration Under Test

- App type: GUI
- User/caller surface: preview window
- Invocation route: admitted build result -> UI-thread state handler -> preview window
- Wiring owner/module: `src/impression/preview.py`
- Observable result: preserved camera/last-good scene; visible error and recovery status
- Integration validation: `tests/test_preview_controller.py` camera/error/recovery tests; offscreen preview failure/recovery smoke

## Manual Smoke

- Exercise the preview result route and inspect preserved camera/last-good scene; visible error and recovery status.

## Automated Smoke Tests

- A fast offscreen route reaches the declared state handler.

## Automated Acceptance Tests

- Unit/helper behavior:
  - successful-state commit
  - failure/recovery state transition
- Integrated route behavior:
  - preview window asserts every owned outcome
- Failure and stale-result behavior, if applicable:
  - failure never clears the scene
  - late completion cannot overwrite recovered state

## App-Type Proof

- GUI proof:
  - visible preview state and UI-thread handoff
- Console proof:
  - not applicable
- API/service proof:
  - not applicable
- Mixed-surface proof:
  - not applicable
- Library-only proof:
  - not applicable

## Fixtures And Data

- Deterministic admitted/stale/success/failure build results.
- Production-data rule: no user production data.

## Acceptance

- [x] Feature child is canonical.
- [x] GUI route proof exists.
- [x] Helper-only tests cannot satisfy the contract.
- [x] Observable and failure behavior is asserted.
