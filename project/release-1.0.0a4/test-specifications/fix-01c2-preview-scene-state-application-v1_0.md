# Fix 01C2 Test: Preview Scene State Application

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 01C2: Preview Scene State Application](../specifications/fix-01c2-preview-scene-state-application-v1_0.md)
Feature spec canonical status: Archived
Architecture ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)

## Overview

Temporary paired contract for the Fix 01c split child.

## Application Integration Under Test

- App type: GUI
- User/caller surface: preview window scene, camera, and status/error state
- Invocation route: preview window scene, camera, and status/error state -> `src/impression/preview.py`
- Wiring owner/module: `src/impression/preview.py`
- Observable result: current visible scene; preserved camera/last-good scene; visible error and recovery status
- Integration validation: `tests/test_preview_controller.py` state transitions; offscreen preview scene/camera/error smoke

## Manual Smoke

- Exercise preview window scene, camera, and status/error state and inspect current visible scene; preserved camera/last-good scene; visible error and recovery status.

## Automated Smoke Tests

- A fast offscreen/real route test reaches the declared owner.

## Automated Acceptance Tests

- Unit/helper behavior:
  - current-result acceptance check
  - scene/status state application
- Integrated route behavior:
  - the real preview route asserts every owned outcome
- Failure and stale-result behavior, if applicable:
  - stale/failed results cannot replace the scene
  - shutdown ignores late completions

## App-Type Proof

- GUI proof:
  - preview event/state and UI-thread behavior
- Console proof:
  - not applicable
- API/service proof:
  - not applicable
- Mixed-surface proof:
  - not applicable
- Library-only proof:
  - not applicable

## Fixtures And Data

- Temporary entry/helper model and deterministic build results.
- Production-data rule: no user production data.

## Acceptance

- [ ] Feature child is canonical, or this test remains temporary.
- [ ] Route-level proof exists.
- [ ] Helper-only tests cannot satisfy the contract.
- [ ] Observable and failure behavior is asserted.
