# Fix 07 Test: Reference Review Linux Lifecycle

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-07-reference-review-linux-lifecycle-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: `project/release-0.1.0a/architecture/reference-review-async-concurrency.md`

## Overview

Verify deterministic reference-review GUI startup/teardown on Linux and macOS.

## Application Integration Under Test

- App type: GUI.
- User/caller surface: reference-review shell startup and close.
- Invocation route: Qt event loop -> shell -> close -> work/renderer drain -> exit.
- Wiring owner/module: reference-review UI shell/controller.
- Observable result: normal process exit without hang/crash/orphan.
- Integration validation: complete module repeated in one Linux/macOS process.

## Backlink

[Fix 07 specification](../specifications/fix-07-reference-review-linux-lifecycle-v1_0.md)

## Manual Smoke

On Linux headless CI, run the full UI-shell module repeatedly with faulthandler
enabled and confirm normal process exit; run the same focused module on macOS.

## Automated Smoke Tests

Launch and close the smallest reference-review shell in the supported headless Qt
configuration and assert process exit code 0 within a bounded timeout.

## Automated Acceptance Tests

- Run the complete `tests/test_reference_review_ui_shell.py` in one process.
- Repeat it to expose teardown/order instability.
- Assert no timeout, orphan child, fatal Qt/VTK message, signal, or exit 139.
- Exercise success, construction failure, and close-during-pending-work paths.
- Keep the module enabled in Linux and macOS CI lanes.

CI logs must preserve faulthandler output on failure without converting crashes to skips.

## App-Type Proof

- GUI proof: offscreen visible shell lifecycle, event processing, close ordering, and native exit.
- Console, API/service, mixed, and library-only proof: not applicable.

## Fixtures And Data

- Supported Qt platform configuration and local reference-review shell fixtures.
- Production-data rule: no production database or user references.

## Acceptance

- [x] Feature spec is canonical and real GUI route is exercised.
- [x] Observable exit status and failure diagnostics are asserted.
- [x] Construction failure and close-during-work behavior are covered.
