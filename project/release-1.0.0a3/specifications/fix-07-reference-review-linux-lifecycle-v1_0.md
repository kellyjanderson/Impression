# Fix 07: Reference Review Linux Lifecycle (v1.0)

Date: 2026-08-04
Status: Final
Issue: [#227](https://github.com/kellyjanderson/Impression/issues/227)

## Work Units

Count: 1 IWU.

### IWU 1 — Make the reference-review shell lifecycle safe on Linux

- Input: issue #227's Linux headless Qt/VTK UI-shell lifecycle.
- Work: correct application, widget, renderer, and pending-work teardown ownership
  and restore the full module to normal Linux CI.
- Output: a deterministic one-process startup and teardown path on Linux and macOS.
- Complete when: repeated runs exit 0 without timeout, orphan, fatal message, or signal.

## Problem And Outcome

`tests/test_reference_review_ui_shell.py` can hang or exit 139 on Linux under
headless Qt. The supported test lane must initialize and tear down the review UI
in one process without timeout, orphan process, or segmentation fault.

## Scope

- Correct application/widget/renderer setup and teardown ownership for headless Linux.
- Make the supported Qt platform and graphics configuration explicit in tests.
- Restore the test module to normal CI execution.

Not in scope: redesigning the reference-review UI or weakening assertions by
skipping Linux behavior.

## Implementation Routing

- `src/impression/devtools/reference_review/ui/` lifecycle owners.
- `tests/test_reference_review_ui_shell.py` and CI environment configuration.

## Contract

One test process owns one application lifecycle, closes all top-level UI and
graphics resources, and exits normally. Platform setup happens before Qt/VTK
initialization. The same ownership rules must preserve the existing macOS lane.

## Acceptance Criteria

- The full UI-shell test module completes on Linux with exit code 0.
- No timeout, orphan process, fatal Qt message, or exit 139 occurs.
- Repeated execution is stable and does not depend on test order.
- macOS UI-shell coverage remains green.

## Verification

[Paired test specification](../test-specifications/fix-07-reference-review-linux-lifecycle-v1_0.md)
