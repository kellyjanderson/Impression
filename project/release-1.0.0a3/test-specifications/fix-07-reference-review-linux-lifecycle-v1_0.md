# Fix 07 Test: Reference Review Linux Lifecycle

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Verify headless review-shell startup and teardown

- Input: the complete UI-shell module on supported Linux headless and macOS lanes.
- Work: repeat success, construction-failure, and close-during-work paths in one
  process with faulthandler and bounded timeout.
- Output: cross-platform CI lifecycle coverage retaining crash diagnostics.
- Complete when: runs are stable with no skip, orphan, fatal message, signal, or masking.

## Backlink

[Fix 07 specification](../specifications/fix-07-reference-review-linux-lifecycle-v1_0.md)

## Manual Smoke

On Linux headless CI, run the full UI-shell module repeatedly with faulthandler
enabled and confirm normal process exit; run the same focused module on macOS.

## Automated Smoke

Launch and close the smallest reference-review shell in the supported headless Qt
configuration and assert process exit code 0 within a bounded timeout.

## Automated Acceptance

- Run the complete `tests/test_reference_review_ui_shell.py` in one process.
- Repeat it to expose teardown/order instability.
- Assert no timeout, orphan child, fatal Qt/VTK message, signal, or exit 139.
- Exercise success, construction failure, and close-during-pending-work paths.
- Keep the module enabled in Linux and macOS CI lanes.

CI logs must preserve faulthandler output on failure without converting crashes to skips.
