# Fix 07 Test: Reference Review Linux Lifecycle

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One cross-platform process-lifecycle matrix proves deterministic UI-shell startup and teardown.

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
