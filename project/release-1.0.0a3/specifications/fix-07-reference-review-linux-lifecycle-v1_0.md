# Fix 07: Reference Review Linux Lifecycle (v1.0)

Date: 2026-08-04
Status: Final
Issue: [#227](https://github.com/kellyjanderson/Impression/issues/227)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One headless application lifecycle boundary must terminate cleanly in one process on Linux while preserving macOS behavior.

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
