# Fix 08: Safe Documentation Archive Extraction (v1.0)

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One archive-extraction trust boundary validates every member before writing beneath the selected destination.

## Problem And Outcome

`_extract_docs_archive` joins archive member names to the destination without
rejecting absolute paths, `..` traversal, or unsafe link-like members. A crafted
documentation ZIP can therefore target files outside the installation directory.

## Scope

- Normalize and validate every archive member path before any extraction write.
- Reject absolute, drive-qualified, traversal, NUL-containing, and link-like entries.
- Require the resolved target to remain within the resolved destination.
- Preserve clean extraction of the release-generated documentation archive.

Not in scope: supporting arbitrary third-party archive formats or repairing
malformed archives.

## Implementation Routing

- `src/impression/cli.py::_extract_docs_archive`.
- Focused CLI/archive security regressions using in-memory ZIP fixtures.

## Contract

Input is untrusted ZIP bytes and a chosen destination. Validation is all-or-none:
if any member is unsafe, extraction fails before writing any member. Valid regular
files and directories are written only below the destination. The refusal names
the unsafe member without echoing file contents.

## Acceptance Criteria

- `../`, nested traversal, absolute, drive-qualified, NUL, and symlink-style
  members are rejected before filesystem mutation.
- Prefix-confusion paths cannot escape to a sibling directory.
- A normal release docs ZIP installs successfully with and without `clean`.
- Tests assert no sentinel outside the destination is created or changed.

## Verification

[Paired test specification](../test-specifications/fix-08-safe-docs-archive-extraction-v1_0.md)
