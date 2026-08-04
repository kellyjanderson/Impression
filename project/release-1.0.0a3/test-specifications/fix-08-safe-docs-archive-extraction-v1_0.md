# Fix 08 Test: Safe Documentation Archive Extraction

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One hostile-archive matrix proves validation is contained and atomic before extraction writes.

## Backlink

[Fix 08 specification](../specifications/fix-08-safe-docs-archive-extraction-v1_0.md)

## Manual Smoke

Install the release-generated docs ZIP into a temporary directory, then attempt a
ZIP containing `../sentinel` and confirm it is refused without changing sentinel.

## Automated Smoke

Pass an in-memory ZIP with one traversal member to `_extract_docs_archive` and
assert a specific error and an empty destination.

## Automated Acceptance

- Cover direct/nested traversal, absolute POSIX, drive/UNC-style, NUL, prefix
  confusion, and symlink-like external-attribute entries.
- Put a valid member before an invalid one and prove all-or-none prevalidation.
- Assert destination and sibling sentinels remain unchanged on refusal.
- Cover valid files/directories and both `clean` modes.
- Install the actual packaged documentation ZIP as an integration fixture.

All archive bytes are generated locally; tests never target user directories.
