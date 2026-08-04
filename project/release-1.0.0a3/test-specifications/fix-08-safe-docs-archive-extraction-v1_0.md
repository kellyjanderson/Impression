# Fix 08 Test: Safe Documentation Archive Extraction

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-08-safe-docs-archive-extraction-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: not applicable

## Overview

Verify that unsafe documentation archives are refused atomically and valid archives install.

## Application Integration Under Test

- App type: console.
- User/caller surface: documentation installation command.
- Invocation route: command -> archive bytes -> validation -> optional clean -> extraction.
- Wiring owner/module: `src/impression/cli.py`.
- Observable result: installed docs or nonzero refusal with unchanged filesystem.
- Integration validation: actual package ZIP plus hostile ZIP matrix through extractor/CLI route.

## Backlink

[Fix 08 specification](../specifications/fix-08-safe-docs-archive-extraction-v1_0.md)

## Manual Smoke

Install the release-generated docs ZIP into a temporary directory, then attempt a
ZIP containing `../sentinel` and confirm it is refused without changing sentinel.

## Automated Smoke Tests

Pass an in-memory ZIP with one traversal member to `_extract_docs_archive` and
assert a specific error and an empty destination.

## Automated Acceptance Tests

- Cover direct/nested traversal, absolute POSIX, drive/UNC-style, NUL, prefix
  confusion, and symlink-like external-attribute entries.
- Put a valid member before an invalid one and prove all-or-none prevalidation.
- Assert destination and sibling sentinels remain unchanged on refusal.
- Cover valid files/directories and both `clean` modes.
- Install the actual packaged documentation ZIP as an integration fixture.

All archive bytes are generated locally; tests never target user directories.

## App-Type Proof

- Console proof: command/extractor input, error, filesystem side effect, and completion result.
- GUI, API/service, mixed, and library-only proof: not applicable.

## Fixtures And Data

- Generated hostile/valid ZIPs, temporary destination, sibling sentinels, real docs ZIP.
- Production-data rule: temporary directories only; never user paths.

## Acceptance

- [x] Feature spec is canonical and real console extraction route is covered.
- [x] Observable filesystem and refusal results are asserted.
- [x] Failure ordering and both clean modes are covered.
