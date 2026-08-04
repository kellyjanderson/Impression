# Fix 08 Test: Safe Documentation Archive Extraction

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Verify documentation extraction containment and atomicity

- Input: generated hostile ZIP variants, valid ZIPs, and protected filesystem sentinels.
- Work: exercise unsafe member classes, valid-before-invalid ordering, containment,
  and both clean modes.
- Output: an atomicity/containment regression matrix plus real-package integration case.
- Complete when: hostile archives change nothing and the packaged docs archive installs.

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
