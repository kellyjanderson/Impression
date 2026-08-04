# Fix 11 Test: Export Manufacturing Integrity Gate

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Verify the STL integrity gate and atomic write boundary

- Input: valid and five invalid mesh classes plus new/existing output-path sentinels.
- Work: invoke CLI export, inspect failure categories, and verify target atomicity.
- Output: a manufacturing-gate matrix covering ASCII, binary, and test-model exports.
- Complete when: valid output passes QA and every invalid fixture fails pre-write.

## Backlink

[Fix 11 specification](../specifications/fix-11-export-manufacturing-integrity-gate-v1_0.md)

## Manual Smoke

Export a valid cube and an intentionally open surface to fresh targets; confirm
only the cube writes and the refusal reports why the open surface is unsuitable.

## Automated Smoke

Invoke CLI export for a watertight primitive and assert a parseable non-empty STL;
invoke it for an open fixture and assert nonzero result with no file.

## Automated Acceptance

- Parameterize empty, non-finite, degenerate, open, and non-manifold fixtures.
- Assert each failure category is present in the diagnostic.
- Pre-create a target sentinel and prove failed export does not truncate it.
- Cover valid binary/ASCII output and surface-first export policy.
- Export the test-modeling fixtures and assert zero degenerates/watertightness.

Use temporary targets and machine-readable mesh QA, not slicer screenshots.
