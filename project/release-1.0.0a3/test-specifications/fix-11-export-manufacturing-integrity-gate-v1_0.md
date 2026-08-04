# Fix 11 Test: Export Manufacturing Integrity Gate

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One CLI export matrix proves valid output succeeds and each integrity failure refuses atomically.

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
