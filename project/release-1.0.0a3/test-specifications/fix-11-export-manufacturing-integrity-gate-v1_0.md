# Fix 11 Test: Export Manufacturing Integrity Gate

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-11-export-manufacturing-integrity-gate-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: `project/release-0.1.0a/architecture/mesh-execution-tessellation-boundary-architecture.md`

## Overview

Verify manufacturing mesh QA and atomic STL output through the real export command.

## Application Integration Under Test

- App type: console.
- User/caller surface: `impression export` command/options/output path.
- Invocation route: command -> collector -> export tessellation -> QA -> atomic write.
- Wiring owner/module: `src/impression/cli.py`.
- Observable result: STL/zero exit or diagnostic/nonzero exit with untouched target.
- Integration validation: Typer command tests with geometry and path sentinels.

## Backlink

[Fix 11 specification](../specifications/fix-11-export-manufacturing-integrity-gate-v1_0.md)

## Manual Smoke

Export a valid cube and an intentionally open surface to fresh targets; confirm
only the cube writes and the refusal reports why the open surface is unsuitable.

## Automated Smoke Tests

Invoke CLI export for a watertight primitive and assert a parseable non-empty STL;
invoke it for an open fixture and assert nonzero result with no file.

## Automated Acceptance Tests

- Parameterize empty, non-finite, degenerate, open, and non-manifold fixtures.
- Assert each failure category is present in the diagnostic.
- Pre-create a target sentinel and prove failed export does not truncate it.
- Cover valid binary/ASCII output and surface-first export policy.
- Export the test-modeling fixtures and assert zero degenerates/watertightness.

Use temporary targets and machine-readable mesh QA, not slicer screenshots.

## App-Type Proof

- Console proof: command args, stdout/stderr, exit code, and file side effects are asserted.
- GUI, API/service, mixed, and library-only proof: not applicable.

## Fixtures And Data

- Valid, empty, non-finite, degenerate, open, and non-manifold models; path sentinels.
- Production-data rule: temporary models/targets only.

## Acceptance

- [x] Feature spec is canonical and the real console route is covered.
- [x] Observable artifact, diagnostic, exit, and atomic side effects are asserted.
- [x] Every required failure class is covered.
