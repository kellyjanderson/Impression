# Fix 15 Test: Preview PNG Export

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-15-preview-png-export-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: not applicable

## Overview

Verify that the preview command produces a real PNG through the console route
even when the launch directory already contains a live preview control file.

## Application Integration Under Test

- App type: console.
- User/caller surface: `impression preview MODEL --screenshot PATH`.
- Invocation route: CLI parsing -> one-shot route selection -> off-screen PyVista render -> PNG artifact.
- Wiring owner/module: `src/impression/cli.py` and `src/impression/preview.py`.
- Observable result: zero exit, success text, decodable PNG, and unchanged live control-file content.
- Integration validation: fake-renderer route test plus a real installed-command subprocess smoke.

## Manual Smoke

- Keep an interactive preview running, invoke `impression preview` with `--screenshot`, and confirm the PNG appears without changing or closing the original preview.
- Open the PNG and confirm the requested model is visible with normal preview styling.

## Automated Smoke Tests

- Invoke the CLI with a fake preview renderer and a live-looking control file; assert screenshot mode reaches the renderer with watching and control-file handoff disabled.
- Run the installed command against a temporary box model and decode the resulting PNG.

## Automated Acceptance Tests

- Unit/helper behavior:
  - Assert the requested screenshot path and one-shot settings reach `PyVistaPreviewer.show(...)`.
- Integrated route behavior:
  - Assert the command writes a non-empty PNG with the PNG signature, reports its path, exits zero, and preserves the control file.
- Failure and stale-result behavior, if applicable:
  - Assert renderer failure remains a non-zero command result and does not report success.

## App-Type Proof

- GUI proof:
  - not applicable.
- Console proof:
  - command, model argument, screenshot flag, stdout, exit code, PNG side effect, and control-file non-mutation are asserted.
- API/service proof:
  - not applicable.
- Mixed-surface proof:
  - not applicable.
- Library-only proof:
  - not applicable.

## Fixtures And Data

- Temporary Python box model, screenshot directory, and live-looking control file.
- Fake preview renderer for deterministic route selection.
- Real workspace virtual environment and installed `impression` entrypoint for renderer integration.
- Production-data rule: no production models or user output paths are used.

## Acceptance

- [x] Feature spec is canonical.
- [x] Route-level proof exists for the console command.
- [x] Helper-only tests cannot satisfy this feature contract.
- [x] PNG output, success text, exit code, and control-file preservation are asserted.
- [x] Renderer failure behavior is covered.
