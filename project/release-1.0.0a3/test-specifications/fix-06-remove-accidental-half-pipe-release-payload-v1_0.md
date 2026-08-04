# Fix 06 Test: Remove Accidental Half-Pipe Release Payload

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Verify half-pipe payload removal in source and distributions

- Input: candidate source, wheel, sdist, and a fresh installation environment.
- Work: inspect paths/dependencies/artifact contents, install the wheel, and smoke
  supported imports/examples.
- Output: source-and-distribution absence checks tied to release qualification.
- Complete when: no artifact or clean installation contains or pulls the experiment.

## Backlink

[Fix 06 specification](../specifications/fix-06-remove-accidental-half-pipe-release-payload-v1_0.md)

## Manual Smoke

Build the candidate, inspect its dependency metadata and file list, then clean
install it and import the supported package and examples.

## Automated Smoke

Assert the source paths do not exist and parsed project dependencies do not
contain `build123d`.

## Automated Acceptance

- Build wheel and sdist; inspect both archives for `half_pipe`, `cad.py`, and
  `build123d` metadata.
- Clean-install the wheel with dependencies and assert `build123d` is not pulled.
- Run package import and approved example smoke tests.
- Search maintained docs/tests for live references requiring the removed adapter.

The test checks built artifacts, not source absence alone.
