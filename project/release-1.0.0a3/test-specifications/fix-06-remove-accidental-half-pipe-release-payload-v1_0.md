# Fix 06 Test: Remove Accidental Half-Pipe Release Payload

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One source-and-built-artifact inspection proves the entire accidental experiment payload is absent.

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
