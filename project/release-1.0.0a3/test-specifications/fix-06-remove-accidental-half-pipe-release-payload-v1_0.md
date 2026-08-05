# Fix 06 Test: Remove Accidental Half-Pipe Release Payload

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-06-remove-accidental-half-pipe-release-payload-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: not applicable

## Overview

Verify that source, built distributions, and clean installation contain no half-pipe payload.

## Application Integration Under Test

- App type: workflow.
- User/caller surface: package build and clean-install consumer.
- Invocation route: source -> build -> artifact inspection -> dependency install -> import smoke.
- Wiring owner/module: package metadata and release qualification workflow.
- Observable result: absence of files/import/dependency and successful approved imports.
- Integration validation: wheel/sdist inspection and clean installation.

## Backlink

[Fix 06 specification](../specifications/fix-06-remove-accidental-half-pipe-release-payload-v1_0.md)

## Manual Smoke

Build the candidate, inspect its dependency metadata and file list, then clean
install it and import the supported package and examples.

## Automated Smoke Tests

Assert the source paths do not exist and parsed project dependencies do not
contain `build123d`.

## Automated Acceptance Tests

- Build wheel and sdist; inspect both archives for `half_pipe`, `cad.py`, and
  `build123d` metadata.
- Clean-install the wheel with dependencies and assert `build123d` is not pulled.
- Run package import and approved example smoke tests.
- Search maintained docs/tests for live references requiring the removed adapter.

The test checks built artifacts, not source absence alone.

## App-Type Proof

- GUI, console, API/service, mixed, and library-only proof: not applicable.
- Workflow proof: the exact built wheel/sdist are inspected and the wheel is installed fresh.

## Fixtures And Data

- Candidate source, built artifacts, and temporary clean environment.
- Production-data rule: not applicable.

## Acceptance

- [x] Feature spec is canonical and full package route is proved.
- [x] Observable source/artifact/install results are asserted.
- [x] Source-only checks cannot satisfy this contract.
