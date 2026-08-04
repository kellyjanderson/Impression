# Fix 13A Test: Release Artifact Build and Qualification

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-13a-release-artifact-build-qualification-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: not applicable

## Overview

Verify that tests gate a one-time build and that the exact wheel, sdist, and docs
artifacts pass version/content/clean-install checks before a manifest is emitted.

## Application Integration Under Test

- App type: workflow.
- User/caller surface: maintainer tag and downstream publication job.
- Invocation route: tag -> tests -> build -> inspect/install smoke -> manifest.
- Wiring owner/module: `.github/workflows/release.yml`.
- Observable result: qualified manifest or failed workflow with no publish input.
- Integration validation: workflow contract and real candidate qualification run.

## Manual Smoke

- Run the qualification commands for `v1.0.0a3`, inspect the manifest, and install
  each distribution/docs artifact in a fresh temporary environment/destination.

## Automated Smoke Tests

- Parse tag/project versions, build artifacts once, and assert expected manifest entries.
- Install the wheel and run import/model-load/preview-export non-GUI smoke.

## Automated Acceptance Tests

- Unit/helper behavior: version agreement, artifact names/hashes/types, forbidden payload.
- Integrated route behavior: tests precede build; exact outputs feed wheel/sdist/docs smoke.
- Failure behavior: injected test/build/content/install failure prevents manifest emission.

## App-Type Proof

- Workflow proof: trigger, job dependencies, artifact transfer, temp writes, and manifest output.
- GUI, console, API/service, mixed, and library-only proof: not applicable.

## Fixtures And Data

- Alpha tag/project metadata, candidate wheel/sdist/docs, temporary fresh environments.
- Production-data rule: repository artifacts and temporary environments only.

## Acceptance

- [x] Feature spec is canonical and workflow route is covered.
- [x] Exact artifact outputs and manifest are asserted.
- [x] Failure behavior prevents qualification output.
