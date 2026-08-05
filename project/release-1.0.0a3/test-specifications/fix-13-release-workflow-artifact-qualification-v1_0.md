# Fix 13 Test: Release Workflow Artifact Qualification

Date: 2026-08-04
Status: Superseded by split children
Feature spec: `../specifications/fix-13-release-workflow-artifact-qualification-v1_0.md`
Feature spec canonical status: Superseded parent
Architecture ancestor: not applicable

## Overview

Verify the tag workflow qualifies exact artifacts and publishes correct prerelease metadata.

Verification coverage moved to:

- [Fix 13A test](fix-13a-release-artifact-build-qualification-v1_0.md)
- [Fix 13B test](fix-13b-qualified-prerelease-publication-v1_0.md)

## Application Integration Under Test

- App type: workflow.
- User/caller surface: maintainer version tag and GitHub release consumer.
- Invocation route: tag -> tests -> build -> inspect/install smoke -> publish.
- Wiring owner/module: `.github/workflows/release.yml`.
- Observable result: qualified prerelease assets or no release on gate failure.
- Integration validation: workflow contract tests, candidate run, post-publish metadata check.

## Backlink

[Fix 13 specification](../specifications/fix-13-release-workflow-artifact-qualification-v1_0.md)

## Manual Smoke

Run the release qualification commands locally for `v1.0.0a3`, inspect the built
assets, and verify the workflow preview classifies the tag as a prerelease.

## Automated Smoke Tests

Validate workflow structure and execute release helper tests that parse tag and
project versions, classify alpha/beta/rc/final, and inspect artifact names.

## Automated Acceptance Tests

- Build wheel, sdist, and docs once; install wheel and sdist in fresh environments.
- Run installed import, model-load, preview/export, and docs-install smoke checks.
- Inspect archives for version agreement and forbidden experimental payload.
- Test prerelease classification for `a`, `b`, `rc`, and final versions.
- Inject a failing qualification command and assert publish is unreachable.
- On the real candidate, verify GitHub release metadata and attached checksums/file list.

The publish-metadata check may run post-release, but all pre-publish gates must be automated.

## App-Type Proof

- Workflow proof: tag trigger, job dependencies, artifacts, failure reachability, and release side effect.
- GUI, console, API/service, mixed, and library-only proof: not applicable.

## Fixtures And Data

- Alpha/beta/rc/final tags, built wheel/sdist/docs, and injected gate failures.
- Production-data rule: repository artifacts and temporary clean environments only.

## Acceptance

- [x] Feature spec is superseded and both canonical child routes are linked.
- [x] Artifact, release metadata, and gate-failure results are asserted.
- [x] Helper-only tests cannot satisfy publication provenance.
