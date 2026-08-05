# Fix 13B Test: Qualified Prerelease Publication

Date: 2026-08-04
Status: Final
Feature spec: `../specifications/fix-13b-qualified-prerelease-publication-v1_0.md`
Feature spec canonical status: Canonical
Architecture ancestor: not applicable

## Overview

Verify that GitHub publication consumes only Fix 13A's qualified manifest, applies
PEP 440 prerelease semantics, and creates no release after refusal/failure.

## Application Integration Under Test

- App type: workflow.
- User/caller surface: maintainer tag and GitHub release consumer.
- Invocation route: qualified manifest -> publish job -> GitHub release/assets.
- Wiring owner/module: `.github/workflows/release.yml`.
- Observable result: correct prerelease/assets or no release.
- Integration validation: workflow dependency tests and post-publish metadata check.

## Manual Smoke

- Inspect the candidate GitHub release and confirm prerelease status, tag/version,
  asset names, and hashes match the qualified manifest.

## Automated Smoke Tests

- Classify alpha/beta/rc/final versions and select only manifest-listed assets.
- Validate publish job has a hard successful dependency on Fix 13A output.

## Automated Acceptance Tests

- Unit/helper behavior: version classification and manifest asset selection.
- Integrated route behavior: qualified a3 manifest creates one prerelease with exact assets.
- Failure behavior: missing/mismatched manifest/tag/version/asset leaves no release.

## App-Type Proof

- Workflow proof: dependency, scoped permission, external release write, metadata, assets.
- GUI, console, API/service, mixed, and library-only proof: not applicable.

## Fixtures And Data

- Alpha/beta/rc/final versions, valid/invalid manifests, candidate artifact set.
- Production-data rule: repository/test release artifacts only.

## Acceptance

- [x] Feature spec is canonical and external publication route is covered.
- [x] Release metadata/assets and no-release failures are asserted.
- [x] Helper-only tests cannot satisfy external publication proof.
