# Fix 13 Test: Release Workflow Artifact Qualification

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Verify gated qualification and prerelease publication

- Input: release workflow, version/tag fixtures, built artifacts, and injected failures.
- Work: validate job ordering, install/smoke/inspect exact outputs, and test version
  agreement plus alpha/beta/rc/final classification.
- Output: automated pre-publish qualification and post-publish metadata evidence.
- Complete when: injected failures block publishing and a3 exposes only qualified assets.

## Backlink

[Fix 13 specification](../specifications/fix-13-release-workflow-artifact-qualification-v1_0.md)

## Manual Smoke

Run the release qualification commands locally for `v1.0.0a3`, inspect the built
assets, and verify the workflow preview classifies the tag as a prerelease.

## Automated Smoke

Validate workflow structure and execute release helper tests that parse tag and
project versions, classify alpha/beta/rc/final, and inspect artifact names.

## Automated Acceptance

- Build wheel, sdist, and docs once; install wheel and sdist in fresh environments.
- Run installed import, model-load, preview/export, and docs-install smoke checks.
- Inspect archives for version agreement and forbidden experimental payload.
- Test prerelease classification for `a`, `b`, `rc`, and final versions.
- Inject a failing qualification command and assert publish is unreachable.
- On the real candidate, verify GitHub release metadata and attached checksums/file list.

The publish-metadata check may run post-release, but all pre-publish gates must be automated.
