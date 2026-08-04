# Fix 13: Release Workflow Artifact Qualification (v1.0)

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Qualify and publish the exact release artifacts

- Input: one `v*` tag, matching project metadata, and the candidate source commit.
- Work: gate tests; build once; install, smoke, and inspect the exact wheel/sdist/docs;
  and derive GitHub prerelease status from PEP 440 semantics.
- Output: a qualified immutable artifact set passed to the publish step.
- Complete when: every failed gate blocks publish and an alpha publishes only that
  qualified set as a prerelease.

## Problem And Outcome

The tag workflow builds and uploads artifacts without first running tests or
clean-installing those artifacts, and it does not explicitly mark alpha tags as
prereleases. The publish step must consume only artifacts that passed release
qualification in the same workflow run.

## Scope

- Run the configured release test gate before packaging.
- Build wheel, sdist, and docs archive once, then clean-install and smoke-test them.
- Inspect filenames, versions, contents, and accidental experimental dependencies.
- Publish `a`, `b`, and `rc` PEP 440 versions as GitHub prereleases.
- Prevent the release step from running after any qualification failure.

Not in scope: changing versioning schemes, signing infrastructure, or publishing
to a package index.

## Implementation Routing

- `.github/workflows/release.yml` and focused release helper scripts/tests.
- `scripts/release/package_docs_zip.py` for docs packaging contract.

## Contract

Input is a `v*` tag whose version matches project metadata. Qualified artifacts
are immutable outputs passed to the publish step. Smoke tests run from fresh
environments against the built wheel and sdist. GitHub prerelease status is
derived from parsed version semantics, not a manual after-publish correction.

## Acceptance Criteria

- A failing test, build, metadata check, or clean-install smoke prevents release.
- Smoke covers import, minimal model load, preview/export non-GUI path, and docs ZIP.
- Tag, package version, and artifact names agree.
- `v1.0.0a3` is published with prerelease metadata and only qualified assets.

## Verification

[Paired test specification](../test-specifications/fix-13-release-workflow-artifact-qualification-v1_0.md)
