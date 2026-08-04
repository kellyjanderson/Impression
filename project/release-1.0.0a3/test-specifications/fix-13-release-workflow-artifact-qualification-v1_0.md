# Fix 13 Test: Release Workflow Artifact Qualification

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One workflow-contract suite proves ordering, exact-artifact qualification, clean installation, and prerelease classification.

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
