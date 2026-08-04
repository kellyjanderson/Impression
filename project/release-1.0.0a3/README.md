# Impression v1.0.0a3 Fix Release Definition

Date: 2026-08-04
Status: Planned
Tracking issue: [#232](https://github.com/kellyjanderson/Impression/issues/232)

## Intent

`v1.0.0a3` is a corrective alpha release. It restores trustworthy modeling,
preview/export, installation, and release behavior without adding new product
scope. The release starts from `v1.0.0a2` and is complete only when its source,
built artifacts, clean-install behavior, and test-modeling reproductions agree.

## User-Visible Outcomes

- The five current failures documented by the test-modeling project work without
  its grouped-body, solid-wall, sampling, or identity-loss workarounds.
- User models keep one coherent `impression.modeling` type universe when loaded.
- Surface-first results can travel through the normal preview and export path.
- Export refuses invalid manufacturing output instead of silently writing it.
- The Linux reference-review shell exits cleanly under the supported headless
  test configuration.
- The accidental half-pipe experiment and its dependency are absent from the
  released package.
- Release artifacts are tested, clean-installed, and published as prereleases.

## Planned Specifications

The release contains 13 final implementation leaves and 13 paired verification
leaves. The indexed lists live in [Specifications](specifications/README.md) and
[Test Specifications](test-specifications/README.md); execution order lives in
[Progression](planning/progression.md).

The evidence and disposition for every known issue considered for this release
is recorded in [Known-Issue Intake](planning/known-issue-intake.md).

## Release Gates

The release candidate may be tagged only after all of the following are true:

1. Every implementation and paired verification leaf in progression is complete.
2. The five test-modeling reproductions pass against the candidate build without
   their documented workarounds.
3. The full configured test suite is green on supported macOS and Linux lanes,
   including issue #227's headless lifecycle case.
4. Wheel, source distribution, and documentation archive are built once from the
   candidate commit and tested by clean installation in fresh environments.
5. Installed-artifact smoke tests cover import, model load, preview, STL export,
   and documentation installation.
6. Package contents contain no accidental `half_pipe` example, `cad.py` adapter,
   or runtime `build123d` dependency.
7. Documentation archive traversal abuse cases are rejected before any write.
8. GitHub publishes `v1.0.0a3` as a prerelease and attaches only the qualified
   artifacts produced by the gated workflow.

## Exclusions

- No new modeling feature family or UI redesign.
- No broad public-API curation, generated-code governance, unit-system redesign,
  text-engine rewrite, or general cancellation architecture.
- No reintroduction of the SDF or half-pipe experiments in another form.
- No claim that deferred architectural work is fixed merely because this release
  strengthens the first preview/export path.

## Version-Level Notes

- Target tag: `v1.0.0a3`.
- Base release: `v1.0.0a2`.
- This folder is active release work. Archive it under `project/releases/` only
  after the release is published and its evidence is frozen.
- GitHub issue #232 is the release-planning anchor; implementation work should use
  linked child issues or pull requests rather than a parallel local issue tracker.
