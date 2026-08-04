# Fix 13: Release Workflow Artifact Qualification (v1.0)

Date: 2026-08-04
Status: Superseded by split children
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `not applicable - release workflow qualification policy`
Source artifact: `.github/workflows/release.yml`
Split provenance: `none`
Canonical status: `Superseded`
Prerequisites:
- `fix-13a-release-artifact-build-qualification-v1_0.md` - owns build and qualification.
- `fix-13b-qualified-prerelease-publication-v1_0.md` - owns gated external publication.

## Review Score

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; obsolete IWU metadata removed.
- Adversarial rescore basis: counted version/artifact records; five build/test/publish
  dependencies; four artifact/release outputs; four reused tools; two existing workflow/
  script additions; one external publication write; provenance-sensitive publication;
  bounded CI execution, and explicit job-ordering/concurrency. Workflow steps are
  not counted as Python methods.
- Functions/methods: 0 x 2 = 0
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 5 x 1 = 5
- Returns/outputs/signals: 4 x 1 = 4
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 4 x 0.5 = 2
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 26
- Split decision: split required. Fix 13A owns tests/build/install/inspection and emits
  an immutable qualified manifest; Fix 13B owns the gated external release write and
  prerelease metadata using only that manifest.

## Source Field Carryover

- Source purpose: test and qualify the exact artifacts before alpha publication.
- Source responsibilities by category:
  - Data structures/models: parsed version and qualified artifact manifest.
  - Dependencies/services: GitHub Actions, Python build, pip/fresh environments,
    docs packager, and GitHub release action.
  - Returns/outputs/signals: wheel, sdist, docs ZIP, and GitHub release.
  - Reusable code plan: existing CI tests, build, docs packager, and release action.
  - Destructive/write behavior: publish external GitHub release/assets.
  - Security-sensitive behavior: artifact/tag provenance and release permission.
  - Performance-sensitive behavior: bounded clean-install qualification in CI.
  - Async/concurrency behavior: job dependencies order qualification before publication.
  - Functions, UI, database, and cross-screen behavior: not applicable.
- Source open questions / nuance discovered: prerelease status derives from PEP 440 tag version.
- Source split/provenance notes: score 26 forced the Fix 13A/Fix 13B split.

## Purpose

Turn the tag workflow into a gated provenance chain that publishes only qualified artifacts.

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

## Split Coverage

- Parent spec: `none`
- Parent coverage status: 100% covered
- Parent responsibilities owned by this child:
  - Fix 13A covers tag/version validation, tests, one-time build, artifact manifest,
    archive inspection, and clean-install smoke.
  - Fix 13B covers dependency-gated publication, PEP 440 prerelease metadata,
    scoped permissions, release assets, and failure/no-release behavior.
- Parent responsibilities still missing from children: none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../planning/spec-review-ledger-20260804-040607.md` | 1 | a3 implementation specs 01-13 | Fix 13A; Fix 13B | continue |

## Implementation Routing

- `.github/workflows/release.yml` and focused release helper scripts/tests.
- `scripts/release/package_docs_zip.py` for docs packaging contract.

## Chosen Defaults / Parameters

- Trigger remains `v*`; tag and project version must agree.
- Build wheel, sdist, and docs exactly once and pass those files between jobs/steps.
- Smoke fresh installs of wheel and sdist; alpha/beta/rc tags set GitHub prerelease true.
- Any failed qualification prevents the publish step.

## Data Ownership

- Source of truth: tag, candidate commit, project version, and built artifact hashes/names.
- Read ownership: qualification job reads metadata/artifacts.
- Write ownership: release job alone creates GitHub release/assets after all gates.
- Derived/cache data: artifacts and manifest are immutable workflow-run outputs.
- Privacy/logging constraints: use scoped token; do not print secrets/environment credentials.

## Dependencies And Routes

- Domain/service dependencies: GitHub Actions, `python -m build`, pip/fresh venvs,
  docs packager, `softprops/action-gh-release`.
- Database and GUI routes: not applicable.
- Workflow route: tag -> tests -> build once -> artifact inspection/install smoke -> publish.
- Background/concurrency route: jobs may run only when artifact dependencies enforce ordering.

## Prerequisite Handling

- Architecture feedback artifacts/status: none; not applicable for release policy.
- Already implemented prerequisites: CI suite and docs packaging script.
- Missing prerequisite architecture/specifications: none.
- Unimplemented prerequisite specifications: none for workflow code; candidate publish waits all progression items.
- Progression handling: workflow fix may proceed now; actual tag is blocked by release gates, not by missing spec work.

## Application Integration

- App type: workflow.
- User/caller surface: maintainer pushing a version tag and GitHub release consumers.
- Invocation route: `v*` push -> gated workflow -> GitHub release.
- Wiring owner/module: `.github/workflows/release.yml`.
- Observable result: qualified assets and correct prerelease metadata, or no release on failure.
- Integration validation: workflow structure/helper tests, candidate run, post-publish metadata check.
- Incomplete status risk: source tests alone do not prove clean-installed artifacts or release metadata.

## Reuse And Extraction Plan

- Existing code to reuse: CI test command, Python build, docs packager, GitHub release action.
- Current reuse readiness: add gates/manifest handoff to existing workflow/script boundary.
- Extraction/wrapping needed: helper only if version classification cannot stay testable inline.
- Additions to existing library/modules: release workflow and existing release helper tests/scripts.
- New reusable modules to expose: none.
- One-off code justification: tag workflow orchestration is repository-specific.

## Required DTOs / Functions / Components

- DTOs/models: parsed PEP 440 version; artifact manifest with names/hashes/types.
- Functions/methods: no required Python API; workflow steps/optional private helper only.
- UI fields/elements/components: not applicable.

## Performance Contract

- One build per tag; qualification reuses artifacts and completes within normal CI limits.

## Error And State Behavior

- Any test/build/install/inspection mismatch fails before publish; no release is created.
- Publish retry consumes the same qualified artifacts or reruns full qualification.

## Test Strategy

- Unit tests: version classification and artifact manifest/content checks.
- Integrated route tests: workflow ordering plus fresh wheel/sdist/docs installation smoke.
- Post-publish check: release prerelease flag and attached asset list.
- Service/DB and GUI tests: not applicable; no production data.

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

## Readiness Checklist

- [x] Ancestors, full score, carryover, canonical status, and terminal ledger are explicit.
- [x] The 26-point score forced a split and both children cover 100% of parent responsibilities.
- [x] Workflow dependencies/order, outputs, publication write/security, defaults, ownership,
  reuse, performance, errors, and integration proof are explicit.
- [x] No readiness gap or missing child coverage remains; this parent is superseded.
- [x] Qualification and post-publish evidence avoid production data.
