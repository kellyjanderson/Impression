# Fix 13A: Release Artifact Build and Qualification (v1.0)

Date: 2026-08-04
Status: Final
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `not applicable - release workflow qualification policy`
Source artifact: `fix-13-release-workflow-artifact-qualification-v1_0.md`
Split provenance: `fix-13-release-workflow-artifact-qualification-v1_0.md`
Canonical status: `Canonical`
Prerequisites:
- `none` - qualification workflow code can land before candidate feature completion.

## Review Score

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; new split child independently scored in review pass 2.
- Adversarial rescore basis: counted version/manifest records, four test/build/install
  dependencies, wheel/sdist/docs/manifest outputs, four reused tools, two existing
  workflow/helper additions, job ordering, ephemeral artifact/environment writes, and
  bounded CI cost. This child has no release permission or external publication.
- Functions/methods: 0 x 2 = 0
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 4 x 1 = 4
- Returns/outputs/signals: 4 x 1 = 4
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 4 x 0.5 = 2
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 22
- Split decision: remain whole after mandatory split review. Tests, one-time build,
  artifact inspection, and clean-install smoke collectively produce one qualified
  immutable manifest; none is independently releasable as qualification evidence.

## Source Field Carryover

- Source purpose: produce and qualify the exact a3 wheel, sdist, and docs archive.
- Source responsibilities by category:
  - Data structures/models: parsed project/tag version and qualified artifact manifest.
  - Dependencies/services: CI tests, Python build, fresh pip environments, docs packager.
  - Returns/outputs/signals: wheel, sdist, docs ZIP, and qualified manifest.
  - Reusable code plan: current CI test command, build, docs packager, and smoke commands.
  - Async/concurrency behavior: build follows tests; qualification follows build artifacts.
  - Destructive/write behavior: ephemeral artifact and clean-environment writes only.
  - Performance-sensitive behavior: build once; reuse exact outputs across checks.
  - Functions, UI, database, external security permission, and cross-screen behavior: not applicable.
- Source open questions / nuance discovered: tag/project/artifact versions must match exactly.
- Source split/provenance notes: owns parent responsibilities through qualified manifest emission.

## Purpose

Create a single immutable artifact set and prove it installs and behaves correctly
before any external release write is possible.

## Scope

Owns:

- release test gate, one-time wheel/sdist/docs build, artifact manifest, content/version
  inspection, and fresh wheel/sdist/docs smoke.

Does not own:

- GitHub release creation, prerelease flag, or external asset upload (Fix 13B).

## Split Coverage

- Parent spec: `fix-13-release-workflow-artifact-qualification-v1_0.md`
- Parent coverage status: 100% covered with Fix 13B
- Parent responsibilities owned by this child: tests, build, manifest, inspection, clean install.
- Parent responsibilities still missing from children: none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../planning/spec-review-ledger-20260804-040607.md` | 2 | a3 specs 01-12, 13A, 13B | none | reached |

## Implementation Routing

- Primary modules/files: `.github/workflows/release.yml` - tests/build/qualification jobs.
- Supporting modules/files: `scripts/release/package_docs_zip.py` - docs artifact.
- GUI/QML files: not applicable.
- Reusable library/module files: existing release helper tests/scripts only.
- Tests: workflow contract tests and fresh artifact smoke commands.

## Chosen Defaults / Parameters

- Build each artifact once after tests pass; tag and project version must match.
- Qualify wheel and sdist in separate fresh environments; install the docs ZIP to temp.
- Manifest records names, types, versions, and hashes; forbidden experiment payload fails.

## Data Ownership

- Source of truth: candidate commit, tag/project version, built files, and manifest hashes.
- Read ownership: qualification jobs.
- Write ownership: build job creates ephemeral artifacts; qualification creates temp environments.
- Derived/cache data: manifest and artifacts are immutable workflow-run outputs.
- Privacy/logging constraints: no release token or secrets are required/printed in this child.

## Dependencies And Routes

- Domain/service dependencies: CI test command, `python -m build`, pip/fresh environments,
  docs packager.
- Database and GUI routes: not applicable.
- Workflow route: tag -> tests -> build once -> inspect/install smoke -> qualified manifest.
- Background/concurrency route: explicit job dependencies prevent qualification before build.

## Prerequisite Handling

- Architecture feedback artifacts/status: none; not applicable.
- Already implemented prerequisites: CI tests and docs packager.
- Missing prerequisite architecture/specifications: none.
- Unimplemented prerequisite specifications: none for workflow code.
- Progression handling: current item precedes Fix 13B.

## Application Integration

- App type: workflow.
- User/caller surface: maintainer tag and downstream publication job.
- Invocation route: tag-triggered qualification jobs.
- Wiring owner/module: `.github/workflows/release.yml`.
- Observable result: qualified artifact manifest or failed workflow with no publish input.
- Integration validation: real workflow candidate plus local/helper contract tests.
- Incomplete status risk: source tests without exact artifact install/inspection are insufficient.

## Reuse And Extraction Plan

- Existing code to reuse: CI tests, Python build, docs packager, pip smoke commands.
- Current reuse readiness: add orchestration and assertions to existing workflow/helpers.
- Extraction/wrapping needed: testable private version/manifest helper only if needed.
- Additions to existing library/modules: workflow and release helper tests/scripts.
- New reusable modules to expose: none.
- One-off code justification: repository-specific qualification orchestration.

## Required DTOs / Functions / Components

- DTOs/models: parsed version and artifact manifest.
- Functions/methods: no public API; optional private testable helper only.
- UI fields/elements/components: not applicable.

## Performance Contract

- One build per tag; checks reuse artifacts and complete within normal CI limits.

## Error And State Behavior

- Any test/build/version/content/install/docs failure stops without emitting a qualified manifest.
- Temporary environments/artifacts are isolated to the workflow run.

## Test Strategy

- Unit tests: version/manifest/content classification helpers.
- Integrated route tests: workflow order and fresh wheel/sdist/docs smoke.
- Service/DB and GUI tests: not applicable.
- Production-data rule: repository artifacts and temporary environments only.

## Acceptance Criteria

- Tests pass before one-time artifact build.
- Exact wheel, sdist, and docs outputs pass version/content/install smoke.
- A qualified immutable manifest is emitted only after every check passes.

## Verification

[Paired test specification](../test-specifications/fix-13a-release-artifact-build-qualification-v1_0.md)

## Readiness Checklist

- [x] Ancestors, complete score, carryover, canonical status, split coverage, and ledger are explicit.
- [x] The 22-point split review documents one qualification-evidence transaction.
- [x] Workflow route/order, outputs, ephemeral writes, defaults, ownership, reuse, performance,
  errors, and integrated proof are explicit.
- [x] No readiness gap, missing prerequisite, or unresolved deferral marker remains.
