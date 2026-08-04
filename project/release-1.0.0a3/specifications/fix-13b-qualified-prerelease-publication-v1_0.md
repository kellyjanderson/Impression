# Fix 13B: Qualified Prerelease Publication (v1.0)

Date: 2026-08-04
Status: Final
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `not applicable - release publication policy`
Source artifact: `fix-13-release-workflow-artifact-qualification-v1_0.md`
Split provenance: `fix-13-release-workflow-artifact-qualification-v1_0.md`
Canonical status: `Canonical`
Prerequisites:
- `fix-13a-release-artifact-build-qualification-v1_0.md` - must emit the qualified artifact manifest first.

## Review Score

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; new split child independently scored in review pass 2.
- Adversarial rescore basis: counted manifest/version records, GitHub release action/API,
  release output, two reused boundaries, one workflow addition, dependency ordering,
  external release write, and scoped-token/provenance security behavior.
- Functions/methods: 0 x 2 = 0
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 0 x 2 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 16
- Split decision: remain whole after mandatory split review. Prerelease classification,
  asset selection, scoped permission, and release creation are one external publication
  transaction with one rollback/refusal boundary.

## Source Field Carryover

- Source purpose: publish only the artifact set qualified by Fix 13A with correct metadata.
- Source responsibilities by category:
  - Data structures/models: qualified artifact manifest and parsed prerelease version.
  - Dependencies/services: GitHub release action and GitHub release API.
  - Returns/outputs/signals: GitHub release with attached assets.
  - Reusable code plan: current release action and PEP 440 parsing boundary.
  - Async/concurrency behavior: publication depends on successful qualification artifact.
  - Destructive/write behavior: external release and asset creation.
  - Security-sensitive behavior: scoped contents permission and token secrecy/provenance.
  - Functions, UI, database, performance, and cross-screen behavior: not applicable.
- Source open questions / nuance discovered: `a`, `b`, and `rc` publish as prereleases; final does not.
- Source split/provenance notes: owns parent responsibilities after qualified manifest emission.

## Purpose

Create the GitHub release only from Fix 13A's qualified manifest and apply deterministic
PEP 440 prerelease metadata.

## Scope

Owns:

- manifest dependency, asset selection, release permission, prerelease classification,
  GitHub release creation, and failure/no-release behavior.

Does not own:

- tests, artifact build, inspection, or clean-install smoke (Fix 13A).

## Split Coverage

- Parent spec: `fix-13-release-workflow-artifact-qualification-v1_0.md`
- Parent coverage status: 100% covered with Fix 13A
- Parent responsibilities owned by this child: gated release write, asset upload, prerelease metadata.
- Parent responsibilities still missing from children: none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../planning/spec-review-ledger-20260804-040607.md` | 2 | a3 specs 01-12, 13A, 13B | none | reached |

## Implementation Routing

- Primary modules/files: `.github/workflows/release.yml` - dependency-gated publish job.
- Supporting modules/files: existing release helper tests if version classification is extracted.
- GUI/QML files: not applicable.
- Reusable library/module files: current GitHub release action configuration.
- Tests: workflow dependency, version classification, asset list, post-publish metadata.

## Chosen Defaults / Parameters

- Consume only files named/hashed in Fix 13A's manifest.
- PEP 440 `a`, `b`, and `rc` versions set prerelease true; final versions set false.
- `contents: write` is scoped to publish; any missing/mismatched manifest refuses.

## Data Ownership

- Source of truth: qualified manifest, tag version, and candidate commit.
- Read ownership: publish job verifies manifest/version before release action.
- Write ownership: publish job alone creates external release/assets.
- Derived/cache data: prerelease boolean derives from parsed version.
- Privacy/logging constraints: never log tokens; expose only artifact/version metadata.

## Dependencies And Routes

- Domain/service dependencies: GitHub release action and GitHub release API.
- Database and GUI routes: not applicable.
- Workflow route: qualified manifest -> verify tag/version/assets -> create release/upload assets.
- Background/concurrency route: publish job has a hard dependency on successful Fix 13A output.

## Prerequisite Handling

- Architecture feedback artifacts/status: none; not applicable.
- Already implemented prerequisites: current release action.
- Missing prerequisite architecture/specifications: none.
- Unimplemented prerequisite specifications: Fix 13A, linked above.
- Progression handling: implement and verify Fix 13A before this item.

## Application Integration

- App type: workflow.
- User/caller surface: maintainer tag and GitHub release consumer.
- Invocation route: successful qualification -> publish job -> GitHub release.
- Wiring owner/module: `.github/workflows/release.yml`.
- Observable result: release with correct prerelease flag/assets or no release on refusal.
- Integration validation: workflow contract plus post-publish GitHub metadata/asset check.
- Incomplete status risk: release-action configuration without manifest dependency can publish unqualified files.

## Reuse And Extraction Plan

- Existing code to reuse: current GitHub release action and version parsing helper if created by Fix 13A.
- Current reuse readiness: add dependency/manifest inputs to existing workflow.
- Extraction/wrapping needed: none beyond private version helper shared with Fix 13A if present.
- Additions to existing library/modules: publish job configuration/tests.
- New reusable modules to expose: none.
- One-off code justification: repository release publication is workflow-specific.

## Required DTOs / Functions / Components

- DTOs/models: qualified artifact manifest and parsed version.
- Functions/methods: no public API; optional shared private version parser.
- UI fields/elements/components: not applicable.

## Performance Contract

- Not performance-sensitive; upload each qualified asset once.

## Error And State Behavior

- Missing/mismatched manifest, tag, version, or asset stops before release creation.
- Publish failure is visible and never falls back to an unqualified rebuild.

## Test Strategy

- Unit tests: prerelease classification and manifest asset selection.
- Integrated route tests: workflow dependency/failure reachability and post-publish metadata.
- Service/DB and GUI tests: not applicable.
- Production-data rule: repository artifacts and test tags only.

## Acceptance Criteria

- Publish is unreachable until Fix 13A succeeds and its manifest verifies.
- `v1.0.0a3` produces a prerelease with exactly the qualified assets.
- Any manifest/version/asset mismatch creates no release.

## Verification

[Paired test specification](../test-specifications/fix-13b-qualified-prerelease-publication-v1_0.md)

## Readiness Checklist

- [x] Ancestors, complete score, carryover, canonical status, prerequisite, coverage, and ledger are explicit.
- [x] The 16-point split review documents one external publication transaction.
- [x] Workflow dependency, external write, permission/security, defaults, ownership, reuse,
  errors, and post-publish proof are explicit.
- [x] No readiness gap, missing prerequisite artifact, or unresolved deferral marker remains.
