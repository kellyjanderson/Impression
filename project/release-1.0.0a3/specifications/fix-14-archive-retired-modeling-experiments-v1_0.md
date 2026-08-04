# Fix 14: Archive Retired Modeling Experiments Specification

Date: 2026-08-04
Status: Final
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `not applicable - repository and release hygiene`
Source artifact: `project/release-1.0.0a3/planning/known-issue-intake.md`
Split provenance: `none`
Canonical status: `Canonical`
Review Score: 13
Prerequisites:
- The exact hinge and SDF experiment sources must be locally recoverable before removal proceeds.

## Source Field Carryover

- Source purpose:
  - Remove retired experimental product surfaces without losing their source history.
- Source responsibilities by category:
  - Functions/methods: not applicable.
  - Data structures/models: the hinge and SDF standalone Git repositories.
  - Dependencies/services: not applicable.
  - Returns/outputs/signals: two clean archives, one clean core payload, and one absent remote experiment ref.
  - UI surfaces/components: not applicable.
  - UI fields/elements: not applicable.
  - Reusable code plan: reuse Git object storage and existing package-absence tests.
  - Database queries/tables/migrations: not applicable.
  - Async/concurrency behavior: not applicable.
  - Destructive/write behavior: named core-tree removals and one named remote-branch deletion.
  - Security/privacy-sensitive behavior: not applicable.
  - Performance-sensitive behavior: not applicable.
  - Cross-screen reusable behavior: not applicable.
- Source open questions / nuance discovered:
  - Historical research and release records remain as evidence; only live product ownership is removed.
- Source split/provenance notes:
  - No parent split; both archives participate in one preserve-before-remove transaction.

## Purpose

Preserve the retired hinge and SDF experiments outside Impression, then remove
their remaining live product files and experiment branch from the core repository.

## Scope

Owns:

- Locally committed, independently verifiable hinge and SDF archives.
- Removal of the hinge bridge, end-user docs/examples, hinge-only tests and active
  specifications, and core inventory entries that claim hinge ownership.
- Removal of `feature/sdf-endcaps-shelved` from the Impression remote after archive verification.
- Package regression coverage proving both experiment families are absent.

Does not own:

- Publishing either standalone archive or creating a new remote for it.
- Rewriting Impression history or deleting the SDF cleanup branch.
- Reintroducing either experiment as supported product scope.
- Rewriting historical research or previously published release records.

## Split Coverage

- Parent spec: `none`
- Parent coverage status: not applicable
- Parent responsibilities owned by this child:
  - not applicable
- Parent responsibilities still missing from children:
  - none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../planning/spec-review-ledger-20260804-071535.md` | 1 | Fix 14 | none | reached |

## Implementation Routing

- Primary modules/files:
  - `/Users/k/Documents/Projects/impression-hinges` - hinge archive ownership.
  - `/Users/k/Documents/Projects/impression-sdf-experiments` - SDF archive ownership.
  - `src/impression/modeling/hinges.py` - removed core compatibility bridge.
- Supporting modules/files:
  - `docs/`, `impression-docs/`, and `project/release-0.1.0a/` - remove live hinge product documentation and specification leaves.
  - `src/impression/modeling/csg.py` - remove retired hinge caller inventory.
  - `project/release-1.0.0a3/` - record release scope and progression.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - no new reusable product module.
- Tests:
  - `tests/test_release_metadata.py` - hinge/SDF package-absence contract.
  - documentation, CSG inventory, and no-hidden-fallback suites - surviving core route integrity.

## Chosen Defaults / Parameters

- Archives remain local Git repositories with no automatic remote publication.
- The SDF archive preserves original commit `3b35e4490be4c4de4592c0a5bb445655c4c1efe6`.
- Only `feature/sdf-endcaps-shelved` is deleted; the correction branch remains as audit history.
- Historical records may name retired experiments when their historical context is explicit.
- Missing experimental imports fail collection; no compatibility fallback remains in core.

## Data Ownership

- Source of truth: standalone Git object databases for experiments; the Impression Git tree for release content.
- Read ownership: local developers may inspect archives; build/test tooling reads the Impression tree.
- Write ownership: this change writes archive metadata and removes only the named core files/ref.
- Derived/cache data: wheel, sdist, docs ZIP, Python caches, and test outputs are rebuilt from source.
- Privacy/logging constraints: archive verification must not introduce credentials or machine-private data.

## Dependencies And Routes

- Domain/service dependencies:
  - local Git and the existing Impression GitHub remote for the single ref deletion.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable; preservation and removal run sequentially.

## Prerequisite Handling

- Architecture feedback artifacts:
  - none; repository ownership correction does not change product architecture.
- Architecture feedback status:
  - not applicable.
- Already implemented prerequisites:
  - existing Git history and package-content regression test route.
- Missing prerequisite architecture:
  - none.
- Missing prerequisite specifications:
  - none.
- Unimplemented prerequisite specifications:
  - none.
- Progression handling:
  - archive verification must complete before core or remote removal.

## Application Integration

- App type: mixed workflow and library-only.
- User/caller surface: release maintainer workflow and `impression.modeling` consumers.
- Invocation route: archive commit/fsck -> core cleanup -> pytest collection and package tests -> remote ref deletion.
- Wiring owner/module: Git repositories, package tree, documentation indexes, and release metadata tests.
- Observable result: recoverable external archives, no experimental core import/docs surface, and no SDF experiment remote ref.
- Integration validation: Git fsck/ref queries plus full collection and focused source/package regression suites.
- Incomplete status risk: a partial transaction either loses recoverability or leaves unsupported product surfaces in a3.

App-type-specific proof:

- Mixed workflow proof: Git verifies archive integrity and remote-ref absence; library tests verify imports and packaging independently.
- Library-only proof: `tests/test_release_metadata.py` and complete pytest collection exercise the core consumer boundary.

## Reuse And Extraction Plan

- Existing code to reuse:
  - Git clone/init/commit/fsck/ref operations - durable source preservation and verification.
  - `tests/test_release_metadata.py` - established release-payload absence assertions.
- Current reuse readiness:
  - both mechanisms are reusable as-is.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - no production-library additions; extend release metadata tests only.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - named file/ref removal is the intended one-time repository correction.

## Required DTOs / Functions / Components

- DTOs/models:
  - standalone hinge Git repository - archived hinge source and provenance.
  - standalone SDF Git repository - archived SDF source and provenance.
- Functions/methods:
  - not applicable.
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Not performance-sensitive; bounded Git verification and normal package-test limits apply.

## Error And State Behavior

- Removal does not proceed until each archive is committed and `git fsck` succeeds.
- A stale experimental import fails test collection instead of loading a sibling fallback.
- Package absence tests fail when an experiment module, export, doc, example, or dependency returns.
- Remote deletion targets one fully resolved ref and leaves correction/history branches intact.

## Test Strategy

- Unit tests:
  - source, export, documentation, example, and dependency absence assertions.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - archive fsck/ref checks, full pytest collection, focused release metadata,
    documentation, CSG inventory, and no-hidden-fallback suites.
- Production-data rule:
  - tests use repository fixtures and temporary outputs only.

## Acceptance Criteria

- Both standalone archives are committed, independent, and pass full Git object verification.
- Core contains no hinge runtime bridge, current hinge end-user docs/examples, or hinge-owned tests/specification leaves.
- Core release tests explicitly assert hinge and SDF absence and collect without experimental imports.
- `feature/sdf-endcaps-shelved` is absent from the Impression remote.
- Historical release/research evidence remains available.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Review Score appears in the front matter and exactly matches the total in the final Review Score Calculation section.
- [x] The current implementation-spec template was loaded and its source path is recorded in the final Review Score Calculation section.
- [x] Review Score is adversarially recounted from the current spec text; prior scores are challenged instead of trusted.
- [x] Unresolved deferral/gap markers are absent or explicitly resolved.
- [x] Source fields are carried into spec sections or preserved as explicit provenance/history.
- [x] Canonical status is explicit.
- [x] Prerequisites are linked, implemented, or marked not applicable.
- [x] Missing or stale prerequisite architecture is marked not applicable.
- [x] Missing prerequisite behavior is marked not applicable.
- [x] Split coverage is marked not applicable.
- [x] Per-request review ledger records the terminal new-leaf list.
- [x] Implementation owners and modules are named.
- [x] Existing code reuse/extraction decisions are explicit.
- [x] Library additions and new reusable module boundaries are named or marked not applicable.
- [x] UI fields/elements are marked not applicable.
- [x] Chosen defaults are explicit.
- [x] Data source of truth and write owner are explicit.
- [x] GUI/concurrency routes are marked not applicable and sequential ordering is explicit.
- [x] App type and application integration route are explicit.
- [x] Integrated route validation is named.
- [x] Mixed and library-only proof match the app type.
- [x] Performance bounds are explicit.
- [x] Privacy/logging constraints are explicit.
- [x] Test strategy does not depend on production data.
- [x] Acceptance criteria are testable.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: 13; treated as an adversarial input rather than trusted.
- Adversarial rescore basis: checked for uncounted branch/service coupling,
  hidden archive outputs, missing ownership, ambiguous removal targets, absent
  mixed-route proof, and unresolved deferral markers; no additional responsibility remained.
- Functions/methods: 0 x 2 = 0
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 0 x 1 = 0
- Returns/outputs/signals: 4 x 1 = 4
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 2 x 3 = 6
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 0 x 2 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 13
- If total matches prior score, adversarial survival reason: the preserved archives,
  cleaned core payload, absent remote ref, two reuse mechanisms, and two destructive
  boundaries account for every explicit responsibility without adding a service,
  function, UI, database, async, security, or performance surface.
