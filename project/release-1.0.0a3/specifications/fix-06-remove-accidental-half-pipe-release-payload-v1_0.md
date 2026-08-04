# Fix 06: Remove Accidental Half-Pipe Release Payload (v1.0)

Date: 2026-08-04
Status: Final
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `not applicable - release payload correction`
Source artifact: Git history and current `main` payload inventory
Split provenance: `none`
Canonical status: `Canonical`
Prerequisites:
- `none` - removal is independent of modeling fixes.

## Review Score

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; obsolete IWU metadata removed.
- Adversarial rescore basis: counted dependency metadata, one removed dependency,
  source/wheel/sdist outputs, existing build reuse, and one destructive source removal.
- Functions/methods: 0 x 2 = 0
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 0 x 2 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 8.5
- Split decision: remain whole; files, dependency, and distribution metadata are one
  experiment payload and partial extraction would leave the release contaminated.

## Source Field Carryover

- Source purpose: remove the unapproved half-pipe experiment from a3.
- Source responsibilities by category:
  - Data structures/models: `pyproject.toml` dependency metadata.
  - Dependencies/services: `build123d`.
  - Returns/outputs/signals: corrected source, wheel, and sdist payloads.
  - Reusable code plan: existing packaging workflow.
  - Destructive/write behavior: delete two tracked experiment files and metadata entries.
  - Functions, UI, database, async, security, performance, and cross-screen behavior: not applicable.
- Source open questions / nuance discovered: Git history is the recovery surface.
- Source split/provenance notes: not applicable.

## Purpose

Remove the accidentally merged half-pipe experiment from source and release artifacts.

## Problem And Outcome

The experimental half-pipe branch was merged into `main` even though experimental
branches were not approved for release. The example, build123d adapter, and
`build123d` dependency must be absent from Impression while their history remains
recoverable in Git.

## Scope

- Remove `examples/half_pipe.py` and `src/impression/cad.py`.
- Remove `build123d` from runtime dependencies when no approved code uses it.
- Remove or update only references that exist solely for that experiment.

Not in scope: deleting Git history, publishing the experiment elsewhere, or
removing unrelated CAD functionality.

## Split Coverage

- Parent spec: `none`
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../planning/spec-review-ledger-20260804-040607.md` | 2 | a3 specs 01-12, 13A, 13B | none | reached |

## Implementation Routing

- `examples/half_pipe.py`, `src/impression/cad.py`, `pyproject.toml`.
- Package-content and dependency regression tests.

## Chosen Defaults / Parameters

- Delete the experiment from the product tree; preserve commit history.
- `build123d` is absent from runtime dependencies unless approved code independently requires it.

## Data Ownership

- Source of truth: Git tree and `pyproject.toml` dependency metadata.
- Read ownership: build/package tooling.
- Write ownership: this change removes only the named payload and metadata.
- Derived/cache data: wheel/sdist are rebuilt from source.
- Privacy/logging constraints: not applicable.

## Dependencies And Routes

- Domain/service dependencies: Python package build metadata.
- Database, GUI, and concurrency routes: not applicable.

## Prerequisite Handling

- Architecture feedback artifacts/status: none; not applicable.
- Already implemented prerequisites: package build workflow.
- Missing prerequisite architecture/specifications: none.
- Unimplemented prerequisite specifications: none.
- Progression handling: current item may proceed first.

## Application Integration

- App type: workflow.
- User/caller surface: clean installation of wheel/sdist.
- Invocation route: source tree -> package build -> dependency resolution/install.
- Wiring owner/module: `pyproject.toml` and package file selection.
- Observable result: no experiment files/import/dependency in installed artifacts.
- Integration validation: build, inspect, and clean-install both distributions.
- Incomplete status risk: source-only removal can leave metadata or artifacts contaminated.

## Reuse And Extraction Plan

- Existing code to reuse: current build workflow and artifact inspection.
- Current reuse readiness: reusable as-is.
- Extraction/wrapping/additions/new modules: none.
- One-off code justification: direct removal is the intended correction.

## Required DTOs / Functions / Components

- DTOs/models: project dependency metadata; no new DTO.
- Functions/methods and UI components: not applicable.

## Performance Contract

- Not performance-sensitive; normal package build bounds apply.

## Error And State Behavior

- Build fails if stale imports/references require removed files.
- Removal is recoverable from Git; no history rewrite occurs.

## Test Strategy

- Unit tests: source path/dependency absence.
- Integrated route tests: wheel/sdist inspection and clean installation.
- Service/DB and GUI tests: not applicable.
- Production-data rule: not applicable.

## Contract

Input is the current release tree. Output is a tree and built distribution with
no half-pipe files, import surface, or `build123d` runtime requirement. Git commit
history remains the recovery mechanism. No unresolved product decision remains:
the experiment is excluded from a3.

## Acceptance Criteria

- Repository and built wheel/sdist exclude both experimental modules.
- Clean installation does not install `build123d` through Impression metadata.
- Package imports and approved examples still pass.
- Release notes identify the extraction as payload correction, not a feature loss.

## Verification

[Paired test specification](../test-specifications/fix-06-remove-accidental-half-pipe-release-payload-v1_0.md)

## Readiness Checklist

- [x] Ancestors, full score, carryover, canonical status, and ledger are explicit.
- [x] Removal targets, destructive scope, recovery, outputs, and integration route are explicit.
- [x] No blocker, missing prerequisite, unresolved gap, or split coverage remains.
- [x] Tests cover source and built artifacts without production data.
