# Current and Planned Project Structure

Date: 2026-07-23

Status: Proposed structural migration

## Summary

The current project structure makes a historical `release-0.1.0a` workspace
look like the active source of truth even though the package is now
`1.0.0a0`. The planned structure promotes current architecture, specs, plans,
and evidence to stable top-level project folders. Releases become history and
snapshots rather than containers that hide current truth.

## Current Structure

```text
project/
  README.md
  project-dna.md
  future-features/                  7 Markdown files
  meetings/
  notes/                            current review notes; some untracked
  reference-images/                 second artifact location
  reference-stl/                    second artifact location
  release-0.1.0a/                  active-looking historical workspace
    README.md
    adhoc/                          39 Markdown files
    architecture/                   85 Markdown files
    codeimprovement/
    coverage/
    planning/                       25 Markdown files
    prs/
    reference-images/
    reference-stl/
    spec-refinement-history/         5 Markdown files
    specifications/                699 Markdown files
    test-specifications/           215 Markdown files
  releases/
    release-0.0.3a1/
    release-0.0.3a2/
  research/
```

The counts below describe the pre-plan snapshot immediately before this
four-file `project/adhoc/` package was added.

Structural problems:

- canonical and historical material are mixed under a stale release name;
- architecture, ACDs, reviews, plans, inventories, and trackers share one
  directory;
- active leaves, parents, programs, reports, proposed work, and retired specs
  share one directory;
- current and release-nested reference artifact trees overlap;
- release path references occur hundreds of times, making moves expensive and
  discouraging cleanup;
- the `project/` tree contains 1,171 Markdown files and 1,355 total files;
- `project/` occupies about 575 MB, including about 518 MB of release-nested
  reference STL data;
- exact duplicate artifacts exist across the two reference trees, while other
  same-named artifacts differ and require provenance review;
- some individual tracked STL files are tens to hundreds of megabytes.

## Planned Structure

```text
project/
  README.md                         one navigation and lifecycle authority
  project-dna.md                    durable product values
  architecture/                     current conformed architecture only
    README.md
    surface-body-model-architecture.md
    tessellation-export-and-mesh-boundary-architecture.md
    surface-family-capability-architecture.md
    surface-csg-architecture.md
    loft-system-architecture.md
    inference-and-curve-fitting-architecture.md
    impress-persistence-architecture.md
    preview-and-reference-review-architecture.md
    testing-and-reference-artifact-architecture.md
    compatibility-and-standalone-mesh-tooling-architecture.md
  acds/                             active transitions only
    README.md
  specifications/                   final current implementation leaves
    README.md
  test-specifications/              paired current test leaves
    README.md
  planning/                         current progression and release intent
    README.md
  codeimprovement/                  accepted code-quality discovery records
    README.md
  future-features/                  explicitly non-committed directions
    README.md
  research/                         durable research
  evidence/                         authoritative generated/reference evidence
    README.md
    reference-images/
    reference-stl/
    coverage/
  records/                          meetings, PR notes, and durable reviews
    meetings/
    prs/
    reviews/
  adhoc/                            temporary cross-release analysis
  archive/
    README.md
    removed-in-vX.Y.Z/
      architecture/
      acds/
      specifications/
      test-specifications/
      planning/
      indexes/
  releases/
    README.md
    v0.0.3a1/
      release-summary.md
    v0.0.3a2/
      release-summary.md
    v1.0.0a0/
      release-summary.md
```

The exact canonical architecture file count may change during consolidation.
The invariant is one current home per architectural fact, not a fixed number of
files.

## Key Structural Decisions

### Stable active paths

Architecture and specifications should not move every release. Stable paths
reduce link churn and make “current truth” obvious.

### Releases as history

Release folders should contain release summaries, evidence manifests, and links
to tagged repository state. They should not duplicate the full active
architecture/spec tree unless there is a regulatory or offline-snapshot reason.
Git tags already preserve the exact historical tree.

### Active ACDs separated from canonical architecture

Putting active ACDs in `project/acds/` makes the distinction between true
architecture and intended transition visible. Closed ACDs move into the
versioned archive after canonical reconciliation.

### Active specs are leaves

`project/specifications/` and `project/test-specifications/` contain current
final leaves. Parent/program/superseded artifacts live in the archive after
coverage proof. Completion state belongs in metadata and the conformance
ledger, not in path nesting by release.

### One evidence root

Reference images, STL files, coverage evidence, and generated matrices need one
authoritative root and a manifest. “Dirty” and “gold” describe evidence state;
they should not be duplicated by release and top-level location.

## Path Migration Map

| Current | Planned |
| --- | --- |
| `project/release-0.1.0a/architecture/*.md` | classify into `project/architecture/`, `project/acds/`, `project/records/reviews/`, `project/planning/`, or versioned archive |
| `project/release-0.1.0a/specifications/*.md` | final leaves to `project/specifications/`; parents/deprecated/history to archive |
| `project/release-0.1.0a/test-specifications/*.md` | active paired leaves to `project/test-specifications/`; obsolete/parent/history to archive |
| `project/release-0.1.0a/planning/` | current plans to `project/planning/`; completed/stale plans to archive |
| `project/release-0.1.0a/codeimprovement/` | `project/codeimprovement/` |
| `project/release-0.1.0a/prs/` | `project/records/prs/` |
| `project/meetings/` | `project/records/meetings/` |
| `project/notes/` | accepted durable reviews to `project/records/reviews/`; temporary work remains outside cleanup until accepted |
| both reference artifact trees | deduplicated `project/evidence/` with manifest |
| `project/release-0.1.0a/adhoc/` | current cross-release work to `project/adhoc/`; historical bookkeeping to archive |
| `project/release-0.1.0a/spec-refinement-history/` | versioned archive or a compact audit index |
| `project/releases/release-*` | normalized release summaries under `project/releases/v*/` |

## Migration Order

1. Create classification ledger and link map.
2. Establish `project/README.md`, archive rules, and stable active paths.
3. Consolidate architecture without moving specs yet.
4. Reconcile/close ACDs.
5. Verify and canonicalize specs and paired test specs by domain.
6. Move planning, records, and code-improvement artifacts.
7. Consolidate evidence trees.
8. Rewrite links and remove the empty old release workspace.
9. Add release summary for `v1.0.0a0` and the cleanup release.

## Repository Cleanup Tasks to Consider

### High value

- Add an automated Markdown link checker and run it in CI.
- Generate architecture/spec indexes from metadata to prevent manual index
  drift.
- Add a document-lifecycle validator for required fields such as `status`,
  `kind`, `domain`, `superseded_by`, and `removed_in`.
- Generate surface-family, intersection, and CSG capability matrices from code.
- Add a spec-to-code/test/route conformance ledger.
- Deduplicate the current and release-nested reference artifact roots.
- Decide whether large reference STL files belong in Git LFS, release assets,
  or reproducible generated evidence.

### Story and navigation

- Add a “How Impression Fits Together” document.
- Add domain owners and canonical-document links to the architecture index.
- Keep future features out of canonical architecture until adopted.
- Replace broad checkbox plans with links to final active leaves.
- Make the surface-body transition a short, explicit historical narrative:
  mesh-first origin, migration, current surface-body authority, retained mesh
  boundaries.

### Hygiene

- Keep generated coverage HTML and caches out of Git.
- Review stale root-level experimental folders such as `wow/`, `examples/`,
  and `design-assets/` for intended ownership.
- Review duplicate documentation packaging under `docs/` and
  `impression-docs/`.
- Add artifact manifests with generator command, source revision, state
  (`dirty`/`gold`), and expected retention.
- Keep cleanup commits separate from large binary storage changes.

### Governance

- Require an ACD for post-architecture structural changes.
- Require ACD closure in the same change that reconciles canonical
  architecture.
- Require 100% child coverage before parent-spec archival.
- Require the release containing an archival move to populate `removed_in`.
- Require route-level evidence before a user-facing feature is marked complete.

## Success Test

A new contributor should be able to answer these questions from
`project/README.md` in under five minutes:

1. What is Impression’s canonical geometry model?
2. Where and why do meshes still exist?
3. Which architecture documents are current?
4. Which features are implemented, partial, or unimplemented?
5. Which specs are active implementation leaves?
6. Which architectural transitions are still open?
7. Where is historical material and when was it removed?
8. What is the next verified work?
