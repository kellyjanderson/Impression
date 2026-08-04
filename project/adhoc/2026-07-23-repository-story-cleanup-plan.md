# Repository Story Cleanup Plan

Date: 2026-07-23

Status: Proposed cleanup plan

Scope: `project/` architecture, ACDs, specifications, test specifications,
planning records, release nesting, and closely related repository hygiene

Related:

- [Defined but Unimplemented Functionality](2026-07-23-defined-but-unimplemented-functionality.md)
- [Current and Planned Project Structure](2026-07-23-current-and-planned-project-structure.md)
- [Project DNA](../project-dna.md)

## Outcome

Make the project tell one clear story:

1. Impression is a surface-body modeling system.
2. Meshes are explicit consumer-boundary products, compatibility values, or
   standalone analysis/repair inputs; they are not the canonical modeled
   representation.
3. A small set of canonical architecture documents describes what is true.
4. Active specifications are final implementation leaves only.
5. Implemented behavior, unimplemented behavior, and historical decisions are
   visibly distinct.
6. Releases record history without hiding the active architecture and spec
   truth under a version-nested workspace.

This is a classification-and-conformance migration. It is not a bulk move based
on filenames or existing checkboxes.

## Baseline

The pre-plan snapshot, taken immediately before adding this four-file cleanup
package, had:

| Surface | Current count | Main issue |
| --- | ---: | --- |
| `project/release-0.1.0a/architecture/` | 85 Markdown files | 72 non-ACD documents, 12 ACDs, and one index share one flat directory |
| `project/release-0.1.0a/specifications/` | 699 Markdown files | Parents, leaves, programs, proposed work, retired work, and reports share one flat directory |
| `project/release-0.1.0a/test-specifications/` | 215 Markdown files | Active and historical verification contracts are not visibly separated |
| `project/release-0.1.0a/planning/` | 25 Markdown files | Multiple planning models disagree about completion |
| `project/release-0.1.0a/adhoc/` | 39 Markdown files | Temporary process artifacts remain inside an obsolete release-nested workspace |
| `project/future-features/` | 7 Markdown files | At least one “future” feature overlaps implemented and actively architected inference work |
| all of `project/` | 1,171 Markdown files; 1,355 files total | The active story is difficult to distinguish from history |

The package version is `1.0.0a0`, while the active project workspace is still
named `release-0.1.0a`. The main progression marks 594 items complete, but 51
specifications explicitly say `Status: Proposed`, 10 are explicitly
superseded/retired, and all 12 ACDs remain `Proposed` or `Manifesting`. Existing
completion checkboxes are therefore evidence inputs, not final truth.

`Manifesting` is not one of the current ACD lifecycle states
(`Proposed`, `Accepted`, `Drafting Specs`, `In Progress`,
`Conformance Review`, and `Closed`), so ACD status normalization is itself part
of the cleanup.

## Non-Negotiable Rules

### Preserve user work

- Do not alter or absorb the existing untracked `project/notes/` review files
  as part of this cleanup unless they are separately accepted into scope.
- Move and rewrite in small reviewable commits.
- Never delete historical material before it has a classified archive target
  and a link audit.

### Keep canonical architecture truthful

- Canonical architecture describes the conformed implementation.
- An ACD remains active until code, tests, routes, docs, and spec state conform.
- Merge an ACD into canonical architecture only during explicit conformance
  review.
- Archive the ACD whole after its accepted content is merged and active
  documents no longer depend on it.

### Archive parent specs only after coverage proof

- Identify parent, umbrella, program, split-parent, and superseded specs.
- Build a parent-to-child responsibility matrix.
- Require 100% child coverage with no `Partial`, `Missing`, or `Parent-only`
  responsibilities.
- Promote the children to canonical leaves.
- Only then archive the parent and update indexes and progression links.

### Treat the mesh-to-surface transition semantically

Classify every mesh-related document into one of these lanes:

1. `obsolete modeled-mesh authority` — archive;
2. `historical migration or rollback evidence` — archive;
3. `surface-to-mesh tessellation/export boundary` — keep if current;
4. `explicit compatibility API` — keep if current and intentionally supported;
5. `standalone mesh analysis/repair tooling` — keep if current;
6. `debug-only or test-fixture mesh route` — keep outside canonical product
   architecture, or archive if no longer used.

The word `mesh` is not, by itself, a deprecation marker.

## Artifact Classification Ledger

Before moving documents, create a machine-readable or Markdown ledger with one
row per architecture document, ACD, implementation spec, and test spec.

Required fields:

| Field | Purpose |
| --- | --- |
| Current path | Stable source identity during migration |
| Artifact kind | Architecture, ACD, feature spec, test spec, report, plan, inventory, or research |
| Domain | Surface core, tessellation/export, CSG, loft, inference, persistence, preview/reference review, testing, mesh tooling, or process |
| Lifecycle state | Canonical, active transition, proposed, implemented, superseded, deprecated, retired, historical, or unknown |
| Parent/leaf state | Leaf, parent, umbrella/program, split parent, or not applicable |
| Surface-transition lane | One of the six mesh/surface lanes above |
| Implementation owner | Code module, route, or explicit `none found` |
| Verification evidence | Tests, reference artifacts, smoke route, or explicit `none found` |
| Canonical destination | New path or archive path |
| Replacement/successor | Canonical document or child leaves |
| Removed in | Exact release containing removal, or an explicit unknown value |
| Confidence | Confirmed, likely, or needs review |

Do not use `unknown` as a stopping state. Unknown rows form the next review
batch.

## Execution Phases

### Phase 0 — Freeze the truth surface

- [ ] Choose the release that will contain the cleanup.
- [ ] Record that exact release as `cleanup_release`.
- [ ] Snapshot `git status`, current tag, package version, document counts, and
      link counts.
- [ ] Preserve existing untracked notes and unrelated user work.
- [ ] Create the classification ledger and archive metadata template.
- [ ] Add a link checker that understands both active and archived paths.

Exit: every in-scope file has a ledger row and no destructive move has started.

### Phase 1 — Define the new project story

- [ ] Replace the release-nested active-document policy with the planned
      top-level project structure.
- [ ] Write `project/README.md` as the single navigation and lifecycle
      authority.
- [ ] Define the canonical architecture document set and domain ownership.
- [ ] Define active-spec, completed-spec, and archive boundaries.
- [ ] Define how releases snapshot project state without becoming the active
      location of architecture and specs.
- [ ] Define archive metadata and `removed_in` rules.

Exit: a reviewer can locate current architecture, current specs, known gaps,
history, and release records from `project/README.md`.

### Phase 2 — Architecture consolidation

Review the 72 non-ACD architecture-area documents by domain.

- [ ] Separate canonical architecture from trackers, plans, inventories,
      product definitions, reviews, and evidence reports.
- [ ] Consolidate overlapping surface-core documents.
- [ ] Consolidate patch-family documents into one surface-family architecture
      with an explicit capability matrix.
- [ ] Consolidate CSG documents into one architecture plus narrowly separated
      solver/capability appendices only where needed.
- [ ] Consolidate loft documents into one system architecture plus a topology
      and diagnostics appendix if the size warrants it.
- [ ] Consolidate inference/curve-fitting documents into one architecture.
- [ ] Consolidate Reference Review and preview documents into one tool
      architecture; move UI product definitions, delta reviews, and remediation
      plans to their correct artifact types.
- [ ] Consolidate file-format/persistence material.
- [ ] Consolidate testing/reference-artifact architecture.
- [ ] Give every canonical architecture document a change history and links to
      source ACDs and archived predecessors.

Suggested canonical set:

1. `surface-body-model-architecture.md`
2. `tessellation-export-and-mesh-boundary-architecture.md`
3. `surface-family-capability-architecture.md`
4. `surface-csg-architecture.md`
5. `loft-system-architecture.md`
6. `inference-and-curve-fitting-architecture.md`
7. `impress-persistence-architecture.md`
8. `preview-and-reference-review-architecture.md`
9. `testing-and-reference-artifact-architecture.md`
10. `compatibility-and-standalone-mesh-tooling-architecture.md`

This is a target set, not a forced ten-file quota. Split only where a document
has an independently meaningful ownership and change boundary.

Exit: each architectural fact has one canonical home and all replaced
architecture documents have archive destinations.

### Phase 3 — ACD conformance and closure

Process each of the 12 ACDs independently:

- [ ] Normalize legacy `Manifesting` statuses to the current ACD lifecycle.
- [ ] Re-read target architecture and closure criteria.
- [ ] Map the ACD to source modules, tests, public route, docs, specs, and
      reference evidence.
- [ ] Classify it as `not implemented`, `implemented in isolation`, `wired`,
      `integrated`, or `conformant`.
- [ ] If not conformant, keep the ACD active and list the missing work in the
      unimplemented-functionality report.
- [ ] If conformant, merge accepted architectural truth into the appropriate
      canonical document.
- [ ] Remove active spec/plan dependency on the ACD.
- [ ] Mark it closed and archive it under the cleanup release.

The current source already contains many names corresponding to the loft CSG
ACDs, so ACD status text alone must not be used to call those features
unimplemented. They are high-priority conformance-review candidates.

Exit: no closed ACD remains live authority, and no active ACD has been silently
folded into canonical architecture.

### Phase 4 — Specification verification and canonicalization

Run by domain, not as a 914-file all-at-once rewrite.

- [ ] Identify explicit and implicit parents/programs.
- [ ] Recount leaf status from current content; do not trust old IWU or review
      scores.
- [ ] Verify 100% child coverage before archiving any parent.
- [ ] For each intended leaf, locate implementation, focused tests, integrated
      route proof when applicable, and durable documentation.
- [ ] Use these status values consistently:
      `Designed`, `Implemented in isolation`, `Wired`, `Integrated`,
      `User-accessible`, and `Complete`.
- [ ] Mark specs with missing implementation evidence as active work and add
      them to the unimplemented-functionality report.
- [ ] Archive superseded and deprecated specs with replacement links.
- [ ] Archive obsolete mesh-first specs only after surface-transition
      classification.
- [ ] Keep valid tessellation, export, compatibility, and standalone mesh-tool
      specs active.
- [ ] Apply the same decision to paired test specs.
- [ ] Rebuild progression from final active leaves only.

Suggested audit order:

1. surface-body core and migration;
2. tessellation/export and mesh boundaries;
3. surface families and persistence;
4. surface CSG;
5. loft;
6. inference and curve fitting;
7. Reference Review/preview;
8. testing/reference artifacts;
9. remaining tools and compatibility surfaces.

Exit: active specifications contain final leaves only, and every active leaf
has an honest implementation status.

### Phase 5 — Flatten and move

- [ ] Create the planned top-level project folders.
- [ ] Move canonical architecture and active leaf specs with `git mv`.
- [ ] Move closed/superseded material into the versioned archive.
- [ ] Move non-architecture documents out of `architecture/`.
- [ ] Move non-spec reports/inventories out of `specifications/`.
- [ ] Preserve completed release summaries under `project/releases/`, not full
      duplicate active truth trees.
- [ ] Rewrite links by ledger mapping.
- [ ] Run the link checker and search for the old
      `project/release-0.1.0a/` prefix.
- [ ] Remove the empty old release workspace only after all rows are resolved.

Exit: no active document is hidden under the old release tree and no active
link depends on it.

### Phase 6 — Story and hygiene pass

- [ ] Rewrite root and project documentation maps.
- [ ] Add a one-page “How Impression Fits Together” narrative from public
      modeling API through `SurfaceBody`, operations, tessellation, preview,
      persistence, and STL export.
- [ ] Reconcile planning claims with verified spec status.
- [ ] Remove or relocate obsolete trackers and process scaffolding.
- [ ] Decide one authoritative location for dirty/gold reference artifacts.
- [ ] Evaluate Git LFS or generated-artifact storage for very large STL files.
- [ ] Remove exact duplicate reference artifacts only after path consumers are
      migrated.
- [ ] Validate tests, docs links, package contents, and release metadata.

Exit: the repository has one current narrative and one history path.

## Archive Metadata

Every archived document should begin with consistent metadata, either YAML
front matter or a standard Markdown block:

```yaml
status: archived
removed_in: vNEXT
archive_reason: superseded-parent
superseded_by:
  - project/specifications/example-leaf-v1_0.md
original_path: project/release-0.1.0a/specifications/example-parent-v1_0.md
```

Rules:

- `removed_in` is the release containing the archival/removal commit.
- For historical removals proven by Git history, use the first containing tag.
- If the historical release cannot be proven, write
  `removed_in: unknown-pre-v1.0.0a0`; do not invent a version.
- Replace `vNEXT` before merging the cleanup release.
- Preserve the original path and successor link.

Recommended archive shape:

```text
project/archive/
  removed-in-vX.Y.Z/
    architecture/
    acds/
    specifications/
    test-specifications/
    planning/
    indexes/
```

## Verification Gates

The cleanup is not complete until:

- every architecture/ACD/spec/test-spec row has a terminal classification;
- every archived parent spec has a 100% child coverage matrix;
- every closed ACD has conformance evidence and canonical merge history;
- every active spec has an implementation and verification status;
- every mesh-related spec has a surface-transition lane;
- all active Markdown links resolve;
- no active index points into the old release workspace;
- the package version, project lifecycle docs, and release story agree;
- focused and full test suites selected for the affected domains pass;
- `git diff --check` passes;
- unrelated untracked user notes remain untouched.

## Recommended Commit Sequence

1. Add inventory, ledger, and link-check tooling.
2. Add new project lifecycle and archive policy.
3. Consolidate one architecture domain at a time.
4. Close and archive ACDs one at a time or in tightly related groups.
5. Canonicalize specs one domain at a time.
6. Flatten paths and rewrite links.
7. Deduplicate reference artifacts and large-file policy separately.
8. Finish with navigation/story documentation.

Each commit should remain reviewable and reversible. Do not combine large STL
storage changes with architecture/spec semantic changes.

## Decisions Needed Before Execution

- Which release will contain the cleanup and therefore supply `removed_in`?
- Should completed specs remain in an active `completed/` index, or should
  active specs include both complete and incomplete final leaves?
- Should release archives preserve full frozen doc trees or only release
  summaries plus versioned removed artifacts?
- Should large generated STL evidence move to Git LFS, release assets, or a
  reproducible generation workflow?

The plan can start with inventory and classification before these decisions,
but path moves and final archive metadata should wait for them.
