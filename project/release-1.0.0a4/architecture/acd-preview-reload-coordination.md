# Preview Reload Coordination Architectural Change Document

Date: 2026-08-04
Status: Drafting Specs
Canonical architecture targets:

- `project/release-0.1.0a/architecture/reference-review-preview-engine-sharing-architecture.md`

Related:

- Release / plan / issue: `project/release-1.0.0a4/README.md`; GitHub #242
- Parent ACD, if any: none

## Change Intent

Make preview reload one bounded coordination contract shared by watcher events,
manual refresh, control-file switching, and background scene building. Manual
refresh must force user-module invalidation; automatic reload may use tracked
file versions.

## Current Architecture

`src/impression/preview.py` owns an unbounded watcher queue, a single build
executor, one boolean queued flag, renderer-thread polling, and the `R` event.
`src/impression/cli.py` owns user-module caching, tracked transitive paths, and
mtime comparison. The reload message contains no reason or invalidation intent,
so `R` reaches the same cache decision as an ordinary rebuild.

## Target Architecture

- A bounded latest-request reload coordinator owns request coalescing.
- Reload requests distinguish automatic file change, forced manual refresh,
  control-file switch, signal refresh, and animation rebuild.
- The user-module loader owns a monotonic invalidation generation. Forced
  requests advance it before the background build reads the cache.
- One build is active at a time; at most one latest replacement request is
  retained. Replacement preserves forced intent.
- Watcher callbacks do no model work and deliver local filesystem changes
  within a 250 ms pre-build budget.
- The renderer thread alone applies successful current-generation datasets.
  Failed or stale builds leave the last good scene and camera intact.

## Non-Goals

- New preview controls, renderer replacement, process isolation, or model-build
  performance optimization.

## Canonical Document Impact

- Architecture docs to update on closure:
  - shared preview engine architecture - record reload request ownership and
    CLI cache invalidation boundary.
- Specs or plans affected:
  - `../specifications/fix-01-preview-watch-and-forced-refresh-v1_0.md`.

## Readiness Blocker Resolution

- Blocker being resolved: issue #242 did not define ownership between watcher,
  cache, build lane, and renderer lane.
- Source artifact: GitHub #242 and current CLI/preview implementation.
- Resolution provided by this ACD: typed reason, bounded latest-wins queue,
  loader-owned generation, and renderer-thread handoff.
- Follow-on artifact: Fix 01 draft spec and paired test spec.
- Resolution status: resolved.

## Compatibility And Migration Strategy

The `impression preview` command and existing `R` binding remain stable. The
change is internal coordination. Existing animation rebuilds do not force module
reload unless a manual or file-change request also arrives.

## Application Integration Contract

- App type: mixed GUI and console.
- User/caller surface: live preview window, `R` key, watched files, control file.
- Invocation route: event producer -> reload coordinator -> build executor ->
  renderer-thread scene application.
- Wiring owner/module: `src/impression/preview.py`, with cache invalidation in
  `src/impression/cli.py`.
- Observable result: changed model appears after build without watcher delay;
  errors remain in the console and the last good scene stays visible.
- Integration validation: real CLI preview smoke for save and `R`, plus actual
  filesystem-event timing and adjacent-request tests.

## Specification Sources

- One implementation leaf owns request records, bounded coalescing, forced
  cache generation, watcher timing, build replacement, and visible CLI wiring.
- Reuse the existing executor, timer handoff, tracked-module set, and scene
  application path; add coordination to existing modules rather than creating
  a parallel preview engine.

## Specification Conformance

- Parent specs created or affected:
  - [Fix 01 draft](../specifications/fix-01-preview-watch-and-forced-refresh-v1_0.md) - created from this ACD; independent review pending.
- Canonical child specs: none yet.
- Paired test specs:
  - [Fix 01 test draft](../test-specifications/fix-01-preview-watch-and-forced-refresh-v1_0.md) - verifies the mixed CLI/GUI route.

## Conformance Checklist

- [ ] Implementation conforms to the target architecture.
- [ ] Draft leaf is independently reviewed and canonicalized.
- [ ] Paired test spec points to the canonical leaf.
- [ ] Final progression points to the canonical leaf.
- [ ] Canonical preview architecture is reconciled after implementation.

## Closure Criteria

Close only after the save and `R` routes pass integrated validation, canonical
architecture records the conformed coordination boundary, and active release
artifacts no longer depend on this ACD.

## Closure Notes

- Canonical architecture updated: none yet.
- Archived or removed scaffolding: none.
- Follow-up ACDs: none.

## Change History

- 2026-08-04 - Linked the full-template Fix 01 and paired test drafts. Reason: complete the `do specs` creation handoff.
- 2026-08-04 - Initial draft. Reason: plan issue #242 for `v1.0.0a4`.
