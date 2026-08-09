# Preview Reload Coordination Architectural Change Document

Date: 2026-08-04
Status: Complete
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

- Fix 01A owns request records, bounded coalescing, watcher timing, and build replacement.
- Fix 01B owns forced cache generation and transitive user-module invalidation.
- Fix 01C1 owns save and `R` event wiring into those two boundaries.
- Fix 01C2A owns current-generation renderer-thread scene application.
- Fix 01C2B owns camera, error, recovery, and last-good-scene state.
- Reuse the existing executor, timer handoff, tracked-module set, and scene
  application path; add coordination to existing modules rather than creating
  a parallel preview engine.

## Specification Conformance

- Archived split parents:
  - [Fix 01](../specifications/fix-01-preview-watch-and-forced-refresh-v1_0.md)
  - [Fix 01C](../specifications/fix-01c-preview-refresh-route-integration-v1_0.md)
  - [Fix 01C2](../specifications/fix-01c2-preview-scene-state-application-v1_0.md)
- Canonical child specs:
  - [Fix 01A](../specifications/fix-01a-preview-watch-request-coordination-v1_0.md)
  - [Fix 01B](../specifications/fix-01b-preview-module-cache-invalidation-v1_0.md)
  - [Fix 01C1](../specifications/fix-01c1-preview-refresh-input-wiring-v1_0.md)
  - [Fix 01C2A](../specifications/fix-01c2a-preview-current-generation-scene-apply-v1_0.md)
  - [Fix 01C2B](../specifications/fix-01c2b-preview-last-good-camera-error-state-v1_0.md)
- Paired canonical test specs use the matching filenames under
  [test specifications](../test-specifications/README.md).
- Progression: [v1.0.0a4 corrective release progression](../planning/progression.md).

## Conformance Checklist

- [x] Implementation conforms to the target architecture.
- [x] Final leaves are independently reviewed and canonicalized.
- [x] Paired test specs point to the canonical leaves.
- [x] Final progression points to the canonical leaves.
- [x] Canonical preview architecture is reconciled after implementation.

## Closure Criteria

Close only after the save and `R` routes pass integrated validation, canonical
architecture records the conformed coordination boundary, and active release
artifacts no longer depend on this ACD.

## Closure Notes

- Canonical architecture updated:
  `project/release-0.1.0a/architecture/reference-review-preview-engine-sharing-architecture.md`.
- Archived or removed scaffolding: none.
- Follow-up ACDs: none.

## Change History

- 2026-08-04 - Closed after implementation and route validation. Reason: the
  bounded coordinator, forced transitive invalidation, current-generation scene
  admission, camera/error preservation, and canonical architecture reconciliation
  are complete.
- 2026-08-04 - Linked the final dependency-ordered progression. Reason: make the canonical preview leaves executable without routing archived parents.
- 2026-08-04 - Recorded the five canonical preview leaves and archived split parents after fixed-point review.
- 2026-08-04 - Linked the full-template Fix 01 and paired test drafts. Reason: complete the `do specs` creation handoff.
- 2026-08-04 - Initial draft. Reason: plan issue #242 for `v1.0.0a4`.
