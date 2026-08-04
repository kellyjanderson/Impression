# Fix 01: Preview Watch And Forced Refresh

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Preview Watch And Forced Refresh ACD](../architecture/acd-preview-reload-coordination.md)
Architecture ancestor: [Preview Watch And Forced Refresh ACD](../architecture/acd-preview-reload-coordination.md)
Source artifact: [GitHub issue #242](https://github.com/kellyjanderson/Impression/issues/242)
Split provenance: none
Canonical status: Draft
Review Score: pending independent review

## Source Field Carryover

The issue's observed behavior, expected behavior, reproduction geometry, and a4 milestone are retained. This specification adds an implementation boundary and measurable acceptance contract without weakening the issue.

## Purpose

Restore the a3-missed live-preview contract: file changes schedule a rebuild promptly, and the visible `R` command always performs a cache-invalidating rebuild.

## Scope

The preview controller, filesystem event normalization, reload request coalescing, Python module invalidation, CLI scene factory, status reporting, and preview-focused tests.

## Split Coverage

This leaf owns the complete responsibility stated above. It does not claim adjacent leaves indexed by the release intake.

## Refinement History

Initial do-specs draft. Independent refinement has not yet occurred.

## Implementation Routing

Feature branch after canonical review; integrate through the future a4 working branch. Back-reference issue #242 and this specification in commits and PRs.

## Chosen Defaults / Parameters

Use one active build plus one latest replacement request. Preserve a forced-refresh bit while coalescing. Budget no more than 250 ms from a supported local filesystem event to build submission, excluding build/render time.

## Data Ownership

The preview controller owns reload intent and generation state. The scene factory consumes a generation token; the watcher only emits normalized change intent.

## Dependencies And Routes

Existing `PreviewController`, CLI module loader, watchdog adapter, Qt event handoff, and preview status surface. No new watcher framework.

## Prerequisite Handling

None beyond the current preview command. This is mandatory carried scope from v1.0.0a3.

## Application Integration

`impression preview` remains the entry point. Watcher and keyboard requests use the same bounded coordinator; `R` sets forced intent and increments the cache generation even when mtimes appear unchanged.

## Reuse And Extraction Plan

Extract only the shared records and validators named here. Do not create a parallel execution stack or copy planner logic between public and internal routes.

## Required DTOs / Functions / Components

`ReloadRequest(force: bool, changed_paths: frozenset[Path])`; a bounded request coordinator; generation-aware scene factory invalidation; status events for queued, rebuilding, succeeded, and failed.

## Performance Contract

Watcher processing must submit eligible work within 250 ms on supported local filesystems. At most one build and one replacement request may be retained; bursts must not grow memory or replay stale builds.

## Error And State Behavior

Build failures retain the last valid scene, show the error, and do not consume a newer queued request. Forced refresh remains observable and cannot be downgraded by a watcher event.

## Test Strategy

Unit-test coalescing, force-bit retention, generation invalidation, and failure recovery. Integration-test real file writes and `R` against a transitive imported model module. The paired contract is [Fix 01 Test](../test-specifications/fix-01-preview-watch-and-forced-refresh-v1_0.md).

## Acceptance Criteria

- [ ] A saved top-level or imported model change reaches build submission within 250 ms, excluding build/render time.
- [ ] A burst during an active build produces exactly one latest replacement build.
- [ ] Pressing `R` rebuilds from freshly invalidated modules even when mtimes and paths appear unchanged.
- [ ] A failed build leaves the last valid preview visible and a later request still runs.

## Readiness Checklist

- [x] Source issue and release ownership recorded.
- [x] Architecture transition and paired test contract identified.
- [x] Ownership, failure behavior, and measurable acceptance drafted.
- [ ] Independent review specs completed.
- [ ] Valid Review Score assigned and canonical status confirmed.
- [ ] Final progression responsibility coverage verified.

## Review Score Calculation

Template source: /Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md

Prior score: none

- Intent and scope: pending independent review
- Architecture and ownership: pending independent review
- Dependencies and integration: pending independent review
- Error, performance, and test contracts: pending independent review
- Acceptance and implementability: pending independent review

Total: pending independent review

