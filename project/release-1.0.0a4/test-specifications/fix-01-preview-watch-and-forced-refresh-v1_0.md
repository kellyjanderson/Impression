# Fix 01 Test: Preview Watch And Forced Refresh

Date: 2026-08-04
Status: Proposed
Feature spec: [Fix 01: Preview Watch And Forced Refresh](../specifications/fix-01-preview-watch-and-forced-refresh-v1_0.md)
Feature spec canonical status: Draft
Architecture ancestor: [Active ACD](../architecture/acd-preview-reload-coordination.md)

## Overview

This paired contract verifies the complete draft feature boundary for Fix 01. It remains a draft until independent `review specs` canonicalizes the feature leaf.

## Application Integration Under Test

- App type: mixed
- User/caller surface: live `impression preview` command and preview window
- Invocation route: saved top-level/transitive file or `R` key -> reload coordinator -> loader/build lane -> UI-thread scene application
- Wiring owner/module: `src/impression/preview.py` and `src/impression/cli.py`
- Observable result: fresh visible scene after real build time, preserved camera/last-good scene, and visible failure status
- Integration validation: real command/GUI smoke plus actual filesystem timing and module-cache assertions

## Manual Smoke

- Launch `impression preview` on an entry model importing a local helper.
- Save the entry model and helper separately; confirm each schedules promptly and displays the new value.
- Rewrite the helper while restoring its prior mtime, press `R`, and confirm the new value appears.
- Trigger a build error, confirm the last good scene/camera remains, repair it, and confirm recovery.

## Automated Smoke Tests

- A temporary real file event reaches captured build submission within 250 ms, excluding build/render.
- The actual `R` event reaches forced cache invalidation and a visible scene apply.

## Automated Acceptance Tests

- Unit/helper behavior:
  - request coalescing, force-bit retention, generation invalidation, dependency rediscovery, stale completion, failure recovery, camera preservation
- Integrated route behavior:
  - top-level save, transitive save, mtime-neutral forced refresh, burst saves, error/recovery, and orderly shutdown through the CLI/preview route
- Failure and stale-result behavior, if applicable:
  - older completions cannot overwrite newer state; errors retain last-good scene; forced intent survives adjacent watcher events

## App-Type Proof

- GUI proof: visible preview, `R` event, UI-thread apply, stale/failure behavior, and camera state
- Console proof: actual command startup, watched paths, status/error output, and shutdown
- API/service proof:
  - not applicable
- Mixed-surface proof: separate command/watcher submission and visible renderer assertions
- Library-only proof: not applicable

## Fixtures And Data

- temporary entry/helper Python modules with observable scene values
- burst edit sequence, mtime-restored helper edit, deliberate syntax/build failure
- Production-data rule: tests use project-local deterministic fixtures and temporary directories; no user production data is required.

## Acceptance

- [ ] Feature spec is canonical, or this test spec remains explicitly temporary while review/split coverage is incomplete.
- [ ] Route-level proof exists for the declared app type.
- [ ] Helper-only tests cannot satisfy this contract.
- [ ] Every observable result and feature acceptance criterion is asserted through the intended route.
- [ ] Failure, stale-result, refusal, or no-cut behavior is covered where applicable.
- [ ] Focused and full configured suites pass without mesh modeling fallback or test-model workaround geometry.

