# Fix 01 Test: Preview Watch And Forced Refresh

Date: 2026-08-04
Status: Proposed
Feature specification: [Fix 01: Preview Watch And Forced Refresh](../specifications/fix-01-preview-watch-and-forced-refresh-v1_0.md)
Canonical status: Draft

## Overview

This contract proves the user-visible behavior, internal invariants, failure behavior, and release regression boundary for Fix 01. It becomes binding only when the paired feature spec is independently reviewed and canonicalized.

## Application Integration Under Test

Desktop/CLI integration proof: drive the actual watcher adapter, controller, module loader, and visible preview status rather than invoking the scene factory alone.

## Manual Smoke

Run `impression preview` on a model importing a helper module; edit each file and press `R` after an mtime-neutral rewrite. Confirm the last valid result remains usable after any deliberate failure.

## Automated Smoke Tests

Controller tests cover one-active/one-latest coalescing, force-bit retention, failure recovery, and cache-generation changes.

## Automated Acceptance Tests

A real temporary filesystem write must reach captured build submission within 250 ms; `R` must load changed transitive module content. Include deterministic positive, negative, and regression assertions and require actionable diagnostic content for refusals.

## App-Type Proof

Desktop/CLI integration proof: drive the actual watcher adapter, controller, module loader, and visible preview status rather than invoking the scene factory alone.

## Fixtures And Data

Top-level and transitive model modules; burst edits; mtime-neutral content replacement; one intentionally failing revision. Fixtures must be deterministic, project-local, and small enough for normal CI. Preserve the exact issue reproduction where it is the acceptance fixture.

## Acceptance

- [ ] Manual smoke succeeds on a supported macOS development environment.
- [ ] Automated smoke covers the primary state transition and failure recovery.
- [ ] Automated acceptance proves every criterion in the paired implementation specification.
- [ ] The real application/public route is exercised; helper-only proof is rejected.
- [ ] The focused suite and full configured suite pass without workaround geometry or mesh fallback.
- [ ] Test names and failure output identify the violated contract and relevant fixture.

