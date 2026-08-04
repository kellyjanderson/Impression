# Fix 08 Test: Loft Surface Difference Cut Execution

Date: 2026-08-04
Status: Proposed
Feature specification: [Fix 08: Loft Surface Difference Cut Execution](../specifications/fix-08-loft-surface-difference-cut-execution-v1_0.md)
Canonical status: Draft

## Overview

This contract proves the user-visible behavior, internal invariants, failure behavior, and release regression boundary for Fix 08. It becomes binding only when the paired feature spec is independently reviewed and canonicalized.

## Application Integration Under Test

Surface-kernel/test-model proof: execute the real model route with no mesh or grouped-body fallback.

## Manual Smoke

Run the diagonal audio-cube USB, acoustic, and snap-pocket cuts without workaround geometry; inspect new boundaries and closed results. Confirm the last valid result remains usable after any deliberate failure.

## Automated Smoke Tests

Kernel tests cover intersection evidence, trim fragmentation, classification, cutter caps, branch recomposition, and invalid cases.

## Automated Acceptance Tests

Each qualifying public difference changes geometry with witnesses and returns validated closed shells within the fixture timeout. Include deterministic positive, negative, and regression assertions and require actionable diagnostic content for refusals.

## App-Type Proof

Surface-kernel/test-model proof: execute the real model route with no mesh or grouped-body fallback.

## Fixtures And Data

USB/acoustic/snap cutters; branched loft; tangential and no-intersection controls; invalid recomposition. Fixtures must be deterministic, project-local, and small enough for normal CI. Preserve the exact issue reproduction where it is the acceptance fixture.

## Acceptance

- [ ] Manual smoke succeeds on a supported macOS development environment.
- [ ] Automated smoke covers the primary state transition and failure recovery.
- [ ] Automated acceptance proves every criterion in the paired implementation specification.
- [ ] The real application/public route is exercised; helper-only proof is rejected.
- [ ] The focused suite and full configured suite pass without workaround geometry or mesh fallback.
- [ ] Test names and failure output identify the violated contract and relevant fixture.

