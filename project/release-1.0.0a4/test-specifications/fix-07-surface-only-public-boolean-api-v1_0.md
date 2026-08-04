# Fix 07 Test: Surface-Only Public Boolean API

Date: 2026-08-04
Status: Proposed
Feature specification: [Fix 07: Surface-Only Public Boolean API](../specifications/fix-07-surface-only-public-boolean-api-v1_0.md)
Canonical status: Draft

## Overview

This contract proves the user-visible behavior, internal invariants, failure behavior, and release regression boundary for Fix 07. It becomes binding only when the paired feature spec is independently reviewed and canonicalized.

## Application Integration Under Test

Packaging/API proof: verify source checkout and clean wheel plus rendered/reference documentation inventory.

## Manual Smoke

From a clean installed wheel, call public booleans with surfaces, meshes, and mixed operands; follow the mesh migration message. Confirm the last valid result remains usable after any deliberate failure.

## Automated Smoke Tests

Signature/export tests, runtime operand matrix, docs scan, and installed-package import tests cover the contract.

## Automated Acceptance Tests

Surface calls work; mesh/mixed calls fail before kernel work; separately named mesh utilities remain usable if retained. Include deterministic positive, negative, and regression assertions and require actionable diagnostic content for refusals.

## App-Type Proof

Packaging/API proof: verify source checkout and clean wheel plus rendered/reference documentation inventory.

## Fixtures And Data

SurfaceBody operands, mesh operands, mixed pairs, public imports, docs snippets, and migration examples. Fixtures must be deterministic, project-local, and small enough for normal CI. Preserve the exact issue reproduction where it is the acceptance fixture.

## Acceptance

- [ ] Manual smoke succeeds on a supported macOS development environment.
- [ ] Automated smoke covers the primary state transition and failure recovery.
- [ ] Automated acceptance proves every criterion in the paired implementation specification.
- [ ] The real application/public route is exercised; helper-only proof is rejected.
- [ ] The focused suite and full configured suite pass without workaround geometry or mesh fallback.
- [ ] Test names and failure output identify the violated contract and relevant fixture.

