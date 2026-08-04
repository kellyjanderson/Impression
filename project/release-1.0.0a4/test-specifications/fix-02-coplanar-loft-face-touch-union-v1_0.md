# Fix 02 Test: Coplanar Loft Face-Touch Union

Date: 2026-08-04
Status: Proposed
Feature specification: [Fix 02: Coplanar Loft Face-Touch Union](../specifications/fix-02-coplanar-loft-face-touch-union-v1_0.md)
Canonical status: Draft

## Overview

This contract proves the user-visible behavior, internal invariants, failure behavior, and release regression boundary for Fix 02. It becomes binding only when the paired feature spec is independently reviewed and canonicalized.

## Application Integration Under Test

Surface-kernel proof: use real `SurfaceBody` patches and the public boolean route; mesh-derived assertions are insufficient.

## Manual Smoke

Build the two reproduced face-touching loft bodies and inspect the union shell, shared face removal, and seams. Confirm the last valid result remains usable after any deliberate failure.

## Automated Smoke Tests

Unit tests cover coincident domain/orientation classification, interior-pair filtering, and shell assembly negatives.

## Automated Acceptance Tests

Public surface union must produce one closed shell for the fixture and refuse near-coplanar/partial-domain controls. Include deterministic positive, negative, and regression assertions and require actionable diagnostic content for refusals.

## App-Type Proof

Surface-kernel proof: use real `SurfaceBody` patches and the public boolean route; mesh-derived assertions are insufficient.

## Fixtures And Data

Exact face-touching loft pair; reversed orientation; near-coplanar gap; partial overlap; open-seam negative. Fixtures must be deterministic, project-local, and small enough for normal CI. Preserve the exact issue reproduction where it is the acceptance fixture.

## Acceptance

- [ ] Manual smoke succeeds on a supported macOS development environment.
- [ ] Automated smoke covers the primary state transition and failure recovery.
- [ ] Automated acceptance proves every criterion in the paired implementation specification.
- [ ] The real application/public route is exercised; helper-only proof is rejected.
- [ ] The focused suite and full configured suite pass without workaround geometry or mesh fallback.
- [ ] Test names and failure output identify the violated contract and relevant fixture.

