# Surface Spec 46 Test: Public/Internal API Transition Boundary

## Overview

This test specification defines verification for the boundary between internal
surface-native adoption work and what is exposed through the public
`impression.modeling` API during migration.

## Backlink

- [Surface Spec 46: Public/Internal API Transition and Documentation Boundary (v1.0)](../specifications/surface-46-public-internal-api-transition-boundary-v1_0.md)

## Manual Smoke Check

- Review the public modeling namespace and user-facing docs.
- Confirm public examples and docs do not promise surface-native returns before
  promotion.
- Confirm internal/private helpers remain available only through private
  modules.

## Automated Smoke Tests

- private surface-native helpers are not re-exported by `impression.modeling`
- private helpers remain importable from their explicit internal modules
- public modeling APIs in transition still return their documented mesh-native
  values

## Automated Acceptance Tests

- the public/internal API boundary is explicit in code shape and import surface
- user-facing documentation does not get ahead of supported public behavior
- internal experimentation remains available without creating accidental public
  contracts

## Notes

- Cover at least one primitive-side internal helper and one modeling-op-side
  internal helper.
