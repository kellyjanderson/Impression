# Surface Spec 50 Test: Surface-to-Mesh Adapter

## Overview

This test specification defines verification for the explicit adapter path from
surface-native objects to mesh consumers.

## Backlink

- [Surface Spec 50: Surface-to-Mesh Adapter Contract (v1.0)](../specifications/surface-50-surface-to-mesh-adapter-contract-v1_0.md)

## Manual Smoke Check

- Convert a representative surface-native body through the adapter.
- Confirm the adapter produces the expected mesh output and does not mutate the
  source surface object.

## Automated Smoke Tests

- adapter accepts documented surface-native inputs
- adapter produces non-empty mesh output for valid inputs

## Automated Acceptance Tests

- adapter behavior is deterministic for identical inputs and requests
- adapter records or exposes documented lossiness boundaries where relevant
- adapter is explicit in consumer paths that still require meshes

## Notes

- Pair with watertight and open-surface fixtures.
