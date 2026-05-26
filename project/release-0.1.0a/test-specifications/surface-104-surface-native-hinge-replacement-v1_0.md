# Surface Spec 104 Test: Surface-Native Hinge Replacement

## Overview

This test specification defines verification for the surface-first replacement of deprecated hinge generators.

## Backlink

- [Surface Spec 104: Surface-Native Hinge Replacement (v1.0)](../specifications/surface-104-surface-native-hinge-replacement-v1_0.md)

## Manual Smoke Check

- Build representative traditional, living, and bistable hinge outputs through the surface-native path.
- Confirm the resulting objects preview/export without mesh-first construction.

## Automated Smoke Tests

- hinge replacement terminates in canonical non-mesh-first outputs
- assembly/group semantics remain deterministic

## Automated Acceptance Tests

- representative hinge outputs tessellate non-empty and structurally valid
- representative examples gain reference images and STL artifacts once implemented
