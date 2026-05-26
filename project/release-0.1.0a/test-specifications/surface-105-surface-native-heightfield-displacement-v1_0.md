# Surface Spec 105 Test: Surface-Native Heightfield and Displacement Replacement

## Overview

This test specification defines verification for the surface-first replacement of deprecated heightfield and displacement features.

## Backlink

- [Surface Spec 105: Surface-Native Heightfield and Displacement Replacement (v1.0)](../specifications/surface-105-surface-native-heightfield-displacement-v1_0.md)

## Manual Smoke Check

- Build representative height-derived geometry and displaced surface-native geometry.
- Confirm the output remains canonical before tessellation.

## Automated Smoke Tests

- heightfield/displacement replacement produces non-empty canonical outputs
- image sampling and projection rules remain deterministic

## Automated Acceptance Tests

- representative terrain or relief fixtures tessellate validly
- representative examples gain reference images and STL artifacts once implemented
