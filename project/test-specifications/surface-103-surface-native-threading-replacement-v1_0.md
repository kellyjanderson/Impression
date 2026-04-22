# Surface Spec 103 Test: Surface-Native Threading Replacement

## Overview

This test specification defines verification for the surface-first replacement of deprecated threading capability.

## Backlink

- [Surface Spec 103: Surface-Native Threading Replacement (v1.0)](../specifications/surface-103-surface-native-threading-replacement-v1_0.md)

## Manual Smoke Check

- Build representative external/internal threads and convenience threaded parts through the surface-native path.
- Confirm preview/export still works through boundary tessellation.

## Automated Smoke Tests

- thread generation returns canonical non-mesh-first outputs
- fit and quality inputs remain deterministic

## Automated Acceptance Tests

- representative thread bodies and cutters tessellate validly
- nuts, rods, and runout cases remain covered
- representative results gain reference images and STL artifacts once implemented
