# Surface Spec 98 Test: Loft Surface Output Consumer Handoff

## Overview

This test specification defines verification for loft as a surface-native
producer feeding preview, export, and compatibility consumers.

## Backlink

- [Surface Spec 98: Loft Surface Output Consumer Handoff (v1.0)](../specifications/surface-98-loft-surface-output-consumer-handoff-v1_0.md)

## Manual Smoke Check

- Run a representative loft workflow on the canonical surface-native path.
- Confirm loft returns a surface-native result, preview/export render through
  tessellation, and any legacy mesh consumer uses the explicit adapter path.

## Automated Smoke Tests

- loft canonical output type is surface-native
- preview/export tessellation succeeds for loft-produced surfaces
- legacy mesh-consumer bridge remains explicit and callable where documented

## Automated Acceptance Tests

- loft-produced closed bodies tessellate to watertight meshes through the
  standard tessellation boundary
- loft-produced open or invalid surfaces classify correctly and do not silently
  bypass kernel validation
- preview/export consume the same loft surface truth rather than separate mesh
  generation paths

## Notes

- Use one stable loft fixture and one split/merge-capable fixture once the loft
  surface path reaches those capabilities.
