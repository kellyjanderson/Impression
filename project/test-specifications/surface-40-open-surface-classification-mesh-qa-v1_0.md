# Surface Spec 40 Test: Open-Surface Classification and Mesh QA

## Overview

This test specification defines verification for tessellation of open surfaces
and the associated QA/classification path.

## Backlink

- [Surface Spec 40: Open-Surface Classification and Mesh QA Contract (v1.0)](../specifications/surface-40-open-surface-classification-mesh-qa-v1_0.md)

## Manual Smoke Check

- Tessellate an intentionally open surface.
- Confirm output is classified as open rather than incorrectly promoted to a
  closed-valid watertight body.

## Automated Smoke Tests

- open surfaces tessellate without crashing
- open classification is reported explicitly

## Automated Acceptance Tests

- QA output distinguishes open surfaces from closed-valid bodies
- open meshes report expected boundary edges while remaining otherwise valid
- tessellation does not silently heal open surfaces into a different modeled
  meaning

## Notes

- Use at least one intentionally open fixture with a single obvious boundary.
