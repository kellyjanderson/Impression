# Surface Spec 44 Test: Primitive API Surface Return Migration

## Overview

This test specification defines verification for primitive APIs that migrate
from mesh-returning to surface-returning behavior.

## Backlink

- [Surface Spec 44: Primitive API Surface Return-Type Migration (v1.0)](../specifications/surface-44-primitive-api-surface-return-migration-v1_0.md)

## Manual Smoke Check

- Construct representative primitives through the public modeling API.
- Confirm they now return surface-native objects on the canonical path and
  still preview/export successfully.

## Automated Smoke Tests

- migrated primitive APIs return `SurfaceBody` or documented surface-native
  containers
- compatibility bridge remains available where promised

## Automated Acceptance Tests

- primitive API documentation/examples match the migrated return contract
- migrated primitives flow through preview/export tessellation without special
  case mesh emitters
- compatibility shims are explicit rather than silent alternate return paths

## Notes

- Cover at least one box-like and one rotational primitive.
