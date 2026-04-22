# Surface Spec 52 Test: Adapter Lossiness and Lifecycle

## Overview

This test specification defines verification for documented adapter lossiness
and lifecycle behavior.

## Backlink

- [Surface Spec 52: Adapter Lossiness and Lifecycle Rules (v1.0)](../specifications/surface-52-adapter-lossiness-lifecycle-rules-v1_0.md)

## Manual Smoke Check

- Convert representative surface-native objects through the adapter and inspect
  the documented boundaries of what is preserved versus derived.

## Automated Smoke Tests

- adapter lifecycle hooks or cache behavior follow the documented contract
- repeated conversions remain deterministic

## Automated Acceptance Tests

- documented lossy boundaries are observable and stable
- adapter output changes only when source identity or tessellation request
  changes in a documented way
- cosmetic metadata does not invalidate geometry output unless explicitly part
  of the adapter contract

## Notes

- Prefer fixtures that also exercise cache invalidation boundaries.
