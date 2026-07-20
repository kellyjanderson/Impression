# Surface Spec 94: Cache-Key Dependency and Identity Usage Rules (v1.0)

## Overview

This specification defines how stable identity participates in cache keys and
which cached operations may rely on identity alone.

## Backlink

Parent specification:

- [Surface Spec 31: Stable Identity and Caching Keys for Surface Objects (v1.0)](surface-31-stable-identity-caching-keys-v1_0.md)

## Scope

This specification covers:

- identity participation in cache keys
- content-sensitive versus identity-sensitive caching
- prohibited cache-key shortcuts

## Behavior

This branch must define:

- body tessellation caches may use:
  - body identity
  - tessellation request
  - any non-baked transform state
- seam sampling caches may use:
  - seam identity
  - tessellation request
- cosmetic consumer metadata must not invalidate kernel identity by itself
- content-derived fields are still required whenever identity alone would
  collapse structurally distinct states

## Constraints

- cache-key usage must be explicit
- identity-only caching must be bounded to safe cases
- prohibited shortcuts must be explicit

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- identity participation in cache keys is explicit
- content-sensitive cache requirements are explicit
- prohibited cache-key shortcuts are explicit

## Current Preferred Answer

The preferred first-pass cache structure is:

- body tessellation cache keyed by body identity plus request and transform
  state
- seam sampling cache keyed by seam identity plus request
