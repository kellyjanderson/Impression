# Surface Body Families Planning Bundle

## Purpose

This bundle collects execution plans for finishing the surface-body direction.

It is organized around three related tracks:

- [Remove Fallback Code Plan](remove-fallback-code-plan.md)
- [Surface Support Libraries Plan](surface-support-libraries-plan.md)
- [Full Surface Body Implementation Plan](full-surface-body-implementation-plan.md)

These plans are intentionally implementation-facing. They should be refined
into final leaf specifications and paired test specifications before large code
changes begin.

## Inputs

- [Surface Body Patch Families Research](../../../research/2026-05-21-surface-body-patch-families-research.md)
- [Current Feature Surface Lost-Capability Audit](../../../research/2026-04-26-current-feature-surface-lost-capability-audit.md)
- [Surface-first internal model architecture](../../architecture/surface-first-internal-model.md)
- [SurfaceBody seam and adjacency architecture](../../architecture/surfacebody-seam-adjacency-architecture.md)
- [SurfaceBody CSG architecture](../../architecture/surfacebody-csg-architecture.md)
- [Surface-native capability replacement architecture](../../architecture/surface-native-capability-replacement-architecture.md)

## Dependency Shape

The recommended sequence is:

1. create shared libraries that remove duplication and define common surface
   kernel services
2. remove hidden mesh fallback and placeholder behavior
3. finish the required v1 patch families
4. expand into deferred/freeform patch families
5. promote complete surface-body modeling as the canonical path

Fallback removal depends on enough shared surface libraries to replace the
mesh wrappers cleanly. Full patch-family implementation depends on both the
library track and the removal of hidden fallback behavior.
