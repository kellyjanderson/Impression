# Surface Spec 98: Loft Surface Output Consumer Handoff (v1.0)

## Overview

This specification defines the surfaced behavior of a loft operation that now
produces surface-native results and hands them off cleanly to preview, export,
and mesh-adapter consumers.

## Backlink

Parent specification:

- [Surface Spec 06: Loft Surface Refactor Track (v1.0)](surface-06-loft-surface-refactor-track-v1_0.md)

## Scope

This specification covers:

- loft-produced `SurfaceBody` consumer handoff
- preview/export tessellation of loft-produced surfaces
- compatibility behavior for mesh-facing consumers

## Behavior

This branch must define:

- loft returns a surface-native result on the canonical path
- preview and export tessellate that surface-native result through the standard
  tessellation boundary
- legacy mesh consumers reach loft output only through the documented
  compatibility adapter path

## Constraints

- consumer-visible loft behavior must not rely on a hidden mesh-first path
- preview/export must consume the same loft surface truth
- compatibility adapters must remain explicit and testable

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- loft consumer handoff rules are explicit
- preview/export behavior from loft-produced surfaces is explicit
- legacy mesh-consumer fallback behavior is explicit
