# Surface Spec 96: Loft Surface-Native Cap Construction (v1.0)

## Overview

This specification defines how loft endcaps are rebuilt as surface-native
construction rather than mesh-native closure geometry.

## Backlink

Parent specification:

- [Surface Spec 06: Loft Surface Refactor Track (v1.0)](surface-06-loft-surface-refactor-track-v1_0.md)

## Scope

This specification covers:

- flat/chamfer/round/cove cap execution against `SurfaceBody`
- cap patch generation
- cap trim/seam integration

## Behavior

This branch must define:

- cap modes produce surface patches and trim loops instead of final triangles
- cap construction reuses shell seam ownership rules
- cap outputs attach cleanly to loft sidewall patches before tessellation

## Constraints

- cap construction must not duplicate seam truth already owned by the shell
- cap behavior must remain deterministic across identical inputs
- cap construction must terminate in surface-kernel records valid for the
  tessellation boundary

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- surface-native cap construction rules are explicit
- cap/seam/trim integration is explicit
- mesh-native cap shortcuts are excluded from the canonical path
