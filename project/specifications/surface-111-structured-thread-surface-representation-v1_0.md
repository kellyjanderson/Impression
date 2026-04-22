# Surface Spec 111: Structured Thread Surface Representation (v1.0)

## Overview

This specification defines the canonical surface-native representation of thread geometry.

## Backlink

- [Surface Spec 103: Surface-Native Threading Replacement (v1.0)](surface-103-surface-native-threading-replacement-v1_0.md)

## Scope

This specification covers:

- external and internal thread surface representation
- structured or analytic thread surface families
- surfaced representation of runout and end-treatment behavior

## Behavior

This branch must define:

- the canonical surface-native output of thread generation
- how handedness, starts, taper, and profile shape are represented
- how thread-local structure remains deterministic before tessellation

## Constraints

- thread generation must not make triangle mesh the authored truth
- fit and quality semantics must remain separable from canonical geometry
- representation choices must be explicit enough to support downstream verification

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- thread surface representation is explicit for the supported thread forms
- the canonical output remains surface-native
- verification requirements are defined by its paired test specification

