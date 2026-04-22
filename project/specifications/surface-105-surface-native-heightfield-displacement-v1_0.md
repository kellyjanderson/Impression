# Surface Spec 105: Surface-Native Heightfield and Displacement Replacement (v1.0)

## Overview

This specification defines the surface-first replacement path for deprecated
mesh-primary heightfield and displacement features.

## Backlink

- [Surface Spec 99: Surface-Native Replacement Program (v1.0)](surface-99-surface-native-replacement-program-v1_0.md)

## Scope

This specification covers:

- heightmap-driven surface generation
- displacement of canonical modeled geometry
- replacement posture for the deprecated mesh-only APIs

## Behavior

- the surface-native representation of height-derived geometry
- how displacement applies to surface-native inputs
- how image-derived geometry remains consumer-compatible without becoming mesh-first

## Constraints

- heightfield replacement must not make mesh the primary document
- image sampling and projection rules must remain explicit
- public heightfield/displacement replacement must have durable docs

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- a surface-native heightfield/displacement replacement contract exists for the deprecated capability
- verification requirements are defined by its paired test specification
- documentation requirements for the public replacement are explicit
