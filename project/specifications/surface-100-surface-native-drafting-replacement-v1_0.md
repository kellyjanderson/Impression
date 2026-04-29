# Surface Spec 100: Surface-Native Drafting Replacement (v1.0)

## Overview

This specification defines the surface-first replacement path for deprecated
mesh-primary drafting helpers.

## Backlink

- [Surface Spec 99: Surface-Native Replacement Program (v1.0)](surface-99-surface-native-replacement-program-v1_0.md)

## Scope

This specification covers:

- line, plane, arrow, and dimension drafting capability
- surface-native or topology-native output forms
- public replacement posture for deprecated mesh drafting helpers

## Behavior

This branch must define:

- the canonical surface-first drafting outputs
- how drafting artifacts participate in scene and consumer handoff
- how dimension and text attachment work without mesh as authored truth

## Constraints

- drafting replacement must not depend on mesh as primary document
- drafting outputs must remain usable by preview and export through boundary tessellation only
- user-facing drafting APIs must gain durable documentation

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- a surface-first drafting replacement contract exists for the deprecated drafting capability
- verification requirements are defined by its paired test specification
- documentation requirements for the public drafting replacement are explicit
