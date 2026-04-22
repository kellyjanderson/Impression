# Surface Spec 30: Surface Metadata Placement Contract (v1.0)

## Overview

This specification defines the branch responsible for where metadata lives
across surface bodies, shells, and patches.

## Backlink

Parent specification:

- [Surface Spec 10: Surface Transform, Metadata, and Identity Policy (v1.0)](surface-10-surface-transform-metadata-identity-v1_0.md)

## Scope

This specification covers:

- metadata placement
- body/shell/patch metadata ownership
- kernel-native versus consumer-specific metadata distinction

## Behavior

This branch must define:

- which metadata belongs at each level
- which metadata is inherited or overridden
- which metadata is kernel-native versus consumer-specific

## Constraints

- metadata placement must be explicit
- ownership and override rules must be explicit
- kernel-native and consumer-specific metadata must not be mixed ambiguously

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 89: Metadata Placement by Body, Shell, and Patch Level (v1.0)](surface-89-metadata-placement-by-level-v1_0.md)
- [Surface Spec 90: Metadata Inheritance and Override Rules (v1.0)](surface-90-metadata-inheritance-override-rules-v1_0.md)
- [Surface Spec 91: Kernel-Native Versus Consumer Metadata Boundary (v1.0)](surface-91-kernel-vs-consumer-metadata-boundary-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- metadata placement is explicit
- ownership/override rules are explicit
- kernel-native versus consumer-specific distinction is explicit
