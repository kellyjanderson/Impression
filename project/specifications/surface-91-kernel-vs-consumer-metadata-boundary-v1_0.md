# Surface Spec 91: Kernel-Native Versus Consumer Metadata Boundary (v1.0)

## Overview

This specification defines which metadata is native to the geometric kernel and
which metadata belongs to preview/export/other consumer layers.

## Backlink

Parent specification:

- [Surface Spec 30: Surface Metadata Placement Contract (v1.0)](surface-30-surface-metadata-placement-contract-v1_0.md)

## Scope

This specification covers:

- kernel-native metadata classes
- consumer-specific metadata classes
- boundary rules between the two

## Behavior

This branch must define:

- which metadata the kernel owns directly
- which metadata is carried only for consumers
- how consumer-specific metadata is attached without contaminating kernel meaning

## Constraints

- the boundary must be explicit
- kernel-native meaning must remain free of renderer-only concerns
- consumer metadata attachment must not alter geometric truth

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- kernel-native metadata classes are explicit
- consumer-specific metadata classes are explicit
- attachment rules across the boundary are explicit

