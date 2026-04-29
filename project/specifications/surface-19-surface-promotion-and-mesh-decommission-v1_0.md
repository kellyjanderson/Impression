# Surface Spec 19: Surface Promotion and Mesh-First Decommission Gate (v1.0)

## Overview

This specification defines the branch responsible for deciding when surfaces are
canonical and when mesh-first internal paths can be considered deprecated or
removed.

## Backlink

Parent specification:

- [Surface Spec 05: Migration and Compatibility Path (v1.0)](surface-05-migration-and-compatibility-path-v1_0.md)

## Scope

This specification covers:

- promotion criteria for surface-native truth
- decommission criteria for mesh-first internal paths
- completion gates and rollback expectations

## Behavior

This branch must define:

- what has to be true before surfaces are canonical
- what legacy behavior must still be supported until that point
- what evidence or tests are required before mesh-first internals are removed

## Constraints

- promotion criteria must be explicit and testable
- decommission must not happen just because most of the migration is done
- rollback expectations must remain possible until the promotion gate is met

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 56: Surface Canonical Promotion Criteria (v1.0)](surface-56-surface-canonical-promotion-criteria-v1_0.md)
- [Surface Spec 57: Mesh-First Decommission and Rollback Policy (v1.0)](surface-57-mesh-first-decommission-rollback-v1_0.md)
- [Surface Spec 58: Promotion Verification Matrix and Evidence Burden (v1.0)](surface-58-promotion-verification-matrix-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- the promotion gate is explicit
- decommission conditions are explicit
- the verification burden for promotion is explicit
