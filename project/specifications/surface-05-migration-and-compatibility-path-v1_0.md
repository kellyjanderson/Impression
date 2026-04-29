# Surface Spec 05: Migration and Compatibility Path (v1.0)

## Overview

This specification defines the branch responsible for migrating Impression from
mesh-first internals to surface-first internals without breaking the system in
an uncontrolled way.

## Backlink

Parent specification:

- [Surface Spec 01: Surface-First Internal Model Program (v1.0)](surface-01-surface-first-internal-model-program-v1_0.md)

## Scope

This specification covers:

- migration ordering
- compatibility adapters
- temporary mesh-boundary support
- promotion criteria for making surfaces the canonical internal truth

## Behavior

This branch must define:

- how surface-native tools coexist with mesh-native legacy paths during
  transition
- what compatibility adapters exist
- which subsystems migrate first
- when a subsystem is considered fully migrated
- how the architecture avoids getting stuck in a permanent dual-kernel state

## Constraints

- migration steps must be reversible enough to debug safely
- temporary adapters must not quietly become permanent architectural centers
- compatibility must preserve existing preview/export behavior where practical
- migration ordering must put surface foundation work ahead of loft-specific
  surface refactor work

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 17: Compatibility Adapter Contracts (v1.0)](surface-17-compatibility-adapter-contracts-v1_0.md)
- [Surface Spec 18: Surface Migration Sequencing and Subsystem Order (v1.0)](surface-18-surface-migration-sequencing-v1_0.md)
- [Surface Spec 19: Surface Promotion and Mesh-First Decommission Gate (v1.0)](surface-19-surface-promotion-and-mesh-decommission-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- the migration order is explicit
- temporary adapters are named and bounded
- a clear completion gate exists for when surfaces become canonical
- the child branches define migration and compatibility concerns as final
  implementation-sized leaves
