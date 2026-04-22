# Surface Spec 18: Surface Migration Sequencing and Subsystem Order (v1.0)

## Overview

This specification defines the branch responsible for the order in which
subsystems migrate from mesh-first internals to surface-first internals.

## Backlink

Parent specification:

- [Surface Spec 05: Migration and Compatibility Path (v1.0)](surface-05-migration-and-compatibility-path-v1_0.md)

## Scope

This specification covers:

- migration order across subsystems
- prerequisites between migration steps
- the sequencing boundary between surface-foundation work and loft refactor work

## Behavior

This branch must define:

- which subsystems migrate first
- which later steps depend on earlier ones
- how the migration order avoids circular or premature adoption

## Constraints

- the sequence must keep the system debuggable
- loft-specific migration must remain downstream of surface-foundation work
- migration order must be explicit enough to drive planning later

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 53: Surface Migration Phase Ordering (v1.0)](surface-53-surface-migration-phase-ordering-v1_0.md)
- [Surface Spec 54: Migration Phase Gates and Dependency Rules (v1.0)](surface-54-migration-phase-gates-dependency-rules-v1_0.md)
- [Surface Spec 55: Surface-Foundation to Loft-Track Handoff Gate (v1.0)](surface-55-surface-foundation-to-loft-handoff-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- migration order is explicit
- subsystem dependencies are explicit
- the loft handoff point is explicit
