# Surface Spec 54: Migration Phase Gates and Dependency Rules (v1.0)

## Overview

This specification defines the gate conditions and dependency rules that govern
movement from one migration phase to the next.

## Backlink

Parent specification:

- [Surface Spec 18: Surface Migration Sequencing and Subsystem Order (v1.0)](surface-18-surface-migration-sequencing-v1_0.md)

## Scope

This specification covers:

- phase entry and exit gates
- dependency declarations between phases
- blocked-progress conditions

## Behavior

Entry and exit gates are:

- phase 1 entry: none beyond approved surface architecture/spec tree
- phase 1 exit: explicit `SurfaceBody` / patch / seam kernel contracts exist in
  code
- phase 2 entry: phase 1 complete
- phase 2 exit: seam-first tessellation and closed/open classification are
  backed by passing tests
- phase 3 entry: phases 1 and 2 complete
- phase 3 exit: scene handoff, adapter, and compatibility bridge contracts are
  implemented
- phase 4 entry: phases 1 through 3 complete
- phase 4 exit: public surfaced compatibility paths exist for the first bounded
  primitive/op family
- phase 5 entry: phases 1 through 4 complete
- phase 5 exit: loft surfaced executor/cap/orchestration/handoff path is
  implemented
- phase 6 entry: phases 1 through 5 complete
- phase 6 exit: promotion criteria, rollback policy, and evidence matrix are
  explicitly satisfied

Blocked-progress conditions:

- unresolved seam ownership or tessellation truth blocks phases 3 through 6
- missing explicit public compatibility paths blocks promotion
- missing surfaced loft path blocks canonical promotion

## Constraints

- phase gates must be testable
- dependency rules must be explicit
- blocked-progress conditions must be visible rather than inferred

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one phase-gating policy.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- entry and exit gates are explicit
- inter-phase dependencies are explicit
- blocked-progress conditions are explicit
