# Surface Spec 06: Loft Surface Refactor Track (v1.0)

## Overview

This specification defines the dedicated loft refactor track that will follow
the surface-foundation work.

It exists so loft can be treated as an explicit downstream migration target
rather than forcing the surface kernel to be invented inside loft-specific
implementation work.

## Backlink

Parent specification:

- [Surface Spec 01: Surface-First Internal Model Program (v1.0)](surface-01-surface-first-internal-model-program-v1_0.md)

## Scope

This specification covers:

- the boundary between surface-foundation work and loft-specific refactor work
- the role of the existing loft planner in a surface-native future
- the expectation that loft will move from mesh execution to surface execution
- the loft-specific target branches required to begin implementation

## Behavior

The loft refactor track defines:

- how the current loft planner targets surface-native execution
- how caps become surface-native rather than mesh-native
- how split/merge planning interacts with surface patch generation
- how tessellation consumes loft-produced surfaces

## Constraints

- loft refactor work must not begin by inventing missing surface-kernel
  semantics ad hoc
- loft-specific implementation must build on the surface kernel and tessellation
  contracts rather than bypassing them
- loft-specific surface execution must terminate in `SurfaceBody`/consumer
  contracts rather than preserving a hidden mesh-first fast path

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
surface-kernel or loft-architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 95: Loft Plan-to-Surface Executor Contract (v1.0)](surface-95-loft-plan-to-surface-executor-contract-v1_0.md)
- [Surface Spec 96: Loft Surface-Native Cap Construction (v1.0)](surface-96-loft-surface-native-cap-construction-v1_0.md)
- [Surface Spec 97: Loft Split/Merge Surface Patch Orchestration (v1.0)](surface-97-loft-surface-patch-orchestration-v1_0.md)
- [Surface Spec 98: Loft Surface Output Consumer Handoff (v1.0)](surface-98-loft-surface-output-consumer-handoff-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- loft is explicitly represented as a downstream dedicated track
- it is clear that loft depends on the surface foundation rather than preceding
  it
- executor-target, cap, split/merge, and consumer-handoff leaves are explicit
