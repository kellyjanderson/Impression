# Loft Planner / Executor Architecture

## Overview

This document defines the architectural boundary between loft planning and loft
execution.

The key rule is:

> The planner resolves structural evolution.
> The executor realizes resolved evolution as surfaces.

This boundary exists to keep structural reasoning separate from geometric
construction.

## Working Assumption

Most planner intervals should resolve without extraordinary machinery.

The planner should therefore optimize for:

- direct resolved correspondence where available
- direct event decomposition where one clear structural answer exists
- explicit constraint-request escalation only for the residual hard cases

## Components

### Planner Inputs

The planner receives:

- progression-ordered placed topology states
- optional predecessor/successor constraints
- optional previously propagated local constraints from neighboring intervals

Each placed topology state should already provide:

- normalized structural regions
- containment hierarchy
- placement and frame context
- progression coordinate

### Planner Outputs

The planner emits:

- one evolution plan for the whole sequence
- interval plan records
- resolved correspondences
- planner-internal relationship graph records where event decomposition requires
  them
- transition operators
- unresolved constraint-request records
- interval-level diagnostics

Planner output is geometric-plan data, not meshes and not surface patches.

### Executor Inputs

The executor receives:

- resolved planner output only
- surface-kernel construction context

Placement data should already be referenced through the resolved plan rather
than arriving through a second parallel channel.

### Executor Outputs

The executor emits:

- surface patches
- seam and shared-boundary relationships
- trim and cap structures where required
- shell/body assembly data

The canonical executor output is a `SurfaceBody`.

## Planner Phases

Architecturally, planning should proceed in ordered phases.

### 1. State Normalization

Normalize each placed topology state into canonical structural form.

### 2. Deterministic Reduction

Resolve:

- predecessor/successor-constrained matches
- obvious direct correspondences
- obvious birth/death cases
- straightforward split/merge decompositions with one clear answer

### 3. Residual Subset Formation

Partition unresolved structure into independent subsets.

### 4. Ambiguity Classification

Classify remaining unresolved subsets into explicit ambiguity classes.

At this point the planner should prepare a minimal directional-constraint
request rather than continuing toward execution with unresolved structure.

### 5. Plan Assembly

Emit one full-sequence evolution plan containing:

- resolved interval records
- unresolved interval records marked as requiring additional constraint
- diagnostics
- execution eligibility

## Plan Structure

At the architectural level, the evolution plan should contain at least:

- sequence metadata
- normalized state references
- interval records
- operator records
- planner-internal relationship graph references
- ambiguity records
- summary diagnostics

Each interval record should state whether it is:

- executable
- blocked pending additional constraint
- blocked by invalid input

## Transition Operators

Transition operators should be explicit enough that the executor does not need
to reinterpret structure.

At minimum the architecture expects operators for:

- continuity bridge
- birth expansion
- death collapse
- split fan-out
- merge fan-in

The architecture also expects operator composition, so a complex interval may
resolve into a short deterministic sequence of operators instead of a single
monolithic event.

## Relationships

- the planner owns structural interpretation
- the executor owns geometric realization
- the executor must not reinterpret unresolved planner ambiguity
- the planner must not emit geometric approximation artifacts

## Data Flow

```text
placed topology states
-> correspondence reduction
-> transition classification
-> evolution plan
-> patch generation
-> stitched surface body
```

## Cross-Domain Solutions

### Planning is Non-Terminating Over the Sequence

The planner must process the full progression sequence even when some intervals
remain ambiguous.

This allows the system to:

- report all ambiguous intervals
- keep deterministic intervals fully resolved
- support tooling that helps the user apply minimal additional constraints

### Execution is Resolution-Bound

Execution may only run on intervals whose structural evolution is resolved.

This prevents the executor from quietly becoming a fallback decision-maker.

Straightforward split and merge cases are still expected to execute normally,
because they should already have been decomposed into resolved plan structure by
the planner.

If an interval is not resolved, the correct next step is additional
predecessor/successor constraint input and re-planning, not executor fallback.

### Directional Constraints Versus Planner Bookkeeping

The authored correspondence model is directional.

Users constrain correspondence through:

- `predecessor_ids`
- `successor_ids`

This is sufficient for the known authored loft problems.

The planner may still maintain internal stable handles or synthetic graph nodes
for:

- diagnostics
- decomposition bookkeeping
- execution references

This distinction is especially important for:

- `1 -> N`
- `N -> 1`
- `N -> M`

transition classes.

Architecturally, the planner-owned graph should be treated as an internal
bookkeeping artifact rather than a required authored concept.

### Transition Operators as the Planner / Executor Interface

The planner should not hand the executor raw topology-event labels alone.

It should hand the executor resolved transition operators that are explicit
enough to construct geometry, such as:

- continuity bridge
- birth expansion
- death collapse
- split fan-out
- merge fan-in

This avoids forcing the executor to infer missing structural meaning.

In practical terms, this means the executor should consume:

- explicit operator type
- explicit source/target structural references
- explicit placement references
- explicit seam / boundary intent where needed

instead of relying on loose event labels.

The executor should never receive unresolved ambiguity records.

### Bounded Constraint Propagation

The planner may use limited neighboring interval information to preserve
consistency.

This does not turn planning into global optimization.

It means only that interval-local solving is allowed to inherit already-resolved
constraints where needed to keep progression coherent.

In practice, this means the planner may propagate already-resolved
predecessor/successor structure across neighboring intervals without inventing a
separate authored lineage concept.

The default propagation radius should be:

- one interval to either side of the active participants

In schematic form:

```text
A < B - C > D
```

When resolving the active interval `B -> C`, the planner may consult the
already-resolved neighboring intervals:

- `A -> B`
- `C -> D`

This default should remain explicitly configurable in planner policy rather
than being hard-coded into deeper execution logic.

The architecture also leaves room for future dynamic propagation policy when
neighboring changes are slight enough that a wider or narrower radius becomes
appropriate.

## Surface Construction Guidance

The architecture expects executor output to remain compatible with the surface
program already underway.

That means next-gen loft should initially target:

- ruled patches for bridged interval surfaces
- planar patches for explicit planar closures and caps

Additional patch families may be introduced later if the surface kernel grows,
but loft should not assume unsupported surface forms in its first pass.

## Execution Boundary

The planner / executor boundary is satisfied when:

- all geometric guessing has been eliminated from the executor
- all unresolved constraint requests remain visible as planner output
- the executor can construct surfaces entirely from resolved operators and
  placement data

## Specifications

This architecture branch should eventually feed the dedicated next-gen loft
specification tree.

No child specifications are created here yet because the broader loft evolution
branch is still being finalized.
