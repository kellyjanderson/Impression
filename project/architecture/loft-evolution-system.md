# Loft Evolution System Architecture

## Overview

Impression loft is evolving from a profile-bridging mesh generator into a
deterministic surface construction system.

The core architectural statement is:

> Loft constructs surfaces by evolving topology over progression.

Loft is therefore:

- a surface constructor
- topology-aware
- deterministic
- ambiguity-sensitive

Loft is not:

- an interpolation system
- a spline system
- a mesh morph
- a best-effort heuristic guesser

In the target architecture, loft operates within the broader surface-first
kernel:

```text
placed topology sequence
-> loft planner
-> loft executor
-> surface body
-> downstream surface operations
-> tessellation (optional / consumer-driven)
```

Meshes are downstream artifacts only.

## Architectural Bias

Most loft intervals should be straightforward.

The architecture should therefore optimize first for:

- deterministic one-to-one continuity
- deterministic hole birth and hole death
- deterministic `1 -> N` and `N -> 1` cases where one good structural
  decomposition is obvious

Only a smaller subset of cases should escalate into ambiguity handling.

This means the default loft path should be:

- explicit
- local
- mostly direct

and only branch into ambiguity reporting when structural symmetry or missing
constraints truly make a unique answer unavailable.

When that happens, ambiguity should be treated as a request for additional
directional correspondence constraints rather than as a persistent execution
mode.

## Components

### Progression Sequence

Loft operates over an ordered scalar progression.

Progression is:

- monotonic
- non-temporal, though analogous to time
- allowed to be non-uniform
- attached to each placed topology state

Progression defines:

- ordering of structural states
- adjacency of intervals
- where topology evolution is evaluated

Progression does not define surface behavior by itself.

### Placed Topology States

Each progression sample contributes one placed topology state.

A placed topology state combines:

- a progression coordinate
- placement information
- one normalized topology state

Placement information includes:

- origin
- local frame or equivalent transform
- any section-local orientation information required by the executor

This keeps structural truth and placement truth distinct while still giving loft
one canonical per-sample input object.

### Topology States

Inside a placed topology state, the topology state remains structural.

A topology state defines:

- what regions exist
- how they are nested by containment
- region polarity through containment alternation
- optional directional correspondence constraints
- section-level normalization invariants

Topology states describe structure, not transition behavior.

Directional correspondence constraints are expressed through:

- `predecessor_ids`
- `successor_ids`

These are relationship labels, not standalone authored region identities.

### Evolution Plan

The loft planner does not hand geometry directly to the executor.

It produces an evolution plan.

An evolution plan contains:

- interval records
- resolved correspondences
- transition operators
- planner-internal relationship graph references where needed
- execution eligibility
- ambiguity and diagnostic records

This plan is the canonical contract between structural reasoning and geometric
construction.

### Loft Planner

The loft planner is responsible for converting adjacent topology states into a
resolved evolution plan.

It performs:

- predecessor/successor-constrained correspondence reduction
- region and containment interpretation
- topology-event classification
- planner-internal relationship graph construction where event decomposition
  requires it
- ambiguity detection
- decomposition of complex transitions into locally resolvable operations
- full-sequence accumulation of resolved and unresolved interval records

The planner does not generate geometry.

### Loft Executor

The loft executor consumes a resolved plan and constructs surface patches.

It performs:

- local patch generation
- shared-boundary coordination
- patch stitching into shells and bodies
- trim and cap construction where required by resolved operators
- surface-native output emission

The executor does not guess correspondence.

### Surface Kernel

The canonical output of loft is a `SurfaceBody`.

This keeps loft aligned with the larger Impression kernel direction:

- surfaces are canonical
- tessellation is deferred
- preview/export/analysis consume derived meshes rather than owning truth

Loft should initially target the v1 surface kernel using the surface families
already in scope for the broader surface program:

- ruled patches
- planar patches

Additional families may be added later if the surface program grows, but nextgen
loft should not depend on unsupported patch families in its first pass.

### Diagnostics Layer

The diagnostics layer records:

- ambiguity reports
- unresolved intervals
- conflicting constraints
- structural summaries of planned transitions

Diagnostics are first-class outputs of planning, not afterthought logging.

## Default Structural Cases

The architecture assumes that many cases admit one clear structural evolution.

These should be treated as first-class default paths rather than exotic cases.

### Direct Continuity

When one region clearly corresponds to one region across an interval, loft
should produce direct bridging without invoking ambiguity machinery.

### Hole Birth and Hole Death

When a hole clearly appears or disappears within a containing region, loft
should treat this as a normal resolved topology event rather than a speculative
branching case.

### Straightforward `1 -> N` and `N -> 1`

Some split and merge cases have one clear structural interpretation due to
containment, geometry, directional correspondence, or strong locality.

Those should be treated as resolved decomposition cases.

They do not need to be escalated merely because cardinality changed.

### Escalation Threshold

Ambiguity handling should begin only when:

- multiple structurally valid decompositions remain
- no explicit constraint or deterministic rule separates them

This keeps the architecture focused on solving the common case cleanly while
still surfacing truly ambiguous structure honestly.

### Resolution After Constraint Injection

If an interval remains ambiguous, the planner should request minimal additional
directional correspondence from the user.

Once those predecessor/successor constraints are supplied, the remaining solve
should be deterministic.

Ambiguity is therefore a planning-time request for more constraint, not a
normal post-planning system state.

## Relationships

- progression orders placed topology states
- placed topology states provide the canonical per-sample input to planning
- topology states provide the structural truth inside each placed state
- optional predecessor/successor annotations constrain correspondence
- the planner converts adjacent placed topology states into evolution-plan
  interval records
- the executor converts resolved transition operators into surface patches
- the surface kernel receives the final surface body
- tessellation remains downstream of loft and is never part of loft’s canonical
  output

## Data Flow

### Nominal Flow

```text
stations / sections over progression
-> placed topology normalization
-> interval planning
-> resolved evolution plan
-> local patch construction
-> stitched surface body
```

### Diagnostic Flow

```text
stations / sections over progression
-> placed topology normalization
-> interval planning
-> ambiguity detection
-> structured diagnostics
```

Planning must complete over the full progression sequence even when ambiguity is
encountered.

Execution requires full resolution for the intervals it is asked to execute.

## Cross-Domain Solutions

### Topology Over Progression

The fundamental problem loft solves is not interpolation between shapes.

It is deterministic construction of surface continuity from a sequence of
discrete structural states.

This architecture resolves the mismatch between:

- topology, which is discrete and structural
- surfaces, which are continuous and geometric

by introducing an explicit planning stage between them.

### Deterministic, Explicit Decision Rules

Loft rejects hidden or non-deterministic heuristics.

Decision rules may still rank, reduce, or tie-break, but they must be:

- explicit
- deterministic
- testable
- visible in diagnostics when they fail to fully resolve ambiguity

### Local Solving With Limited Constraint Propagation

The primary solving unit is the interval between adjacent progression states.

However, interval solving may use bounded neighboring constraints where needed
to preserve consistency of resolved relationships or continuity across
intervals.

This architecture rejects mandatory global optimization as the baseline solving
model.

### Placed-State Input Instead of Bare Topology

Loft should consume one canonical per-sample object that includes:

- progression
- placement
- normalized topology

This avoids scattering placement data across parallel argument paths while still
preserving the conceptual separation between structural truth and placement
truth.

### Regions as Structural Primitives

Loft operates on regions arranged in a containment hierarchy.

Human concepts such as:

- islands
- holes
- positive space
- negative space

are interpretations of region structure rather than separate kernel primitives.

This keeps the loft planner aligned with topology-native data structures.

### Directional Correspondence Instead of Standalone Authored Identity

The authored loft model is relationship-first.

Regions may declare:

- `predecessor_ids`
- `successor_ids`

These directional references constrain how structure relates across progression.

The architecture does not require a standalone authored region `id` field for
known loft problems.

This means:

- authored correspondence is expressed through forward/backward relationship
  labels
- matching is constrained by predecessor/successor agreement
- planner bookkeeping may still use internal stable handles where needed for
  diagnostics, synthetic nodes, and execution records

### Transition Operators Instead of Event Guessing

The planner classifies topology change into structural transition operators,
such as:

- continuity bridge
- birth expansion
- death collapse
- split fan-out
- merge fan-in

These operators are the executor-facing representation of evolution.

They are not interpolations.

The architecture expects these operators to be decomposable and composable.

That means an interval may be represented by:

- one direct operator in simple cases
- multiple locally ordered operators in split/merge cases

without changing the planner / executor boundary.

### Evolution Plan as the Canonical Handoff

The planner / executor handoff should be one explicit evolution-plan object,
not an ad hoc bundle of partial structures.

That plan should be rich enough to support:

- execution
- diagnostics
- future tooling
- regression testing at the plan layer

### Surface Output Contract

Loft output should be surface-native and stitchable before tessellation.

At the architectural level, this means the executor must produce:

- patch-local parameterization
- explicit shared-boundary references
- shell/body assembly structure
- output suitable for downstream surface operations and consumer-driven
  tessellation

The architecture does not assume that watertight tessellation is achieved by
post-mesh repair.

### Ambiguity as a Valid Outcome

Ambiguity is not treated as a malformed input condition.

It is a planning-time signal indicating that:

- multiple structurally valid evolutions remain
- available constraints are insufficient to choose one uniquely

The system must surface ambiguity explicitly instead of guessing.

The intended recovery path is:

- report the unresolved subset
- request minimal additional predecessor/successor constraints
- re-plan deterministically

## Areas Requiring Extra Attention

The architecture is confident about the broad direction, but these areas still
deserve extra care before or during specification work:

### Split / Merge Relationship Graphs

Straightforward split and merge cases should resolve normally, but the exact
planner-internal representation of decomposed predecessor/successor structure
still needs careful specification.

### Containment-Driven Edge Cases

Cases where region containment changes in subtle ways may need more precise
rules than the broad architecture defines here.

### SurfaceBody Boundary Contracts

Loft execution depends on exact rules for:

- shared-boundary ownership
- trim responsibility
- cap attachment
- patch parameter alignment

Those rules should be defined by the SurfaceBody architecture and consumed by
loft rather than invented locally inside the loft branch.

### Tolerance Policy

The loft branch still needs explicit tolerance families and spec-level
operational rules for:

- containment evaluation
- collapse detection
- deterministic tie-break thresholds

Numeric tolerance values should emerge downstream through specifications and
implementation evidence rather than being frozen in the umbrella architecture
document.

### Surface Family Limits

The first pass should stay aligned with the v1 surface kernel rather than
assuming richer patch families that the broader surface program has not yet
adopted.

## System Properties

### Determinism

Identical inputs and constraints produce identical plans, diagnostics, and
surface results.

### Explicitness

All assumptions, constraints, and unresolved ambiguities are made visible.

### Composability

Loft outputs surface-native geometry that composes cleanly with later surface
operations and consumer-driven tessellation.

### Diagnostic Transparency

Failure to resolve does not collapse into generic “loft failed” behavior.

The system reports where and why structural ambiguity remains.

Execution, however, is expected to operate only on a fully resolved plan.

## Specifications

This document is an architecture branch for the next-generation loft refactor.

The dedicated loft specification tree should be written after the remaining
architectural targets for next-gen loft are stabilized.
