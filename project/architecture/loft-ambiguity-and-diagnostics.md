# Loft Ambiguity and Diagnostics Architecture

## Overview

This document defines how ambiguity is represented, detected, and surfaced by
the next-generation loft system.

The central rule is:

> Ambiguity is surfaced, not hidden.

Loft must never silently choose among multiple structurally valid evolutions
without an explicit deterministic rule that fully resolves the choice.

If no such rule exists, the planner must report ambiguity.

The intended response to ambiguity is to request additional directional
correspondence from the user and then re-plan deterministically.

## Architectural Bias

Ambiguity should be the exception, not the default.

Most loft structure should resolve through:

- explicit directional correspondence
- direct geometry
- locality
- containment
- straightforward event decomposition

The ambiguity system exists to expose the residual hard cases honestly, not to
define the normal loft path.

## Components

### Ambiguity Detector

The planner contains an ambiguity detector that evaluates unresolved structural
subsets after deterministic reduction steps have completed.

### Ambiguity Record

An ambiguity record describes:

- stable diagnostic id
- progression interval
- the topology state that holds the ambiguity
- the regions inside that topology state that remain ambiguous
- the relationship group when the ambiguity exists inside an already-related
  subset

Additional classification or debugging detail may be attached where useful for:

- tooling
- testing
- developer-oriented inspection

### Diagnostic Surface

Diagnostics should be consumable by:

- developer logs
- tests
- IDE tooling
- future AI-assisted workflows

This requires ambiguity output to be structured rather than free-form.

### Invalid-Input Record

The architecture should also distinguish invalid-input diagnostics from
ambiguity diagnostics.

Invalid input includes issues such as:

- malformed topology
- broken containment
- contradictory predecessor/successor attachment

These should not be collapsed into ambiguity.

## Ambiguity Classes

At the architectural level, loft should distinguish at least these ambiguity
classes:

### Symmetry Ambiguity

Multiple candidate correspondences are structurally identical under symmetry,
for example:

```text
ivan, ivan -> ivan, ivan
```

### Insufficient Geometry Ambiguity

Available geometry does not provide enough asymmetry or locality cues to
deterministically prefer one structural mapping.

### Directional Constraint Residual Ambiguity

Explicit predecessor/successor correspondence exists, but multiple valid
resolutions still remain after those directional constraints have been honored.

### Split / Merge Residual Ambiguity

The planner detects that a split or merge structure exists, but multiple valid
fan-out or fan-in interpretations remain after normal deterministic
decomposition has been applied inside the related subset.

### Containment Ambiguity

The region hierarchy or polarity interpretation is insufficiently constrained to
produce a unique structural interpretation across the interval.

## Relationships

- ambiguity detection happens after deterministic reduction has resolved all
  certain matches
- ambiguity records are planner outputs
- invalid-input records are distinct from ambiguity records
- the executor never resolves ambiguity on its own
- diagnostics may be emitted even when some other intervals are fully resolved

Ambiguity records are planner-facing and user-facing.

They are not executor-facing.

## Data Flow

```text
topology states
-> deterministic reduction
-> unresolved subsets
-> ambiguity classification
-> structured diagnostics
```

## Cross-Domain Solutions

### Minimal Fix Principle

The system should favor diagnostics that point toward the smallest additional
constraint likely to resolve ambiguity.

Typical examples include:

- adding one predecessor or successor reference
- disambiguating one containment interpretation

The goal is to help users resolve ambiguity without redesigning the whole loft.

The preferred fix is a minimal predecessor/successor annotation that breaks the
tie cleanly.

### Reduction-Stage Visibility

Diagnostics should say not only what ambiguity class occurred, but also where
in planning it remained unresolved, for example:

- after directional-correlation reduction
- after direct locality reduction
- after split/merge decomposition

This makes the diagnostics more actionable for both humans and tools.

### Related-Subset Determinism

When ambiguity exists inside an already-related subset, the planner should not
switch to a different special rule set.

Instead it should:

- isolate the related subset
- apply the same deterministic reduction order used elsewhere
- honor predecessor/successor constraints inside that subset
- report only the residual ambiguity that remains after decomposition

This keeps related-region ambiguity aligned with the general loft determinism
model rather than creating a separate secondary ambiguity system.

### Full-Sequence Reporting

Ambiguity reporting should be collected across the full progression sequence.

This supports:

- full planning diagnostics in one run
- editor highlighting across multiple intervals
- AI-assisted correction suggestions

After constraint injection, the planner should be rerun and the expectation is
that the affected intervals become deterministic.

### Explicit, Structured Diagnostics

Diagnostic output should remain structured enough to support:

- test assertions
- IDE overlays
- future quick-fix tooling

Free-form error strings are not sufficient as the primary ambiguity interface.

### Clear Boundary Between Ambiguity and Failure

Ambiguity is not identical to invalid input.

Invalid input may still exist, for example:

- malformed topology
- broken containment
- contradictory predecessor/successor attachment

But ambiguity specifically means that multiple valid structural evolutions still
remain after valid reduction.

### Execution Blocking Semantics

Diagnostics should make it explicit whether a record:

- blocks execution of one interval
- blocks execution of the whole sequence
- is advisory only

This matters because full-sequence planning is non-terminating, while execution
is resolution-bound.

The normal ambiguity contract should be:

- ambiguity blocks execution of the affected intervals
- user supplies additional directional correspondence
- planner reruns
- execution proceeds only from the resolved result

## System Properties

### Visibility

Ambiguity must be observable and attributable to:

- a specific interval
- the topology state holding the ambiguity
- the ambiguous regions inside that topology state
- the relationship group when applicable

### Determinism

The same underconstrained input must produce the same ambiguity records.

The same added directional constraints must then produce the same resolved plan.

### Toolability

Diagnostics must be structured for machine consumption as well as human
inspection.

## Specifications

This architecture branch is intended to feed the eventual loft diagnostic and
ambiguity specification leaves once the broader next-gen loft architecture is
fully stabilized.
