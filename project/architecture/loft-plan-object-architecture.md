# Loft Plan Object Architecture

## Overview

This document defines the architectural shape of the next-generation loft plan
object.

The plan object is the canonical handoff between loft planning and loft
execution.

It exists so that:

- structural reasoning is explicit
- execution is geometry-only
- diagnostics have a stable home
- plan-layer regression is possible without invoking geometry construction

## Purpose

The loft plan object must be rich enough to represent:

- normalized placed topology inputs
- interval-by-interval resolved structure
- transition operators
- closure ownership
- ambiguity and invalid-input diagnostics
- execution eligibility

The plan object must not contain:

- tessellated mesh artifacts
- executor-generated patches
- hidden geometry guesses

## Components

### Plan Header

The plan header records whole-plan configuration and summary metadata.

At minimum it should contain:

- schema version
- plan-wide sampling count
- planner identity/version
- split/merge controls
- ambiguity controls
- fairness controls
- skeleton usage mode
- summary counts and diagnostics

This keeps planning assumptions explicit and auditable.

### Planned States

The plan should store normalized per-sample placed topology states.

Each planned state should include:

- station index
- progression value
- placement frame
- normalized region loops

These states are the executor’s read-only reference for section-local geometry
placement.

### Planned Intervals

The plan should store one interval record for each adjacent state pair.

Each interval record should include:

- source and destination state indices
- topology case
- ambiguity classification
- ordered branch emission
- one or more region-pair or operator-group records

Intervals are the primary execution units of loft.

### Planned Region-Pair Records

Each planned interval should contain explicit region-level records describing
how structural units relate across the interval.

These records should include:

- source structural reference
- destination structural reference
- region-level action classification
- branch identifier
- loop-pair records
- closure ownership records

These records are executor-facing and must be complete enough that the executor
does not reinterpret topology.

### Planned Loop-Pair Records

Each region-pair record should contain loop-level correspondence records.

These should include:

- source loop reference
- destination loop reference
- normalized source loop
- normalized destination loop
- loop role classification

Loop roles should distinguish at least:

- stable
- synthetic birth
- synthetic death

### Closure Records

Closure ownership should be explicit in the plan rather than inferred later.

Closure records should define:

- side (`prev` or `curr`)
- scope (`loop` or `region`)
- loop index when loop-scoped

This allows the executor to perform closure deterministically.

### Diagnostic Records

The plan should contain explicit plan-layer diagnostics.

This includes:

- ambiguity records
- invalid-input records
- summary class counts
- fairness diagnostics
- execution-blocking status

The plan object is the natural home for these records because they arise during
planning rather than execution.

Ambiguity diagnostics in the plan should carry enough information to:

- identify the blocked interval
- identify the topology state that holds the ambiguity
- identify the ambiguous regions inside that topology state
- identify the relationship group when the ambiguity is internal to an already-
  related subset

The plan should not require the executor to infer or reinterpret this
diagnostic structure.

## Relationships

- planned states provide stable references for intervals
- intervals own ordered execution structure
- region-pair and loop-pair records provide executor-facing correspondence
- closure records define deterministic ownership of closure geometry
- diagnostics annotate the plan without requiring execution

## Data Flow

```text
placed topology states
-> normalization
-> deterministic reduction
-> interval records
-> plan metadata and diagnostics
-> loft plan
```

## Cross-Domain Solutions

### Explicit Execution Eligibility

Each interval should be explicitly classifiable as:

- executable
- partially resolved
- blocked by ambiguity
- blocked by invalid input

This keeps the non-terminating planner compatible with a resolution-bound
executor.

### Stable Branch Identity

Each region-pair record should carry a stable branch identifier within the plan.

This is not an authored identity primitive.

It is plan-local bookkeeping that allows:

- deterministic branch ordering
- continuity diagnostics
- closure accounting
- executor reference stability

### Operator-Ready Rather Than Event-Only

The plan should not stop at broad event labels.

It should provide enough structure that each interval can be executed from:

- references
- roles
- closures
- branch ordering

without a second structural interpretation pass.

### Validation Before Execution

The plan object should be fully self-validating before it reaches the executor.

Validation should cover:

- interval ordering
- reference ranges
- loop sample counts
- closure ownership consistency
- metadata completeness for execution controls

## Current Implementation Reference

The current loft implementation already embodies much of this plan shape
through:

- `PlannedStation`
- `PlannedLoopRef`
- `PlannedRegionRef`
- `PlannedLoopPair`
- `PlannedClosure`
- `PlannedRegionPair`
- `PlannedTransition`
- `LoftPlan`

This architecture document captures the direction those structures should be
preserved and generalized toward rather than reinvented.

## Areas Requiring Extra Attention

The following details still need tighter definition during specification work:

- how ambiguity and invalid-input records are embedded versus summarized
- whether operator records remain region-pair centric or become a separate plan
  layer
- how plan-local branch ids relate to future planner-internal synthetic graph
  nodes

## Specifications

This document should feed the next-generation loft plan-object specification
branch once the loft architecture tree is ready for specification refinement.
