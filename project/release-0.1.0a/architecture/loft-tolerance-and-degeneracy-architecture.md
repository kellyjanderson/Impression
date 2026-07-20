# Loft Tolerance and Degeneracy Architecture

## Overview

This document defines the architectural role of tolerances, collapse behavior,
and degeneracy handling in next-generation loft.

Loft is deterministic only if its thresholds are explicit.

That means the architecture must define not just the happy path, but also:

- when structure is considered valid
- when it is considered collapsed
- when ambiguity thresholds are crossed
- when resolution should fail instead of guessing

## Purpose

Tolerance and degeneracy rules exist to keep the loft system:

- deterministic
- stable under small numeric perturbations
- explicit about collapse and closure behavior
- compatible with both planning and surface execution

They should not be used as hidden escape hatches for poor structural planning.

## Components

### Tolerance Taxonomy

At the architectural level, loft should distinguish several tolerance families
rather than treating all numeric judgments as one shared epsilon.

The current loft-side tolerance families are:

- `input_validity`
- `structural_classification`
- `decomposition_resolution`
- `collapse_degeneracy`
- `plan_validation`

Architecture defines these families as categories of decision.

Specifications and implementation evidence should define the specific
operational rules and any durable numeric thresholds that later become stable
contract.

### Input Validity Thresholds

The loft system must define minimum validity expectations for input structure.

These include:

- minimum loop vertex count
- non-degenerate loop area
- finite coordinates
- strictly increasing progression values
- valid split/merge and ambiguity control ranges

Invalid input should be reported as invalid input, not as ambiguity.

This belongs to the `input_validity` family.

### Structural Classification Tolerances

The loft system must separate tolerances used to classify structural situations
from tolerances used to collapse geometry or limit planner search.

These include:

- continuity versus birth/death classification
- containment-driven interpretation
- hole escape or hole persistence classification
- event classification near structural boundaries

These tolerances belong to the `structural_classification` family.

### Collapse Thresholds

The loft system must define when a structure is considered collapsed for
planning purposes.

At the architectural level this includes:

- hole collapse during death
- synthetic birth seeds approaching degenerate size
- region collapse during cap/end treatment
- closure ownership when a synthetic loop stands in for a collapsed structure

These thresholds belong to the `collapse_degeneracy` family.

### Ambiguity Thresholds

The planner must define deterministic limits around ambiguity search and
reduction, including:

- candidate enumeration limits
- deterministic tie-break stages
- residual ambiguity gates

These thresholds are planner controls, not geometry semantics, but they still
shape how loft behaves under difficult topology.

These thresholds belong to the `decomposition_resolution` family.

### Synthetic Seed Policy

Birth and death decomposition often require synthetic loops.

The architecture should treat synthetic seeds as deterministic geometric
proxies, not arbitrary guesses.

Seed generation should therefore be governed by explicit controls such as:

- split/merge staging count
- split/merge bias
- deterministic seed scale policy

### Closure Policy

When synthetic birth/death or region-level collapse occurs, closure ownership
must become explicit.

This means the architecture must define when closure occurs at:

- loop scope
- region scope
- interval boundary

and how those closure decisions are surfaced in the plan.

### Plan Validation Thresholds

The loft plan must also distinguish thresholds used to validate plan integrity
before execution.

These include:

- minimum viable sampled structure
- reference-range validity
- closure consistency checks
- execution-control completeness checks

These thresholds belong to the `plan_validation` family.

## Relationships

- input validity thresholds gate planning entry
- collapse thresholds influence region/loop action classification
- ambiguity thresholds govern residual search and escalation
- synthetic seed policy affects decomposition geometry
- closure policy turns collapse into deterministic executable structure

## Data Flow

```text
input states
-> validity checks
-> deterministic reduction and decomposition
-> collapse / seed / ambiguity thresholds
-> plan records and diagnostics
```

## Cross-Domain Solutions

### Validation Before Interpretation

Malformed input should be rejected before the planner attempts structural
interpretation.

This avoids conflating:

- invalid topology
- numeric degeneracy
- genuine ambiguity

### Deterministic Synthetic Birth / Death

Synthetic loops are acceptable only when they are:

- explicitly introduced
- deterministically sized
- explicitly closed

They must not act as hidden random seeds for structure creation.

### Ambiguity Limits as Policy, Not Guessing

Controls such as `ambiguity_max_branches` are legitimate planner policies.

They are not permission to guess.

They define when the planner should stop claiming it can deterministically
distinguish residual structure.

### Tolerance Values Emerge Downstream

Architecture should define tolerance families and their roles, not freeze
numeric doctrine too early.

In this branch:

- architecture defines the tolerance taxonomy
- specifications define the operational tolerance rules
- implementation and tests provide the evidence needed before numeric values
  become durable contract

This keeps loft tolerant of evidence-driven refinement while still preserving a
clear conceptual separation between tolerance classes.

### Collapse as Structural Event

Collapse should be treated as a structural event with explicit consequences:

- synthetic death
- synthetic birth
- loop closure
- region closure

rather than as a silent numeric artifact.

### Staging Controls Shape Decomposition

Controls such as:

- `split_merge_steps`
- `split_merge_bias`

are not just UI knobs.

They define how deterministic intermediate structure is introduced across
progression.

The architecture should therefore treat them as first-class decomposition
controls.

## Current Implementation Reference

The current loft implementation already provides concrete behavior that should
inform the architectural direction:

- `samples >= 3`
- station progression must be strictly increasing
- `split_merge_steps >= 1`
- `split_merge_bias in [0, 1]`
- `ambiguity_max_branches >= 1`
- synthetic birth/death loops created via deterministic `_shrunken_loop(...)`
  and `_synthetic_seed_scale(...)`
- closure ownership explicitly tracked as loop or region closures
- ambiguity failures surfaced with structured interval/class/stage details

These existing behaviors should be treated as the first concrete reference set
for spec work rather than discarded.

## Areas Requiring Extra Attention

The following details still need more definition before specification work is
complete:

- exact numeric collapse thresholds for loop/region validity
- which thresholds remain loft-local versus later align with broader kernel
  policy
- how future surface-body tolerances should interact with loft-local collapse
  and closure decisions
- how endcap collapse rules align with the broader next-gen loft model

## Specifications

This document should feed the next-generation loft tolerance / degeneracy
specification branch once the loft architecture tree is ready for specification
refinement.
