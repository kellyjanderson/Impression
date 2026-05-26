# Loft N->M / M->N Decomposition Architecture

## Overview

This document defines the architectural approach for many-to-many loft
transitions.

These are intervals where:

- multiple source regions contribute to multiple target regions, or
- the inverse relationship is equally many-to-many when viewed across
  progression

The architectural goal is not to treat `N -> M` as one monolithic event.

The goal is to:

- decompose it into smaller structurally meaningful relationships where possible
- request additional directional correspondence where decomposition is not
  uniquely determined
- preserve deterministic execution after decomposition is resolved

## Working Principle

`N -> M` and `M -> N` should be treated as decomposition problems.

The planner should attempt to reduce these transitions into:

- resolved direct continuities
- resolved births and deaths
- resolved `1 -> N`
- resolved `N -> 1`

If the planner reaches a point where no more regions can be deterministically
decomposed, it should stop and emit a constraint request.

## Components

### Many-to-Many Candidate Set

The planner first identifies the subset of source and target regions that
participate in the unresolved many-to-many relationship.

This subset should be isolated from already-resolved neighboring structure.

### Decomposition Pass

The planner then tries to decompose the subset into a short ordered set of
smaller structural transitions.

The decomposition should proceed in this order:

1. isolate the full unresolved many-to-many subset
2. apply explicit predecessor/successor constraints inside that subset
3. resolve direct correspondences that remain obvious after directional
   constraints are honored
4. resolve explicit births and deaths that become structurally clear
5. reduce the remaining structure into deterministic `1 -> N` and `N -> 1`
   decompositions

This ordering matters because premature local reduction can hide the true shape
of the many-to-many problem before directional correspondence has been applied.

### Constraint Request Surface

If the decomposition pass reaches a state where no more regions can be
deterministically reduced, the planner should emit a constraint request that
asks for minimal additional predecessor/successor relationships.

This request should identify:

- the ambiguous interval
- the ambiguous subset
- the candidate structures needing directional ties

### Resolved Decomposition

Once enough directional correspondence has been supplied, the planner should
re-run and emit a deterministic decomposition.

That resolved decomposition becomes normal plan structure, not a special runtime
mode.

## Relationships

- many-to-many decomposition is planner-owned
- executor never receives unresolved many-to-many structure
- predecessor/successor constraints are the tie-breaking mechanism
- resolved many-to-many decompositions reduce into ordinary transition operators

## Data Flow

```text
placed topology states
-> deterministic reduction
-> unresolved many-to-many subset
-> decomposition attempt
-> either:
   - resolved smaller operators
   - constraint request for directional ties
```

## Cross-Domain Solutions

### Decompose Before Escalate

The planner should make a serious deterministic decomposition attempt before
requesting user input.

This keeps:

- straightforward many-to-many cases automatic
- genuinely ambiguous cases visible

The same rule applies even when the unresolved subset already belongs to a
known related group.

Related structure does not create a separate ambiguity regime. It only defines
the subset within which the normal deterministic decomposition rules must be
applied.

### Automatic Decomposability Gate

A many-to-many subset should be treated as automatically decomposable while the
planner can continue reducing regions under the normal deterministic reduction
order.

The automatic decomposability gate is reached when:

- the current unresolved subset still contains regions
- no additional regions can be deterministically decomposed
- further progress would require added predecessor/successor constraints

At that point the planner should stop automatic reduction and surface the
residual ambiguity for resolution.

### Directional Constraints Are the Tie-Breaker

When decomposition cannot be uniquely determined, the planner should request
predecessor/successor annotations rather than inventing hidden ranking logic.

This keeps the architecture aligned with the relationship-first loft model.

Within the decomposition process, predecessor/successor constraints should be
applied before direct matching heuristics are allowed to simplify the subset.

### No Executor-Level Many-to-Many Guessing

The executor should only ever see:

- resolved direct operators
- resolved decomposed operator sequences

It should never receive an unresolved “many-to-many” operator and decide what
that means geometrically.

### Synthetic Nodes Remain Internal

If the planner needs internal synthetic graph nodes while decomposing many-to-
many structure, those remain planner-internal bookkeeping.

They are not part of the authored correspondence model.

## Areas Requiring Extra Attention

The following details still need tighter definition during specification work:

- how decomposed branch structure propagates into adjacent intervals
- how the default one-interval neighbor propagation radius should evolve if a
  future dynamic policy is introduced

## Specifications

This document should feed the next-generation loft `N -> M` / `M -> N`
decomposition specification branch once the loft architecture tree is ready for
specification refinement.
