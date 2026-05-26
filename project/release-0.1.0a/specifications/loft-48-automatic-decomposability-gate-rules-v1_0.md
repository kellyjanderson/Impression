# Loft Spec 48: Automatic Decomposability Gate Rules (v1.0)

## Overview

This specification defines when an unresolved many-to-many subset remains
automatically decomposable and when the planner must stop deterministic
reduction.

## Backlink

Parent specification:

- [Loft Spec 27: Many-to-Many Decomposition and Automatic Decomposability Gate (v1.0)](loft-27-many-to-many-decomposition-and-decomposability-gate-v1_0.md)

## Scope

This specification covers:

- continuing deterministic region consumption
- gate conditions for stopping automatic decomposition
- residual unresolved-state detection

## Behavior

This branch must define:

- what it means for deterministic decomposition to continue
- what exact conditions mean the automatic decomposability gate has been
  reached
- what planner state must exist at the gate

## Constraints

- gate rules must be explicit
- automatic decomposition must stop when no more regions can be consumed
  deterministically
- gate logic must remain independent of executor concerns

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- continuing-decomposition conditions are explicit
- gate conditions are explicit
- residual-state semantics at the gate are explicit
- executable transitions expose non-blocking decomposability state
