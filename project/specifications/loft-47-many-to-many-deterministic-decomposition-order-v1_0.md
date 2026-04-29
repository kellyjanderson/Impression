# Loft Spec 47: Many-to-Many Deterministic Decomposition Order (v1.0)

## Overview

This specification defines the required deterministic reduction order inside an
isolated many-to-many subset.

## Backlink

Parent specification:

- [Loft Spec 27: Many-to-Many Decomposition and Automatic Decomposability Gate (v1.0)](loft-27-many-to-many-decomposition-and-decomposability-gate-v1_0.md)

## Scope

This specification covers:

- predecessor/successor reduction ordering
- direct-correspondence reduction ordering
- birth/death reduction ordering
- final `1 -> N` / `N -> 1` reduction ordering

## Behavior

This branch must define:

- the required order of deterministic reduction inside a many-to-many subset
- what earlier stages are allowed to simplify before later stages run
- what reduction order is prohibited

## Constraints

- decomposition order must be explicit
- premature local reduction must be prohibited where it would hide the true
  subset shape
- reduction order must remain deterministic

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- stage order is explicit
- allowed simplifications per stage are explicit
- prohibited reduction shortcuts are explicit
- executable transitions expose the required decomposition order
