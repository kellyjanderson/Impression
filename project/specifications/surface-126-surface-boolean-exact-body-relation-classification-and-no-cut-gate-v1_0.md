# Surface Spec 126: Surface Boolean Exact Body-Relation Classification and No-Cut Gate (v1.0)

## Overview

This specification defines the surfaced boolean body-relation stage that
classifies exact no-cut cases before cut-curve generation begins.

## Backlink

- [Surface Spec 117: Surface Boolean Intersection, Classification, and Operand Splitting (v1.0)](surface-117-surface-boolean-intersection-classification-and-operand-splitting-v1_0.md)

## Scope

This specification covers:

- deterministic body-relation classification for prepared operands
- the bounded no-cut gate for disjoint, touching, equal, and exact-containment cases
- exact-containment posture for the initial supported primitive families

## Behavior

This leaf must define:

- the canonical body-relation records produced before cut discovery
- when surfaced execution short-circuits before cut-curve generation
- which no-cut outcomes are exact enough to drive later empty or reuse reconstruction

## Constraints

- no-cut gating must remain surfaced and must not hide mesh fallback
- exact-containment rules must be deterministic for equal operands and equal request state
- unsupported containment inference outside the initial supported families must remain explicit

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- canonical no-cut body relations are explicit
- the pre-cut short-circuit gate is explicit
- exact-containment rules are explicit enough to drive downstream reconstruction leaves
- verification requirements are defined by its paired test specification
