# Historical Spec Audit

Date: 2026-04-18

## Overview

This note records the audit of legacy topology and first-generation loft
specifications before they were deprecated.

## Goal

Confirm whether older specifications still contain implementation-significant
requirements that are missing from the active surface-first and next-generation
loft trees.

## Findings

### Carried Forward

The audit identified two implementation-significant details worth carrying into
the active next-generation loft specs:

1. station placement must use a right-handed orthonormal basis
2. normalized topology must preserve deterministic loop order and a canonical
   loop anchor for stable downstream signatures/anchoring

These were carried into:

- [Loft Spec 28: Placed Topology State Object Shape (v1.0)](../specifications/loft-28-placed-topology-state-object-shape-v1_0.md)
- [Loft Spec 29: Topology State Normalization Invariants (v1.0)](../specifications/loft-29-topology-state-normalization-invariants-v1_0.md)

### Retained As Historical Rationale Only

The remaining historical content was judged to be either:

- already represented in the active surface-first or next-generation loft
  branches
- implemented historical rationale rather than active specification work
- superseded by later architectural decisions

This includes:

- topology ownership boundaries now represented by the surface-first program
- first-generation split/merge and planner/executor refactor milestones now
  superseded by the next-generation loft branch
- first-generation loft example strategy now superseded by broader project DNA
  and later loft/surface planning

### Still Active, Not Deprecated

These specs remain active future-work branches and were not treated as
historical in this audit:

- [Loft Spec 18: Probabilistic Ambiguity Disambiguation (v1.0)](../specifications/loft-18-probabilistic-disambiguation-v1_0.md)
- [Loft Spec 19: Global Fairness and Skeleton Optimization (v1.0)](../specifications/loft-19-global-fairness-skeleton-optimization-v1_0.md)
- [Loft Spec 20: Interactive Branch Picking API (v1.0)](../specifications/loft-20-interactive-branch-picking-v1_0.md)

## Conclusion

The historical topology/first-generation loft documents remain useful as audit
history, but active implementation planning should use the current:

- surface-first spec tree
- next-generation loft spec tree
- paired test-specification tree
