# Surface Spec 130: Surface Boolean No-Cut and Exact-Reuse Result Reconstruction (v1.0)

## Overview

This specification defines result reconstruction for surfaced boolean outcomes
that require no new cut topology.

## Backlink

- [Surface Spec 118: Surface Boolean Result Topology Reconstruction (v1.0)](surface-118-surface-boolean-result-topology-reconstruction-v1_0.md)

## Scope

This specification covers:

- empty-result reconstruction for bounded no-cut cases
- exact source-body, source-shell, or source-fragment reuse when no new cut topology is required
- explicit multi-shell disjoint union reconstruction for the bounded initial slice

## Behavior

This leaf must define:

- when surfaced results are reconstructed as empty without new topology
- when a prepared source body or fragment is reused exactly rather than rebuilt
- how disjoint union results remain surfaced multi-shell results rather than hidden mesh combination

## Constraints

- no-cut reconstruction must remain surface-native
- exact reuse must preserve valid shell, seam, and metadata meaning
- no-cut reconstruction must not invent unnecessary new topology

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- empty and exact-reuse outcomes are explicit for the bounded initial slice
- multi-shell disjoint union reconstruction is explicit
- reuse-versus-empty rules are explicit enough to support downstream result classification
- verification requirements are defined by its paired test specification
