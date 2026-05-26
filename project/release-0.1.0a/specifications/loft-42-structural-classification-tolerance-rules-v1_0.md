# Loft Spec 42: Structural-Classification Tolerance Rules (v1.0)

## Overview

This specification defines the tolerance rules used when loft classifies
structural situations such as continuity, birth/death, and containment-driven
interpretation.

## Backlink

Parent specification:

- [Loft Spec 26: Tolerance and Degeneracy Policy (v1.0)](loft-26-tolerance-and-degeneracy-policy-v1_0.md)

## Scope

This specification covers:

- event classification near structural boundaries
- containment interpretation tolerance
- hole persistence versus escape classification

## Behavior

Current structural-classification use sites include:

- region topology-case classification
- region transition ambiguity classification
- hole persistence versus synthetic birth/death classification
- containment-driven ambiguity classification

Current behavior seeding the first policy pass comes from:

- `_classify_region_topology_case`
- `_classify_region_transition_ambiguity`
- `_stable_loop_transition`
- `_point_in_polygon`

Current interpretation rule boundary:

- classification tolerances determine whether structure is treated as stable,
  synthetic birth/death, or ambiguous
- these rules are distinct from explicit collapse/degeneracy checks

## Constraints

- classification tolerances must remain distinct from collapse tolerances
- rules must support deterministic event classification
- seam/tessellation tolerances must remain out of scope for this branch

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- classification use sites are explicit
- interpretation rules are explicit
- distinction from other tolerance families is explicit
