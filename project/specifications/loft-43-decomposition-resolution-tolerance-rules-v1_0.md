# Loft Spec 43: Decomposition-Resolution Tolerance Rules (v1.0)

## Overview

This specification defines the loft tolerance rules that govern decomposition
progress, residual ambiguity gates, and planner search limits.

## Backlink

Parent specification:

- [Loft Spec 26: Tolerance and Degeneracy Policy (v1.0)](loft-26-tolerance-and-degeneracy-policy-v1_0.md)

## Scope

This specification covers:

- residual ambiguity gates
- candidate enumeration limits
- decomposition-resolution planner controls

## Behavior

Current decomposition-resolution controls include:

- `split_merge_steps`
- `split_merge_bias`
- `ambiguity_mode`
- `ambiguity_cost_profile`
- `ambiguity_max_branches`

These controls influence residual ambiguity escalation by:

- bounding candidate enumeration
- selecting whether ambiguous intervals auto-resolve or block
- controlling the staged synthetic seed/decomposition path

Current policy seeding comes from:

- `_pair_sections_for_transition`
- `_enumerate_region_assignments`
- `_synthetic_seed_scale`
- `_raise_structured_ambiguity_error`

## Constraints

- planner limits must remain policy, not hidden guessing
- decomposition-resolution rules must stay deterministic
- ambiguity escalation must remain explicit

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- decomposition-resolution controls are explicit
- escalation triggers are explicit
- current policy seeding is explicit
