# Loft Spec 44: Collapse and Degeneracy Tolerance Rules (v1.0)

## Overview

This specification defines the loft tolerance rules used to decide when loops
or regions are treated as collapsed or degenerate.

## Backlink

Parent specification:

- [Loft Spec 26: Tolerance and Degeneracy Policy (v1.0)](loft-26-tolerance-and-degeneracy-policy-v1_0.md)

## Scope

This specification covers:

- loop collapse
- region collapse
- synthetic birth/death closure thresholds

## Behavior

Current collapse/degeneracy decisions appear in:

- synthetic seed generation for births/deaths
- shrunken-loop collapse handling in split/merge resolution
- endcap collapse checks during endcap generation

Current structural consequences are:

- collapsed unmatched loops become `synthetic_birth` or `synthetic_death`
- collapse drives loop or region closure ownership
- severe collapse during endcap generation is rejected explicitly

Current policy seeding comes from:

- `_synthetic_seed_scale`
- `_shrunken_loop`
- endcap collapse checks in `loft_endcaps`

## Constraints

- collapse must remain a structural event, not a silent numeric artifact
- collapse rules must be explicit
- collapse rules must remain separable from seam tolerances

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- collapse conditions are explicit
- structural consequences are explicit
- initial policy seeding is explicit
