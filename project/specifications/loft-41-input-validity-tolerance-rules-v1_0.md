# Loft Spec 41: Input-Validity Tolerance Rules (v1.0)

## Overview

This specification defines the loft tolerance rules used to decide whether
planner input is valid enough to enter deterministic interpretation.

## Backlink

Parent specification:

- [Loft Spec 26: Tolerance and Degeneracy Policy (v1.0)](loft-26-tolerance-and-degeneracy-policy-v1_0.md)

## Scope

This specification covers:

- minimum loop validity
- finite-coordinate requirements
- progression-order validity
- planner-control validity ranges

## Behavior

Current pre-planning validity checks include:

- `samples >= 3`
- at least two stations
- strictly increasing station progression `t`
- finite station frame vectors
- valid orthonormal station frame
- valid split/merge controls
- valid ambiguity controls
- valid fairness and skeleton controls
- non-degenerate normalized profile/section topology

Immediate rejection cases currently include:

- non-finite or degenerate input geometry
- non-monotonic progression ordering
- invalid planner control ranges
- malformed directional correspondence arity

Initial policy seeding from current loft comes from:

- `_validate_section_stations`
- `_validate_station_frame`
- `_validate_split_merge_controls`
- `_validate_ambiguity_*`
- `_validate_fairness_*`
- `_normalize_profile_inputs`

## Constraints

- invalid input must be rejected before structural interpretation
- rules must remain deterministic
- loft-local validity rules must not depend on mesh-era assumptions

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- pre-planning validity rules are explicit
- rejection conditions are explicit
- initial policy seeding from current loft is explicit
