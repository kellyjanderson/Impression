# Loft Spec 39: Constraint Request Record Contract (v1.0)

## Overview

This specification defines the record used when next-generation loft requests
minimal additional directional correspondence from the user.

## Backlink

Parent specification:

- [Loft Spec 25: Ambiguity, Constraint Request, and Diagnostic Surface (v1.0)](loft-25-ambiguity-constraint-request-and-diagnostic-surface-v1_0.md)

## Scope

This specification covers:

- requested predecessor/successor ties
- request target location
- minimal-fix guidance payload

## Behavior

Constraint requests are currently represented by `LoftConstraintRequest`.

Current fields are:

- `interval`
- `topology_state_index`
- `ambiguous_region_indices`
- `requested_ties`
- `ambiguity_class`
- optional `relationship_group`

Requested directional ties are currently represented minimally as the pair:

- `predecessor_ids`
- `successor_ids`

The request points back to the blocked topology and regions through the same
locator fields used by `LoftAmbiguityRecord`.

## Constraints

- request records must remain minimal-fix oriented
- request records must not over-prescribe user edits
- request payloads must be deterministic

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- request fields are explicit
- directional-tie representation is explicit
- blocked-structure locator semantics are explicit
