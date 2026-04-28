# Feature Spec 07B2: Planner Consumption Boundaries for Explicit Shared Guidance (v1.0)

## Overview

This specification defines how explicit shared guidance is consumed by loft
planning once attached.

## Backlink

- [Feature Spec 07B: Explicit Shared Guidance Attachment and Consumption Rules (v1.0)](feature-07b-explicit-shared-guidance-attachment-and-consumption-rules-v1_0.md)

## Scope

This specification covers:

- planner consumption boundaries
- influence of explicit shared guidance on in-between travel
- protection of topology-owned planning behavior

## Behavior

This leaf must define:

- how explicit shared guidance influences in-between travel
- where planner consumption stops
- how guidance remains subordinate to topology truth

## Constraints

- planner consumption must remain deterministic
- explicit shared guidance must not override topology-owned planning behavior

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- planner-consumption boundaries are explicit
- subordination to topology truth is explicit
- deterministic consumption rules are explicit
