# Feature Spec 04B: Planner Consumption Boundary for Hidden Control Stations (v1.0)

## Overview

This specification defines how hidden control stations are consumed by loft
planning without becoming public authored API.

## Backlink

- [Feature Spec 04: Non-User-Facing Control Stations Program (v1.0)](feature-04-non-user-facing-control-stations-program-v1_0.md)

## Scope

This specification covers:

- planner-consumption boundaries
- use of hidden control stations during planning
- limits on public surface exposure

## Behavior

This leaf must define:

- where in the planner hidden control stations may influence behavior
- where they may not override topology truth
- how the public API avoids exposing them prematurely

## Constraints

- hidden control stations must not become stealth public authored inputs
- planner use must preserve topology-owned behavior

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- planner-consumption boundaries are explicit
- topology-preservation limits are explicit
- public exposure boundaries are explicit
