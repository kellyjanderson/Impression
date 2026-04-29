# Loft Spec 49: Residual Many-to-Many Constraint Escalation (v1.0)

## Overview

This specification defines how unresolved many-to-many residuals escalate into
constraint requests once automatic decomposition stops.

## Backlink

Parent specification:

- [Loft Spec 27: Many-to-Many Decomposition and Automatic Decomposability Gate (v1.0)](loft-27-many-to-many-decomposition-and-decomposability-gate-v1_0.md)

## Scope

This specification covers:

- residual unresolved region reporting
- many-to-many constraint-request escalation
- handoff to the ambiguity/diagnostic surface

## Behavior

This branch must define:

- how residual unresolved regions are surfaced
- how many-to-many residuals become constraint-request records
- what information is handed to the ambiguity/diagnostic branch

## Constraints

- escalation must remain explicit
- executor must never receive unresolved many-to-many structure
- escalation output must remain compatible with the general ambiguity surface

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- residual reporting rules are explicit
- escalation record rules are explicit
- compatibility with general ambiguity handling is explicit
- blocked planning exposes many-to-many locator data through the structured ambiguity surface
