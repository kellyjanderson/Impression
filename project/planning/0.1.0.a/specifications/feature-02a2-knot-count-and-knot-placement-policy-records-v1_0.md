# Feature Spec 02A2: Knot-Count and Knot-Placement Policy Records (v1.0)

## Overview

This specification defines the policy records that govern knot-count and
knot-placement decisions for fit-backed inference.

## Backlink

- [Feature Spec 02A: Parameterization, Knot, and Fit Configuration Records (v1.0)](feature-02a-parameterization-knot-and-fit-configuration-records-v1_0.md)

## Scope

This specification covers:

- knot-count policy
- knot-placement policy
- durable policy linkage to later fit configuration

## Behavior

This leaf must define:

- what knot-count choices are explicit
- what knot-placement choices are explicit
- how those choices remain durable and comparable across fits

## Constraints

- knot policy must not be hidden inside fitting helpers
- knot-count and knot-placement decisions must remain inspectable

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- knot-count policy records are explicit
- knot-placement policy records are explicit
- later fit branches can reference the same durable policy contract
