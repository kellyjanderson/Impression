# Surface Spec 90: Metadata Inheritance and Override Rules (v1.0)

## Overview

This specification defines how metadata inherited from higher structural levels
is overridden at lower levels.

## Backlink

Parent specification:

- [Surface Spec 30: Surface Metadata Placement Contract (v1.0)](surface-30-surface-metadata-placement-contract-v1_0.md)

## Scope

This specification covers:

- inheritance rules
- override rules
- shadowing and merge behavior

## Behavior

This branch must define:

- which metadata participates in inheritance
- how lower levels override or merge higher-level values
- how absent values are resolved through the structural hierarchy

## Constraints

- inheritance behavior must be explicit
- override semantics must be deterministic
- merge behavior must be bounded and testable

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- inheritance rules are explicit
- override semantics are explicit
- merge/shadowing behavior is explicit

