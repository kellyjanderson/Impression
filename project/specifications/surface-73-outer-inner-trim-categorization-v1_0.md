# Surface Spec 73: Outer and Inner Trim Categorization Rules (v1.0)

## Overview

This specification defines how trim loops are categorized as outer boundaries
or inner holes.

## Backlink

Parent specification:

- [Surface Spec 24: Trim-Loop Representation and Ownership (v1.0)](surface-24-trim-loop-representation-v1_0.md)

## Scope

This specification covers:

- outer versus inner trim classification
- classification invariants
- categorization visibility to downstream systems

## Behavior

This branch must define:

- how outer trims are distinguished from inner trims
- what invariants outer and inner categories must satisfy
- how downstream tessellation and cap logic consume the categorization

## Constraints

- categorization must be explicit
- classification must not be inferred ad hoc
- downstream consumers must not need to reclassify trims independently

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- categorization rules are explicit
- category invariants are explicit
- downstream visibility of categories is explicit

