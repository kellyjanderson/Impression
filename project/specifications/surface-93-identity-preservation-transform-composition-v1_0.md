# Surface Spec 93: Identity Preservation Through Transform and Composition (v1.0)

## Overview

This specification defines how object identity survives transformation,
grouping, copying, and composition.

## Backlink

Parent specification:

- [Surface Spec 31: Stable Identity and Caching Keys for Surface Objects (v1.0)](surface-31-stable-identity-caching-keys-v1_0.md)

## Scope

This specification covers:

- transform-time identity preservation
- composition/grouping identity preservation
- copy and clone identity behavior

## Behavior

This branch must define:

- when identity is preserved unchanged
- when new identity must be minted
- how grouping and transforms affect the identity of contained objects

## Constraints

- preservation rules must be explicit
- new-identity triggers must be explicit
- grouping and transform behavior must be deterministic

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- identity preservation rules are explicit
- new-identity triggers are explicit
- grouping/transform effects on identity are explicit

