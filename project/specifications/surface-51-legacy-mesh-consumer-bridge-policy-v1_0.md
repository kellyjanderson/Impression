# Surface Spec 51: Legacy Mesh Consumer Bridge Policy (v1.0)

## Overview

This specification defines which legacy mesh consumers remain supported during
the migration and how they are routed through compatibility adapters.

## Backlink

Parent specification:

- [Surface Spec 17: Compatibility Adapter Contracts (v1.0)](surface-17-compatibility-adapter-contracts-v1_0.md)

## Scope

This specification covers:

- supported legacy consumer classes
- bridge visibility and ownership
- removal expectations for each bridge class

## Behavior

This branch must define:

- which legacy consumers receive bridge support
- whether bridges are internal or publicly visible
- what removal trigger or sunset condition each bridge has

## Constraints

- supported legacy consumers must be explicitly listed
- bridge visibility must be explicit
- no bridge may exist without a sunset condition

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one bounded bridge-support policy.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- supported legacy consumer classes are explicit
- bridge visibility is explicit
- sunset conditions are explicit

