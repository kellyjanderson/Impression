# Surface Spec 48: Composition Flattening and Traversal Rules (v1.0)

## Overview

This specification defines when composed structures remain nested and when they
are flattened for downstream consumers.

## Backlink

Parent specification:

- [Surface Spec 16: Surface Composition and Consumer Handoff Rules (v1.0)](surface-16-surface-composition-consumer-handoff-v1_0.md)

## Scope

This specification covers:

- flattening triggers
- nested composition traversal rules
- deterministic flattening order

## Behavior

This branch must define:

- when flattening is required
- when nested structures must be preserved
- how flattened output ordering is derived from nested composition

## Constraints

- flattening must be deterministic
- flattening must not destroy metadata or identity unexpectedly
- preservation versus flattening must not be decided ad hoc per consumer

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
isolates one flattening/traversal policy.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- flattening triggers are explicit
- preservation cases are explicit
- flattened ordering rules are explicit

