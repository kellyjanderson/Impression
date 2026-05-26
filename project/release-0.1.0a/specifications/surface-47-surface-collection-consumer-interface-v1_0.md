# Surface Spec 47: Surface Collection Consumer Interface (v1.0)

## Overview

This specification defines the interface by which preview, export, and analysis
consumers receive one or more surface-native objects.

## Backlink

Parent specification:

- [Surface Spec 16: Surface Composition and Consumer Handoff Rules (v1.0)](surface-16-surface-composition-consumer-handoff-v1_0.md)

## Scope

This specification covers:

- consumer-facing surface collection shape
- ordering guarantees for handed-off collections
- collection-level metadata and identity expectations

## Behavior

This branch must define:

- whether consumers receive one body, a flat collection, or a composite wrapper
- what ordering guarantees the interface preserves
- what metadata and identity information accompanies the collection

## Constraints

- collection shape must be explicit
- consumers must not need to inspect scene internals directly
- ordering and metadata guarantees must be deterministic

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one consumer interface contract.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the consumer interface shape is explicit
- collection ordering guarantees are explicit
- collection metadata/identity guarantees are explicit

