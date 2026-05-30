# Surface Spec 49: Composition-to-Tessellation Invocation Boundary (v1.0)

## Overview

This specification defines the exact system boundary where composition ends and
tessellation begins.

## Backlink

Parent specification:

- [Surface Spec 16: Surface Composition and Consumer Handoff Rules (v1.0)](surface-16-surface-composition-consumer-handoff-v1_0.md)

## Scope

This specification covers:

- invocation ownership for tessellation
- composition-layer responsibilities prior to invocation
- prohibited hidden tessellation inside composition logic

## Behavior

This branch must define:

- who is allowed to invoke tessellation
- what preparation the composition layer must finish first
- what data is handed over at the invocation point

## Constraints

- the invocation boundary must be explicit
- composition code must not perform hidden eager tessellation
- tessellation entry must receive fully prepared but still surface-native input

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one invocation boundary contract.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- tessellation invocation ownership is explicit
- pre-invocation composition responsibilities are explicit
- handoff payload at the boundary is explicit

