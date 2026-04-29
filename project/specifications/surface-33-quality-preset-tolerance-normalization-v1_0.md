# Surface Spec 33: Quality Presets and Explicit Tolerance Normalization (v1.0)

## Overview

This specification defines how human-facing tessellation quality presets and
explicit numeric tolerances are reconciled into one coherent request policy.

## Backlink

Parent specification:

- [Surface Spec 11: Tessellation Request and Quality Contract (v1.0)](surface-11-tessellation-request-and-quality-contract-v1_0.md)

## Scope

This specification covers:

- named quality presets
- explicit tolerance override semantics
- conflict resolution between preset and numeric inputs

## Behavior

This branch must define:

- the supported named quality presets
- which numeric controls each preset implies
- precedence rules when callers also provide explicit tolerances
- invalid or contradictory request handling

## Constraints

- precedence rules must be deterministic
- quality presets must remain user-facing conveniences rather than hidden magic
- numeric tolerances must always be preserved explicitly in normalized form

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one normalization policy for one request contract.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- preset names and meanings are explicit
- preset-versus-override precedence is explicit
- contradictory inputs have deterministic outcomes

