# Surface Spec 34: Canonical Executor Input and Request Normalization (v1.0)

## Overview

This specification defines the canonical normalized request shape that the
tessellation executor receives after consumer-facing inputs are resolved.

## Backlink

Parent specification:

- [Surface Spec 11: Tessellation Request and Quality Contract (v1.0)](surface-11-tessellation-request-and-quality-contract-v1_0.md)

## Scope

This specification covers:

- normalized executor input fields
- canonical ordering and representation rules
- request canonicalization for caching and deterministic execution

## Behavior

This branch must define:

- the executor-facing normalized request type
- canonical field ordering and value normalization
- which consumer-level distinctions are erased before execution
- which normalized fields participate in cache keys

## Constraints

- canonicalization must be deterministic
- executor input must not preserve ambiguous aliases from caller input
- cache-relevant normalization must be stable across processes

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
focuses on one executor-facing contract plus canonicalization rules.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the normalized executor input type is explicit
- canonicalization rules are explicit
- cache-key relevant fields are explicit

