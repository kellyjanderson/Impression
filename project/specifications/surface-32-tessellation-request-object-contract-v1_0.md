# Surface Spec 32: Tessellation Request Object and Field Contract (v1.0)

## Overview

This specification defines the request object that downstream consumers use to
ask a surface body for tessellation.

## Backlink

Parent specification:

- [Surface Spec 11: Tessellation Request and Quality Contract (v1.0)](surface-11-tessellation-request-and-quality-contract-v1_0.md)

## Scope

This specification covers:

- the required fields of a tessellation request
- request-level intent declaration
- defaulting rules for omitted optional fields

## Behavior

The tessellation request contract must define:

- a stable request object shape
- required fields for consumer intent, target output class, and quality mode
- optional fields for tolerances and overrides
- deterministic defaulting before normalization occurs

## Constraints

- request fields must be serializable and cache-key friendly
- defaults must not depend on hidden caller context
- the request object must remain consumer-facing rather than executor-facing

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one durable request contract and one bounded defaulting concern.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- request fields are explicitly named and typed
- required versus optional fields are explicit
- defaulting rules are explicit and deterministic

