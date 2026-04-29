# Surface Spec 46: Public/Internal API Transition and Documentation Boundary (v1.0)

## Overview

This specification defines the boundary between internal adoption of
surface-returning APIs and what is publicly exposed and documented during the
migration period.

## Backlink

Parent specification:

- [Surface Spec 15: Modeling API Surface Return-Type Adoption (v1.0)](surface-15-modeling-api-surface-return-type-adoption-v1_0.md)

## Scope

This specification covers:

- public versus internal API exposure
- transitional documentation policy
- stability expectations during the migration window

## Behavior

This branch must define:

- which APIs may change internally before public promotion
- what public documentation promises during the transition
- how hidden/internal compatibility layers are kept out of user-facing contracts

## Constraints

- the public/internal boundary must be explicit
- documentation must not get ahead of actual supported behavior
- internal experimentation must not leak as accidental public API

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one exposure boundary plus one documentation policy.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- public/internal boundaries are explicit
- user-facing documentation promises are explicit
- transition-period stability expectations are explicit

