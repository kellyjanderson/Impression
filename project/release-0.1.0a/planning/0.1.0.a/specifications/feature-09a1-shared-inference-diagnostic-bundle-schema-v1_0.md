# Feature Spec 09A1: Shared Inference Diagnostic Bundle Schema (v1.0)

## Overview

This specification defines the schema of the shared diagnostic bundle used by
`0.1.0.a` inference features.

## Backlink

- [Feature Spec 09A: Shared Inference Diagnostic Bundle Structure (v1.0)](feature-09a-shared-inference-diagnostic-bundle-structure-v1_0.md)

## Scope

This specification covers:

- retained vs dropped station explanation fields
- fit drift fields
- structural preservation fields
- inferred trajectory and curve evidence references

## Behavior

This leaf must define:

- what shared diagnostic bundles contain
- how field shape stays stable across inference features
- how the schema remains durable enough for replay and testing

## Constraints

- bundle schema must stay reusable across multiple inference branches
- schema shape must remain stable enough for comparison tooling

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- bundle schema is explicit
- field stability posture is explicit
- replay and testing durability are explicit
