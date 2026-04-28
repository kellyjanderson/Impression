# Feature Spec 06A: Descriptor and Evidence Records for Curve-Intent Inference (v1.0)

## Overview

This specification defines the durable evidence records used for curve-intent
inference.

## Backlink

- [Feature Spec 06: Curve-Intent Inference Program (v1.0)](feature-06-curve-intent-inference-program-v1_0.md)

## Scope

This specification covers:

- descriptor record families
- span-local curve-intent evidence records

## Behavior

This leaf must define:

- the child leaf that owns descriptor record-family contracts
- the child leaf that owns span-local evidence assembly, normalization, and
  ordering

## Constraints

- descriptor evidence must remain deterministic
- evidence must be durable enough for later comparison and replay

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Feature Spec 06A1: Descriptor Record Families for Curve-Intent Inference](feature-06a1-descriptor-record-families-for-curve-intent-inference-v1_0.md)
- [Feature Spec 06A2: Span-Local Evidence Assembly and Ordering for Curve-Intent Inference](feature-06a2-span-local-evidence-assembly-and-ordering-for-curve-intent-inference-v1_0.md)

## Acceptance

This specification is complete when:

- descriptor record families are explicit
- span-local evidence assembly and ordering are explicit
- later candidate classification can consume the same evidence contract
