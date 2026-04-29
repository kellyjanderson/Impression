# Feature Spec 06A1: Descriptor Record Families for Curve-Intent Inference (v1.0)

## Overview

This specification defines the descriptor record families used by the initial
curve-intent inference lane.

## Backlink

- [Feature Spec 06A: Descriptor and Evidence Records for Curve-Intent Inference (v1.0)](feature-06a-descriptor-and-evidence-records-for-curve-intent-inference-v1_0.md)

## Scope

This specification covers:

- section descriptor records
- loop descriptor records
- correspondence-track descriptor records

## Behavior

This leaf must define:

- which descriptor families are part of the initial curve-intent lane
- what each descriptor family records
- how descriptor families remain deterministic and reusable

## Constraints

- descriptor families must remain durable enough for replay and comparison
- descriptor families must remain distinct from later evidence or candidate
  classification layers

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- descriptor record families are explicit
- family-level recorded fields are explicit
- determinism and reuse posture are explicit
