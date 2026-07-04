# Feature Spec 07B: Explicit Shared Guidance Attachment and Consumption Rules (v1.0)

## Overview

This specification defines the explicit shared-guidance branch inside loft.

## Backlink

- [Feature Spec 07: Shared Trajectory Inference and Guidance Program (v1.0)](feature-07-shared-trajectory-inference-and-guidance-program-v1_0.md)

## Scope

This specification covers:

- explicit shared guidance input
- attachment records
- planner consumption boundaries

## Behavior

This branch must define:

- the leaf that owns explicit shared-guidance attachment records
- the leaf that owns planner consumption boundaries for explicit shared guidance

## Constraints

- explicit shared guidance must remain subordinate to topology-aware planning
- attachment resolution must remain deterministic

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Feature Spec 07B1: Explicit Shared Guidance Attachment Record Contract](feature-07b1-explicit-shared-guidance-attachment-record-contract-v1_0.md)
- [Feature Spec 07B2: Planner Consumption Boundaries for Explicit Shared Guidance](feature-07b2-planner-consumption-boundaries-for-explicit-shared-guidance-v1_0.md)

## Acceptance

This specification is complete when:

- explicit shared-guidance attachment and planner-consumption work are split
  into executable leaves
- deterministic attachment rules remain explicit
