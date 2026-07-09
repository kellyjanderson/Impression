# Feature Spec 07B Test: Explicit Shared Guidance Attachment and Consumption Rules

## Overview

This test specification defines verification for the decomposed explicit shared
guidance branch.

## Backlink

- [Feature Spec 07B: Explicit Shared Guidance Attachment and Consumption Rules (v1.0)](../specifications/feature-07b-explicit-shared-guidance-attachment-and-consumption-rules-v1_0.md)

## Behavior

This parent test branch must verify:

- no executable attachment-record work remains hidden in the parent
- no executable planner-consumption boundary work remains hidden in the parent
- every final child leaf has a paired test specification

## Child Test Specifications

- [Feature Spec 07B1 Test: Explicit Shared Guidance Attachment Record Contract](feature-07b1-explicit-shared-guidance-attachment-record-contract-v1_0.md)
- [Feature Spec 07B2 Test: Planner Consumption Boundaries for Explicit Shared Guidance](feature-07b2-planner-consumption-boundaries-for-explicit-shared-guidance-v1_0.md)

## Acceptance

This test specification is complete when the child set fully covers the branch.
