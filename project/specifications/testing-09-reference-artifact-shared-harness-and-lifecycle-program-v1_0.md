# Testing Spec 09: Reference Artifact Shared Harness and Lifecycle Program (v1.0)

## Overview

This specification defines the top-level reference-artifact tooling branch
owned by the testing architecture.

## Backlink

- [Testing Spec 01: Testing Tooling and Verification Program (v1.0)](testing-01-testing-tooling-and-verification-program-v1_0.md)

## Scope

This specification covers:

- reusable reference-image and reference-STL tooling
- dirty and clean baseline lifecycle rules
- grouped model-output completeness rules for shared artifact sets
- the boundary between top-level reference tooling and feature fixtures that
  consume it

## Behavior

This branch must define:

- the child leaves that own baseline lifecycle and grouped artifact
  completeness behavior
- the rule that reference-artifact tooling belongs to testing rather than to
  any one feature trunk

## Constraints

- reusable reference-artifact tooling must live under testing
- feature branches may use the tooling but must not own its structure

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Testing Spec 10: Reference Artifact Baseline Lifecycle and Invalidation Contract](testing-10-reference-artifact-baseline-lifecycle-and-invalidation-contract-v1_0.md)
- [Testing Spec 11: Reference Artifact Grouped Model-Output Completeness and Bootstrap Rules](testing-11-reference-artifact-grouped-model-output-completeness-and-bootstrap-rules-v1_0.md)

## Acceptance

This specification is complete when:

- reusable reference-artifact ownership is explicit at the testing level
- lifecycle and grouped completeness work are pushed into executable child
  leaves
- the parent remains a container rather than an implementation leaf
