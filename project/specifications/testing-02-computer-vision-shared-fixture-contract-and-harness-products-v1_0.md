# Testing Spec 02: Computer Vision Shared Fixture Contract and Harness Products (v1.0)

## Overview

This specification defines the shared-fixture and harness parent branch for the
architecture-level CV verification tooling.

## Backlink

- [Testing Spec 01: Testing Tooling and Verification Program (v1.0)](testing-01-testing-tooling-and-verification-program-v1_0.md)

## Scope

This specification covers:

- the minimum fixture schema for CV-backed verification
- shared harness layers such as builder, renderer, normalization, interpreter,
  decision, and review publication
- required deterministic harness products by lane
- shared result-taxonomy posture and uncertainty handling
- grouped artifact completeness and contract invalidation rules

## Behavior

This branch must define:

- the minimum fields a CV fixture must declare, including fixture identity,
  authoritative lane, expected result classes, and pass/fail mapping
- the reusable harness layers that turn a fixture contract into deterministic
  CV input artifacts
- how lane-specific result classes fit a common pattern of positive,
  transformed, different, and unknown/unreadable outcomes
- how grouped harness products participate in reference-artifact lifecycle
  rules without silently weakening completeness
- when a changed fixture meaning invalidates existing CV-backed reference
  baselines

## Constraints

- CV-backed fixtures must use deterministic harness products rather than
  arbitrary ad hoc renders
- incomplete artifact groups must fail rather than silently degrade
- uncertainty, unreadable, or ambiguous outcomes must stay explicit
- lane-specific contracts may extend the shared taxonomy but must not bypass
  the shared decision posture
- this leaf defines tooling contracts and facilitation rules, not feature-level
  product behavior

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Testing Spec 12: Computer Vision Shared Fixture Contract and Result Taxonomy](testing-12-computer-vision-shared-fixture-contract-and-result-taxonomy-v1_0.md)
- [Testing Spec 13: Computer Vision Shared Harness Pipeline and Artifact Bundle Integration](testing-13-computer-vision-shared-harness-pipeline-and-artifact-bundle-integration-v1_0.md)

## Acceptance

This specification is complete when:

- the shared CV tooling concerns are separated into honest executable child
  leaves
- the parent remains a container rather than an executable implementation leaf
- verification requirements are pushed down into the paired child test
  specifications
