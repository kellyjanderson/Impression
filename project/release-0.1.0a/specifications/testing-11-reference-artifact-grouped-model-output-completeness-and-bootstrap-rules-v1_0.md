# Testing Spec 11: Reference Artifact Grouped Model-Output Completeness and Bootstrap Rules (v1.0)

## Overview

This specification defines the shared completeness and bootstrap rules for
model-output fixture artifact groups.

## Backlink

- [Testing Spec 09: Reference Artifact Shared Harness and Lifecycle Program (v1.0)](testing-09-reference-artifact-shared-harness-and-lifecycle-program-v1_0.md)

## Scope

This specification covers:

- default grouped artifact sets such as image plus STL
- explicit exception posture when one artifact type is legitimately absent
- first-run bootstrap for new named fixtures
- failure posture when an existing fixture is missing only part of its artifact
  group

## Behavior

This leaf must define:

- which artifact groups are complete by default for model-output work
- how the first run bootstraps a brand-new fixture group
- how partial artifact loss differs from genuine first-run bootstrap
- how grouped completeness interacts with optional section or diagnostic
  artifacts

## Constraints

- model-output completeness must default to both image and STL unless a durable
  exception is documented
- partial artifact sets for an existing fixture must fail clearly
- bootstrap must not silently promote anything to clean

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- grouped completeness rules are explicit
- first-run bootstrap versus partial-loss failure is explicit
- model-output default artifact expectations are explicit
- verification requirements are defined by its paired test specification
