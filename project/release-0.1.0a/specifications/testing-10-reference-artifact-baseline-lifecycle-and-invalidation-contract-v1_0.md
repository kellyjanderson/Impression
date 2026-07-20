# Testing Spec 10: Reference Artifact Baseline Lifecycle and Invalidation Contract (v1.0)

## Overview

This specification defines the shared baseline-lifecycle contract for reference
images and reference STL files.

## Backlink

- [Testing Spec 09: Reference Artifact Shared Harness and Lifecycle Program (v1.0)](testing-09-reference-artifact-shared-harness-and-lifecycle-program-v1_0.md)

## Scope

This specification covers:

- dirty versus clean baseline meaning
- baseline selection rules during test execution
- invalidation rules when fixture meaning changes
- promotion boundaries between dirty and clean

## Behavior

This leaf must define:

- when tests compare against clean versus dirty references
- what kinds of fixture-contract change invalidate existing baselines
- how invalid baselines are removed before a new dirty bootstrap
- what dirty references may prove and what they may not prove

## Constraints

- dirty references must remain change detectors rather than truth claims
- clean promotion requires explicit human review
- contract changes must not silently reuse stale baselines

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- dirty and clean lifecycle meaning is explicit
- selection and invalidation rules are explicit
- promotion boundaries are explicit
- verification requirements are defined by its paired test specification
