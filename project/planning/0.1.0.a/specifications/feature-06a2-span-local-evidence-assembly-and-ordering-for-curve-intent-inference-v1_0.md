# Feature Spec 06A2: Span-Local Evidence Assembly and Ordering for Curve-Intent Inference (v1.0)

## Overview

This specification defines how curve-intent evidence is assembled from
descriptor records for span-local inference.

## Backlink

- [Feature Spec 06A: Descriptor and Evidence Records for Curve-Intent Inference (v1.0)](feature-06a-descriptor-and-evidence-records-for-curve-intent-inference-v1_0.md)

## Scope

This specification covers:

- span-local curve-intent evidence records
- normalization of evidence built from descriptor records
- evidence ordering and carry-forward into later candidate classification

## Behavior

This leaf must define:

- how span-local evidence is assembled from descriptor inputs
- how evidence is normalized and ordered
- how later candidate-classification leaves consume the same evidence shape

## Constraints

- evidence ordering must remain deterministic
- evidence assembly must not mutate descriptor record definitions

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- span-local evidence shape is explicit
- normalization and ordering are explicit
- downstream candidate consumption contract is explicit
