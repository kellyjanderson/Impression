# Feature Spec 06A Test: Descriptor and Evidence Records for Curve-Intent Inference

## Overview

This test specification defines verification for descriptor and evidence records
used by curve-intent inference.

## Backlink

- [Feature Spec 06A: Descriptor and Evidence Records for Curve-Intent Inference (v1.0)](../specifications/feature-06a-descriptor-and-evidence-records-for-curve-intent-inference-v1_0.md)

## Behavior

This parent test branch must verify:

- no executable descriptor-record-family work remains hidden in the parent
- no executable span-local evidence assembly or ordering work remains hidden in
  the parent
- every final child leaf has a paired test specification

## Child Test Specifications

- [Feature Spec 06A1 Test: Descriptor Record Families for Curve-Intent Inference](feature-06a1-descriptor-record-families-for-curve-intent-inference-v1_0.md)
- [Feature Spec 06A2 Test: Span-Local Evidence Assembly and Ordering for Curve-Intent Inference](feature-06a2-span-local-evidence-assembly-and-ordering-for-curve-intent-inference-v1_0.md)

## Acceptance

This test specification is complete when the child set fully covers the branch.
