# Feature Spec 06A2 Test: Span-Local Evidence Assembly and Ordering for Curve-Intent Inference

## Overview

This test specification defines verification for span-local evidence assembly
and ordering used by curve-intent inference.

## Backlink

- [Feature Spec 06A2: Span-Local Evidence Assembly and Ordering for Curve-Intent Inference (v1.0)](../specifications/feature-06a2-span-local-evidence-assembly-and-ordering-for-curve-intent-inference-v1_0.md)

## Automated Smoke Tests

- span-local evidence records remain inspectable
- identical descriptor inputs produce identical evidence ordering

## Automated Acceptance Tests

- span-local evidence shape is stable for identical inputs
- normalization and ordering remain explicit
- later candidate-classification branches can consume the same evidence shape
