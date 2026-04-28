# Feature Spec 09A1 Test: Shared Inference Diagnostic Bundle Schema

## Overview

This test specification defines verification for the shared inference
diagnostic bundle schema.

## Backlink

- [Feature Spec 09A1: Shared Inference Diagnostic Bundle Schema (v1.0)](../specifications/feature-09a1-shared-inference-diagnostic-bundle-schema-v1_0.md)

## Automated Smoke Tests

- shared diagnostic bundles expose retained/dropped, fit drift, structural
  preservation, and evidence-reference fields
- schema shape remains inspectable and reusable

## Automated Acceptance Tests

- bundle schema remains stable across inference branches
- schema is durable enough for replay and testing
- schema supports later reporting consumers without branch-specific mutation
