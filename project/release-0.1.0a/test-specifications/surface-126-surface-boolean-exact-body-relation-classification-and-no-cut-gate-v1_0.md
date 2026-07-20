# Surface Spec 126 Test: Surface Boolean Exact Body-Relation Classification and No-Cut Gate

## Overview

This test specification defines verification for exact body-relation
classification and the surfaced no-cut gate.

## Backlink

- [Surface Spec 126: Surface Boolean Exact Body-Relation Classification and No-Cut Gate (v1.0)](../specifications/surface-126-surface-boolean-exact-body-relation-classification-and-no-cut-gate-v1_0.md)

## Automated Smoke Tests

- representative disjoint, touching, equal, and containment inputs produce deterministic body-relation records
- eligible no-cut cases short-circuit surfaced execution without mesh fallback

## Automated Acceptance Tests

- exact-containment gating remains bounded to the supported initial families
- unsupported containment inference remains explicit surfaced unsupported behavior
- equal request state yields identical no-cut gate results
