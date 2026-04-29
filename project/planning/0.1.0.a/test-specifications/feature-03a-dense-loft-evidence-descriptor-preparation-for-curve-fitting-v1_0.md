# Feature Spec 03A Test: Dense Loft Evidence Descriptor Preparation for Curve Fitting

## Overview

This test specification defines verification for dense-evidence descriptor
preparation ahead of fit-backed curve analysis.

## Backlink

- [Feature Spec 03A: Dense Loft Evidence Descriptor Preparation for Curve Fitting (v1.0)](../specifications/feature-03a-dense-loft-evidence-descriptor-preparation-for-curve-fitting-v1_0.md)

## Automated Smoke Tests

- dense loft fixtures emit descriptor records in deterministic order
- descriptor normalization preserves station ordering and correspondence meaning

## Automated Acceptance Tests

- prepared descriptor bands are replayable for identical inputs
- ordering and continuity survive descriptor preparation
- prepared evidence is consumable by later candidate-fit branches
