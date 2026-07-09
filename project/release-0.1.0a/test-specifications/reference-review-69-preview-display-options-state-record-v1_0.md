# Reference Review Spec 69 Test: Preview Display Options State Record

## Overview

Verify preview display option defaults and deterministic updates.

## Backlink

- [Reference Review Spec 69: Preview Display Options State Record](../specifications/reference-review-69-preview-display-options-state-record-v1_0.md)

## Manual Smoke Check

- Inspect default display options and confirm they match the product definition.

## Automated Smoke Tests

- Construct default options and assert all defaults.

## Automated Acceptance Tests

- Copy/update helpers change only requested fields.
- Invalid color or lighting mode values fail deterministically.
- Equality comparisons support routing tests.
