# Reference Review Spec 64 Test: Preview Display Icon Metadata Registry

## Overview

Verify stable metadata lookup for preview display-control icons.

## Backlink

- [Reference Review Spec 64: Preview Display Icon Metadata Registry](../specifications/reference-review-64-preview-display-icon-metadata-registry-v1_0.md)

## Manual Smoke Check

- Inspect the registry output and confirm every display control has a readable label and tooltip.

## Automated Smoke Tests

- Registry accessor returns all expected ids.
- Every registry record points to an existing packaged SVG.

## Automated Acceptance Tests

- Records include id, resource path, tooltip, and accessible name.
- Missing ids fail deterministically.
- Consumers can resolve records without hard-coding filesystem paths.
