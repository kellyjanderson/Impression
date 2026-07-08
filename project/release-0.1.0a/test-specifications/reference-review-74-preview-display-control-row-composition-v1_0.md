# Reference Review Spec 74 Test: Preview Display Control Row Composition

## Overview

Verify the composed preview display-control row.

## Backlink

- [Reference Review Spec 74: Preview Display Control Row Composition](../specifications/reference-review-74-preview-display-control-row-composition-v1_0.md)

## Manual Smoke Check

- Launch the workbench with a real `.impress` fixture and confirm the row order,
  separators, hover tooltips, and enabled controls.

## Automated Smoke Tests

- Shell/component test finds color group, lighting group, separators, and independent toggles.
- Ready-state test confirms controls become enabled when preview payload is ready.

## Automated Acceptance Tests

- Control order matches the product definition.
- Group controls use exclusive icon group components.
- Independent controls use icon toggle components.
- Disabled state applies when no preview payload is ready.
- Commands route through preview display command routing.
