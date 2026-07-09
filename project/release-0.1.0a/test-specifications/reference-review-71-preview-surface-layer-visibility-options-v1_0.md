# Reference Review Spec 71 Test: Preview Surface Layer Visibility Options

## Overview

Verify layer visibility options affect repaint output without cache churn.

## Backlink

- [Reference Review Spec 71: Preview Surface Layer Visibility Options](../specifications/reference-review-71-preview-surface-layer-visibility-options-v1_0.md)

## Manual Smoke Check

- Toggle fill, object edges, triangle wireframe, grid, axes, background, and polylines on a real fixture.

## Automated Smoke Tests

- Render a fixture with default layer options and with all overlays disabled.

## Automated Acceptance Tests

- Each layer option independently affects projected or painted scene output.
- Object edges and triangle wireframe can be enabled together.
- Fill can be disabled while overlays remain visible.
- Layer option changes do not recompute object-edge topology.
