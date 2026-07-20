# Reference Review Spec 73 Test: Preview Pane Display Control Slot

## Overview

Verify the preview pane exposes a stable display-control slot instead of the old title.

## Backlink

- [Reference Review Spec 73: Preview Pane Display Control Slot](../specifications/reference-review-73-preview-pane-display-control-slot-v1_0.md)

## Manual Smoke Check

- Launch the workbench and confirm `Selected Fixture` no longer appears above the preview.

## Automated Smoke Tests

- Shell test finds the display-control slot.
- Shell test confirms the old title label is absent.

## Automated Acceptance Tests

- The slot remains visible during empty, loading, ready, and failed preview states.
- Preview surface geometry remains stable when the slot state changes.
