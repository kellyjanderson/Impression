# Reference Review Spec 72 Test: Preview Surface Color And Lighting Options

## Overview

Verify preview surface color and lighting modes.

## Backlink

- [Reference Review Spec 72: Preview Surface Color And Lighting Options](../specifications/reference-review-72-preview-surface-color-and-lighting-options-v1_0.md)

## Manual Smoke Check

- Switch authored/inspection color and all lighting modes on a real fixture.

## Automated Smoke Tests

- Render one mesh in inspection color mode and authored color mode.
- Render one mesh in flat, face-normal, and camera-light modes.

## Automated Acceptance Tests

- Authored mesh colors are used when present.
- Authored face colors are used when present.
- Authored mode falls back to inspection color when no authored data exists.
- Lighting changes do not recompute object-edge topology.
