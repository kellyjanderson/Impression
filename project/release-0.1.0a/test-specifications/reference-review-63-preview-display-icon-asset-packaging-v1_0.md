# Reference Review Spec 63 Test: Preview Display Icon Asset Packaging

## Overview

Verify generated preview display-control SVG assets are packaged as durable UI resources.

## Backlink

- [Reference Review Spec 63: Preview Display Icon Asset Packaging](../specifications/reference-review-63-preview-display-icon-asset-packaging-v1_0.md)

## Manual Smoke Check

- Open the generated SVG files and confirm they render as small line icons.

## Automated Smoke Tests

- Resource layout check includes every preview display-control SVG.
- Package data includes `qml/icons/preview-display/*.svg`.

## Automated Acceptance Tests

- Every required icon file exists.
- Every icon parses as SVG XML.
- Every icon uses `viewBox="0 0 24 24"`.
- Every icon uses `currentColor`.
