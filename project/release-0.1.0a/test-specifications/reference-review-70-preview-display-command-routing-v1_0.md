# Reference Review Spec 70 Test: Preview Display Command Routing

## Overview

Verify display-control commands update preview display options safely.

## Backlink

- [Reference Review Spec 70: Preview Display Command Routing](../specifications/reference-review-70-preview-display-command-routing-v1_0.md)

## Manual Smoke Check

- Toggle controls in the workbench and confirm state changes without preview reload.

## Automated Smoke Tests

- Route one color command, one lighting command, and one independent toggle command.

## Automated Acceptance Tests

- Color mode updates are exclusive.
- Lighting mode updates are exclusive.
- Independent toggles do not alter unrelated toggles.
- Unsupported commands return deterministic diagnostics.
- Disabled preview state rejects display changes.
