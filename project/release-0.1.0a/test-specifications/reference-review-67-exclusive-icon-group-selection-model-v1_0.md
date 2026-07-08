# Reference Review Spec 67 Test: Exclusive Icon Group Selection Model

## Overview

Verify reusable exactly-one icon option selection state.

## Backlink

- [Reference Review Spec 67: Exclusive Icon Group Selection Model](../specifications/reference-review-67-exclusive-icon-group-selection-model-v1_0.md)

## Manual Smoke Check

- Review a three-option group model and confirm one option is selected.

## Automated Smoke Tests

- Build a three-option model and select each option in turn.

## Automated Acceptance Tests

- Exactly one option is selected when enabled.
- Unknown option ids fail deterministically.
- Disabled state prevents selection changes.
- Option order is preserved.
