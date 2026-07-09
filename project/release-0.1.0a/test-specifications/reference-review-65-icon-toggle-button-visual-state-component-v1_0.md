# Reference Review Spec 65 Test: Icon Toggle Button Visual State Component

## Overview

Verify reusable icon toggle visual states.

## Backlink

- [Reference Review Spec 65: Icon Toggle Button Visual State Component](../specifications/reference-review-65-icon-toggle-button-visual-state-component-v1_0.md)

## Manual Smoke Check

- Inspect off, hover, pressed, checked, focused, and disabled states in the component gallery.

## Automated Smoke Tests

- Construct the component with a known icon id.
- Toggle checked state and assert the selected state is observable.

## Automated Acceptance Tests

- State changes do not resize the component.
- Checked state is visible without relying only on color.
- Disabled and focus states are represented.
- No text label is visible inside the icon button.
