# Reference Review Spec 66 Test: Icon Toggle Button Interaction Contract

## Overview

Verify reusable icon toggle activation, command emission, and accessibility.

## Backlink

- [Reference Review Spec 66: Icon Toggle Button Interaction Contract](../specifications/reference-review-66-icon-toggle-button-interaction-contract-v1_0.md)

## Manual Smoke Check

- Activate an enabled icon toggle and confirm the owner receives the command.
- Confirm tooltip and accessible name match the intended command.

## Automated Smoke Tests

- Enabled activation emits one command.
- Disabled activation emits no command.

## Automated Acceptance Tests

- Tooltip and accessible name are set from supplied properties.
- Checked/enabled state can be owner-controlled.
- The component does not import preview-renderer modules.
