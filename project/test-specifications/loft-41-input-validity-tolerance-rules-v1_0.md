# Loft Spec 41 Test: Input-Validity Tolerance Rules

## Overview

This test specification defines verification for loft input-validity rules.

## Backlink

- [Loft Spec 41: Input-Validity Tolerance Rules (v1.0)](../specifications/loft-41-input-validity-tolerance-rules-v1_0.md)

## Automated Smoke Tests

- invalid station ordering and planner controls are rejected before planning

## Automated Acceptance Tests

- minimum samples and progression ordering are enforced
- invalid control ranges are rejected
- malformed input is blocked before structural interpretation
