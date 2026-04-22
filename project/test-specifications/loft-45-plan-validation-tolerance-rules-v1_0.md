# Loft Spec 45 Test: Plan-Validation Tolerance Rules

## Overview

This test specification defines verification for tolerance-sensitive plan
validation before execution.

## Backlink

- [Loft Spec 45: Plan-Validation Tolerance Rules (v1.0)](../specifications/loft-45-plan-validation-tolerance-rules-v1_0.md)

## Automated Smoke Tests

- malformed plans are rejected before execution

## Automated Acceptance Tests

- invalid sample count is rejected at plan validation
- invalid summary metadata is rejected
- closure/reference/order validation remains explicit and executor-independent
