# Loft Spec 30 Test: Directional Correspondence Field Contract

## Overview

This test specification defines verification for authored directional
correspondence input fields.

## Backlink

- [Loft Spec 30: Directional Correspondence Field Contract (v1.0)](../specifications/loft-30-directional-correspondence-field-contract-v1_0.md)

## Automated Smoke Tests

- `predecessor_ids` and `successor_ids` are explicit on `Station`
- directional correspondence aligns to normalized region order

## Automated Acceptance Tests

- correspondence without topology is rejected
- arity mismatch against normalized region count is rejected
- directional correspondence remains relationship-first rather than
  standalone-id-first
