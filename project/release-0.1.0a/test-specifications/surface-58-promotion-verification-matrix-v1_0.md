# Surface Spec 58 Test: Promotion Verification Matrix and Evidence Burden

## Overview

This test specification defines verification for the promotion evidence matrix.

## Backlink

- [Surface Spec 58: Promotion Verification Matrix and Evidence Burden (v1.0)](../specifications/surface-58-promotion-verification-matrix-v1_0.md)

## Manual Smoke Check

- Review each verification category and its named evidence owner.
- Confirm the matrix covers kernel, tessellation, public API, loft, and
  rollback concerns.

## Automated Smoke Tests

- verification categories are explicit
- required evidence per category is explicit
- ownership is explicit

## Automated Acceptance Tests

- verification categories map to promotion criteria
- evidence burden is not left informal
- ownership is unambiguous across subsystem boundaries
