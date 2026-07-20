# Surface Spec 57 Test: Mesh-First Decommission and Rollback Policy

## Overview

This test specification defines verification for mesh-first decommission
triggers and rollback posture.

## Backlink

- [Surface Spec 57: Mesh-First Decommission and Rollback Policy (v1.0)](../specifications/surface-57-mesh-first-decommission-rollback-v1_0.md)

## Manual Smoke Check

- Review the explicit rollback mechanisms that must remain available before
  decommission.
- Confirm rollback retirement is treated as an explicit follow-on decision.

## Automated Smoke Tests

- decommission triggers are explicit
- rollback requirements are explicit

## Automated Acceptance Tests

- rollback posture remains available until canonical promotion is met
- rollback retirement conditions are explicit and non-implicit
- decommission policy aligns with surfaced compatibility-path evidence
