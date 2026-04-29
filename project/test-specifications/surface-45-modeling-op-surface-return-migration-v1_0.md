# Surface Spec 45 Test: Modeling Operation Surface Return Migration

## Overview

This test specification defines verification for modeling operations that
migrate from mesh-returning to surface-returning behavior.

## Backlink

- [Surface Spec 45: Modeling Operation Surface Return-Type Migration (v1.0)](../specifications/surface-45-modeling-op-surface-return-migration-v1_0.md)

## Manual Smoke Check

- Run representative modeling operations through the public API.
- Confirm the canonical outputs are surface-native and remain consumable by
  preview/export.

## Automated Smoke Tests

- migrated modeling operations return documented surface-native results
- operations still produce non-empty preview/export output through tessellation

## Automated Acceptance Tests

- operation return contracts are stable and documented
- no hidden mesh-first fallback remains on the canonical path
- compatibility adapters are explicit where legacy mesh consumers still exist

## Notes

- Include loft or extrude once those operations have entered the surface path.
