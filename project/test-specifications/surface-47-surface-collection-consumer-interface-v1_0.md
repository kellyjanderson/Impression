# Surface Spec 47 Test: Surface Collection Consumer Interface

## Overview

This test specification defines verification for consumer-facing interfaces that
accept collections of surface-native objects.

## Backlink

- [Surface Spec 47: Surface Collection Consumer Interface (v1.0)](../specifications/surface-47-surface-collection-consumer-interface-v1_0.md)

## Manual Smoke Check

- Hand a small collection of surface-native bodies or shells to a consumer path.
- Confirm traversal and downstream tessellation proceed without requiring manual
  mesh conversion.

## Automated Smoke Tests

- consumer interface accepts documented collection/container shapes
- collection traversal is deterministic

## Automated Acceptance Tests

- consumer-facing collection handoff preserves ordering where documented
- downstream tessellation/export operates correctly over a collection input
- unsupported collection members fail with structured errors rather than silent
  omission

## Notes

- Prefer fixtures that mix more than one body to prove ordering and traversal.
