# Surface Spec 51 Test: Legacy Mesh Consumer Bridge

## Overview

This test specification defines verification for legacy mesh-consumer support
once surfaces become canonical.

## Backlink

- [Surface Spec 51: Legacy Mesh Consumer Bridge Policy (v1.0)](../specifications/surface-51-legacy-mesh-consumer-bridge-policy-v1_0.md)

## Manual Smoke Check

- Exercise a legacy mesh consumer with a surface-native source object.
- Confirm the documented bridge path is used and the consumer still functions.

## Automated Smoke Tests

- legacy mesh consumer bridge remains callable where promised
- bridge path is explicit and deterministic

## Automated Acceptance Tests

- legacy consumers do not require hidden mesh-first kernel behavior
- bridge usage is documented and testable
- unsupported legacy paths fail clearly rather than silently bypassing the
  surface boundary

## Notes

- Cover at least one legacy consumer that predates the surface-first transition.
