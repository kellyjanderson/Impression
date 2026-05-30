# Surface Spec 100 Test: Surface-Native Drafting Replacement

## Overview

This test specification defines verification for the surface-first replacement of deprecated drafting capability.

## Backlink

- [Surface Spec 100: Surface-Native Drafting Replacement (v1.0)](../specifications/surface-100-surface-native-drafting-replacement-v1_0.md)

## Manual Smoke Check

- Build representative surface-native line, plane, arrow, and dimension outputs.
- Confirm they appear in preview/export without relying on mesh as authored truth.

## Automated Smoke Tests

- drafting replacement returns surface-native or topology-native canonical outputs
- drafting outputs can flow through standard consumer handoff and tessellation

## Automated Acceptance Tests

- dimension/text attachment remains stable without mesh-first drafting
- surfaced drafting outputs remain visible and non-empty in preview artifacts

## Notes

- Include at least one reference image once surfaced drafting visuals exist.
