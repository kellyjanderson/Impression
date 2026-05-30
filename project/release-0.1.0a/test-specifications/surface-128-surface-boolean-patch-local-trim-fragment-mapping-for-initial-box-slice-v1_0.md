# Surface Spec 128 Test: Surface Boolean Patch-Local Trim-Fragment Mapping for Initial Box Slice

## Overview

This test specification defines verification for mapping surfaced cut curves
into patch-local trim fragments on the initial box slice.

## Backlink

- [Surface Spec 128: Surface Boolean Patch-Local Trim-Fragment Mapping for Initial Box Slice (v1.0)](../specifications/surface-128-surface-boolean-patch-local-trim-fragment-mapping-for-initial-box-slice-v1_0.md)

## Automated Smoke Tests

- each supported cut curve produces per-patch trim fragments
- trim-fragment endpoints land inside the affected patch domains

## Automated Acceptance Tests

- trim-fragment ordering remains deterministic for equal inputs
- cut-fragment payloads stay aligned with their source cut-curve identifiers
- unsupported patch-local mapping cases remain explicit rather than fabricating trim-space data
