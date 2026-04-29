# Surface Spec 127 Test: Surface Boolean Initial Box/Box Cut-Curve Discovery

## Overview

This test specification defines verification for the initial surfaced box/box
cut-curve discovery lane.

## Backlink

- [Surface Spec 127: Surface Boolean Initial Box/Box Cut-Curve Discovery (v1.0)](../specifications/surface-127-surface-boolean-initial-box-box-cut-curve-discovery-v1_0.md)

## Automated Smoke Tests

- representative overlapping box operands produce deterministic surfaced cut curves
- unsupported overlap shapes remain explicit surfaced unsupported results

## Automated Acceptance Tests

- cut-curve identifiers and ordering remain deterministic
- boundary-touch and coplanar cases do not silently masquerade as bounded supported overlap cuts
- representative overlapping face-cut cases produce the expected count of 3D cut segments
