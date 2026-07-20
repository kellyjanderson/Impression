# Surface Spec 137 Test: Surface Boolean Initial Reference Fixture Matrix and Promotion Gates

## Overview

This test specification defines verification for the initial surfaced CSG
reference-fixture matrix and its promotion gates.

## Backlink

- [Surface Spec 137: Surface Boolean Initial Reference Fixture Matrix and Promotion Gates (v1.0)](../specifications/surface-137-surface-boolean-initial-reference-fixture-matrix-and-promotion-gates-v1_0.md)

## Manual Smoke Check

- run the named surfaced union, difference, and intersection fixtures
- preview or export the surfaced results and confirm the expected bounded boolean shape is visible

## Automated Smoke Tests

- the named surfaced CSG fixtures produce durable rendered and exported artifacts
- promotion checks fail clearly when required surfaced references are missing

## Automated Acceptance Tests

- the named reference set includes `surfacebody/csg_union_box_post`,
  `surfacebody/csg_difference_slot`, and `surfacebody/csg_intersection_box_sphere`
- dirty and clean reference-image and reference-STL posture remain aligned with project reference-artifact rules
- promotion checks fail clearly when required named surfaced CSG references are
  missing
- unexpected surfaced CSG reference changes fail clearly unless dirty-reference regeneration was requested intentionally
