# Feature Spec 02A2 Test: Knot-Count and Knot-Placement Policy Records

## Overview

This test specification defines verification for knot-count and knot-placement
policy records.

## Backlink

- [Feature Spec 02A2: Knot-Count and Knot-Placement Policy Records (v1.0)](../specifications/feature-02a2-knot-count-and-knot-placement-policy-records-v1_0.md)

## Automated Smoke Tests

- knot-count and knot-placement policy records accept explicit initial scope
  choices
- policy records remain inspectable from fit configuration inputs

## Automated Acceptance Tests

- knot-count policy is durable and replayable
- knot-placement policy is durable and replayable
- knot decisions are not hidden inside fit helper implementations
