# Loft Spec 32 Test: Planned State and Interval Record Contract

## Overview

This test specification defines verification for planned-station and
planned-transition records.

## Backlink

- [Loft Spec 32: Planned State and Interval Record Contract (v1.0)](../specifications/loft-32-planned-state-and-interval-record-contract-v1_0.md)

## Automated Smoke Tests

- `PlannedStation` exposes normalized regions and placement frame
- `PlannedTransition` exposes deterministic state-index references

## Automated Acceptance Tests

- planned-state fields are explicit
- interval references are explicit and deterministic
- interval records remain geometry-light outside paired loop geometry
