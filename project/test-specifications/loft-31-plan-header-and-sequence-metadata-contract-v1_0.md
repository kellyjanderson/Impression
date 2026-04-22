# Loft Spec 31 Test: Plan Header and Sequence Metadata Contract

## Overview

This test specification defines verification for the loft plan header and
sequence metadata views.

## Backlink

- [Loft Spec 31: Plan Header and Sequence Metadata Contract (v1.0)](../specifications/loft-31-plan-header-and-sequence-metadata-contract-v1_0.md)

## Automated Smoke Tests

- `LoftPlan` exposes explicit header fields
- sequence metadata is auditable and geometry-free

## Automated Acceptance Tests

- schema/version/planner fields are explicit
- sequence-control metadata is explicitly accessible
- summary metadata is separated from the header contract
