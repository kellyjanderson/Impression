# Loft Spec 28 Test: Placed Topology State Object Shape

## Overview

This test specification defines verification for the canonical placed-topology
state input object.

## Backlink

- [Loft Spec 28: Placed Topology State Object Shape (v1.0)](../specifications/loft-28-placed-topology-state-object-shape-v1_0.md)

## Automated Smoke Tests

- `Station` exposes progression, topology, and placement-frame fields
- structure and placement remain distinguishable at the API boundary

## Automated Acceptance Tests

- progression is represented as a scalar field
- placement frame exposes `origin`, `u`, `v`, `n`
- canonical placed-state shape is stable and explicit
