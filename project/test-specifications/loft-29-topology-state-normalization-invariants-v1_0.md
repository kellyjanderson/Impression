# Loft Spec 29 Test: Topology State Normalization Invariants

## Overview

This test specification defines verification for planner-entry topology
normalization invariants.

## Backlink

- [Loft Spec 29: Topology State Normalization Invariants (v1.0)](../specifications/loft-29-topology-state-normalization-invariants-v1_0.md)

## Automated Smoke Tests

- station topology is canonicalized before planning
- deterministic region ordering and loop anchoring are preserved

## Automated Acceptance Tests

- malformed correspondence arity is rejected before planning
- normalized topology is exposed explicitly at the station boundary
