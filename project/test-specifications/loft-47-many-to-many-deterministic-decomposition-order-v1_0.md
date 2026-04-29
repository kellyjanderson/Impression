# Loft Spec 47 Test: Many-to-Many Deterministic Decomposition Order

## Overview

This test specification defines verification for deterministic many-to-many
decomposition ordering in next-generation loft planning.

## Backlink

- [Loft Spec 47: Many-to-Many Deterministic Decomposition Order (v1.0)](../specifications/loft-47-many-to-many-deterministic-decomposition-order-v1_0.md)

## Automated Smoke Tests

- many-to-many transitions expose the required reduction stage order

## Automated Acceptance Tests

- decomposition order is explicit and stable
- candidate-set isolation occurs before direct and synthetic reduction
- decomposition ordering remains deterministic under region reordering
