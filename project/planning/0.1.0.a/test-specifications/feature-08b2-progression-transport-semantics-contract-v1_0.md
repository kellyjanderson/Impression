# Feature Spec 08B2 Test: Progression Transport Semantics Contract

## Overview

This test specification defines verification for progression transport
semantics.

## Backlink

- [Feature Spec 08B2: Progression Transport Semantics Contract (v1.0)](../specifications/feature-08b2-progression-transport-semantics-contract-v1_0.md)

## Automated Smoke Tests

- progression records expose explicit transport semantics
- loft consumes transport semantics through an inspectable contract

## Automated Acceptance Tests

- transport semantics remain deterministic for identical inputs
- transport policy remains separate from twist and scale semantic slots
- transport ownership is durable enough for replay and diagnostics
