# Feature Spec 08: Progression Model Upgrade Program (v1.0)

## Overview

This specification defines the `0.1.0.a` architectural correction that upgrades
progression into a path-backed semantic object.

## Backlink

- [Feature 08 — Progression Model Upgrade Architecture](../architecture/feature-08-progression-model-upgrade-architecture.md)

## Scope

This specification covers:

- path-backed progression object shape
- station attachment and transport semantics

## Behavior

This branch must define:

- the leaf that owns progression object structure and provenance
- the leaf that owns station attachment, transport, and twist/scale semantics

## Constraints

- progression must stop being treated as only loose scalar arrays
- this branch is the replacement path for generic path-driven body-construction
  cases inside loft enhancement

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Feature Spec 08A: Path-Backed Progression Object and Provenance Contract](feature-08a-path-backed-progression-object-and-provenance-contract-v1_0.md)
- [Feature Spec 08B: Station Attachment, Transport, and Twist/Scale Semantics](feature-08b-station-attachment-transport-and-twist-scale-semantics-v1_0.md)

## Acceptance

This specification is complete when:

- progression object ownership is explicit
- station attachment and transport semantics are explicit
