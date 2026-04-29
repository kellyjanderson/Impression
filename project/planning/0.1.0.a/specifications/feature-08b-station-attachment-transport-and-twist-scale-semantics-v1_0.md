# Feature Spec 08B: Station Attachment, Transport, and Twist/Scale Semantics (v1.0)

## Overview

This specification defines the branch for station attachment and loft travel
semantics on top of the new progression model.

## Backlink

- [Feature Spec 08: Progression Model Upgrade Program (v1.0)](feature-08-progression-model-upgrade-program-v1_0.md)

## Scope

This specification covers:

- station attachment to progression
- transport semantics
- twist law slots
- scale law slots

## Behavior

This branch must define:

- the leaf that owns station attachment to progression
- the leaf that owns transport semantics
- the leaf that owns twist and scale semantic slots

## Constraints

- station attachment must not destroy topology-owned station truth
- transport semantics must remain explicit and deterministic

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Feature Spec 08B1: Station Attachment to Path-Backed Progression](feature-08b1-station-attachment-to-path-backed-progression-v1_0.md)
- [Feature Spec 08B2: Progression Transport Semantics Contract](feature-08b2-progression-transport-semantics-contract-v1_0.md)
- [Feature Spec 08B3: Progression Twist and Scale Semantic Slots](feature-08b3-progression-twist-and-scale-semantic-slots-v1_0.md)

## Acceptance

This specification is complete when:

- station attachment, transport, and twist/scale work are split into executable
  leaves
- each semantic layer has explicit ownership
