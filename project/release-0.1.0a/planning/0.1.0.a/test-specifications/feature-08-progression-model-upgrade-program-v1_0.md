# Feature Spec 08 Test: Progression Model Upgrade Program

## Overview

This test specification defines verification for the decomposed progression
model-upgrade branch.

## Backlink

- [Feature Spec 08: Progression Model Upgrade Program (v1.0)](../specifications/feature-08-progression-model-upgrade-program-v1_0.md)

## Behavior

This parent test branch must verify:

- no executable progression-object work remains hidden in the parent
- no executable station-attachment or travel-semantic work remains hidden in
  the parent
- every final child leaf has a paired test specification

## Child Test Specifications

- [Feature Spec 08A Test: Path-Backed Progression Object and Provenance Contract](feature-08a-path-backed-progression-object-and-provenance-contract-v1_0.md)
- [Feature Spec 08B Test: Station Attachment, Transport, and Twist/Scale Semantics](feature-08b-station-attachment-transport-and-twist-scale-semantics-v1_0.md)

## Acceptance

This test specification is complete when the child set fully covers the branch.
