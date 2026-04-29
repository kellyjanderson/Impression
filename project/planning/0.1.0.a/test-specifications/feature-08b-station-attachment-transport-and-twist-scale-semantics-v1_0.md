# Feature Spec 08B Test: Station Attachment, Transport, and Twist/Scale Semantics

## Overview

This test specification defines verification for the decomposed station
attachment and travel-semantics branch.

## Backlink

- [Feature Spec 08B: Station Attachment, Transport, and Twist/Scale Semantics (v1.0)](../specifications/feature-08b-station-attachment-transport-and-twist-scale-semantics-v1_0.md)

## Behavior

This parent test branch must verify:

- no executable station-attachment work remains hidden in the parent
- no executable transport-semantic work remains hidden in the parent
- no executable twist/scale semantic-slot work remains hidden in the parent
- every final child leaf has a paired test specification

## Child Test Specifications

- [Feature Spec 08B1 Test: Station Attachment to Path-Backed Progression](feature-08b1-station-attachment-to-path-backed-progression-v1_0.md)
- [Feature Spec 08B2 Test: Progression Transport Semantics Contract](feature-08b2-progression-transport-semantics-contract-v1_0.md)
- [Feature Spec 08B3 Test: Progression Twist and Scale Semantic Slots](feature-08b3-progression-twist-and-scale-semantic-slots-v1_0.md)

## Acceptance

This test specification is complete when the child set fully covers the branch.
