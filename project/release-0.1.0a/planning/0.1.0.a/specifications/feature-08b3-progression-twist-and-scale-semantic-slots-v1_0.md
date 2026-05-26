# Feature Spec 08B3: Progression Twist and Scale Semantic Slots (v1.0)

## Overview

This specification defines the semantic slots for twist and scale behavior in
the progression model.

## Backlink

- [Feature Spec 08B: Station Attachment, Transport, and Twist/Scale Semantics (v1.0)](feature-08b-station-attachment-transport-and-twist-scale-semantics-v1_0.md)

## Scope

This specification covers:

- twist law slots
- scale law slots
- ownership even when the first implementation remains incomplete

## Behavior

This leaf must define:

- where twist semantics live
- where scale semantics live
- how those semantic slots remain explicit even if some are not fully executed
  in the first milestone

## Constraints

- twist and scale semantics must not be hidden in transport policy
- semantic-slot ownership must remain explicit from the start

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- twist semantic-slot ownership is explicit
- scale semantic-slot ownership is explicit
- first-milestone incompleteness does not hide the semantic contract
