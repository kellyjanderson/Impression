# Feature Spec 08B2: Progression Transport Semantics Contract (v1.0)

## Overview

This specification defines where transport semantics live in the progression
model and how loft consumes them.

## Backlink

- [Feature Spec 08B: Station Attachment, Transport, and Twist/Scale Semantics (v1.0)](feature-08b-station-attachment-transport-and-twist-scale-semantics-v1_0.md)

## Scope

This specification covers:

- transport semantics
- transport-policy ownership
- deterministic transport behavior

## Behavior

This leaf must define:

- where transport semantics live in the progression model
- how loft consumes those semantics
- how deterministic transport behavior is preserved

## Constraints

- transport semantics must remain explicit and deterministic
- transport policy must remain separate from twist and scale semantics

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- transport ownership is explicit
- deterministic transport behavior is explicit
- separation from twist/scale semantics is explicit
