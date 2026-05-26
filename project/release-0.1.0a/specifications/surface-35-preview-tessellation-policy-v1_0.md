# Surface Spec 35: Preview Tessellation Policy Contract (v1.0)

## Overview

This specification defines the tessellation policy used for interactive preview
and rendering-oriented inspection of surface bodies.

## Backlink

Parent specification:

- [Surface Spec 12: Preview / Export Tessellation Policy Split (v1.0)](surface-12-preview-export-tessellation-policy-v1_0.md)

## Scope

This specification covers:

- preview quality targets
- responsiveness tradeoffs permitted in preview
- stable visual expectations for interactive consumers

## Behavior

This branch must define:

- the default preview tessellation mode
- which quality and tolerance bounds are acceptable in preview
- what preview may simplify without changing modeled meaning

## Constraints

- preview must remain derived from canonical surface truth
- preview-specific tradeoffs must be explicit and bounded
- preview policy must remain deterministic for identical inputs

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
isolates one consumer policy with one bounded output goal.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- preview defaults are explicit
- allowed preview tradeoffs are explicit
- deterministic preview guarantees are explicit

