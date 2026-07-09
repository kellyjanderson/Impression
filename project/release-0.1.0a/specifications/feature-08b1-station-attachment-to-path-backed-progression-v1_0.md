# Feature Spec 08B1: Station Attachment to Path-Backed Progression (v1.0)

## Overview

This specification defines how stations attach to path-backed progression
instead of to loose parallel scalar arrays.

## Backlink

- [Feature Spec 08B: Station Attachment, Transport, and Twist/Scale Semantics (v1.0)](feature-08b-station-attachment-transport-and-twist-scale-semantics-v1_0.md)

## Scope

This specification covers:

- station attachment to progression
- attachment ordering and identity
- protection of topology-owned station truth

## Behavior

This leaf must define:

- how stations attach to progression
- how attachment ordering and identity remain durable
- how station attachment preserves topology-owned truth

## Constraints

- station attachment must not collapse into loose scalar progression arrays
- topology-owned station truth must remain intact

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- station attachment shape is explicit
- attachment identity and ordering are explicit
- topology-preservation posture is explicit
