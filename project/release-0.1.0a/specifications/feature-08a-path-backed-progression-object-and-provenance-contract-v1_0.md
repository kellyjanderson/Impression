# Feature Spec 08A: Path-Backed Progression Object and Provenance Contract (v1.0)

## Overview

This specification defines the owned shape of a path-backed progression object.

## Backlink

- [Feature Spec 08: Progression Model Upgrade Program (v1.0)](feature-08-progression-model-upgrade-program-v1_0.md)

## Scope

This specification covers:

- progression object structure
- underlying path or spine reference
- parameter domain
- exact-vs-inferred provenance

## Behavior

This leaf must define:

- what a progression object owns
- how it references the underlying path spine
- how provenance distinguishes explicit from inferred progression

## Constraints

- progression must remain distinct from the raw path primitive
- provenance must remain durable and inspectable

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- progression object ownership is explicit
- relation to the underlying path is explicit
- explicit-vs-inferred provenance is explicit
