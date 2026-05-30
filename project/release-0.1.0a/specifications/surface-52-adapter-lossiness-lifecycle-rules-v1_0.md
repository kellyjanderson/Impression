# Surface Spec 52: Adapter Lossiness and Lifecycle Rules (v1.0)

## Overview

This specification defines how compatibility adapters are classified as lossless
or lossy and how that classification affects their allowed lifecycle.

## Backlink

Parent specification:

- [Surface Spec 17: Compatibility Adapter Contracts (v1.0)](surface-17-compatibility-adapter-contracts-v1_0.md)

## Scope

This specification covers:

- lossless versus lossy adapter classification
- warning/visibility expectations around lossy conversion
- lifecycle limits tied to adapter class

## Behavior

This branch must define:

- what makes an adapter lossless or lossy
- what caller-facing or internal warnings each class requires
- which classes are allowed as temporary bridges versus longer-lived boundaries

## Constraints

- lossiness classification must be explicit
- lossy conversion must not be silently treated as neutral
- lifecycle policy must reinforce temporary migration intent

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one classification and lifecycle policy.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- lossless versus lossy rules are explicit
- warning/visibility expectations are explicit
- lifecycle consequences by class are explicit

