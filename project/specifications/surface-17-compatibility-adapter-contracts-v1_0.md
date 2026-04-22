# Surface Spec 17: Compatibility Adapter Contracts (v1.0)

## Overview

This specification defines the branch responsible for the temporary adapters
that allow mesh-native consumers or APIs to coexist with surface-native kernel
truth during migration.

## Backlink

Parent specification:

- [Surface Spec 05: Migration and Compatibility Path (v1.0)](surface-05-migration-and-compatibility-path-v1_0.md)

## Scope

This specification covers:

- compatibility adapter responsibilities
- permitted adapter directions
- temporary versus durable adapter policy

## Behavior

This branch must define:

- what adapters exist
- which direction each adapter operates in
- whether adapters are public, internal, or both
- which guarantees adapters must preserve

## Constraints

- adapters must not become a permanent architectural center
- adapter semantics must be explicit
- lossy versus lossless transitions must be distinguished clearly

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 50: Surface-to-Mesh Adapter Contract (v1.0)](surface-50-surface-to-mesh-adapter-contract-v1_0.md)
- [Surface Spec 51: Legacy Mesh Consumer Bridge Policy (v1.0)](surface-51-legacy-mesh-consumer-bridge-policy-v1_0.md)
- [Surface Spec 52: Adapter Lossiness and Lifecycle Rules (v1.0)](surface-52-adapter-lossiness-lifecycle-rules-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- compatibility adapters are explicitly named
- their responsibilities are explicit
- their temporary status is enforced by written constraints
