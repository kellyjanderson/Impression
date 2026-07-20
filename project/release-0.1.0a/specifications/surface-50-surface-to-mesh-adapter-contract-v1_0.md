# Surface Spec 50: Surface-to-Mesh Adapter Contract (v1.0)

## Overview

This specification defines the compatibility adapter that converts canonical
surface-native objects into mesh outputs for legacy consumers.

## Backlink

Parent specification:

- [Surface Spec 17: Compatibility Adapter Contracts (v1.0)](surface-17-compatibility-adapter-contracts-v1_0.md)

## Scope

This specification covers:

- adapter inputs and outputs
- the conversion contract
- adapter-owned guarantees inherited from tessellation

## Behavior

This branch must define:

- what surface inputs the adapter accepts
- what mesh outputs it produces
- what guarantees the adapter preserves from canonical tessellation

## Constraints

- the adapter must remain downstream of canonical surface truth
- conversion semantics must be explicit
- the adapter must not become a backdoor modeling kernel

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one adapter contract.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- adapter input and output shapes are explicit
- preserved guarantees are explicit
- prohibited backdoor-kernel behavior is explicit

