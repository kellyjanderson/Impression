# Surface Spec 62: SurfacePatch Interface and Required Methods (v1.0)

## Overview

This specification defines the minimum method surface every `SurfacePatch`
implementation must expose.

## Backlink

Parent specification:

- [Surface Spec 21: Surface Patch Base Contract (v1.0)](surface-21-surface-patch-base-contract-v1_0.md)

## Scope

This specification covers:

- required patch methods
- shared method signatures
- mandatory query capabilities

## Behavior

This branch must define:

- the required method set for all patch families
- what each required method returns
- which methods are allowed to fail only for invalid patch state

## Constraints

- the interface must be family-agnostic
- required methods must be sufficient for tessellation and adjacency consumers
- optional capabilities must not be disguised as required methods

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- required method names and contracts are explicit
- return semantics are explicit
- invalid-state versus unsupported-state behavior is explicit

