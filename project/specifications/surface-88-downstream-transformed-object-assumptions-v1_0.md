# Surface Spec 88: Downstream Transformed-Object Assumptions (v1.0)

## Overview

This specification defines what downstream systems may assume about transformed
surface objects regardless of whether transforms are attached or baked.

## Backlink

Parent specification:

- [Surface Spec 29: Transform Attachment Versus Baked Geometry Policy (v1.0)](surface-29-transform-attachment-vs-baked-policy-v1_0.md)

## Scope

This specification covers:

- downstream assumptions about transformed objects
- invariants preserved across attached and baked states
- prohibited consumer assumptions

## Behavior

This branch must define:

- which invariants downstream systems may rely on for transformed objects
- which assumptions remain valid before and after baking
- what consumers may not assume without forcing canonicalization

## Constraints

- assumptions must be explicit
- invariants must be consistent across attached and baked representations
- prohibited assumptions must be explicit

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- allowed downstream assumptions are explicit
- cross-state invariants are explicit
- prohibited assumptions are explicit

