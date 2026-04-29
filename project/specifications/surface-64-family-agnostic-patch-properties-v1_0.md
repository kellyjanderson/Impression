# Surface Spec 64: Family-Agnostic Patch Properties and Capability Flags (v1.0)

## Overview

This specification defines the non-method properties every patch carries,
including capability flags shared across patch families.

## Backlink

Parent specification:

- [Surface Spec 21: Surface Patch Base Contract (v1.0)](surface-21-surface-patch-base-contract-v1_0.md)

## Scope

This specification covers:

- required patch properties
- family-agnostic capability flags
- patch-level invariants visible to downstream systems

## Behavior

This branch must define:

- the required non-method patch properties
- which capability flags downstream systems may inspect
- which invariants are guaranteed regardless of patch family

## Constraints

- properties must remain family-agnostic
- capability flags must not replace the formal interface contract
- invariants must be explicit and testable

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- required properties are explicit
- capability flags are explicit
- family-agnostic invariants are explicit

