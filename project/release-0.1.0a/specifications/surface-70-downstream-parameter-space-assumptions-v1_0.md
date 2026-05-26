# Surface Spec 70: Downstream Parameter-Space Assumptions Contract (v1.0)

## Overview

This specification defines what downstream systems may assume about a patch's
parameter space.

## Backlink

Parent specification:

- [Surface Spec 23: Patch Parameter-Domain Contract (v1.0)](surface-23-patch-parameter-domain-contract-v1_0.md)

## Scope

This specification covers:

- tessellation assumptions
- trim assumptions
- adjacency/seam assumptions about parameter space

## Behavior

This branch must define:

- what downstream systems may assume about continuity and bounds in parameter space
- which assumptions are guaranteed versus prohibited
- how unsupported assumptions are surfaced as invalid usage

## Constraints

- downstream assumptions must be explicit
- prohibited assumptions must be explicit
- the contract must align with required v1 patch families

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- guaranteed downstream assumptions are explicit
- prohibited assumptions are explicit
- invalid usage behavior is explicit

