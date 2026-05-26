# Surface Spec 21: Surface Patch Base Contract (v1.0)

## Overview

This specification defines the branch responsible for the base contract every
surface patch must satisfy.

## Backlink

Parent specification:

- [Surface Spec 07: Surface Body / Shell / Patch Core Contracts (v1.0)](surface-07-surface-body-shell-patch-contracts-v1_0.md)

## Scope

This specification covers:

- the minimum patch interface
- required patch-local properties
- evaluation and traversal expectations common to all patch families

## Behavior

This branch must define:

- what all patch types must expose
- what evaluation semantics downstream systems may rely on
- what minimal metadata all patches carry regardless of family

## Constraints

- the patch base contract must be family-agnostic enough for reuse
- the contract must be strong enough for tessellation and adjacency consumers
- the branch must avoid embedding family-specific behavior in the shared base

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 62: SurfacePatch Interface and Required Methods (v1.0)](surface-62-surfacepatch-interface-required-methods-v1_0.md)
- [Surface Spec 63: Patch Evaluation Semantics and Parameter Queries (v1.0)](surface-63-patch-evaluation-semantics-v1_0.md)
- [Surface Spec 64: Family-Agnostic Patch Properties and Capability Flags (v1.0)](surface-64-family-agnostic-patch-properties-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- the base patch contract is explicit
- downstream required semantics are explicit
- family-specific behavior is clearly separated from the base
