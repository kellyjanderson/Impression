# Surface Spec 23: Patch Parameter-Domain Contract (v1.0)

## Overview

This specification defines the branch responsible for the parameter-domain
contract of surface patches.

## Backlink

Parent specification:

- [Surface Spec 08: Surface Parameter Domains and Trim Representation (v1.0)](surface-08-surface-parameter-domains-and-trims-v1_0.md)

## Scope

This specification covers:

- patch-local parameter domains
- domain normalization or non-normalization policy
- required domain semantics for downstream consumers

## Behavior

This branch must define:

- whether every patch has a parameter domain
- what that domain looks like
- what downstream systems may assume about parameter coordinates

## Constraints

- parameter-domain rules must be deterministic
- the contract must be broad enough for multiple patch families
- tessellation and trim logic must not need to invent domain semantics

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 68: Patch Domain Existence and Shape Contract (v1.0)](surface-68-patch-domain-existence-shape-v1_0.md)
- [Surface Spec 69: Parameter Domain Normalization Policy (v1.0)](surface-69-parameter-domain-normalization-policy-v1_0.md)
- [Surface Spec 70: Downstream Parameter-Space Assumptions Contract (v1.0)](surface-70-downstream-parameter-space-assumptions-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- domain rules are explicit
- normalization policy is explicit
- downstream consumer assumptions are explicit
