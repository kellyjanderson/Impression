# Surface Spec 22: V1 Patch Family Scope and Explicit Exclusions (v1.0)

## Overview

This specification defines the branch responsible for deciding which surface
patch families are mandatory in the first surface program and which are
explicitly deferred.

## Backlink

Parent specification:

- [Surface Spec 07: Surface Body / Shell / Patch Core Contracts (v1.0)](surface-07-surface-body-shell-patch-contracts-v1_0.md)

## Scope

This specification covers:

- required v1 patch families
- deferred patch families
- the reasoning boundary between mandatory and deferred family scope

## Behavior

This branch must define:

- which patch families are required in v1
- which are explicitly excluded from v1
- which modeling operations each included family is expected to support

## Constraints

- family scope must be explicit
- deferred families must be named rather than implied
- the v1 family set must remain achievable without a full CAD kernel

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 65: Required V1 Patch Families (v1.0)](surface-65-required-v1-patch-families-v1_0.md)
- [Surface Spec 66: Deferred Patch Families and Explicit Exclusions (v1.0)](surface-66-deferred-patch-families-exclusions-v1_0.md)
- [Surface Spec 67: Patch-Family to Feature Coverage Matrix (v1.0)](surface-67-patch-family-feature-coverage-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- required v1 families are explicit
- deferred families are explicit
- the link between family scope and feature coverage is explicit
