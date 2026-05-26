# Surface Spec 20: Surface Body and Shell Ownership Rules (v1.0)

## Overview

This specification defines the branch responsible for ownership and containment
rules between `SurfaceBody` and `SurfaceShell`.

## Backlink

Parent specification:

- [Surface Spec 07: Surface Body / Shell / Patch Core Contracts (v1.0)](surface-07-surface-body-shell-patch-contracts-v1_0.md)

## Scope

This specification covers:

- body-to-shell containment rules
- shell multiplicity rules
- disconnected versus connected shell policy
- traversal and ordering expectations

## Behavior

This branch must define:

- what a body owns
- what a shell owns
- whether bodies may contain multiple shells
- whether shells may be disconnected

## Constraints

- ownership must be deterministic
- multiplicity rules must be explicit
- traversal order must be stable enough for downstream consumers

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 59: SurfaceBody Ownership and Containment Contract (v1.0)](surface-59-surfacebody-ownership-containment-v1_0.md)
- [Surface Spec 60: SurfaceShell Multiplicity and Connectivity Policy (v1.0)](surface-60-surfaceshell-multiplicity-connectivity-v1_0.md)
- [Surface Spec 61: Deterministic Body/Shell Traversal and Ordering Rules (v1.0)](surface-61-body-shell-traversal-ordering-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- body/shell ownership is explicit
- shell multiplicity rules are explicit
- traversal expectations are explicit
