# Surface Spec 07: Surface Body / Shell / Patch Core Contracts (v1.0)

## Overview

This specification defines the branch responsible for the fundamental object
model of the surface kernel: bodies, shells, patches, ownership hierarchy, and
the minimum patch-family scope required for the first migration phase.

## Backlink

Parent specification:

- [Surface Spec 02: Surface Core Data Model (v1.0)](surface-02-surface-core-data-model-v1_0.md)

## Scope

This specification covers:

- the core hierarchy of `SurfaceBody`, `SurfaceShell`, and `SurfacePatch`
- containment and ownership rules between those types
- patch-family scope for the first surface program
- deferred patch-family classes that are explicitly out of scope for v1

## Behavior

This branch must define:

- what constitutes one body versus one shell
- whether shells may be disconnected
- whether patches are always trimmed or may also be untrimmed
- which patch families are mandatory in v1
- which families are explicitly deferred

The resulting contracts must be broad enough to support:

- primitive realization
- extrusion / revolution style tools
- later loft surface execution

without requiring full CAD-grade generality immediately.

## Constraints

- the hierarchy must be simple enough to use consistently across the whole
  modeling stack
- the first patch-family set must be explicitly bounded
- deferred surface classes must be named rather than left ambiguous
- the contracts must support deterministic traversal and serialization

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 20: Surface Body and Shell Ownership Rules (v1.0)](surface-20-surface-body-shell-ownership-rules-v1_0.md)
- [Surface Spec 21: Surface Patch Base Contract (v1.0)](surface-21-surface-patch-base-contract-v1_0.md)
- [Surface Spec 22: V1 Patch Family Scope and Explicit Exclusions (v1.0)](surface-22-v1-patch-family-scope-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- the body/shell/patch hierarchy is explicit
- required v1 patch families are named and bounded
- deferred families are explicitly recorded
- the ownership model is clear enough that downstream specs do not need to
  invent kernel structure
- the child branches define the core contracts as final
  implementation-sized leaves
