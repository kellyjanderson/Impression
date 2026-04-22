# Surface Spec 02: Surface Core Data Model (v1.0)

## Overview

This specification defines the branch responsible for the new core geometric
representation of Impression: surface-native bodies, shells, patches, trims,
metadata, and adjacency semantics.

## Backlink

Parent specification:

- [Surface Spec 01: Surface-First Internal Model Program (v1.0)](surface-01-surface-first-internal-model-program-v1_0.md)

## Scope

This specification covers:

- the surface-native data model
- patch and shell ownership structure
- transform and metadata attachment
- seam / adjacency invariants
- the minimum set of surface families required for the first migration phase

## Behavior

The data model branch must define:

- what a `SurfaceBody` is
- what a `SurfaceShell` is
- what a `SurfacePatch` is
- how trims are represented
- how patch adjacency and seam ownership are represented
- how transforms and colors/material metadata are stored

The v1 surface kernel should be broad enough to support:

- primitives
- extrude/revolve style tools
- loft migration

without requiring full CAD-kernel complexity up front.

## Constraints

- the surface representation must be deterministic and serializable enough for
  stable tooling
- the representation must be expressive enough to survive deferred tessellation
- the representation must not assume that mesh connectivity is the source of
  truth
- the first version must explicitly decide which surface families are in scope
  and which are deferred

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 07: Surface Body / Shell / Patch Core Contracts (v1.0)](surface-07-surface-body-shell-patch-contracts-v1_0.md)
- [Surface Spec 08: Surface Parameter Domains and Trim Representation (v1.0)](surface-08-surface-parameter-domains-and-trims-v1_0.md)
- [Surface Spec 09: Surface Adjacency and Seam Invariants (v1.0)](surface-09-surface-adjacency-and-seam-invariants-v1_0.md)
- [Surface Spec 10: Surface Transform, Metadata, and Identity Policy (v1.0)](surface-10-surface-transform-metadata-identity-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- the core surface types are explicitly named and bounded
- invariants are defined clearly enough that tessellation and modeling tools can
  target them without inventing missing semantics
- deferred concerns such as full NURBS or CAD parity are explicitly called out
  rather than left ambiguous
- the child branches define the kernel contracts as final
  implementation-sized leaves
