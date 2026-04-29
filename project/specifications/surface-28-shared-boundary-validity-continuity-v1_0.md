# Surface Spec 28: Shared-Boundary Validity and Continuity Rules (v1.0)

## Overview

This specification defines the branch responsible for shared-boundary validity
rules and the continuity information that the surface kernel exposes.

## Backlink

Parent specification:

- [Surface Spec 09: Surface Adjacency and Seam Invariants (v1.0)](surface-09-surface-adjacency-and-seam-invariants-v1_0.md)

## Scope

This specification covers:

- shared-boundary validity
- open-boundary versus shared-boundary distinction
- continuity classifications exposed by the kernel

## Behavior

This branch must define:

- when a shared boundary is valid
- how open boundaries differ from shared ones
- what continuity metadata is recorded and how downstream consumers use it

## Constraints

- validity rules must be explicit
- open/shared distinctions must be explicit
- continuity metadata must be bounded to what downstream consumers actually need

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 83: Shared-Boundary Validity Rules (v1.0)](surface-83-shared-boundary-validity-rules-v1_0.md)
- [Surface Spec 84: Open Boundary Versus Shared Boundary Distinction (v1.0)](surface-84-open-vs-shared-boundary-v1_0.md)
- [Surface Spec 85: Surface Continuity Metadata Contract (v1.0)](surface-85-surface-continuity-metadata-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- shared-boundary validity is explicit
- open/shared distinctions are explicit
- continuity metadata expectations are explicit
