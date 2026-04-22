# Surface Spec 08: Surface Parameter Domains and Trim Representation (v1.0)

## Overview

This specification defines the branch responsible for how surface patches expose
parameter space and how trims are represented against those parameter domains.

## Backlink

Parent specification:

- [Surface Spec 02: Surface Core Data Model (v1.0)](surface-02-surface-core-data-model-v1_0.md)

## Scope

This specification covers:

- patch parameter-domain expectations
- trim-loop representation
- trim ownership relative to patches and shells
- outer-boundary versus hole trim semantics
- the minimum trim behavior needed for deterministic tessellation

## Behavior

This branch must define:

- whether every patch owns a parameter domain
- whether trims live in world space, parameter space, or both
- how outer trims and inner trims are distinguished
- how trim orientation and validity are represented
- which trim guarantees tessellation and adjacency code may rely on

These rules must be strong enough for:

- patch trimming
- patch capping
- seam-consistent tessellation
- future loft cap and transition work

## Constraints

- trim representation must remain deterministic
- trim semantics must not rely on triangle connectivity
- trim ownership must be clear enough to avoid duplicated or dangling boundary
  meaning
- the branch must explicitly decide the minimum trim complexity supported in v1

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 23: Patch Parameter-Domain Contract (v1.0)](surface-23-patch-parameter-domain-contract-v1_0.md)
- [Surface Spec 24: Trim-Loop Representation and Ownership (v1.0)](surface-24-trim-loop-representation-v1_0.md)
- [Surface Spec 25: Trim Validity, Orientation, and Boundary Semantics (v1.0)](surface-25-trim-validity-orientation-boundary-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- the patch parameter-domain contract is explicit
- trim representation is clearly defined
- boundary and hole trim semantics are unambiguous
- tessellation consumers can rely on the resulting trim rules without guessing
- the child branches define parameter and trim concerns as final
  implementation-sized leaves
