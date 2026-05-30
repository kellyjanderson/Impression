# Surface Spec 25: Trim Validity, Orientation, and Boundary Semantics (v1.0)

## Overview

This specification defines the branch responsible for trim validity rules,
orientation semantics, and their meaning for patch boundaries.

## Backlink

Parent specification:

- [Surface Spec 08: Surface Parameter Domains and Trim Representation (v1.0)](surface-08-surface-parameter-domains-and-trims-v1_0.md)

## Scope

This specification covers:

- trim validity conditions
- orientation semantics
- the meaning of trim boundaries for patch interiors and holes

## Behavior

This branch must define:

- when a trim is valid
- what trim orientation means
- how trims define retained versus removed patch regions

## Constraints

- validity and orientation semantics must be explicit
- downstream code must not infer boundary meaning ad hoc
- the branch must support deterministic tessellation and capping behavior

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 74: Trim Validity Conditions and Failure Modes (v1.0)](surface-74-trim-validity-failure-modes-v1_0.md)
- [Surface Spec 75: Trim Orientation Semantics (v1.0)](surface-75-trim-orientation-semantics-v1_0.md)
- [Surface Spec 76: Boundary Inclusion and Interior Meaning Rules (v1.0)](surface-76-boundary-inclusion-interior-meaning-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- trim validity is explicit
- orientation semantics are explicit
- patch-boundary meaning is explicit
