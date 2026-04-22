# Surface Spec 24: Trim-Loop Representation and Ownership (v1.0)

## Overview

This specification defines the branch responsible for trim-loop representation
and ownership relative to patches and shells.

## Backlink

Parent specification:

- [Surface Spec 08: Surface Parameter Domains and Trim Representation (v1.0)](surface-08-surface-parameter-domains-and-trims-v1_0.md)

## Scope

This specification covers:

- trim-loop data representation
- ownership of trim loops
- outer versus inner trim categories

## Behavior

This branch must define:

- what a trim loop is
- who owns it
- how outer and inner trim loops are distinguished

## Constraints

- trim ownership must be explicit
- trim representation must remain deterministic
- trim categories must not be inferred ad hoc by downstream code

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 71: Trim Loop Data Structure Contract (v1.0)](surface-71-trim-loop-data-structure-v1_0.md)
- [Surface Spec 72: Trim Ownership and Attachment Policy (v1.0)](surface-72-trim-ownership-attachment-policy-v1_0.md)
- [Surface Spec 73: Outer and Inner Trim Categorization Rules (v1.0)](surface-73-outer-inner-trim-categorization-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- trim-loop representation is explicit
- ownership is explicit
- outer/inner categorization is explicit
