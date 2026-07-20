# Surface Spec 84: Open Boundary Versus Shared Boundary Distinction (v1.0)

## Overview

This specification defines how the kernel distinguishes external/open
boundaries from boundaries shared with adjacent patches.

## Backlink

Parent specification:

- [Surface Spec 28: Shared-Boundary Validity and Continuity Rules (v1.0)](surface-28-shared-boundary-validity-continuity-v1_0.md)

## Scope

This specification covers:

- open-boundary classification
- shared-boundary classification
- visibility of the distinction to downstream systems

## Behavior

This branch must define:

- how open boundaries are represented
- how shared boundaries are represented
- how downstream consumers detect the distinction

## Constraints

- the distinction must be explicit
- representation must be deterministic
- downstream systems must not infer open/shared state indirectly

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- open-boundary representation is explicit
- shared-boundary representation is explicit
- downstream visibility of the distinction is explicit

