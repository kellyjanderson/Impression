# Surface Spec 109: Surface Boolean Result Contract and Failure Modes (v1.0)

## Overview

This specification defines the result shape and failure behavior for surface-body boolean operations.

## Backlink

- [Surface Spec 102: Surface-Body Boolean Replacement (v1.0)](surface-102-surface-body-boolean-replacement-v1_0.md)

## Scope

This specification covers:

- union, difference, and intersection results
- result classification and continuity expectations
- boolean failure and unsupported-case reporting

## Behavior

This branch must define:

- the canonical result type for surfaced booleans
- when booleans return closed bodies, open bodies, or structured failure
- how failure modes are surfaced to callers

## Constraints

- boolean outputs must remain surface-native
- failure cases must not silently collapse into mesh-only success
- result semantics must stay explicit at preview/export boundaries

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- boolean result semantics are explicit for the supported operation set
- failure modes are explicit and testable
- verification requirements are defined by its paired test specification

