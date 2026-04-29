# Surface Spec 66: Deferred Patch Families and Explicit Exclusions (v1.0)

## Overview

This specification defines which patch families are intentionally excluded from
v1 and how that exclusion is recorded.

## Backlink

Parent specification:

- [Surface Spec 22: V1 Patch Family Scope and Explicit Exclusions (v1.0)](surface-22-v1-patch-family-scope-v1_0.md)

## Scope

This specification covers:

- excluded family list
- deferral rationale
- constraints imposed by those exclusions

## Behavior

This branch must define:

- which patch families are deferred out of v1
- why each deferred family is excluded
- what v1 callers may not assume because of those exclusions

## Constraints

- exclusions must be explicit
- deferral rationale must be explicit
- deferred families must not leak in through ad hoc special cases

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- deferred family list is explicit
- deferral rationale is explicit
- caller constraints caused by exclusions are explicit

