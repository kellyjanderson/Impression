# Loft Spec 50: Simple Correspondence Regression Fixtures (v1.0)

## Overview

This specification defines simple loft correspondence fixtures that visibly and
numerically detect twist or deformation regressions.

## Backlink

- [Loft Spec 21: Next-Gen Loft Evolution Program (v1.0)](loft-21-nextgen-loft-evolution-program-v1_0.md)

## Scope

This specification covers:

- simple rectangular correspondence fixtures
- simple cylindrical correspondence fixtures
- reference images and STL artifacts for those fixtures
- lightweight geometric checks that detect twist or deformation drift

## Behavior

This branch must define:

- deterministic simple loft fixtures with stable station layouts
- at least one rectangular case whose mid-body orientation exposes twist
- at least one circular case whose mid-body roundness exposes deformation
- regression artifacts and automated checks for those fixtures

## Constraints

- fixtures must stay simple enough to diagnose visually
- checks must not rely only on exact mesh identity
- reference artifacts must use the documented dirty/clean lifecycle

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- rectangular and cylindrical correspondence fixtures exist
- automated checks catch twist or deformation drift in those fixtures
- reference images and STL artifacts exist with paired verification
