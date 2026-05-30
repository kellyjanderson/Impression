# Loft Spec 26: Tolerance and Degeneracy Policy (v1.0)

## Overview

This specification defines how the loft tolerance taxonomy becomes operational
policy in next-generation loft.

## Backlink

Parent specification:

- [Loft Spec 21: Next-Gen Loft Evolution Program (v1.0)](loft-21-nextgen-loft-evolution-program-v1_0.md)

## Scope

This specification covers:

- input-validity tolerance rules
- structural-classification tolerance rules
- decomposition-resolution tolerance rules
- collapse/degeneracy tolerance rules
- plan-validation tolerance rules

## Behavior

This branch must define:

- how each loft tolerance family is used operationally
- which rules are durable versus implementation-tunable
- how current implementation behavior seeds the first concrete policy set

## Constraints

- architecture-level tolerance families must not collapse back into one shared
  epsilon
- numeric thresholds must emerge from specs and evidence rather than umbrella
  architecture
- loft-local tolerance rules must remain separable from future surface-kernel
  tolerances

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Loft Spec 41: Input-Validity Tolerance Rules (v1.0)](loft-41-input-validity-tolerance-rules-v1_0.md)
- [Loft Spec 42: Structural-Classification Tolerance Rules (v1.0)](loft-42-structural-classification-tolerance-rules-v1_0.md)
- [Loft Spec 43: Decomposition-Resolution Tolerance Rules (v1.0)](loft-43-decomposition-resolution-tolerance-rules-v1_0.md)
- [Loft Spec 44: Collapse and Degeneracy Tolerance Rules (v1.0)](loft-44-collapse-and-degeneracy-tolerance-rules-v1_0.md)
- [Loft Spec 45: Plan-Validation Tolerance Rules (v1.0)](loft-45-plan-validation-tolerance-rules-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- each loft tolerance family is specified explicitly
- operational rules are separated by tolerance family
