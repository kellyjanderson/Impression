# Loft Spec 25: Ambiguity, Constraint Request, and Diagnostic Surface (v1.0)

## Overview

This specification defines how next-generation loft surfaces unresolved
planning state and requests minimal additional directional constraints.

## Backlink

Parent specification:

- [Loft Spec 21: Next-Gen Loft Evolution Program (v1.0)](loft-21-nextgen-loft-evolution-program-v1_0.md)

## Scope

This specification covers:

- ambiguity detection output
- constraint-request records
- invalid-input versus underconstrained-input distinction
- diagnostic locator payload

## Behavior

This branch must define:

- what information an ambiguity record must carry
- how constraint requests are surfaced
- how invalid input is kept distinct from ambiguity
- what deterministic guarantees apply when constraints are later added

## Constraints

- executor must never receive unresolved ambiguity
- diagnostics must stay structured and machine-consumable
- the diagnostic payload must remain minimal and topology-locating rather than
  over-specified

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Loft Spec 38: Ambiguity Record Minimal Locator Contract (v1.0)](loft-38-ambiguity-record-minimal-locator-contract-v1_0.md)
- [Loft Spec 39: Constraint Request Record Contract (v1.0)](loft-39-constraint-request-record-contract-v1_0.md)
- [Loft Spec 40: Invalid-Input Versus Underconstrained Taxonomy (v1.0)](loft-40-invalid-input-versus-underconstrained-taxonomy-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- ambiguity records are explicit
- constraint-request records are explicit
- invalid-input and underconstrained taxonomy is explicit
