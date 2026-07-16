# Surface Spec 402A: B-Spline/NURBS Body CSG Evidence Contract And Prerequisite Audit (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: this leaf owns one reusable evidence contract and one prerequisite audit harness for the B-spline/NURBS patch routes consumed by Surface Spec 401.

## Overview

Define the normalized patch-level evidence payload that Surface Spec 401 will
consume and audit checked Specs 341-344 against that payload.

## Backlinks

- [Parent: Surface Spec 402](surface-402-b-spline-nurbs-patch-evidence-completion-for-body-csg-v1_0.md)
- [Specification: Surface Spec 401](surface-401-b-spline-nurbs-body-level-csg-route-integration-v1_0.md)
- [Specification: Surface Spec 341](surface-341-analytic-to-b-spline-csg-intersections-v1_0.md)
- [Specification: Surface Spec 342](surface-342-analytic-to-nurbs-csg-intersections-v1_0.md)
- [Specification: Surface Spec 343](surface-343-spline-and-nurbs-pair-curve-intersections-v1_0.md)
- [Specification: Surface Spec 344](surface-344-spline-and-nurbs-coincident-region-csg-intersections-v1_0.md)

## Scope

This specification covers:

- a normalized evidence record or payload for B-spline/NURBS body CSG inputs
- an audit helper that evaluates the current patch-level route records against
  the normalized payload contract
- deterministic audit diagnostics for missing route ids, patch ids, residuals,
  convergence state, orientation, trim-readiness, or no-mesh-fallback evidence
- focused tests proving checked Specs 341-344 either satisfy the contract or
  produce explicit gaps for later child specs

This specification does not cover implementing the route-specific repairs.
Those belong to 402B, 402C, and 402D.

## Required Evidence Contract

The normalized payload must include:

- operation
- route id
- family pair
- source patch refs and stable patch ids
- source body or operand refs when available
- evidence kind: `curve`, `coincident-region`, or `diagnostic-refusal`
- curve ids or region ids
- patch-local curve points or overlap-boundary loops for every affected patch
- orientation metadata for every patch-local curve or loop
- residual max, tolerance, iteration count, and convergence state
- crossing, tangent, boundary, singular, degenerate, coincident, or refusal classification
- ownership status for coincident regions
- trim-readiness status and reason
- diagnostics with stage, family pair, patch refs, and no hidden mesh fallback
- `no_mesh_fallback=true`

## Verification

Focused tests must cover:

- contract serialization/canonical payload shape
- successful audit rows for already-complete evidence
- failed audit rows for missing residual, patch id, orientation, trim-readiness,
  route id, or no-mesh-fallback fields
- audit coverage for analytic/B-spline, analytic/NURBS, spline-pair curve, and
  coincident-region source records

## Readiness And Sequencing

This spec is the first child of Surface Spec 402 and must complete before
402B-402E.

## Five-Pass Review History

- Pass 1 - Scope Completeness: Narrowed to the shared evidence contract and audit harness.
- Pass 2 - Dependency Check: No lower-level blocker found; existing route records can be audited before repairs.
- Pass 3 - Rescore: Count remains 1 IWU because the deliverable is one contract plus one audit surface.
- Pass 4 - Split: No child split needed; route repairs are split into 402B-402D.
- Pass 5 - Final Review: Final leaf ready for implementation.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when the normalized evidence contract and audit
helper exist, tests cover both passing and failing evidence rows, and later
402 children can target concrete audit failures.
