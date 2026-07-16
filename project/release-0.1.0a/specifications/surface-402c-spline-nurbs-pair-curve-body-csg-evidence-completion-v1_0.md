# Surface Spec 402C: Spline/NURBS Pair Curve Body CSG Evidence Completion (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: this leaf owns spline/spline-family curve evidence for B-spline/B-spline, B-spline/NURBS, and NURBS/NURBS pairs through one shared spline-pair route.

## Overview

Complete body-route-consumable curve evidence for spline/NURBS patch-pair CSG
intersections.

## Backlinks

- [Parent: Surface Spec 402](surface-402-b-spline-nurbs-patch-evidence-completion-for-body-csg-v1_0.md)
- [Prerequisite: Surface Spec 402A](surface-402a-b-spline-nurbs-body-csg-evidence-contract-and-prerequisite-audit-v1_0.md)
- [Specification: Surface Spec 343](surface-343-spline-and-nurbs-pair-curve-intersections-v1_0.md)

## Scope

This specification covers:

- normalizing `intersect_spline_nurbs_patch_pair` output to the 402A evidence contract
- B-spline/B-spline, B-spline/NURBS, and NURBS/NURBS route rows
- crossing, tangent, boundary, trimmed, non-convergent, unsupported, degenerate,
  and unmappable curve cases
- residual/convergence/tangent event metadata required by body CSG

This specification does not cover coincident-region overlap ownership. That is
owned by 402D.

## Behavior

The implementation must:

- produce route ids, family pairs, patch ids, curve ids, patch-local curves,
  orientation, residuals, convergence state, trim-readiness, and no-mesh evidence
- keep tangent and boundary events explicit in evidence rather than silently
  accepting them as ordinary crossing curves
- refuse deterministically for non-convergence, missing patch-local parameters,
  degenerate curve emission, unsupported pairs, and unmappable curves

## Verification

Focused tests must cover:

- B-spline/B-spline crossing, tangent, boundary, trimmed, and non-convergent cases
- B-spline/NURBS crossing, tangent, boundary, trimmed, and non-convergent cases
- NURBS/NURBS crossing, tangent, boundary, trimmed, and non-convergent cases
- unsupported or malformed patch pair diagnostics
- normalized evidence payload fields required by 402A
- no-hidden-mesh-fallback assertions

## Readiness And Sequencing

Blocked by Surface Spec 402A. This spec must complete before Surface Spec 402E.

## Five-Pass Review History

- Pass 1 - Scope Completeness: Limited to curve intersections, excluding coincident regions.
- Pass 2 - Dependency Check: Depends on 402A contract only.
- Pass 3 - Rescore: Count remains 1 IWU; one spline-pair curve evidence outcome.
- Pass 4 - Split: Coincident-region handling remains split into 402D.
- Pass 5 - Final Review: Final leaf ready after 402A.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when spline/NURBS pair curve routes produce
402A-compliant success and refusal evidence for all required pair classes and
focused tests pass without mesh fallback.
