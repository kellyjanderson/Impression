# Surface Spec 402B2: Analytic To NURBS Body CSG Evidence Completion (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: this leaf owns one route outcome: `intersect_analytic_nurbs_patch_pair` must emit 402A-compliant body-route evidence including rational-weight diagnostics.

## Overview

Complete body-route-consumable evidence for analytic/NURBS CSG patch
intersections.

## Backlinks

- [Parent: Surface Spec 402B](surface-402b-analytic-to-b-spline-nurbs-body-csg-evidence-completion-v1_0.md)
- [Prerequisite: Surface Spec 402A](surface-402a-b-spline-nurbs-body-csg-evidence-contract-and-prerequisite-audit-v1_0.md)
- [Specification: Surface Spec 342](surface-342-analytic-to-nurbs-csg-intersections-v1_0.md)

## Scope

This specification covers:

- normalizing `intersect_analytic_nurbs_patch_pair` output to the 402A evidence contract
- retaining NURBS weight, rational residual, and exact-conic-compatible metadata
- planar/NURBS, ruled/NURBS, and revolution/NURBS route evidence
- tangent, boundary, trimmed, non-convergent, invalid-weight, singular rational,
  degenerate, and unmappable diagnostics
- no-hidden-mesh-fallback evidence for every success and refusal path

This specification does not cover analytic/B-spline residual evidence.

## Behavior

The implementation must:

- emit route id, family pair, source patch refs, stable patch ids, residuals,
  weight diagnostics, convergence state, trim-readiness, and `no_mesh_fallback=true`
- classify tangent and boundary cases explicitly instead of treating them as
  ordinary crossings
- preserve invalid-weight and singular rational diagnostics in the normalized
  evidence payload
- refuse before body-level execution when patch-local curve mapping is missing,
  degenerate, or ambiguous

## Verification

Focused tests must cover:

- planar/NURBS, ruled/NURBS, revolution/NURBS, and exact-conic-compatible evidence
- tangent and boundary diagnostics
- trimmed analytic patch evidence
- invalid NURBS weights, singular rational diagnostics, non-convergence,
  degenerate curve emission, and unmappable local-parameter refusal
- normalized evidence payload fields required by 402A
- no-hidden-mesh-fallback assertions

## Readiness And Sequencing

Blocked by Surface Spec 402A. This spec must complete before Surface Spec 402E.

## Five-Pass Review History

- Pass 1 - Scope Completeness: Narrowed from analytic-to-spline to analytic/NURBS only.
- Pass 2 - Dependency Check: Depends only on 402A.
- Pass 3 - Rescore: Count remains 1 IWU for one route function and verification surface.
- Pass 4 - Split: No further split required.
- Pass 5 - Final Review: Final leaf ready after 402A.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when analytic/NURBS routes produce
402A-compliant success and refusal evidence and all focused tests pass without
mesh fallback.
