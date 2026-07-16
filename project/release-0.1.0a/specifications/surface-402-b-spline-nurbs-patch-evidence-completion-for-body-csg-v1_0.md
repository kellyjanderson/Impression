# Surface Spec 402: B-Spline/NURBS Patch Evidence Completion For Body CSG (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 6 IWU branch rollup.
Basis: this parent covers the complete blocker-removal program for Surface Spec 401. It is split into six final implementation leaves: evidence contract/audit, analytic/B-spline evidence, analytic/NURBS evidence, spline-pair curve evidence, coincident-region evidence, and the normalized collector/readiness gate.

## Overview

Complete the B-spline/NURBS patch-level CSG evidence contract required by
Surface Spec 401.

Surface Spec 401 is blocked because the existing checked patch-level specs
produce isolated route records, but the body-level bridge needs stronger
evidence: patch-local curves or overlap regions with route ids, residuals,
convergence state, tangent/degeneracy classification, trim-readiness,
orientation, source patch identities, and no-hidden-mesh-fallback metadata.

This parent defines the whole blocker-removal path. It is not an implementation
leaf and must not appear as the executable checklist item in progression.

## Backlinks

- [Architecture: Higher-Order Parametric CSG Routes Architecture](../architecture/higher-order-parametric-csg-routes-architecture.md)
- [Architecture: Surface CSG Executable Completion Architecture](../architecture/surface-csg-executable-completion-architecture.md)
- [Specification: Surface Spec 341: Analytic To B-Spline CSG Intersections](surface-341-analytic-to-b-spline-csg-intersections-v1_0.md)
- [Specification: Surface Spec 342: Analytic To NURBS CSG Intersections](surface-342-analytic-to-nurbs-csg-intersections-v1_0.md)
- [Specification: Surface Spec 343: Spline And NURBS Pair Curve Intersections](surface-343-spline-and-nurbs-pair-curve-intersections-v1_0.md)
- [Specification: Surface Spec 344: Spline And NURBS Coincident Region CSG Intersections](surface-344-spline-and-nurbs-coincident-region-csg-intersections-v1_0.md)
- [Specification: Surface Spec 401: B-Spline/NURBS Body-Level CSG Route Integration](surface-401-b-spline-nurbs-body-level-csg-route-integration-v1_0.md)

## Scope

This parent covers:

- auditing the checked state of Specs 341-344 against the evidence needs of
  Surface Spec 401
- defining the normalized patch-level evidence payload consumed by body CSG
- completing analytic/B-spline patch evidence
- completing analytic/NURBS patch evidence
- completing B-spline/B-spline, B-spline/NURBS, and NURBS/NURBS curve evidence
- completing spline/NURBS coincident-region evidence
- providing the collector/readiness gate that lets Surface Spec 401 consume
  patch evidence without family-specific special casing

This parent does not cover:

- assembling a body-level result shell
- returning public `SurfaceBooleanResult` success for B-spline/NURBS operands
- creating dirty STL reference fixtures
- performing mesh CSG, tessellated boolean fallback, or triangle-fragment truth

## Blocker Analysis

Surface Spec 401 is blocked until all child leaves of this parent are complete.

The blocker is not that B-spline/NURBS evaluators are absent. The lower-level
surface patch records, spline/NURBS evaluators, route registry, and basic
patch-local mapping are already present. The blocker is that the body-level CSG
route cannot rely on the current patch-level route records as a complete
execution contract.

This parent is not blocked by another new spec. If a child implementation
discovers a missing lower-level spline evaluator or parameter-domain primitive,
that repair belongs in the relevant child unless it would require a new
unrelated public API.

## Child Specifications

- [Surface Spec 402A: B-Spline/NURBS Body CSG Evidence Contract And Prerequisite Audit](surface-402a-b-spline-nurbs-body-csg-evidence-contract-and-prerequisite-audit-v1_0.md)
- [Surface Spec 402B: Analytic To B-Spline/NURBS Body CSG Evidence Completion](surface-402b-analytic-to-b-spline-nurbs-body-csg-evidence-completion-v1_0.md)
- [Surface Spec 402B1: Analytic To B-Spline Body CSG Evidence Completion](surface-402b1-analytic-to-b-spline-body-csg-evidence-completion-v1_0.md)
- [Surface Spec 402B2: Analytic To NURBS Body CSG Evidence Completion](surface-402b2-analytic-to-nurbs-body-csg-evidence-completion-v1_0.md)
- [Surface Spec 402C: Spline/NURBS Pair Curve Body CSG Evidence Completion](surface-402c-spline-nurbs-pair-curve-body-csg-evidence-completion-v1_0.md)
- [Surface Spec 402D: Spline/NURBS Coincident Region Body CSG Evidence Completion](surface-402d-spline-nurbs-coincident-region-body-csg-evidence-completion-v1_0.md)
- [Surface Spec 402E: B-Spline/NURBS Patch Evidence Collector And Body-Route Readiness Gate](surface-402e-b-spline-nurbs-patch-evidence-collector-and-body-route-readiness-gate-v1_0.md)

## Five-Pass Review History

- Pass 1 - Scope Completeness: Identified the original 402 as the correct blocker-removal area for 401, but too broad to remain a final leaf.
- Pass 2 - Dependency Check: Confirmed the blocker is patch-level evidence consumability, not body shell assembly or reference STL generation.
- Pass 3 - Rescore: Rescored the branch as 6 IWU after splitting analytic/B-spline and analytic/NURBS evidence into separate leaves.
- Pass 4 - Split: Split into 402A, 402B1, 402B2, 402C, 402D, and 402E final leaves, with 402B retained as a branch parent.
- Pass 5 - Final Review: Confirmed the parent remains a branch specification and progression must list only the final child leaves.

## Refinement Status

Parent branch specification. Not an implementation leaf.

## Acceptance

This parent is complete when final child specifications 402A, 402B1, 402B2, 402C, 402D, and 402E are complete
and Surface Spec 401 can depend on their normalized patch-level evidence
contract without adding new patch-level route behavior.
