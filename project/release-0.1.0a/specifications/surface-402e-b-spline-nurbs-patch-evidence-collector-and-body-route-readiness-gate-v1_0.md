# Surface Spec 402E: B-Spline/NURBS Patch Evidence Collector And Body-Route Readiness Gate (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: this leaf owns one collector/readiness gate that gathers 402A-compliant evidence from completed child routes and exposes it to Surface Spec 401.

## Overview

Build the B-spline/NURBS patch evidence collector and readiness gate consumed by
Surface Spec 401.

## Backlinks

- [Parent: Surface Spec 402](surface-402-b-spline-nurbs-patch-evidence-completion-for-body-csg-v1_0.md)
- [Prerequisite: Surface Spec 402A](surface-402a-b-spline-nurbs-body-csg-evidence-contract-and-prerequisite-audit-v1_0.md)
- [Prerequisite: Surface Spec 402B1](surface-402b1-analytic-to-b-spline-body-csg-evidence-completion-v1_0.md)
- [Prerequisite: Surface Spec 402B2](surface-402b2-analytic-to-nurbs-body-csg-evidence-completion-v1_0.md)
- [Prerequisite: Surface Spec 402C](surface-402c-spline-nurbs-pair-curve-body-csg-evidence-completion-v1_0.md)
- [Prerequisite: Surface Spec 402D](surface-402d-spline-nurbs-coincident-region-body-csg-evidence-completion-v1_0.md)
- [Specification: Surface Spec 401](surface-401-b-spline-nurbs-body-level-csg-route-integration-v1_0.md)

## Scope

This specification covers:

- a collector that dispatches analytic/B-spline, analytic/NURBS, spline-pair
  curve, and coincident-region routes and returns normalized 402A evidence
- deterministic readiness diagnostics for missing route coverage or mixed
  success/refusal evidence
- a body-route readiness gate that Surface Spec 401 can call before trim
  arrangement and fragment graph construction
- regression tests proving the collector can be called without family-specific
  special casing in the body-level route

This specification does not cover body-level trim arrangement, shell assembly,
or public `SurfaceBooleanResult` success. Those belong to Surface Spec 401.

## Behavior

The implementation must:

- gather all participating B-spline/NURBS patch-pair evidence for prepared
  operand patch refs
- preserve per-route diagnostics rather than replacing them with generic
  unsupported messages
- classify readiness as success-ready, diagnostic-refusal-ready, or blocked by
  missing patch evidence
- emit no-hidden-mesh-fallback evidence for the collected route result
- expose a stable API for Surface Spec 401

## Verification

Focused tests must cover:

- collector success for analytic/B-spline and analytic/NURBS evidence
- collector success for spline-pair curve evidence
- collector success/refusal for coincident-region evidence
- readiness refusal for missing route coverage
- stable payload shape consumed by a 401-facing smoke test
- no-hidden-mesh-fallback assertions

## Readiness And Sequencing

Blocked by Surface Specs 402A, 402B1, 402B2, 402C, and 402D. This spec must
complete before Surface Spec 401.

## Five-Pass Review History

- Pass 1 - Scope Completeness: Limited to collection and readiness gating.
- Pass 2 - Dependency Check: Depends on completed 402A, 402B1, 402B2, 402C, and 402D route evidence.
- Pass 3 - Rescore: Count remains 1 IWU; one collector/readiness outcome.
- Pass 4 - Split: Body-level trim and shell work remains split into 401.
- Pass 5 - Final Review: Final leaf ready after 402A, 402B1, 402B2, 402C, and 402D.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when Surface Spec 401 has a stable collector API
that returns normalized B-spline/NURBS patch evidence or deterministic
diagnostic refusals for all completed child routes without hidden mesh fallback.
