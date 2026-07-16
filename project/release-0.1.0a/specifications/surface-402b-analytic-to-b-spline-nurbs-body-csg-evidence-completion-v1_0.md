# Surface Spec 402B: Analytic To B-Spline/NURBS Body CSG Evidence Completion (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 2 IWU branch rollup.
Basis: the original 402B bundled two independently reviewable route outcomes: analytic-to-B-spline evidence and analytic-to-NURBS evidence. They share the 402A contract but have different route functions, diagnostics, and verification surfaces.

## Overview

Complete body-route-consumable evidence for analytic/B-spline and
analytic/NURBS CSG patch intersections.

This parent is not an implementation leaf. It is split into 402B1 and 402B2.

## Backlinks

- [Parent: Surface Spec 402](surface-402-b-spline-nurbs-patch-evidence-completion-for-body-csg-v1_0.md)
- [Prerequisite: Surface Spec 402A](surface-402a-b-spline-nurbs-body-csg-evidence-contract-and-prerequisite-audit-v1_0.md)
- [Specification: Surface Spec 341](surface-341-analytic-to-b-spline-csg-intersections-v1_0.md)
- [Specification: Surface Spec 342](surface-342-analytic-to-nurbs-csg-intersections-v1_0.md)

## Scope

This parent covers:

- analytic-to-B-spline body-route evidence completion
- analytic-to-NURBS body-route evidence completion

This parent does not cover spline-pair route evidence, coincident-region
evidence, or the final collector/readiness gate.

## Child Specifications

- [Surface Spec 402B1: Analytic To B-Spline Body CSG Evidence Completion](surface-402b1-analytic-to-b-spline-body-csg-evidence-completion-v1_0.md)
- [Surface Spec 402B2: Analytic To NURBS Body CSG Evidence Completion](surface-402b2-analytic-to-nurbs-body-csg-evidence-completion-v1_0.md)

## Five-Pass Review History

- Pass 1 - Scope Completeness: Found the original 402B bundled analytic/B-spline and analytic/NURBS repairs.
- Pass 2 - Dependency Check: Both children depend on 402A but not on each other.
- Pass 3 - Rescore: Rescored as a 2 IWU branch rollup.
- Pass 4 - Split: Split into 402B1 and 402B2 final leaves.
- Pass 5 - Final Review: Parent remains a branch specification and must not appear as a progression implementation item.

## Refinement Status

Parent branch specification. Not an implementation leaf.

## Acceptance

This parent is complete when 402B1 and 402B2 are complete.
