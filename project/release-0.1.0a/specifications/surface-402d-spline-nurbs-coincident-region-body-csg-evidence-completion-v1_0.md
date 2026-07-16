# Surface Spec 402D: Spline/NURBS Coincident Region Body CSG Evidence Completion (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: this leaf owns coincident-region and overlap-boundary evidence for spline/NURBS patch pairs through one shared overlap route.

## Overview

Complete body-route-consumable coincident-region evidence for B-spline/NURBS
CSG overlap cases.

## Backlinks

- [Parent: Surface Spec 402](surface-402-b-spline-nurbs-patch-evidence-completion-for-body-csg-v1_0.md)
- [Prerequisite: Surface Spec 402A](surface-402a-b-spline-nurbs-body-csg-evidence-contract-and-prerequisite-audit-v1_0.md)
- [Specification: Surface Spec 344](surface-344-spline-and-nurbs-coincident-region-csg-intersections-v1_0.md)

## Scope

This specification covers:

- normalizing `detect_spline_nurbs_coincident_regions` output to the 402A evidence contract
- identical patch overlap
- partial overlap
- reversed orientation
- near-coincident refusal
- ownership ambiguity diagnostics
- overlap-boundary loops and trim-readiness metadata

This specification does not cover non-coincident curve intersections. Those are
owned by 402C.

## Behavior

The implementation must:

- avoid silently promoting partial overlap to full-domain coincidence
- preserve loop orientation and source patch identities
- emit overlap-boundary loops that can be consumed by trim arrangement when
  ownership is resolved
- refuse with tolerance metadata when near-coincident samples exceed the policy
- refuse with ownership diagnostics when fragment ownership is ambiguous
- include no-hidden-mesh-fallback evidence on every path

## Verification

Focused tests must cover:

- identical B-spline/B-spline, B-spline/NURBS, and NURBS/NURBS overlap
- partial overlap loop emission
- reversed-orientation loop metadata
- near-coincident refusal with sampled distance and tolerance
- ownership ambiguity refusal
- normalized evidence payload fields required by 402A

## Readiness And Sequencing

Blocked by Surface Spec 402A. This spec must complete before Surface Spec 402E.

## Five-Pass Review History

- Pass 1 - Scope Completeness: Limited to coincident-region evidence and ownership.
- Pass 2 - Dependency Check: Depends on 402A contract only.
- Pass 3 - Rescore: Count remains 1 IWU; one overlap-region evidence outcome.
- Pass 4 - Split: Non-coincident curve evidence remains split into 402C.
- Pass 5 - Final Review: Final leaf ready after 402A.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when spline/NURBS coincident-region routes
produce 402A-compliant success and refusal evidence for full, partial, reversed,
near-coincident, and ambiguous overlap cases without mesh fallback.
