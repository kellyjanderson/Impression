# Testing Spec 08: Computer Vision Diagnostic Triptych and Panel-Region Presentation (v1.0)

## Overview

This specification defines the parent branch for architecture-level diagnostic
presentation tooling for deterministic triptych and multi-panel artifacts such
as operand/result panels and expected/actual/diff groups.

## Backlink

- [Testing Spec 01: Testing Tooling and Verification Program (v1.0)](testing-01-testing-tooling-and-verification-program-v1_0.md)

## Scope

- deterministic multi-panel layout and ordering
- panel-region extraction, cropping, and labeling rules
- diagnostic presentation posture for triptych and grouped review artifacts
- the tooling contract for panel segmentation and presentation

## Behavior

This branch must define:

- the panel-order and panel-label contract for deterministic review layouts
- the tooling boundary between authoritative verification products and
  diagnostic panel presentation
- how panel regions are extracted or cropped reproducibly
- the boundary between diagnostic presentation and authoritative proof lanes

## Constraints

- diagnostic panels must not be mistaken for authoritative semantic proof unless
  another leaf delegates a narrow proof use explicitly
- panel layout and extraction must be deterministic
- grouped diagnostic artifacts must remain interpretable by humans during
  failure review
- this leaf defines presentation tooling rather than feature-level geometry
  truth

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Testing Spec 22: Computer Vision Diagnostic Panel Layout, Ordering, and Region Extraction Contract](testing-22-computer-vision-diagnostic-panel-layout-ordering-and-region-extraction-contract-v1_0.md)
- [Testing Spec 23: Computer Vision Diagnostic Panel Honesty Boundary and Proof Delegation Rules](testing-23-computer-vision-diagnostic-panel-honesty-boundary-and-proof-delegation-rules-v1_0.md)

## Acceptance

This specification is complete when:

- diagnostic panel layout/extraction and honesty-boundary work are separated
  into honest executable child leaves
- the parent remains a container rather than an executable implementation leaf
- verification requirements are pushed down into the paired child test
  specifications
