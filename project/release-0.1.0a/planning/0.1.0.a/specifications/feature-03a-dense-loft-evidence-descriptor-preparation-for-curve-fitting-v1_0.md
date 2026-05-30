# Feature Spec 03A: Dense Loft Evidence Descriptor Preparation for Curve Fitting (v1.0)

## Overview

This specification defines the descriptor-preparation lane for fit-backed loft
analysis.

## Backlink

- [Feature Spec 03: Curve Fitting From Dense Loft Evidence Program (v1.0)](feature-03-curve-fitting-from-dense-loft-evidence-program-v1_0.md)

## Scope

This specification covers:

- station-derived descriptor extraction
- descriptor normalization for fitting
- ordering and continuity preservation across dense evidence

## Behavior

This leaf must define:

- which dense loft descriptors are prepared for candidate curve fitting
- how descriptor extraction stays deterministic
- how descriptor bands preserve enough continuity for later fitting

## Constraints

- descriptor extraction must not destroy ordering or correspondence meaning
- descriptor preparation must remain replayable for identical inputs

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- descriptor preparation inputs and outputs are explicit
- ordering and continuity behavior are explicit
- the prepared evidence is durable enough for candidate comparison
