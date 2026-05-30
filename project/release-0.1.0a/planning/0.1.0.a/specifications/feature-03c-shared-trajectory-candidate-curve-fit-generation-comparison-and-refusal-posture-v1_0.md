# Feature Spec 03C: Shared-Trajectory Candidate Curve-Fit Generation, Comparison, and Refusal Posture (v1.0)

## Overview

This specification defines the shared-trajectory candidate-fit and comparison
behavior for dense loft evidence.

## Backlink

- [Feature Spec 03: Curve Fitting From Dense Loft Evidence Program (v1.0)](feature-03-curve-fitting-from-dense-loft-evidence-program-v1_0.md)

## Scope

This specification covers:

- shared-trajectory candidate fitted-curve generation
- shared-trajectory candidate comparison
- residual-based acceptance or refusal for the shared-trajectory lane

## Behavior

This leaf must define:

- how shared-trajectory candidate fits are produced from prepared evidence
- how shared-trajectory candidates are compared
- how refusal is reported when no shared-trajectory candidate is trustworthy
  enough

## Constraints

- comparison must reference explicit residual diagnostics
- refusal must stay explicit rather than silently inferring a weak trajectory

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- shared-trajectory candidate generation behavior is explicit
- comparison posture is explicit
- refusal behavior is explicit
