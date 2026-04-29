# Feature Spec 03B: Station-Derived Candidate Curve-Fit Generation, Comparison, and Refusal Posture (v1.0)

## Overview

This specification defines the station-derived candidate-fit and comparison
behavior for dense loft evidence.

## Backlink

- [Feature Spec 03: Curve Fitting From Dense Loft Evidence Program (v1.0)](feature-03-curve-fitting-from-dense-loft-evidence-program-v1_0.md)

## Scope

This specification covers:

- station-derived candidate fitted-curve generation
- station-derived candidate comparison
- residual-based acceptance or refusal

## Behavior

This leaf must define:

- how station-derived candidate fits are produced from prepared evidence
- how station-derived candidates are compared
- how refusal is reported when no station-derived candidate is trustworthy
  enough

## Constraints

- comparison must reference explicit residual diagnostics
- refusal must stay explicit rather than silently picking a weak candidate

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- candidate generation behavior is explicit
- comparison posture is explicit
- refusal behavior is explicit
