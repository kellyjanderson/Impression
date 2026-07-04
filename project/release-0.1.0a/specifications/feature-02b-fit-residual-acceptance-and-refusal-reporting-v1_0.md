# Feature Spec 02B: Fit Residual, Acceptance, and Refusal Reporting (v1.0)

## Overview

This specification defines how fit-backed workflows report residuals and
decision outcomes.

## Backlink

- [Feature Spec 02: Explicit Fit Policy and Diagnostics Program (v1.0)](feature-02-explicit-fit-policy-and-diagnostics-program-v1_0.md)

## Scope

This specification covers:

- residual reports
- acceptance reports
- refusal reports
- exact-vs-approximate fit posture where relevant

## Behavior

This leaf must define:

- how fit drift is measured and reported
- how acceptance and refusal are expressed durably
- how later inference branches reuse those reports

## Constraints

- refusal must be explicit rather than silently degrading
- residual reporting must remain durable enough for replay and comparison

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- residual and acceptance reporting are explicit
- refusal posture is explicit
- later inference consumers can rely on the report contract
