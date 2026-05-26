# Feature 02 — Explicit Fit Policy And Diagnostics Architecture

## Status

`0.1.0.a` feature-path architecture branch.

This document corresponds directly to Feature `2` in
[0.1.0.a Feature List](../feature-list.md).

Related architecture:

- [Priority 02 — Parameterization, Knot, And Fit Policy Architecture](priority-02-parameterization-knot-and-fit-policy-architecture.md)
- [Feature 03 — Curve Fitting From Dense Loft Evidence Architecture](feature-03-curve-fitting-from-dense-loft-evidence-architecture.md)

## Purpose

Define the owned fitting-policy and diagnostic layer that makes inference
trustworthy and replayable.

## Included Scope

- parameterization policy
- knot-count policy
- knot-placement policy
- fit configuration record
- residual report
- acceptance or refusal report
- exact-vs-approximate fit posture where relevant

## Core Rule

Fitting decisions must become explicit project truth rather than hidden helper
behavior.

This feature exists so that later inference branches can answer:

- what policy was used
- why this fit was chosen
- where it drifted
- why a reduction or trajectory candidate was accepted or refused

## System Role

```text
sampled geometric evidence
-> explicit fit policy
-> B-spline fitting or equivalent
-> residual and acceptance diagnostics
-> downstream inference consumer
```

## Consumers

- control-station inference
- curve-intent inference
- trajectory inference
- progression model upgrade
- explainability and diagnostics

## Architectural Conclusion

Feature `02` is where fitting becomes owned and inspectable. It is the contract
that keeps all later inference branches deterministic and debuggable.
