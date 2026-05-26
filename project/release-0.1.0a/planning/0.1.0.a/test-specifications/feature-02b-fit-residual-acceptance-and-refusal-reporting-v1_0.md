# Feature Spec 02B Test: Fit Residual, Acceptance, and Refusal Reporting

## Overview

This test specification defines verification for residual, acceptance, and
refusal reporting in fit-backed workflows.

## Backlink

- [Feature Spec 02B: Fit Residual, Acceptance, and Refusal Reporting (v1.0)](../specifications/feature-02b-fit-residual-acceptance-and-refusal-reporting-v1_0.md)

## Automated Smoke Tests

- representative fits emit residual reports and a decision outcome
- refusal remains inspectable as a first-class outcome

## Automated Acceptance Tests

- fit drift is reported using durable metrics
- acceptance and refusal remain distinguishable and replayable
- weak fits are not silently promoted to accepted outputs
