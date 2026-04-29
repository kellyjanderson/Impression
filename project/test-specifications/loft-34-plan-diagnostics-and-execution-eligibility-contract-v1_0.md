# Loft Spec 34 Test: Plan Diagnostics and Execution Eligibility Contract

## Overview

This test specification defines verification for plan diagnostics and execution
eligibility surfaces.

## Backlink

- [Loft Spec 34: Plan Diagnostics and Execution Eligibility Contract (v1.0)](../specifications/loft-34-plan-diagnostics-and-execution-eligibility-contract-v1_0.md)

## Automated Smoke Tests

- interval execution eligibility is explicit
- whole-plan blocking status is explicit

## Automated Acceptance Tests

- diagnostic metadata is accessible without executor reinterpretation
- executable and blocked states are distinguishable at the plan boundary
- plan-layer blocking status is deterministic
