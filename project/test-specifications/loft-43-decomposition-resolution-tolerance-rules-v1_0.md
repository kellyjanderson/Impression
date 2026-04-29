# Loft Spec 43 Test: Decomposition-Resolution Tolerance Rules

## Overview

This test specification defines verification for decomposition-resolution
controls and ambiguity escalation.

## Backlink

- [Loft Spec 43: Decomposition-Resolution Tolerance Rules (v1.0)](../specifications/loft-43-decomposition-resolution-tolerance-rules-v1_0.md)

## Automated Smoke Tests

- split/merge and ambiguity controls remain explicit

## Automated Acceptance Tests

- branch-budget exhaustion raises structured blocked-planning diagnostics
- auto-resolved ambiguous transitions remain executable
- decomposition controls are deterministic and not hidden heuristics
