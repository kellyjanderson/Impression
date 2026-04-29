# Surface Spec 136 Test: Surface Boolean Initial Executable Scope and Unsupported-Case Matrix

## Overview

This test specification defines verification for the initial executable surfaced
boolean scope and its unsupported-case matrix.

## Backlink

- [Surface Spec 136: Surface Boolean Initial Executable Scope and Unsupported-Case Matrix (v1.0)](../specifications/surface-136-surface-boolean-initial-executable-scope-and-unsupported-case-matrix-v1_0.md)

## Automated Smoke Tests

- every initially supported surfaced boolean case executes without mesh fallback
- every explicitly unsupported case returns structured surfaced unsupported status

## Automated Acceptance Tests

- only the explicitly in-scope surfaced boolean operand classes execute successfully
- out-of-scope operand families, shell structures, or trim complexity remain structured unsupported results
- the documented initial scope remains aligned with the bounded execution leaves under this branch
