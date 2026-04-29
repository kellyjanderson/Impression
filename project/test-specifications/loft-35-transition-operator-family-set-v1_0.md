# Loft Spec 35 Test: Transition Operator Family Set

## Overview

This test specification defines verification for the explicit operator-family
set at the planner/executor boundary.

## Backlink

- [Loft Spec 35: Transition Operator Family Set (v1.0)](../specifications/loft-35-transition-operator-family-set-v1_0.md)

## Automated Smoke Tests

- resolved region pairs expose explicit operator families
- interval-level operator-family views are explicit

## Automated Acceptance Tests

- continuity, split, and merge families are explicit
- operator families are not guessed by the executor
