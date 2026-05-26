# Loft Spec 37 Test: Planner / Executor Execution-Boundary Rules

## Overview

This test specification defines verification for the exact boundary between
planning and execution.

## Backlink

- [Loft Spec 37: Planner / Executor Execution-Boundary Rules (v1.0)](../specifications/loft-37-planner-executor-execution-boundary-rules-v1_0.md)

## Automated Smoke Tests

- executable plans pass `require_executable()`
- execution only uses validated plan input

## Automated Acceptance Tests

- unresolved planning states block execution before executor fallback
- executor assumptions are explicit and binary
