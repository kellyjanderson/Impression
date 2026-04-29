# Loft Spec 48 Test: Automatic Decomposability Gate Rules

## Overview

This test specification defines verification for the automatic decomposability
gate in many-to-many loft planning.

## Backlink

- [Loft Spec 48: Automatic Decomposability Gate Rules (v1.0)](../specifications/loft-48-automatic-decomposability-gate-rules-v1_0.md)

## Automated Smoke Tests

- successful many-to-many plans expose a non-blocking decomposability state

## Automated Acceptance Tests

- automatic consumption remains explicit while deterministic reduction continues
- executable plans do not retain a reached decomposability gate
- blocked many-to-many states surface through the planning error path instead of the executor
