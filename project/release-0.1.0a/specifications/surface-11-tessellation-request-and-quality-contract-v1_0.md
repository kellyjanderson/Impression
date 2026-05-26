# Surface Spec 11: Tessellation Request and Quality Contract (v1.0)

## Overview

This specification defines the branch responsible for how consumers request
tessellation from a surface body and how those requests express quality,
tolerance, and output intent.

## Backlink

Parent specification:

- [Surface Spec 03: Tessellation Boundary and Rendering Contract (v1.0)](surface-03-tessellation-boundary-v1_0.md)

## Scope

This specification covers:

- tessellation request inputs
- quality-facing versus tolerance-facing controls
- deterministic request normalization
- the minimum output-shape guarantees attached to a tessellation request

## Behavior

This branch must define:

- what a tessellation request object is
- how callers specify preview/export/analysis intent
- how quality presets interact with explicit tolerances
- what normalized tessellation parameters downstream executors receive

## Constraints

- request normalization must be deterministic
- quality controls must not alter modeled meaning
- tolerance controls must be explicit rather than hidden in consumer code
- the request contract must be stable enough for multiple downstream clients

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 32: Tessellation Request Object and Field Contract (v1.0)](surface-32-tessellation-request-object-contract-v1_0.md)
- [Surface Spec 33: Quality Presets and Explicit Tolerance Normalization (v1.0)](surface-33-quality-preset-tolerance-normalization-v1_0.md)
- [Surface Spec 34: Canonical Executor Input and Request Normalization (v1.0)](surface-34-executor-input-request-canonicalization-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- tessellation request structure is explicit
- quality and tolerance controls are bounded clearly
- downstream tessellation code can consume normalized requests without guessing
